"""
bithuman_expression/src/pipeline/expression_pipeline.py
Python replacement for expression_pipeline.cpython-310-x86_64-linux-gnu.so

Orchestrates the full expression-avatar inference pipeline:
  1. Load ExpressionModel + Wav2Vec2 audio encoder from model_dir
  2. preprocess_audio()  — raw waveform → Wav2Vec2 features
  3. prepare_params()    — build scheduler / conditioning tensors
  4. generate()          — flow-matching denoising loop → video frames

Classes:
  ExpressionPipeline

Free functions:
  timestep_transform(t, ...)  — map [0,1] timestep to scheduler scale
  get_cond_image_dict(...)    — build conditioning image dictionary
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def timestep_transform(
    t: torch.Tensor,
    shift: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Transform a [0, 1] timestep to the scheduler's internal scale.

    Uses a shifted logit-normal reparameterisation common in flow-matching:
        t' = sigmoid(shift + scale * logit(t))

    Args:
        t:     (B,) float tensor in (0, 1)
        shift: additive shift before sigmoid
        scale: multiplicative scale

    Returns:
        transformed timestep, same shape as t
    """
    t = t.clamp(1e-6, 1 - 1e-6)
    logit_t = torch.log(t / (1.0 - t))
    return torch.sigmoid(shift + scale * logit_t)


def get_cond_image_dict(
    embeddings: torch.Tensor,
    videos: torch.Tensor,
    model_dir: Union[str, Path, None] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build the conditioning dictionary used by ExpressionPipeline.generate().

    Args:
        embeddings: (B, D) face/identity embedding tensor
        videos:     (B, C, T, H, W) reference video frames
        model_dir:  optional path for loading auxiliary assets

    Returns:
        dict with keys 'embeddings', 'ref_image', and 'ref_frames'
    """

    def get_image(video: torch.Tensor) -> torch.Tensor:
        """Extract the first frame from a (C, T, H, W) video."""
        return video[:, 0:1, :, :]   # (C, 1, H, W)

    ref_images = torch.stack([get_image(v) for v in videos], dim=0)  # (B, C, 1, H, W)

    return {
        "embeddings": embeddings,
        "ref_image":  ref_images,
        "ref_frames": videos,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ExpressionPipeline:
    """
    End-to-end expression-avatar generation pipeline.

    Initialises the image-to-video generation model components.

    Args:
        model_dir: path to the model weights directory (contains config.json,
                   expression_model weights, wav2vec2 checkpoint, etc.)
    """

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self._person_name: Optional[str] = None
        self._model: Optional[torch.nn.Module] = None
        self._audio_model: Optional[torch.nn.Module] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_models(self):
        """Load ExpressionModel and Wav2Vec2 audio encoder from model_dir."""
        import json

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            logger.warning(f"[ExpressionPipeline] config.json not found at {config_path}")
            return

        with open(config_path) as f:
            cfg = json.load(f)

        logger.info(f"[ExpressionPipeline] Loading models from {self.model_dir}")

        # Load expression model
        try:
            from bithuman_expression.src.modules.expression_model import ExpressionModel
            model_cfg = cfg.get("expression_model", {})
            self._model = ExpressionModel(
                in_dim     = model_cfg.get("in_dim",     512),
                out_dim    = model_cfg.get("out_dim",    512),
                freq_dim   = model_cfg.get("freq_dim",   256),
                ffn_dim    = model_cfg.get("ffn_dim",    2048),
                num_heads  = model_cfg.get("num_heads",  8),
                num_layers = model_cfg.get("num_layers", 12),
                text_dim   = model_cfg.get("text_dim",   768),
            ).to(self._device)

            ckpt = self.model_dir / "expression_model.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self._device)
                self._model.load_state_dict(state, strict=False)
                logger.info("[ExpressionPipeline] Loaded expression model weights")

            self._model.eval()
        except Exception as e:
            logger.error(f"[ExpressionPipeline] Failed to load ExpressionModel: {e}")

        # Load Wav2Vec2 audio encoder
        try:
            from bithuman_expression.audio_analysis.wav2vec2 import (
                Wav2Vec2Config,
                Wav2Vec2Model,
            )
            audio_cfg = cfg.get("wav2vec2", {})
            wav_config = Wav2Vec2Config(**audio_cfg) if audio_cfg else Wav2Vec2Config()
            self._audio_model = Wav2Vec2Model(wav_config).to(self._device)

            wav_ckpt = self.model_dir / "wav2vec2.pt"
            if wav_ckpt.exists():
                state = torch.load(wav_ckpt, map_location=self._device)
                self._audio_model.load_state_dict(state, strict=False)
                logger.info("[ExpressionPipeline] Loaded Wav2Vec2 weights")

            self._audio_model.eval()
        except Exception as e:
            logger.error(f"[ExpressionPipeline] Failed to load Wav2Vec2: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_person_name(self, person_name: str):
        """
        Reset the current avatar identity (clears any cached state).

        Args:
            person_name: identifier string for the avatar
        """
        self._person_name = person_name
        logger.info(f"[ExpressionPipeline] Reset to person: {person_name}")

    def preprocess_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        num_frames: int = 25,
    ) -> torch.Tensor:
        """
        Convert raw audio waveform to Wav2Vec2 feature vectors.

        Args:
            audio:       (B, T_audio) or (T_audio,) float waveform at sample_rate
            sample_rate: audio sample rate in Hz (model expects 16 kHz)
            num_frames:  target number of output feature frames (matches video fps)

        Returns:
            (B, num_frames, hidden_size) audio feature tensor
        """
        if self._audio_model is None:
            raise RuntimeError("Audio model not loaded. Check model_dir.")

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self._device)
        with torch.no_grad():
            features = self._audio_model(
                input_values=audio,
                seq_len=num_frames,
            ).last_hidden_state

        return features  # (B, num_frames, hidden_size)

    def prepare_params(
        self,
        cond_image_dict: Dict[str, torch.Tensor],
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Prepare denoising scheduler parameters and conditioning tensors.

        Args:
            cond_image_dict:      output of get_cond_image_dict()
            num_inference_steps:  number of flow-matching steps
            guidance_scale:       classifier-free guidance scale
            seed:                 RNG seed for reproducibility

        Returns:
            params dict consumed by generate()
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Build timestep schedule: linearly spaced in (0, 1]
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

        return {
            "cond_image_dict":    cond_image_dict,
            "timesteps":          timesteps,
            "guidance_scale":     guidance_scale,
            "num_inference_steps": num_inference_steps,
        }

    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        params: Dict,
        latent_shape: Optional[Tuple[int, ...]] = None,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ) -> torch.Tensor:
        """
        Run the flow-matching denoising loop to generate video latents.

        Args:
            audio_features: (B, L_audio, text_dim) from preprocess_audio()
            params:         dict from prepare_params()
            latent_shape:   (B, C, T, H, W) shape of the output latent;
                            inferred from cond_image_dict if None
            patch_size:     patchification stride

        Returns:
            (B, C, T, H, W) denoised latent tensor
        """
        if self._model is None:
            raise RuntimeError("Expression model not loaded. Check model_dir.")

        timesteps      = params["timesteps"].to(self._device)
        guidance_scale = params["guidance_scale"]
        cond_dict      = params["cond_image_dict"]

        # Infer latent shape from reference image if not provided
        if latent_shape is None:
            ref = cond_dict["ref_image"]           # (B, C, 1, H, W)
            B, C, _, H, W = ref.shape
            T = audio_features.shape[1]
            latent_shape = (B, C, T, H, W)

        # Start from pure noise
        x = torch.randn(*latent_shape, device=self._device, dtype=audio_features.dtype)

        audio_features = audio_features.to(self._device)

        for i, t_val in enumerate(timesteps):
            t = t_val.expand(latent_shape[0]).to(self._device)

            # Conditional prediction
            v_cond = self._model(x, t, audio_features, patch_size)

            if guidance_scale != 1.0:
                # Unconditional prediction (zero audio context)
                null_audio = torch.zeros_like(audio_features)
                v_uncond = self._model(x, t, null_audio, patch_size)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            # Euler step
            dt = (timesteps[i + 1] - t_val) if i + 1 < len(timesteps) else -t_val
            x = x + dt * v

        return x
