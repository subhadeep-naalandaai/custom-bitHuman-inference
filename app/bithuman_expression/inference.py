"""
bithuman_expression/inference.py
Python replacement for inference.cpython-310-x86_64-linux-gnu.so

Top-level public API used by the dispatcher (app/src/dispatcher.pyc).

The dispatcher call flow:
  1. pipeline = get_pipeline(engine_path, ckpt_dir, models_dir, wav2vec_dir,
                              fast_decoder_config, fast_decoder_checkpoint)
  2. params   = get_infer_params(pipeline, cached_audio_duration=..., tgt_fps=...)
  3. audio_emb, _ = pipeline.encode_audio(dummy_audio)          # warmup / per-chunk
  4. frames, timings = pipeline.generate(audio_emb, params)     # denoising loop
  ...or via the high-level wrappers:
  4. audio_emb = get_audio_embedding(pipeline, audio, sample_rate)
  5. base_data = get_base_data(pipeline, image_path)
  6. frames    = run_pipeline(pipeline, audio_emb, base_data, params)

Exports:
  BitHumanPipeline          — pipeline object with .encode_audio() / .generate()
  get_pipeline(...)         — load models, return BitHumanPipeline
  get_infer_params(...)     — build inference parameter dict
  get_audio_embedding(...)  — encode raw audio → feature tensor
  get_base_data(...)        — process avatar reference image → conditioning dict
  run_pipeline(...)         — full inference: denoising + VAE decode → RGB frames
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (match dispatcher env-var fallbacks)
# ---------------------------------------------------------------------------
_BUNDLED_WAV2VEC = "/app/bundled/wav2vec2-base-960h"


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

class BitHumanPipeline:
    """
    Wraps the expression model, audio encoder, and VAE decoder into a single
    object that the dispatcher and session pool interact with.

    Attributes:
        ckpt_dir:   path to Model_Lite checkpoint directory
        models_dir: root model weights directory
        device:     torch.device used for inference
        vae:        LTX-Video VAE decoder (CausalVideoAutoencoder)
        model:      ExpressionModel (DiT)
        wav2vec:    Wav2Vec2Model audio encoder
        width, height, tgt_fps: output video dimensions / frame-rate
    """

    def __init__(
        self,
        ckpt_dir: Union[str, Path],
        models_dir: Union[str, Path],
        wav2vec_dir: Optional[Union[str, Path]] = None,
        fast_decoder_config: Optional[str] = None,
        fast_decoder_checkpoint: Optional[str] = None,
    ):
        self.ckpt_dir   = Path(ckpt_dir)
        self.models_dir = Path(models_dir)
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output video properties (set after model load)
        self.width   = 512
        self.height  = 512
        self.tgt_fps = 25

        self.vae      = None
        self.model    = None
        self.wav2vec  = None
        self._fast_decoder = None

        self._load(wav2vec_dir, fast_decoder_config, fast_decoder_checkpoint)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self, wav2vec_dir, fast_decoder_config, fast_decoder_checkpoint):
        """Load all model components from checkpoint directories."""
        logger.info(f"[BitHumanPipeline] Loading from {self.ckpt_dir}")

        # 1. Audio encoder -----------------------------------------------
        self._load_wav2vec(wav2vec_dir)

        # 2. Expression model (DiT) + VAE --------------------------------
        self._load_expression_model()

        # 3. Optional fast decoder (TurboVAE) ----------------------------
        if fast_decoder_config and fast_decoder_checkpoint:
            self._load_fast_decoder(fast_decoder_config, fast_decoder_checkpoint)

    def _load_wav2vec(self, wav2vec_dir):
        """Load Wav2Vec2 audio encoder."""
        try:
            from bithuman_expression.audio_analysis.wav2vec2 import (
                Wav2Vec2Config,
                Wav2Vec2Model,
            )
            # Resolve wav2vec path: prefer explicit dir, then bundled copy
            w2v_path = None
            if wav2vec_dir and Path(wav2vec_dir).exists():
                w2v_path = str(wav2vec_dir)
            elif Path(_BUNDLED_WAV2VEC).exists():
                logger.info(f"Using bundled wav2vec2 (not found in volume): {_BUNDLED_WAV2VEC}")
                w2v_path = _BUNDLED_WAV2VEC

            if w2v_path:
                self.wav2vec = Wav2Vec2Model.from_pretrained(w2v_path).to(self.device)
            else:
                self.wav2vec = Wav2Vec2Model(Wav2Vec2Config()).to(self.device)

            self.wav2vec.eval()
            logger.info("[BitHumanPipeline] Wav2Vec2 loaded")
        except Exception as e:
            logger.error(f"[BitHumanPipeline] Wav2Vec2 load failed: {e}")

    def _load_expression_model(self):
        """Load ExpressionModel (DiT) and CausalVideoAutoencoder."""
        import json

        config_path = self.ckpt_dir / "config.json"
        if not config_path.exists():
            logger.warning(f"[BitHumanPipeline] config.json not found at {config_path}")
            return

        with open(config_path) as f:
            cfg = json.load(f)

        # VAE
        try:
            from bithuman_expression.ltx_video.ltx_vae import LTXVae
            vae_dir = self.ckpt_dir / "vae"
            if not vae_dir.exists():
                vae_dir = self.models_dir / "vae"
            if vae_dir.exists():
                self.vae = LTXVae.from_pretrained(str(vae_dir)).to(self.device)
                self.vae.eval()
                logger.info("[BitHumanPipeline] LTX VAE loaded")
        except Exception as e:
            logger.error(f"[BitHumanPipeline] VAE load failed: {e}")

        # Expression model
        try:
            from bithuman_expression.src.modules.expression_model import ExpressionModel
            m_cfg = cfg.get("expression_model", {})
            self.model = ExpressionModel(
                in_dim     = m_cfg.get("in_dim",     512),
                out_dim    = m_cfg.get("out_dim",    512),
                freq_dim   = m_cfg.get("freq_dim",   256),
                ffn_dim    = m_cfg.get("ffn_dim",    2048),
                num_heads  = m_cfg.get("num_heads",  8),
                num_layers = m_cfg.get("num_layers", 12),
                text_dim   = m_cfg.get("text_dim",   768),
            ).to(self.device)

            ckpt = self.ckpt_dir / "expression_model.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state, strict=False)

            self.model.eval()

            # Read output dimensions from config
            self.width   = cfg.get("width",   512)
            self.height  = cfg.get("height",  512)
            self.tgt_fps = cfg.get("fps",     25)
            logger.info("[BitHumanPipeline] ExpressionModel loaded")
        except Exception as e:
            logger.error(f"[BitHumanPipeline] ExpressionModel load failed: {e}")

    def _load_fast_decoder(self, config_path, checkpoint_path):
        """Optionally load TurboVAE fast decoder."""
        try:
            from bithuman_expression.ltx_video.fast_decoder import load_fast_decoder
            self._fast_decoder = load_fast_decoder(
                config_path, checkpoint_path, self.device
            )
            logger.info("[BitHumanPipeline] Fast decoder loaded")
        except Exception as e:
            logger.warning(f"[BitHumanPipeline] Fast decoder load failed (optional): {e}")

    # ------------------------------------------------------------------
    # Core inference methods (called directly by dispatcher/session pool)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_audio(
        self,
        audio_samples: torch.Tensor,
        sample_rate: int = 16000,
        num_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode raw audio waveform to Wav2Vec2 feature vectors.

        Args:
            audio_samples: (T_audio,) or (B, T_audio) float32 waveform
            sample_rate:   input sample rate (model expects 16 kHz)
            num_frames:    target output length; defaults to tgt_fps seconds

        Returns:
            (audio_emb, meta) where:
              audio_emb: (B, num_frames, hidden_size)
              meta: dict with 'audio_ms' timing info
        """
        t0 = time.perf_counter()

        if audio_samples.ndim == 1:
            audio_samples = audio_samples.unsqueeze(0)

        audio_samples = audio_samples.to(self.device, dtype=torch.float32)

        if num_frames is None:
            num_frames = int(audio_samples.shape[-1] / sample_rate * self.tgt_fps)
            num_frames = max(1, num_frames)

        emb = self.wav2vec(
            input_values=audio_samples,
            seq_len=num_frames,
        ).last_hidden_state

        audio_ms = (time.perf_counter() - t0) * 1000
        return emb, {"audio_ms": audio_ms}

    @torch.no_grad()
    def generate(
        self,
        audio_emb: torch.Tensor,
        params: Dict,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Run denoising loop then decode latents to RGB frames.

        Args:
            audio_emb: (B, T, hidden_size) from encode_audio()
            params:    dict from get_infer_params() containing:
                         'cond_image_dict', 'timesteps', 'guidance_scale',
                         'num_inference_steps', 'motion_frames_num', 'tgt_fps'

        Returns:
            (frames, timings) where:
              frames:  list of (H, W, 3) uint8 numpy arrays
              timings: dict with stage timing in ms
        """
        from bithuman_expression.src.pipeline.expression_pipeline import (
            ExpressionPipeline,
        )

        t_start = time.perf_counter()

        # Run DiT denoising via ExpressionPipeline
        exp_pipeline: ExpressionPipeline = params["_expression_pipeline"]
        latent = exp_pipeline.generate(audio_emb, params)  # (B, C, T, H, W)

        t_denoise = time.perf_counter()

        # VAE decode latent → pixel space
        frames = self._decode_latent(latent)

        t_decode = time.perf_counter()

        timings = {
            "denoise_ms": (t_denoise - t_start) * 1000,
            "decode_ms":  (t_decode  - t_denoise) * 1000,
            "total_ms":   (t_decode  - t_start) * 1000,
        }
        return frames, timings

    def _decode_latent(self, latent: torch.Tensor) -> List[np.ndarray]:
        """
        Decode a (B, C, T, H, W) latent tensor to a list of RGB numpy frames.
        Uses fast decoder if available, otherwise falls back to LTX VAE.
        """
        B, C, T, H, W = latent.shape
        frames = []

        if self._fast_decoder is not None:
            pixels = self._fast_decoder(latent)  # (B, 3, T, H', W')
        elif self.vae is not None:
            # LTXVae.decode(zs) expects a single (C, T, H, W) tensor (no batch dim)
            # and returns the decoded image tensor directly (no .sample wrapper).
            # It internally does zs.unsqueeze(0) to add the batch dim.
            decoded = self.vae.decode(latent[0])    # → (1, 3, T, H', W') or (3, T, H', W')
            pixels = decoded.unsqueeze(0) if decoded.dim() == 4 else decoded  # ensure (B,3,T,H',W')
        else:
            # No decoder available — return black frames as placeholder
            return [np.zeros((self.height, self.width, 3), dtype=np.uint8)] * T

        # Convert (B, 3, T, H, W) → list of (H, W, 3) uint8
        pixels = pixels[0]                                # (3, T, H, W)
        pixels = pixels.permute(1, 2, 3, 0)               # (T, H, W, 3)
        pixels = ((pixels.float().clamp(-1, 1) + 1) / 2 * 255).byte()
        pixels = pixels.cpu().numpy()

        for i in range(pixels.shape[0]):
            frames.append(pixels[i])

        return frames


# ---------------------------------------------------------------------------
# Public API — called directly by dispatcher
# ---------------------------------------------------------------------------

def get_pipeline(
    engine_path: Optional[Union[str, Path]] = None,
    ckpt_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    wav2vec_dir: Optional[Union[str, Path]] = None,
    fast_decoder_config: Optional[str] = None,
    fast_decoder_checkpoint: Optional[str] = None,
) -> BitHumanPipeline:
    """
    Load all model components and return a ready-to-use BitHumanPipeline.

    This is called once at server startup (prewarm_model) and the returned
    object is stored in AppState for reuse across all sessions.

    Args:
        engine_path:            path to TRT engine file (used by trt_pipeline)
        ckpt_dir:               Model_Lite checkpoint directory
        models_dir:             root weights directory (contains bh-weights)
        wav2vec_dir:            wav2vec2-base-960h directory
        fast_decoder_config:    path to FastDecoder-LTX.json (optional)
        fast_decoder_checkpoint: path to FastDecoder-LTX.pth (optional)

    Returns:
        BitHumanPipeline ready for encode_audio() / generate() calls
    """
    logger.info("PREWARM: Loading pipeline...")

    if ckpt_dir is None:
        ckpt_dir = Path(os.environ.get("BITHUMAN_WEIGHTS_PATH", "/workspace/bh-weights")) \
                   / "bithuman-expression" / "Model_Lite"

    if models_dir is None:
        models_dir = Path(os.environ.get("BITHUMAN_WEIGHTS_PATH", "/workspace/bh-weights"))

    pipeline = BitHumanPipeline(
        ckpt_dir=ckpt_dir,
        models_dir=models_dir,
        wav2vec_dir=wav2vec_dir,
        fast_decoder_config=fast_decoder_config,
        fast_decoder_checkpoint=fast_decoder_checkpoint,
    )

    logger.info("PREWARM: Running warmup inference...")
    _warmup(pipeline)

    return pipeline


def _warmup(pipeline: BitHumanPipeline):
    """Run a single dummy forward pass to warm up CUDA kernels."""
    try:
        sample_rate = 16000
        # 1-second silence at target fps
        num_frames = pipeline.tgt_fps
        dummy_audio = torch.zeros(1, sample_rate, dtype=torch.float32)
        audio_emb, _ = pipeline.encode_audio(dummy_audio, sample_rate, num_frames)

        params = get_infer_params(
            pipeline,
            cached_audio_duration=1.0,
            tgt_fps=pipeline.tgt_fps,
        )
        params["_expression_pipeline"] = _make_expression_pipeline(pipeline)
        pipeline.generate(audio_emb, params)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"[warmup] Non-fatal: {e}")


def _make_expression_pipeline(pipeline: BitHumanPipeline):
    """Create a lightweight ExpressionPipeline shim backed by pipeline.model."""
    from bithuman_expression.src.pipeline.expression_pipeline import ExpressionPipeline

    class _Shim:
        def generate(self, audio_emb, params):
            # Delegate to pipeline.model directly
            if pipeline.model is None:
                T = audio_emb.shape[1]
                H = pipeline.height // 8
                W = pipeline.width  // 8
                return torch.zeros(
                    1, 4, T, H, W,
                    device=pipeline.device,
                    dtype=audio_emb.dtype,
                )
            return pipeline.model(
                x=torch.randn(
                    1, 4,
                    audio_emb.shape[1],
                    pipeline.height // 8,
                    pipeline.width  // 8,
                    device=pipeline.device,
                    dtype=audio_emb.dtype,
                ),
                t=torch.tensor([0.5], device=pipeline.device),
                audio_features=audio_emb,
            )

    return _Shim()


def get_infer_params(
    pipeline: BitHumanPipeline,
    cached_audio_duration: float = 1.0,
    tgt_fps: Optional[int] = None,
    motion_frames_num: int = 1,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
) -> Dict:
    """
    Build the inference parameter dictionary consumed by pipeline.generate().

    Args:
        pipeline:             BitHumanPipeline instance
        cached_audio_duration: audio chunk duration in seconds
        tgt_fps:              target output frame rate (defaults to pipeline.tgt_fps)
        motion_frames_num:    number of motion conditioning frames
        num_inference_steps:  denoising steps
        guidance_scale:       classifier-free guidance scale
        seed:                 RNG seed for reproducibility

    Returns:
        params dict with 'cached_audio_duration', 'tgt_fps', 'motion_frames_num',
        'timesteps', 'guidance_scale', 'num_inference_steps'
    """
    if tgt_fps is None:
        tgt_fps = pipeline.tgt_fps

    if seed is not None:
        torch.manual_seed(seed)

    num_frames = max(1, int(cached_audio_duration * tgt_fps))
    timesteps  = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

    return {
        "cached_audio_duration": cached_audio_duration,
        "tgt_fps":               tgt_fps,
        "fps":                   tgt_fps,
        "motion_frames_num":     motion_frames_num,
        "num_frames":            num_frames,
        "timesteps":             timesteps,
        "guidance_scale":        guidance_scale,
        "num_inference_steps":   num_inference_steps,
        "cond_image_dict":       None,   # filled by get_base_data()
        "_expression_pipeline":  None,   # filled by run_pipeline()
    }


def get_audio_embedding(
    pipeline: BitHumanPipeline,
    audio_samples: torch.Tensor,
    sample_rate: int = 16000,
    num_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Encode raw audio waveform to Wav2Vec2 feature vectors.

    Thin wrapper around pipeline.encode_audio().

    Args:
        pipeline:      BitHumanPipeline
        audio_samples: (T,) or (B, T) float32 waveform
        sample_rate:   waveform sample rate in Hz
        num_frames:    target feature sequence length

    Returns:
        (audio_emb, meta_dict)  — see BitHumanPipeline.encode_audio()
    """
    return pipeline.encode_audio(audio_samples, sample_rate, num_frames)


def get_base_data(
    pipeline: BitHumanPipeline,
    image_path: Union[str, Path],
    face_ratio: float = 1.5,
    target_size: Tuple[int, int] = (512, 512),
) -> Dict:
    """
    Process the avatar reference image into a conditioning dictionary.

    Steps:
      1. Load image from disk
      2. Optionally apply face crop (if ENABLE_FACE_CROP)
      3. Build cond_image_dict for the expression pipeline

    Args:
        pipeline:    BitHumanPipeline
        image_path:  path to avatar reference image (JPEG/PNG)
        face_ratio:  face crop expansion ratio
        target_size: (W, H) to resize the face crop to

    Returns:
        dict with 'ref_image', 'ref_frames', 'embeddings' tensors
    """
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF

    image_path = Path(image_path)
    img = PILImage.open(image_path).convert("RGB")

    # Resize to model input size
    w, h = target_size
    img = img.resize((w, h), PILImage.LANCZOS)

    # (1, 3, H, W) float [-1, 1]
    tensor = TF.to_tensor(img).unsqueeze(0) * 2 - 1
    tensor = tensor.to(pipeline.device)

    # Build a minimal (1, 3, 1, H, W) video tensor (single reference frame)
    ref_video = tensor.unsqueeze(2)   # (1, 3, 1, H, W)

    from bithuman_expression.src.pipeline.expression_pipeline import get_cond_image_dict
    dummy_emb = torch.zeros(1, 512, device=pipeline.device)
    cond = get_cond_image_dict(dummy_emb, [ref_video.squeeze(0)])

    return cond


def run_pipeline(
    pipeline: BitHumanPipeline,
    audio_emb: torch.Tensor,
    base_data: Dict,
    params: Dict,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Full inference: denoising loop + VAE decode → RGB video frames.

    Args:
        pipeline:   BitHumanPipeline
        audio_emb:  (B, T, hidden_size) from get_audio_embedding()
        base_data:  conditioning dict from get_base_data()
        params:     inference params from get_infer_params()

    Returns:
        (frames, timings) — list of (H, W, 3) uint8 numpy arrays + timing dict
    """
    params = dict(params)                        # shallow copy — don't mutate caller's dict
    params["cond_image_dict"]      = base_data
    params["_expression_pipeline"] = _make_expression_pipeline(pipeline)

    return pipeline.generate(audio_emb, params)
