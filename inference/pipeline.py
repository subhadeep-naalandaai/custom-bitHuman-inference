"""
Naalanda inference pipeline — image + audio → talking-head video.

Public API (same as app/bithuman_expression/inference.py):
    get_pipeline(ckpt_dir, models_dir, wav2vec_dir, ...)
    get_infer_params(pipeline, ...)
    get_audio_embedding(pipeline, audio, sample_rate, num_frames)
    get_base_data(pipeline, image_path, ...)
    run_pipeline(pipeline, audio_emb, base_data, params)

Standalone — does not import from app/.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from inference.model import ExpressionModel
from inference.audio import load_wav2vec, encode_audio
from inference.vae   import LtxVAE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline object
# ─────────────────────────────────────────────────────────────────────────────

class NaalandaPipeline:
    """
    Inference wrapper: loads ExpressionModel, Wav2Vec2, and LtxVAE from disk.

    Attributes:
        ckpt_dir   : path to Model_Lite directory (config.json + safetensors)
        models_dir : root weights directory (contains bithuman-expression/)
        device     : torch.device
        model      : ExpressionModel (DiT)
        vae        : LtxVAE
        wav2vec    : Wav2Vec2Model
        width, height, tgt_fps : output dimensions / fps
    """

    def __init__(
        self,
        ckpt_dir:   Union[str, Path],
        models_dir: Union[str, Path],
        wav2vec_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.ckpt_dir   = Path(ckpt_dir)
        self.models_dir = Path(models_dir)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width      = 512
        self.height     = 512
        self.tgt_fps    = 25
        self.model      = None
        self.vae        = None
        self.wav2vec    = None
        self._load(wav2vec_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, wav2vec_dir):
        self._load_wav2vec(wav2vec_dir)
        self._load_expression_model()

    def _load_wav2vec(self, wav2vec_dir):
        dirs_to_try = []
        if wav2vec_dir:
            dirs_to_try.append(str(wav2vec_dir))
        # bundled path inside project
        bundled = Path(__file__).parent.parent / "app" / "bundled" / "wav2vec2-base-960h"
        if bundled.exists():
            dirs_to_try.append(str(bundled))
        dirs_to_try.append("facebook/wav2vec2-base-960h")

        for d in dirs_to_try:
            try:
                self.wav2vec = load_wav2vec(d, str(self.device))
                logger.info(f"[NaalandaPipeline] Wav2Vec2 loaded from {d}")
                return
            except Exception:
                continue
        logger.error("[NaalandaPipeline] Could not load Wav2Vec2")

    def _load_expression_model(self):
        config_path = self.ckpt_dir / "config.json"
        if not config_path.exists():
            logger.warning(f"[NaalandaPipeline] config.json not found at {config_path}")
            return

        with open(config_path) as f:
            cfg = json.load(f)

        # VAE
        try:
            vae_dir = self.models_dir / "bithuman-expression" / "VAE_LTX"
            if not vae_dir.exists():
                vae_dir = self.ckpt_dir.parent / "VAE_LTX"
            vae_src = str(vae_dir) if vae_dir.exists() else "Lightricks/LTX-Video"
            self.vae = LtxVAE(vae_src, str(self.device), torch.bfloat16)
            logger.info(f"[NaalandaPipeline] VAE loaded from {vae_src}")
        except Exception as e:
            logger.error(f"[NaalandaPipeline] VAE load failed: {e}")

        # ExpressionModel
        try:
            from safetensors.torch import load_file as load_sf
            self.model = ExpressionModel(
                dim          = cfg.get("dim",         1536),
                in_channels  = cfg.get("in_dim",       256),
                out_channels = cfg.get("out_dim",      128),
                freq_dim     = cfg.get("freq_dim",     256),
                ffn_dim      = cfg.get("ffn_dim",     8960),
                num_heads    = cfg.get("num_heads",     12),
                num_layers   = cfg.get("num_layers",    30),
                text_dim     = cfg.get("text_dim",    4096),
            ).to(self.device)

            # Weight file priority: naalanda → branded → legacy
            candidates = [
                self.ckpt_dir / "naalanda_expression_dit.safetensors",
                self.ckpt_dir / "bithuman_expression_dit_1_3b.safetensors",
                self.ckpt_dir / "diffusion_pytorch_model.safetensors",
            ]
            ckpt = next((c for c in candidates if c.exists()), None)
            if ckpt:
                state = load_sf(str(ckpt), device=str(self.device))
                if any(k.startswith("bithuman.") for k in state):
                    state = {k.removeprefix("bithuman."): v for k, v in state.items()}
                self.model.load_state_dict(state, strict=False)
                logger.info(f"[NaalandaPipeline] ExpressionModel weights loaded from {ckpt.name}")

            self.model.eval()
            self.width   = cfg.get("width",  512)
            self.height  = cfg.get("height", 512)
            self.tgt_fps = cfg.get("fps",    25)
        except Exception as e:
            logger.error(f"[NaalandaPipeline] ExpressionModel load failed: {e}")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_audio(
        self,
        audio_samples: torch.Tensor,
        sample_rate:   int = 16000,
        num_frames:    Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.perf_counter()
        if audio_samples.ndim == 1:
            audio_samples = audio_samples.unsqueeze(0)
        if num_frames is None:
            num_frames = max(1, int(audio_samples.shape[-1] / sample_rate * self.tgt_fps))

        emb = encode_audio(self.wav2vec, audio_samples.squeeze(0), num_frames, str(self.device))
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        return emb, {"audio_ms": (time.perf_counter() - t0) * 1000}

    @torch.no_grad()
    def generate(
        self,
        audio_emb: torch.Tensor,
        params:    Dict,
    ) -> Tuple[List[np.ndarray], Dict]:
        t_start = time.perf_counter()

        if self.model is None:
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return [blank] * audio_emb.shape[1], {"denoise_ms": 0, "decode_ms": 0, "total_ms": 0}

        cond_dict      = params.get("cond_image_dict")
        timesteps      = params.get("timesteps", torch.linspace(1.0, 0.0, 21)[:-1]).to(self.device)
        guidance_scale = float(params.get("guidance_scale", 3.5))
        num_frames     = params.get("num_frames", audio_emb.shape[1])

        B         = audio_emb.shape[0]
        audio_emb = audio_emb.to(self.device)

        H_lat = self.height // 32
        W_lat = self.width  // 32
        T_lat = max(1, (num_frames - 1) // 8 + 1)

        # Encode reference frame
        ref_latent = None
        if self.vae is not None and cond_dict is not None:
            ref_video = cond_dict.get("ref_frames")
            if ref_video is None:
                ref_video = cond_dict.get("ref_image")
            if ref_video is not None:
                if ref_video.ndim == 4:
                    ref_video = ref_video.unsqueeze(0)
                ref_frame = ref_video[:B, :, 0:1, :, :]
                if ref_frame.shape[-2:] != (self.height, self.width):
                    ref_frame = F.interpolate(
                        ref_frame.squeeze(2), size=(self.height, self.width),
                        mode="bilinear", align_corners=False,
                    ).unsqueeze(2)
                try:
                    ref_enc    = self.vae.encode(ref_frame.to(dtype=torch.bfloat16))
                    ref_enc    = ref_enc[:, :1, :, :]
                    ref_latent = ref_enc.unsqueeze(0).expand(B, -1, T_lat, -1, -1).to(audio_emb.dtype)
                except Exception as e:
                    logger.warning(f"[NaalandaPipeline] VAE encode failed: {e}")

        if ref_latent is None:
            ref_latent = torch.zeros(B, 128, T_lat, H_lat, W_lat, device=self.device, dtype=audio_emb.dtype)
        else:
            ref_latent = ref_latent.to(self.device)

        x_noisy = torch.randn(B, 128, T_lat, H_lat, W_lat, device=self.device, dtype=audio_emb.dtype)

        t_denoise = time.perf_counter()
        for i, t_val in enumerate(timesteps):
            t      = t_val.expand(B).to(self.device)
            x_in   = torch.cat([x_noisy, ref_latent], dim=1)  # (B, 256, T, H, W)
            v_cond = self.model(x_in, t, audio_emb)
            if guidance_scale != 1.0:
                null_audio = torch.zeros_like(audio_emb)
                v_uncond   = self.model(x_in, t, null_audio)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            dt      = (timesteps[i + 1] - t_val) if i + 1 < len(timesteps) else -t_val
            x_noisy = x_noisy + dt * v
        denoise_ms = (time.perf_counter() - t_denoise) * 1000

        # Decode latents → RGB frames
        t_decode = time.perf_counter()
        frames   = []
        if self.vae is not None:
            for b in range(B):
                decoded = self.vae.decode(x_noisy[b])  # (3, T, H, W) in [-1,1]
                decoded = decoded.clamp(-1, 1)
                decoded = ((decoded + 1) * 127.5).byte()
                for t_i in range(decoded.shape[1]):
                    frames.append(decoded[:, t_i, :, :].permute(1, 2, 0).cpu().numpy())
        else:
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frames = [blank] * (T_lat * B)
        decode_ms  = (time.perf_counter() - t_decode) * 1000
        total_ms   = (time.perf_counter() - t_start)  * 1000

        return frames, {"denoise_ms": denoise_ms, "decode_ms": decode_ms, "total_ms": total_ms}


# ─────────────────────────────────────────────────────────────────────────────
# Public API functions
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline(
    ckpt_dir:    str,
    models_dir:  str,
    wav2vec_dir: Optional[str] = None,
    device:      Optional[str] = None,
    **_,
) -> NaalandaPipeline:
    return NaalandaPipeline(ckpt_dir=ckpt_dir, models_dir=models_dir, wav2vec_dir=wav2vec_dir, device=device)


def get_infer_params(
    pipeline: NaalandaPipeline,
    cached_audio_duration: float = 8.0,
    tgt_fps:               int   = 25,
    num_inference_steps:   int   = 20,
    guidance_scale:        float = 3.5,
    seed:                  int   = 42,
) -> Dict:
    if seed is not None:
        torch.manual_seed(seed)
    num_frames = max(1, int(cached_audio_duration * tgt_fps))
    timesteps  = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
    return {
        "cached_audio_duration":  cached_audio_duration,
        "tgt_fps":                tgt_fps,
        "num_frames":             num_frames,
        "timesteps":              timesteps,
        "guidance_scale":         guidance_scale,
        "cond_image_dict":        None,
        "num_inference_steps":    num_inference_steps,
    }


def get_audio_embedding(
    pipeline:     NaalandaPipeline,
    audio_samples: torch.Tensor,
    sample_rate:  int = 16000,
    num_frames:   Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    return pipeline.encode_audio(audio_samples, sample_rate, num_frames)


def get_base_data(
    pipeline:    NaalandaPipeline,
    image_path:  Union[str, Path],
    face_ratio:  float = 1.3,
    target_size: int   = 512,
) -> Dict:
    """Load and VAE-encode a reference portrait image."""
    from PIL import Image
    import torchvision.transforms.functional as tvf

    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    tensor = tvf.to_tensor(img) * 2.0 - 1.0  # (3, H, W) in [-1, 1]
    # (1, 3, 1, H, W) — batch=1, T=1
    video = tensor.unsqueeze(0).unsqueeze(2)

    return {
        "ref_image":  video,
        "ref_frames": video,
        "embeddings": torch.zeros(1, 512),
    }


def run_pipeline(
    pipeline:   NaalandaPipeline,
    audio_emb:  torch.Tensor,
    base_data:  Dict,
    params:     Dict,
) -> Tuple[List[np.ndarray], Dict]:
    params = dict(params)
    params["cond_image_dict"] = base_data
    return pipeline.generate(audio_emb, params)
