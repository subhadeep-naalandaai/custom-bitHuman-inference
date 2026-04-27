"""
LTX-Video VAE wrapper.

Thin wrapper around diffusers AutoencoderKLLTXVideo.
Stride: [8 temporal, 32 spatial] → 512×512 → 16×16 latents (128 channels).
Standalone — does not import from app/.
"""
import logging

import torch

logger = logging.getLogger(__name__)

_HF_REPO = "Lightricks/LTX-Video"


def load_vae(
    vae_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load AutoencoderKLLTXVideo from local dir, falling back to HuggingFace.

    Args:
        vae_dir: local directory with config.json + safetensors
        device:  target device
        dtype:   compute dtype (bfloat16 recommended)

    Returns:
        AutoencoderKLLTXVideo in eval mode, frozen
    """
    from pathlib import Path
    from diffusers import AutoencoderKLLTXVideo

    p = Path(vae_dir)
    if p.is_dir() and (p / "config.json").exists():
        try:
            vae = AutoencoderKLLTXVideo.from_pretrained(str(p))
            logger.info(f"VAE loaded from {p}")
            return vae.requires_grad_(False).to(device).to(dtype).eval()
        except Exception as e:
            logger.warning(f"Local VAE load failed ({e}); downloading from {_HF_REPO}")

    vae = AutoencoderKLLTXVideo.from_pretrained(_HF_REPO, subfolder="vae")
    logger.info(f"VAE loaded from {_HF_REPO}")
    return vae.requires_grad_(False).to(device).to(dtype).eval()


class LtxVAE:
    """Encode/decode video latents using AutoencoderKLLTXVideo."""

    def __init__(self, vae_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype  = dtype
        self.model  = load_vae(vae_dir, device, dtype)

    def _stats(self, ref: torch.Tensor):
        m = getattr(self.model, "latents_mean", None)
        s = getattr(self.model, "latents_std",  None)
        if m is not None and s is not None:
            mean = torch.as_tensor(m).view(1, -1, 1, 1, 1).to(ref)
            std  = torch.as_tensor(s).view(1, -1, 1, 1, 1).to(ref)
        else:
            # Fall back to scaling_factor (SD-style: latent / scale_factor)
            sf   = float(getattr(self.model.config, "scaling_factor", 0.18215))
            mean = torch.zeros(1, device=ref.device, dtype=ref.dtype)
            std  = torch.full((1,), sf, device=ref.device, dtype=ref.dtype)
        return mean, std

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(latents)
        return (latents - mean) / std

    def unnormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(latents)
        return latents * std + mean

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, 3, T, H, W) in [-1, 1] → (128, T_lat, H_lat, W_lat) normalized"""
        video   = video.to(self.dtype)
        latents = self.model.encode(video).latent_dist.sample()
        return self.normalize(latents)[0]  # drop batch dim

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        """zs: (128, T_lat, H_lat, W_lat) → (3, T, H, W) in [-1, 1]"""
        zs  = zs.to(self.dtype)
        raw = self.unnormalize(zs.unsqueeze(0))  # (1, 128, T_lat, h, w)
        out = self.model.decode(raw).sample       # (1, 3, T, H, W)
        return out[0]
