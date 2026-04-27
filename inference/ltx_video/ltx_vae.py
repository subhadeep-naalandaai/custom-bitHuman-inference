"""
LTX-Video VAE wrapper.

Replaces the decompiled ltx_vae.pyc whose CausalVideoAutoencoder.from_pretrained()
and Decoder.forward() were marked "WARNING: Decompyle incomplete" and produced
garbage tensors (green-screen).

Uses diffusers AutoencoderKLLTXVideo, which is the open-source equivalent of
the same architecture and loads from either:
  - a local directory with a diffusers-format config.json + safetensors
  - the HuggingFace repo Lightricks/LTX-Video (subfolder="vae") as fallback
"""

import logging

import torch

logger = logging.getLogger("expression-avatar")

_OPEN_SOURCE_REPO = "Lightricks/LTX-Video"


def _load_model(pretrained_path: str, device: str, dtype: torch.dtype):
    from pathlib import Path
    from diffusers import AutoencoderKLLTXVideo

    path = Path(pretrained_path)

    if path.is_dir() and (path / "config.json").exists():
        try:
            vae = AutoencoderKLLTXVideo.from_pretrained(str(path))
            logger.info(f"LTX VAE loaded from local {path}")
            return vae.requires_grad_(False).to(device).to(dtype).eval()
        except Exception as e:
            logger.warning(f"Local VAE load failed ({e}); falling back to {_OPEN_SOURCE_REPO}")

    vae = AutoencoderKLLTXVideo.from_pretrained(_OPEN_SOURCE_REPO, subfolder="vae")
    logger.info(f"LTX VAE loaded from {_OPEN_SOURCE_REPO}")
    return vae.requires_grad_(False).to(device).to(dtype).eval()


class LtxVAE:
    """
    Drop-in replacement for the original LtxVAE (from ltx_vae.pyc).
    Interface is identical: encode(video) / decode(zs) + normalize helpers.
    """

    def __init__(
        self,
        pretrained_model_type_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        fast_decoder_config=None,
        fast_decoder_checkpoint=None,
    ):
        self.device = device
        self.dtype  = dtype
        self.model  = _load_model(pretrained_model_type_or_path, device, dtype)

        self.turbo_decoder = None
        if fast_decoder_config and fast_decoder_checkpoint:
            try:
                from inference.ltx_video.fast_decoder import load_fast_decoder
                self.turbo_decoder = load_fast_decoder(
                    fast_decoder_config, fast_decoder_checkpoint, device, dtype
                )
                logger.info("FastDecoder loaded")
            except Exception as e:
                logger.warning(f"FastDecoder not available (optional): {e}")

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ------------------------------------------------------------------
    # Normalization — per-channel stats stored as model buffers
    # ------------------------------------------------------------------

    def _stats(self, ref: torch.Tensor):
        mean = self.model.latents_mean.view(1, -1, 1, 1, 1).to(ref)
        std  = self.model.latents_std.view(1,  -1, 1, 1, 1).to(ref)
        return mean, std

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(latents)
        return (latents - mean) / std

    def un_normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(latents)
        return latents * std + mean

    # ------------------------------------------------------------------
    # Encode / Decode (mirror original LtxVAE interface)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, 3, T, H, W) in [-1, 1] → normalized latents (C, T, h, w)"""
        video   = video.to(self.model_dtype)
        latents = self.model.encode(video).latent_dist.sample()
        return self.normalize_latents(latents)[0]

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        """
        zs: (C, T, h, w) normalized latents, no batch dimension.
        Returns: (3, T, H, W) float in [-1, 1].
        """
        zs      = zs.to(self.model_dtype)
        latents = zs.unsqueeze(0)                    # (1, C, T, h, w)
        raw     = self.un_normalize_latents(latents)  # undo per-channel norm

        if self.turbo_decoder is not None:
            turbo_dtype = next(self.turbo_decoder.parameters()).dtype
            out = self.turbo_decoder(raw.to(turbo_dtype))
            return out.to(self.model_dtype)[0]

        out = self.model.decode(raw).sample           # (1, 3, T, H, W) in [-1, 1]
        return out[0]                                  # (3, T, H, W)
