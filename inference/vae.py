"""
VAE wrapper around diffusers AutoencoderKLLTXVideo.

Encode: PIL image → latent tensor [1, 128, T_lat, H_lat, W_lat]
Decode: latent → video frames [1, 3, T, H, W] in [-1, 1]
"""
import torch
from PIL import Image
import numpy as np

# Latent spatial compression: 512 / 32 = 16
# Latent temporal compression: varies (causal 3D VAE, first frame is not compressed)
LATENT_CHANNELS = 128


class VAEWrapper:
    def __init__(
        self,
        model_path: str = "bh-weights/bithuman-expression/VAE_LTX",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        from diffusers import AutoencoderKLLTXVideo
        self.vae = AutoencoderKLLTXVideo.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(device)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_image(self, image: Image.Image, height: int = 512, width: int = 512) -> torch.Tensor:
        """
        image: PIL RGB image
        Returns latent [1, 128, 1, H_lat, W_lat] for a single-frame "video".
        """
        img = image.convert("RGB").resize((width, height), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0   # → [-1, 1]
        frame = torch.from_numpy(arr).permute(2, 0, 1)           # [3, H, W]
        # Add batch and time dims: [1, 3, 1, H, W]
        video = frame.unsqueeze(0).unsqueeze(2).to(self.device, self.dtype)
        latent = self.vae.encode(video).latent_dist.sample()
        return latent   # [1, 128, T_lat, H_lat, W_lat]

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, 128, T_lat, H_lat, W_lat]
        Returns frames: [B, 3, T, H, W] in [-1, 1]
        """
        return self.vae.decode(latents).sample

    @staticmethod
    def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
        """
        frames: [B, 3, T, H, W] float in [-1, 1]
        Returns: np.ndarray [B, T, H, W, 3] uint8
        """
        frames = (frames.float().clamp(-1, 1) + 1.0) * 127.5
        frames = frames.permute(0, 2, 3, 4, 1).cpu().numpy().astype(np.uint8)
        return frames
