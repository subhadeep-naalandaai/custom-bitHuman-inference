"""
ExpressionPipeline: end-to-end talking-head video generation.

Usage:
    from inference.pipeline import ExpressionPipeline
    pipe = ExpressionPipeline()
    pipe.generate("face.jpg", "speech.wav", output_path="out.mp4")
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .audio import AudioEncoder
from .vae   import VAEWrapper
from .model import BitHumanExpressionModel


# Diffusion config (from infer_params.yaml)
FRAME_NUM   = 33     # video frames per chunk
T_LATENT    = 5      # temporal latent frames (ceil(33/8) with causal VAE offset)
H_LATENT    = 16     # 512 / 32
W_LATENT    = 16
FPS         = 25
SAMPLE_SHIFT = 5.0   # flow matching noise shift


def _build_scheduler(num_steps: int, shift: float = SAMPLE_SHIFT):
    """Return a FlowMatchEulerDiscreteScheduler configured for LTX-Video."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(num_steps)
    return scheduler


class ExpressionPipeline:
    def __init__(
        self,
        weights_dir: str = "bh-weights",
        wav2vec_dir: str = "app/bundled/wav2vec2-base-960h",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype  = dtype
        weights_dir = Path(weights_dir)

        print("[pipeline] Loading VAE...")
        self.vae = VAEWrapper(
            model_path=str(weights_dir / "bithuman-expression" / "VAE_LTX"),
            device=device,
            dtype=dtype,
        )

        print("[pipeline] Loading audio encoder...")
        self.audio_enc = AudioEncoder(model_path=wav2vec_dir, device=device)

        print("[pipeline] Loading DiT...")
        self.model = BitHumanExpressionModel.from_pretrained(
            config_path=str(weights_dir / "bithuman-expression" / "Model_Lite" / "config.json"),
            weights_path=str(weights_dir / "bithuman-expression" / "Model_Lite" / "bithuman_expression_dit_1_3b.safetensors"),
            device=device,
            dtype=dtype,
        )
        print("[pipeline] Ready.")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        image_path:  str,
        audio_path:  str,
        output_path: str = "output.mp4",
        num_steps:   int = 20,
        seed:        int = 42,
    ) -> str:
        """
        Generate a talking-head video from a face image and audio file.

        Returns the path to the saved MP4.
        """
        from PIL import Image

        # ---- 1. Prepare reference latent from image ----
        print("[pipeline] Encoding reference image...")
        image = Image.open(image_path).convert("RGB")
        ref_latent_1 = self.vae.encode_image(image)   # [1, 128, T_ref, 16, 16]
        # Tile reference to T_LATENT frames
        ref_latent = self._tile_latent(ref_latent_1, T_LATENT)  # [1, 128, T_LATENT, 16, 16]

        # ---- 2. Encode audio ----
        print("[pipeline] Encoding audio...")
        audio_feats = self.audio_enc.encode(audio_path)  # [1, 30, 768]
        audio_feats = audio_feats.to(self.dtype)

        # ---- 3. Diffusion denoising ----
        print(f"[pipeline] Denoising ({num_steps} steps)...")
        video_latent = self._denoise(ref_latent, audio_feats, num_steps, seed)
        # [1, 128, T_LATENT, 16, 16]

        # ---- 4. Decode to RGB frames ----
        print("[pipeline] Decoding latents...")
        frames = self.vae.decode(video_latent)           # [1, 3, T, 512, 512]
        frames_np = VAEWrapper.frames_to_uint8(frames)   # [1, T, 512, 512, 3]

        # ---- 5. Save video ----
        print(f"[pipeline] Saving to {output_path}...")
        self._save_video(frames_np[0], output_path, fps=FPS)
        print(f"[pipeline] Done. Video: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    def _tile_latent(self, lat: torch.Tensor, target_t: int) -> torch.Tensor:
        """Repeat a [B, C, T_src, H, W] latent along T to reach target_t."""
        T_src = lat.shape[2]
        if T_src >= target_t:
            return lat[:, :, :target_t]
        reps = (target_t + T_src - 1) // T_src
        lat = lat.repeat(1, 1, reps, 1, 1)
        return lat[:, :, :target_t]

    def _denoise(
        self,
        ref_latent:  torch.Tensor,   # [1, 128, T, H, W]
        audio_feats: torch.Tensor,   # [1, 30, 768]
        num_steps:   int,
        seed:        int,
    ) -> torch.Tensor:
        """Flow-matching denoising loop. Returns denoised latent."""
        scheduler = _build_scheduler(num_steps)
        B = ref_latent.shape[0]
        shape = ref_latent.shape   # [1, 128, T, H, W]

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noisy = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)

        for t in scheduler.timesteps:
            t_batch = t.expand(B).to(self.device)

            with torch.no_grad():
                velocity = self.model(noisy, ref_latent, audio_feats, t_batch)

            noisy = scheduler.step(velocity, t, noisy).prev_sample

        return noisy

    @staticmethod
    def _save_video(frames: np.ndarray, path: str, fps: int = 25) -> None:
        """
        frames: [T, H, W, 3] uint8
        Saves as MP4 using imageio-ffmpeg.
        """
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
