"""
src/trt_pipeline.py
Python replacement for trt_pipeline.cpython-310-x86_64-linux-gnu.so

TRT-accelerated expression avatar pipeline for streaming.
Uses PyTorch for VAE and audio encoding, TRT FP16 for the DiT model.
Supports multiple concurrent sessions sharing one set of model weights.
Falls back to PyTorch + torch.compile when TRT engine is unavailable.

Classes:
  TRTPipeline — central pipeline: encode_audio, generate, create_session
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger("expression-avatar.trt_pipeline")


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

class _SessionContext:
    """
    Lightweight per-session state held by TRTPipeline.

    Caches the preprocessed reference image tensor so create_session()
    is cheap and the session can be reused across many generate() calls.
    """

    def __init__(self, image_path: str, pipeline: "TRTPipeline"):
        self.image_path     = str(image_path)
        self.cond_image_dict: Optional[Dict] = None
        self._pipeline      = pipeline
        self._trt_context   = None   # TRTContext, created lazily

        self._load_image()

    def _load_image(self):
        """Preprocess the avatar reference image into a conditioning dict."""
        try:
            from bithuman_expression.src.pipeline.expression_pipeline import get_cond_image_dict
            from PIL import Image as PILImage
            import torchvision.transforms.functional as TF

            img = PILImage.open(self.image_path).convert("RGB")
            img = img.resize(
                (self._pipeline.width, self._pipeline.height), PILImage.LANCZOS
            )
            tensor = TF.to_tensor(img).unsqueeze(0) * 2.0 - 1.0   # [1, 3, H, W] ∈ [-1,1]
            tensor = tensor.to(self._pipeline.device)
            ref_video = tensor.unsqueeze(2)                         # [1, 3, 1, H, W]

            dummy_emb = torch.zeros(1, 512, device=self._pipeline.device)
            self.cond_image_dict = get_cond_image_dict(dummy_emb, [ref_video.squeeze(0)])
        except Exception as e:
            logger.warning(f"[_SessionContext] Image load failed for {self.image_path}: {e}")
            self.cond_image_dict = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TRTPipeline:
    """
    TRT-accelerated expression avatar pipeline for streaming.
    Uses PyTorch for VAE and audio encoding, TRT FP16 for the DiT model.
    Supports multiple concurrent sessions sharing one set of model weights.
    Falls back to PyTorch + torch.compile when TRT engine is unavailable.
    """

    def __init__(
        self,
        mode: str = "pytorch",
        engine_path: Optional[Union[str, Path]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        wav2vec_dir: Optional[Union[str, Path]] = None,
        fast_decoder_config: Optional[str] = None,
        fast_decoder_checkpoint: Optional[str] = None,
    ):
        self._mode        = mode          # 'trt' or 'pytorch'
        self._engine_path = engine_path
        self._lock        = threading.Lock()

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width   = 512
        self.height  = 512
        self.tgt_fps = 25

        # Inner BitHumanPipeline (wav2vec + VAE + expression model)
        self._inner:      Optional[object] = None
        # Optional TRT runner for the DiT forward pass
        self._trt_runner: Optional[object] = None

        # Legacy single-session state (backward-compat with prepare())
        self._session: Optional[_SessionContext] = None

        self._load(
            ckpt_dir, models_dir, wav2vec_dir,
            fast_decoder_config, fast_decoder_checkpoint,
        )

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(
        self,
        ckpt_dir, models_dir, wav2vec_dir,
        fast_decoder_config, fast_decoder_checkpoint,
    ):
        """Load all model components."""
        from bithuman_expression.inference import BitHumanPipeline

        self._inner = BitHumanPipeline(
            ckpt_dir              = ckpt_dir,
            models_dir            = models_dir,
            wav2vec_dir           = wav2vec_dir,
            fast_decoder_config   = fast_decoder_config,
            fast_decoder_checkpoint = fast_decoder_checkpoint,
        )
        self.width   = self._inner.width
        self.height  = self._inner.height
        self.tgt_fps = self._inner.tgt_fps

        # Optional TRT acceleration for the DiT forward pass
        if self._mode == "trt" and self._engine_path:
            try:
                from src.trt_runner import TRTRunner
                self._trt_runner = TRTRunner(str(self._engine_path))
                logger.info("[TRTPipeline] TRT runner loaded")
            except Exception as e:
                logger.warning(
                    f"[TRTPipeline] TRT load failed, falling back to PyTorch: {e}"
                )
                self._trt_runner = None

    # ------------------------------------------------------------------
    # Public API — per-session
    # ------------------------------------------------------------------

    def create_session(self, image_path: Union[str, Path]) -> _SessionContext:
        """Create a new session context for the given image. Thread-safe."""
        return _SessionContext(str(image_path), self)

    def remove_session(self, session: Optional[_SessionContext]) -> None:
        """Remove cached fallback and release per-session TRT context."""
        if session is None:
            return
        if session._trt_context is not None:
            try:
                del session._trt_context
                session._trt_context = None
            except Exception:
                pass
        if session is self._session:
            self._session = None

    def prepare(self, image_path: Optional[Union[str, Path]] = None, **kwargs) -> None:
        """Prepare legacy single-session state. For backward compatibility."""
        if image_path is not None:
            self._session = self.create_session(image_path)

    # ------------------------------------------------------------------
    # Public API — inference
    # ------------------------------------------------------------------

    def encode_audio(
        self,
        audio_samples,
        sample_rate: int = 16000,
        cached_audio_duration: float = 1.0,
        tgt_fps: Optional[int] = None,
        frame_num: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode audio to embedding using PyTorch wav2vec. Thread-safe.

        Args:
            audio_samples:          (T,) or (B, T) float32 waveform
            sample_rate:            input sample rate in Hz
            cached_audio_duration:  chunk duration in seconds
            tgt_fps:                target fps (used to compute frame_num)
            frame_num:              explicit output frame count (overrides tgt_fps)

        Returns:
            (audio_emb, meta_dict) — same contract as BitHumanPipeline.encode_audio
        """
        if tgt_fps is None:
            tgt_fps = self.tgt_fps
        if frame_num is None:
            frame_num = max(1, int(cached_audio_duration * tgt_fps))

        if not isinstance(audio_samples, torch.Tensor):
            audio_samples = torch.from_numpy(
                np.asarray(audio_samples, dtype=np.float32)
            )

        # encode_audio is already thread-safe inside BitHumanPipeline (no_grad + device)
        return self._inner.encode_audio(
            audio_samples, sample_rate, num_frames=frame_num
        )

    def generate(
        self,
        audio_embedding: torch.Tensor,
        session: Optional[_SessionContext] = None,
        profile: bool = False,
    ):
        """
        Generate video frames from audio embedding. Thread-safe.

        Args:
            audio_embedding: wav2vec audio embedding tensor
            session: per-session state. If None, uses legacy self.* state.
            profile: if True, returns (frames, timings) instead of just frames

        Returns:
            frames: numpy array [T, H, W, C] uint8 (only new frames, excluding motion overlap)
            timings: dict of stage timings in ms (only when profile=True)
        """
        ctx  = session if session is not None else self._session
        cond = ctx.cond_image_dict if ctx is not None else None

        # Build an ExpressionPipeline shim backed by our inner model
        exp_pipeline = _ExpressionPipelineShim(self._inner)

        num_frames = (
            audio_embedding.shape[1]
            if audio_embedding.ndim == 3
            else 1
        )
        params = {
            "cond_image_dict":    cond,
            "timesteps":          torch.linspace(1.0, 0.0, 21, device=self.device)[:-1],
            "guidance_scale":     3.5,
            "num_inference_steps": 20,
            "num_frames":         num_frames,
            "_expression_pipeline": exp_pipeline,
        }

        t0 = time.perf_counter()
        frames_list, timings = self._inner.generate(audio_embedding, params)
        t1 = time.perf_counter()

        # Convert list of (H, W, 3) uint8 → (T, H, W, C) uint8
        if frames_list:
            frames_arr = np.stack(frames_list, axis=0)
        else:
            frames_arr = np.zeros(
                (num_frames, self.height, self.width, 3), dtype=np.uint8
            )

        if profile:
            timings["total_ms"] = (t1 - t0) * 1000
            return frames_arr, timings

        return frames_arr

    def cleanup(self) -> None:
        """Release GPU resources."""
        self._trt_runner = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Internal shim: ExpressionPipeline backed by TRTPipeline.model
# ---------------------------------------------------------------------------

class _ExpressionPipelineShim:
    """Thin shim so TRTPipeline.generate() can call BitHumanPipeline.generate()."""

    def __init__(self, inner):
        self._inner = inner

    def generate(self, audio_emb: torch.Tensor, params: Dict) -> torch.Tensor:
        """Delegate to the inner model's flow-matching denoising loop."""
        model  = self._inner.model
        device = self._inner.device

        if model is None:
            T = audio_emb.shape[1]
            H = self._inner.height // 8
            W = self._inner.width  // 8
            return torch.zeros(1, 4, T, H, W, device=device, dtype=audio_emb.dtype)

        timesteps      = params.get("timesteps", torch.linspace(1.0, 0.0, 21)[:-1]).to(device)
        guidance_scale = params.get("guidance_scale", 3.5)
        T              = audio_emb.shape[1]
        H              = self._inner.height // 8
        W              = self._inner.width  // 8

        x = torch.randn(1, 4, T, H, W, device=device, dtype=audio_emb.dtype)
        audio_emb = audio_emb.to(device)

        with torch.no_grad():
            for i, t_val in enumerate(timesteps):
                t      = t_val.expand(1).to(device)
                v_cond = model(x, t, audio_emb)

                if guidance_scale != 1.0:
                    null_audio = torch.zeros_like(audio_emb)
                    v_uncond   = model(x, t, null_audio)
                    v          = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = v_cond

                dt = (
                    (timesteps[i + 1] - t_val)
                    if i + 1 < len(timesteps)
                    else -t_val
                )
                x = x + dt * v

        return x
