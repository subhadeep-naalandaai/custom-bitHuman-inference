"""
src/streaming_engine.py
Python replacement for streaming_engine.cpython-310-x86_64-linux-gnu.so

Manages audio buffering, TRT inference, and frame output for real-time
avatar generation.  Supports both standalone and shared-pipeline modes
for multi-session operation.

Classes:
  NumpyCircularBuffer — fixed-capacity ring buffer for audio samples
  ListBuffer          — deque-style buffer for arbitrary objects (frame lists)
  StreamingEngine     — full pipeline: audio in → video frames out
"""

import logging
import threading
from collections import deque
from typing import Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("expression-avatar.streaming_engine")


# ---------------------------------------------------------------------------
# NumpyCircularBuffer
# ---------------------------------------------------------------------------

class NumpyCircularBuffer:
    """
    Fixed-capacity ring buffer for float32 audio samples.

    Backed by a pre-allocated numpy array; extend() appends samples from
    the left, old samples fall off the right end.
    """

    def __init__(self, max_size: int, dtype: np.dtype = np.float32):
        """
        Args:
            max_size: maximum number of samples the buffer can hold
            dtype:    numpy dtype (default float32)
        """
        self._buf  = np.zeros(max_size, dtype=dtype)
        self._size = 0
        self._cap  = max_size

    @property
    def size(self) -> int:
        """Number of samples currently stored."""
        return self._size

    def extend(self, samples: np.ndarray) -> None:
        """
        Append `samples` to the buffer, discarding oldest samples if full.

        Args:
            samples: 1-D float32 numpy array
        """
        n = len(samples)
        if n >= self._cap:
            # New data fills or exceeds capacity — keep only the last _cap samples
            self._buf[:] = samples[-self._cap:]
            self._size   = self._cap
            return

        # Shift existing data left to make room
        keep = min(self._size, self._cap - n)
        if keep > 0:
            self._buf[:keep] = self._buf[self._size - keep : self._size]
        self._buf[keep : keep + n] = samples
        self._size = keep + n

    def as_numpy(self) -> np.ndarray:
        """Return a copy of the currently stored samples (oldest → newest)."""
        return self._buf[: self._size].copy()

    def reset(self) -> None:
        """Clear all stored samples."""
        self._size = 0
        self._buf[:] = 0.0

    def truncate_end(self, keep: int) -> None:
        """
        Discard samples from the end (newest) of the buffer, keeping only `keep`.

        Args:
            keep: number of samples to retain from the beginning (oldest)
        """
        self._size = max(0, min(keep, self._size))


# ---------------------------------------------------------------------------
# ListBuffer
# ---------------------------------------------------------------------------

class ListBuffer:
    """
    Deque-style buffer for arbitrary Python objects (e.g. lists of frames).
    """

    def __init__(self, maxlen: Optional[int] = None):
        """
        Args:
            maxlen: maximum capacity; None means unbounded
        """
        self._q: deque = deque(maxlen=maxlen)

    @property
    def size(self) -> int:
        """Number of items in the buffer."""
        return len(self._q)

    def push(self, item) -> None:
        """Append an item to the right end of the buffer."""
        self._q.append(item)

    def get(self) -> object:
        """Remove and return the leftmost (oldest) item."""
        return self._q.popleft()

    def last_n_data(self, n: int) -> list:
        """Return the last `n` items (most recent) without removing them."""
        data = list(self._q)
        return data[-n:] if n < len(data) else data

    def truncate(self, keep: int) -> None:
        """
        Keep only the rightmost `keep` items, discarding older ones.

        Args:
            keep: number of items to retain
        """
        while len(self._q) > keep:
            self._q.popleft()


# ---------------------------------------------------------------------------
# StreamingEngine
# ---------------------------------------------------------------------------

class StreamingEngine:
    """
    Manages audio buffering, TRT inference, and frame output for real-time
    avatar generation.  Supports both standalone and shared-pipeline modes
    for multi-session operation.
    """

    # Defaults
    _DEFAULT_SAMPLE_RATE   = 16000
    _DEFAULT_FPS           = 25
    _DEFAULT_WIDTH         = 512
    _DEFAULT_HEIGHT        = 512
    _MOTION_FRAMES_NUM     = 1       # motion conditioning frames
    _AUDIO_BUFFER_SECONDS  = 4.0    # audio ring-buffer capacity

    def __init__(
        self,
        engine_path:    Optional[Union[str, Path]] = None,
        ckpt_dir:       Optional[Union[str, Path]] = None,
        models_dir:     Optional[Union[str, Path]] = None,
        wav2vec_dir:    Optional[Union[str, Path]] = None,
        image_path:     Optional[Union[str, Path]] = None,
        device_id:      int = 0,
        base_seed:      int = 42,
        shared_pipeline = None,     # TRTPipeline — for multi-session mode
    ):
        """
        Args:
            engine_path:     path to TRT engine file (standalone mode)
            ckpt_dir:        checkpoint directory (standalone mode)
            models_dir:      root weights directory (standalone mode)
            wav2vec_dir:     wav2vec directory (standalone mode)
            image_path:      default reference image path
            device_id:       CUDA device index
            base_seed:       RNG seed for reproducible generation
            shared_pipeline: pre-loaded TRTPipeline (multi-session mode);
                             when provided, engine_path/ckpt_dir are ignored
        """
        self._engine_path    = engine_path
        self._ckpt_dir       = ckpt_dir
        self._models_dir     = models_dir
        self._wav2vec_dir    = wav2vec_dir
        self._default_image  = str(image_path) if image_path else None
        self._device_id      = device_id
        self._base_seed      = base_seed
        self._shared_pipeline = shared_pipeline

        # Runtime state
        self._pipeline       = shared_pipeline   # resolved pipeline
        self._session        = None              # _SessionContext per image
        self._lock           = threading.Lock()

        # Dimensions (set after pipeline load)
        self._width   = self._DEFAULT_WIDTH
        self._height  = self._DEFAULT_HEIGHT
        self._fps     = self._DEFAULT_FPS
        self._sr      = self._DEFAULT_SAMPLE_RATE

        # Chunk size: 1 second of audio at 16 kHz → 25 frames
        self._chunk_frames    = self._fps         # 25
        self._samples_per_fr  = self._sr // self._fps  # 640

        # Audio input ring-buffer
        cap = int(self._sr * self._AUDIO_BUFFER_SECONDS)
        self._audio_buf = NumpyCircularBuffer(max_size=cap)

        # Motion latent state (last N generated latent frames)
        self._motion_latents: Optional[torch.Tensor] = None   # (1, C, N, H, W)

        # Output frame buffer (filled by _generate_impl)
        self._output_buffer = ListBuffer(maxlen=256)

        # Standalone mode: load pipeline immediately
        if shared_pipeline is None and ckpt_dir is not None:
            self.load_model(engine_path, ckpt_dir, models_dir, wav2vec_dir)
        elif shared_pipeline is not None:
            self._width   = getattr(shared_pipeline, "width",   self._DEFAULT_WIDTH)
            self._height  = getattr(shared_pipeline, "height",  self._DEFAULT_HEIGHT)
            self._fps     = getattr(shared_pipeline, "tgt_fps", self._DEFAULT_FPS)
            self._chunk_frames   = self._fps
            self._samples_per_fr = self._sr // self._fps

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def video_fps(self) -> int:
        return self._fps

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def sample_per_chunk(self) -> int:
        """Audio samples per generated video chunk."""
        return self._samples_per_fr * self._chunk_frames

    @property
    def samples_per_frame(self) -> int:
        """Audio samples per single video frame."""
        return self._samples_per_fr

    @property
    def output_buffer(self) -> ListBuffer:
        """Frame output buffer (read by ExpressionVideoGenerator)."""
        return self._output_buffer

    # ------------------------------------------------------------------
    # Model loading (standalone mode)
    # ------------------------------------------------------------------

    def load_model(
        self,
        engine_path    = None,
        ckpt_dir       = None,
        models_dir     = None,
        wav2vec_dir    = None,
    ) -> None:
        """Load TRT engine and PyTorch pipeline (standalone mode only)."""
        from src.engine_builder import resolve_engine
        from src.trt_pipeline   import TRTPipeline

        resolved_path, use_fallback = resolve_engine(engine_path, ckpt_dir)
        mode = "pytorch" if use_fallback else "trt"

        self._pipeline = TRTPipeline(
            mode        = mode,
            engine_path = resolved_path,
            ckpt_dir    = ckpt_dir,
            models_dir  = models_dir,
            wav2vec_dir = wav2vec_dir,
        )
        self._width          = self._pipeline.width
        self._height         = self._pipeline.height
        self._fps            = self._pipeline.tgt_fps
        self._chunk_frames   = self._fps
        self._samples_per_fr = self._sr // self._fps

        logger.info(
            f"[StreamingEngine] Loaded ({mode}) "
            f"{self._width}x{self._height} @ {self._fps} fps"
        )

    # ------------------------------------------------------------------
    # Session / image management
    # ------------------------------------------------------------------

    def _ensure_session(self, image_path: Optional[str]) -> None:
        """Create or update per-session state for the given image."""
        path = str(image_path) if image_path else self._default_image
        if path is None:
            return
        if self._session is None or self._session.image_path != path:
            if self._pipeline is not None:
                self._session = self._pipeline.create_session(path)
                self._reset_motion_frames_to_initial(
                    self._session.cond_image_dict
                )

    # ------------------------------------------------------------------
    # Motion frame state
    # ------------------------------------------------------------------

    def _reset_motion_frames_to_initial(
        self, cond_image_dict: Optional[Dict] = None
    ) -> None:
        """
        Reset motion frames to clean initial state (from session creation).
        Used on interrupt to ensure the first new speech chunk starts from a
        clean visual state rather than carrying mid-speech motion latents.
        """
        self._motion_latents = None

    def _reset_latent_motion_frames(self) -> None:
        """
        Re-encode last N output frames as motion latents (after truncation).
        Called after audio buffer is truncated mid-stream to update the
        motion conditioning from the most recently decoded frames.
        """
        # In PyTorch mode this is a no-op — motion state is managed by
        # the expression pipeline itself.
        self._motion_latents = None

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def _generate_impl(
        self,
        audio_chunk: np.ndarray,
        image_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate frames from audio using TRT pipeline.

        Args:
            audio_chunk: float32 audio, length = sample_per_chunk
            image_path:  reference image (uses default if None)
            seed:        RNG seed (None for random)

        Returns:
            list of (H, W, 3) uint8 numpy arrays
        """
        if self._pipeline is None:
            return []

        self._ensure_session(image_path)

        audio_tensor = torch.from_numpy(audio_chunk)
        audio_emb, _ = self._pipeline.encode_audio(
            audio_tensor,
            sample_rate           = self._sr,
            cached_audio_duration = len(audio_chunk) / self._sr,
            tgt_fps               = self._fps,
        )

        frames = self._pipeline.generate(audio_emb, session=self._session)
        # frames: (T, H, W, C) uint8
        return [frames[i] for i in range(frames.shape[0])]

    def stream_generate(
        self,
        audio:      Union[np.ndarray, torch.Tensor],
        image_path: Optional[Union[str, Path]] = None,
        seed:       Optional[int] = None,
    ) -> Iterator[List[np.ndarray]]:
        """
        Generator that processes `audio` in chunk-sized pieces and yields
        lists of (H, W, 3) uint8 frames for each chunk.

        Args:
            audio:      float32 waveform (1-D numpy or tensor, any length)
            image_path: avatar reference image (uses default_image if None)
            seed:       RNG seed for reproducibility

        Yields:
            List[np.ndarray] — frames for each audio chunk
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy().astype(np.float32).ravel()
        else:
            audio = np.asarray(audio, dtype=np.float32).ravel()

        chunk_size = self.sample_per_chunk
        n_full     = len(audio) // chunk_size

        for i in range(n_full):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            frames = self._generate_impl(chunk, str(image_path) if image_path else None, seed)
            # Deposit into output_buffer for async reader
            for f in frames:
                self._output_buffer.push(f)
            yield frames

    # ------------------------------------------------------------------
    # Reset / cleanup
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset per-session state.
        Called by SessionPool.release() before the slot is returned to the pool.
        """
        self._audio_buf.reset()
        self._motion_latents = None
        self._session = None
        # Clear output buffer
        while self._output_buffer.size > 0:
            try:
                self._output_buffer.get()
            except Exception:
                break

    def cleanup(self) -> None:
        """Cleanup resources (standalone mode releases the pipeline)."""
        self.reset()
        if self._shared_pipeline is None and self._pipeline is not None:
            try:
                self._pipeline.cleanup()
            except Exception:
                pass
