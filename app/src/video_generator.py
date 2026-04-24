"""
src/video_generator.py
Python replacement for video_generator.cpython-310-x86_64-linux-gnu.so

Produces rtc.VideoFrame and rtc.AudioFrame for LiveKit streaming.

ExpressionVideoGenerator wraps a StreamingEngine and:
  - Accepts audio pushed from the LiveKit user (push_audio)
  - Runs inference in a background thread via the StreamingEngine
  - Exposes an async iterator that yields (VideoFrame, AudioFrame) tuples
    for the AvatarRunner to publish

Classes:
  ExpressionVideoGenerator
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("expression-avatar.video_generator")


class ExpressionVideoGenerator:
    """
    Wraps StreamingEngine to produce LiveKit-compatible rtc.VideoFrame +
    rtc.AudioFrame pairs for real-time avatar streaming.

    Usage:
        gen = ExpressionVideoGenerator(engine, image_path, ...)
        gen.start()
        gen.push_audio(audio_bytes, sample_rate)          # from LiveKit user
        async for video_frame, audio_frame in gen:
            await room.local_participant.publish_video(video_frame)
        gen.close()
    """

    def __init__(
        self,
        engine,                                          # StreamingEngine
        image_path:       Optional[Union[str, Path]] = None,
        prompt:           str = "A person is talking naturally.",
        width:            int = 512,
        height:           int = 512,
        video_fps:        int = 25,
        sample_rate:      int = 16000,
        sample_per_chunk: Optional[int] = None,
    ):
        """
        Args:
            engine:          StreamingEngine slot (acquired from SessionPool)
            image_path:      path to avatar reference image
            prompt:          unused text prompt (kept for API compatibility)
            width, height:   output video dimensions
            video_fps:       target output frame rate
            sample_rate:     LiveKit incoming audio sample rate
            sample_per_chunk: audio samples per inference chunk
        """
        self._engine          = engine
        self._image_path      = str(image_path) if image_path else None
        self._prompt          = prompt
        self._width           = getattr(engine, "width",      width)
        self._height          = getattr(engine, "height",     height)
        self._video_fps       = getattr(engine, "video_fps",  video_fps)
        self._sample_rate     = getattr(engine, "sample_rate", sample_rate)
        self._sample_per_chunk = (
            sample_per_chunk
            if sample_per_chunk is not None
            else getattr(engine, "sample_per_chunk", sample_rate)
        )

        # Audio input accumulator (raw float32 samples from user)
        self._audio_input: np.ndarray = np.empty(0, dtype=np.float32)
        self._audio_lock   = threading.Lock()

        # Output queue: (VideoFrame | None, AudioFrame | None) tuples
        self._output_q: asyncio.Queue = asyncio.Queue(maxsize=64)

        # Background threads/tasks
        self._running        = False
        self._inference_thread: Optional[threading.Thread] = None
        self._reader_task:      Optional[asyncio.Task]     = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, image_path=None, prompt=None) -> None:
        """Start inference and reader background tasks."""
        if image_path is not None:
            self._image_path = str(image_path)
        if prompt is not None:
            self._prompt = prompt
        if self._running:
            return
        self._running = True
        # Prefer the running loop (works when called from async context);
        # fall back to get_event_loop() for sync callers.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = None

        # Inference runs in a thread (GPU work + blocking numpy ops)
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="expr-inference",
        )
        self._inference_thread.start()
        logger.info("[ExpressionVideoGenerator] Started")

    def close(self) -> None:
        """Stop all background tasks and release resources."""
        self._running = False

        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
            self._inference_thread = None

        # Drain the output queue
        while not self._output_q.empty():
            try:
                self._output_q.get_nowait()
            except Exception:
                break

        logger.info("[ExpressionVideoGenerator] Closed")

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    def push_audio(self, audio_data, sample_rate: int = 16000) -> None:
        """
        Accept audio samples from the LiveKit user for inference.

        Args:
            audio_data:  float32 numpy array or bytes from rtc.AudioFrame
            sample_rate: input sample rate (will be resampled to model rate)
        """
        async def _push_impl():
            samples = self._resample_audio(audio_data, sample_rate)
            with self._audio_lock:
                self._audio_input = np.concatenate([self._audio_input, samples])

        # Fire-and-forget scheduling on the event loop
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_push_impl(), self._loop)
        else:
            samples = self._resample_audio(audio_data, sample_rate)
            with self._audio_lock:
                self._audio_input = np.concatenate([self._audio_input, samples])

    def _resample_audio(self, audio_data, src_rate: int) -> np.ndarray:
        """Resample audio_data to self._sample_rate."""
        if isinstance(audio_data, (bytes, bytearray)):
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif hasattr(audio_data, "data"):
            # rtc.AudioFrame
            samples = np.frombuffer(audio_data.data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.asarray(audio_data, dtype=np.float32).ravel()

        if src_rate != self._sample_rate and src_rate > 0:
            # Simple linear resampling
            target_len = int(len(samples) * self._sample_rate / src_rate)
            if target_len > 0:
                t_src = np.linspace(0.0, 1.0, len(samples))
                t_dst = np.linspace(0.0, 1.0, target_len)
                samples = np.interp(t_dst, t_src, samples).astype(np.float32)

        return samples

    def clear_buffer(self) -> None:
        """
        Clear pending audio and video buffers.
        Used on interrupt to ensure the first new speech chunk starts from a
        clean visual state rather than carrying mid-speech motion latents.
        """
        async def _clear_input():
            with self._audio_lock:
                self._audio_input = np.empty(0, dtype=np.float32)

        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_clear_input(), self._loop)
        else:
            with self._audio_lock:
                self._audio_input = np.empty(0, dtype=np.float32)

        self._engine._reset_motion_frames_to_initial()
        # Drain output queue
        while not self._output_q.empty():
            try:
                self._output_q.get_nowait()
            except Exception:
                break

    # ------------------------------------------------------------------
    # Background inference loop (runs in thread)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        """Continuously drain the audio input buffer and run inference."""
        from src.idle_audio import get_idle_audio_chunk, init_idle_audio
        init_idle_audio()

        idle_offset = 0
        frame_duration = 1.0 / self._video_fps

        while self._running:
            chunk_size = self._sample_per_chunk

            # Take a chunk from the input buffer (or use idle audio)
            with self._audio_lock:
                if len(self._audio_input) >= chunk_size:
                    audio_chunk        = self._audio_input[:chunk_size].copy()
                    self._audio_input  = self._audio_input[chunk_size:]
                else:
                    audio_chunk = None

            if audio_chunk is None:
                # Use idle audio when there is no real speech
                audio_chunk, idle_offset = get_idle_audio_chunk(idle_offset, chunk_size)
            else:
                idle_offset = 0

            # Run inference
            try:
                frames = self._engine._generate_impl(
                    audio_chunk, self._image_path
                )
            except Exception as exc:
                logger.warning(f"[ExpressionVideoGenerator] Inference error: {exc}")
                frames = []

            # Put each frame into the output queue
            for frame in frames:
                self._put_frame(frame)
                if not self._running:
                    break

    def _put_frame(self, frame: np.ndarray) -> None:
        """
        Convert a (H, W, 3) uint8 frame to rtc.VideoFrame and enqueue it.
        Read from output_buffer and put into async output_queue (no VSR).
        """
        try:
            from livekit import rtc

            # Build VideoFrame (RGBA)
            h, w, _ = frame.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = frame
            rgba[:, :, 3]  = 255

            video_frame = rtc.VideoFrame(
                width  = w,
                height = h,
                type   = rtc.VideoBufferType.RGBA,
                data   = rgba.tobytes(),
            )
        except ImportError:
            video_frame = frame   # fallback: raw numpy

        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._output_q.put((video_frame, None)),
                self._loop,
            )
        else:
            # Synchronous path (tests / no event loop)
            try:
                self._output_q.put_nowait((video_frame, None))
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[Tuple]:
        return self._iter_impl()

    async def _iter_impl(self) -> AsyncIterator[Tuple]:
        """
        Async generator: yields (VideoFrame, AudioFrame) tuples for each frame.
        AudioFrame is None — audio is published separately by the LiveKit runner.
        """
        while self._running:
            try:
                item = await asyncio.wait_for(self._output_q.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.warning(f"[ExpressionVideoGenerator] Iterator error: {exc}")
                break

    async def _reader_loop(self) -> None:
        """Background async task: forward frames from engine output_buffer to output_q."""
        while self._running:
            buf = self._engine.output_buffer
            if buf.size > 0:
                frame = buf.get()
                self._put_frame(frame)
            else:
                await asyncio.sleep(0.005)
