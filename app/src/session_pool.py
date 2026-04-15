"""
src/session_pool.py
Python replacement for session_pool.cpython-310-x86_64-linux-gnu.so

Session pool for multi-session expression-avatar.
Manages N StreamingEngine instances sharing one TRTPipeline.
Inspired by expression-avatar's InProcessWorkerManager pattern.

Classes:
  SessionPool — asyncio.Queue-based pool of StreamingEngine slots
"""

import asyncio
import logging

logger = logging.getLogger("expression-avatar.session_pool")


class SessionPool:
    """
    Session pool for multi-session expression-avatar.
    Manages N StreamingEngine instances sharing one TRTPipeline.
    """

    def __init__(self, pipeline, max_sessions: int = 1):
        """
        Args:
            pipeline:     TRTPipeline instance (shared weights, passed to each engine)
            max_sessions: maximum number of concurrent sessions
        """
        self._pipeline     = pipeline
        self._max_sessions = max_sessions
        self._queue        = asyncio.Queue(maxsize=max_sessions)

        # Pre-populate with StreamingEngine slots (created lazily with shared pipeline)
        from src.streaming_engine import StreamingEngine

        for _ in range(max_sessions):
            engine = StreamingEngine(shared_pipeline=pipeline)
            self._queue.put_nowait(engine)

        logger.info(
            f"[SessionPool] Initialised with {max_sessions} session slot(s) "
            f"sharing one {type(pipeline).__name__}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pipeline(self):
        """The shared TRTPipeline instance."""
        return self._pipeline

    @property
    def max_sessions(self) -> int:
        """Maximum number of concurrent sessions."""
        return self._max_sessions

    @property
    def active_count(self) -> int:
        """Number of currently acquired (active) sessions."""
        return self._max_sessions - self._queue.qsize()

    @property
    def available_count(self) -> int:
        """Number of idle sessions ready to be acquired."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    async def acquire(self, timeout: float = None):
        """
        Acquire an available engine. Raises TimeoutError if timeout exceeded.

        Args:
            timeout: seconds to wait; None means wait indefinitely
        """
        if timeout is not None:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        return await self._queue.get()

    def acquire_nowait(self):
        """
        Acquire an available engine. Raises asyncio.QueueEmpty if none available.
        """
        return self._queue.get_nowait()

    def release(self, engine) -> None:
        """
        Return engine to pool after session ends.

        The engine's per-session state is reset so it is ready for a fresh session.
        """
        try:
            engine.reset()
        except Exception as exc:
            logger.warning(f"[SessionPool] Engine reset failed on release: {exc}")
        try:
            self._queue.put_nowait(engine)
        except asyncio.QueueFull:
            # Should not happen, but guard against double-release
            logger.warning("[SessionPool] Queue full on release — engine discarded")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Clean up all engines and the shared pipeline."""
        while not self._queue.empty():
            try:
                engine = self._queue.get_nowait()
                engine.cleanup()
            except Exception as exc:
                logger.warning(f"[SessionPool] Engine cleanup error: {exc}")

        if hasattr(self._pipeline, "cleanup"):
            try:
                self._pipeline.cleanup()
            except Exception as exc:
                logger.warning(f"[SessionPool] Pipeline cleanup error: {exc}")

        logger.info("[SessionPool] Cleaned up")
