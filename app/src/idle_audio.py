"""
src/idle_audio.py
Python replacement for idle_audio.cpython-310-x86_64-linux-gnu.so

Instead of feeding silent zeros to wav2vec during idle, we feed low-energy
ambient audio. This produces non-zero audio embeddings that drive subtle
facial micro-movements (breathing, eye blinks, head sway) rather than a
frozen pose.

The audio buffer is amplitude-scaled to be perceived as quiet room noise
enough to drive facial micro-movements but not speech.

Exports:
  init_idle_audio()                         — initialise global buffer (idempotent)
  get_idle_audio_chunk(offset, size)        — chunk at offset with wrap-around
  get_idle_audio_full(length)               — buffer tiled to exactly `length` samples
"""

import numpy as np

# ---------------------------------------------------------------------------
# Module-level global buffer
# ---------------------------------------------------------------------------

_idle_audio: np.ndarray = None   # float32, shape (N,)

_SAMPLE_RATE = 16000
_DURATION    = 2.0      # seconds of pink-noise to generate (loops seamlessly)
_AMPLITUDE   = 0.003    # very quiet — below speech threshold for wav2vec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_idle_audio() -> None:
    """Generate and store the global idle audio buffer (loopable pink noise)."""
    global _idle_audio

    n = int(_SAMPLE_RATE * _DURATION)
    rng = np.random.default_rng(42)

    # Approximate pink noise: sum octaves of white noise, then interpolate
    audio = np.zeros(n, dtype=np.float64)
    octave_len = n
    for _ in range(6):
        octave_len = max(1, octave_len // 2)
        noise = rng.normal(0.0, 1.0, octave_len)
        t_src = np.linspace(0.0, 1.0, octave_len)
        t_dst = np.linspace(0.0, 1.0, n)
        audio += np.interp(t_dst, t_src, noise)

    # Normalise, then scale to target amplitude
    peak = np.abs(audio).max() + 1e-8
    audio = (audio / peak * _AMPLITUDE).astype(np.float32)

    _idle_audio = audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_idle_audio() -> None:
    """Initialize the global idle audio buffer. Safe to call multiple times."""
    global _idle_audio
    if _idle_audio is None:
        _load_idle_audio()


def get_idle_audio_chunk(offset: int, size: int):
    """
    Get a chunk of idle audio starting at offset, wrapping around.

    Args:
        offset: sample index into the buffer (will be wrapped)
        size:   number of samples to return

    Returns:
        (chunk, new_offset) where chunk is float32 array of length `size`
        and new_offset is the updated position for the next call.
    """
    global _idle_audio
    if _idle_audio is None:
        init_idle_audio()

    buf = _idle_audio
    n   = len(buf)
    out = np.empty(size, dtype=np.float32)

    pos       = int(offset) % n
    remaining = size
    filled    = 0

    while remaining > 0:
        take = min(remaining, n - pos)
        out[filled : filled + take] = buf[pos : pos + take]
        filled    += take
        remaining -= take
        pos        = (pos + take) % n

    return out, pos


def get_idle_audio_full(length: int) -> np.ndarray:
    """
    Get idle audio tiled to fill `length` samples. Used to pre-fill the
    audio buffer so wav2vec normalization statistics are consistent from
    the start.

    Args:
        length: total number of samples required

    Returns:
        float32 numpy array of shape (length,)
    """
    chunk, _ = get_idle_audio_chunk(0, length)
    return chunk
