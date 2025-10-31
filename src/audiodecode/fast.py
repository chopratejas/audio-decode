"""
Fast path API - zero-overhead audio decoding.

This module provides the absolute fastest decode path by:
- Skipping AudioDecoder wrapper class
- Direct backend calls
- Minimal object creation
- Zero-copy returns

Use this when performance is critical and you don't need the convenience API.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from audiodecode.registry import get_backend_for_file


def decode(
    filepath: str | Path,
    target_sr: int | None = None,
    mono: bool = False,
    offset: float = 0.0,
    duration: float | None = None,
) -> npt.NDArray[np.float32]:
    """
    Fast decode - skips AudioDecoder wrapper for maximum speed.

    This is the FASTEST decode path. Use when:
    - Decoding many files in a loop
    - Performance is critical
    - You don't need the full AudioDecoder API

    Args:
        filepath: Path to audio file
        target_sr: Target sample rate (None = keep original)
        mono: Convert to mono if True
        offset: Start reading after this time (in seconds)
        duration: Only load this much audio (in seconds)

    Returns:
        Decoded audio as float32 numpy array

    Example:
        >>> import audiodecode.fast as fast
        >>> audio = fast.decode("audio.wav", target_sr=16000, mono=True)
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath

    # Get cached backend (singleton)
    backend = get_backend_for_file(filepath)

    # Direct decode - no intermediate objects
    audio_data = backend.decode(filepath, target_sr, mono, offset, duration)

    # Return raw array (zero-copy)
    return audio_data.data


def decode_info(filepath: str | Path) -> dict[str, int | float | str]:
    """
    Fast info extraction - get metadata without decoding.

    Args:
        filepath: Path to audio file

    Returns:
        Dictionary with file metadata
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath

    # Get cached backend
    backend = get_backend_for_file(filepath)

    # Get info
    info = backend.get_info(filepath)

    return {
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "duration": info.duration,
        "samples": info.samples,
        "format": info.format,
    }


# Convenience alias
load = decode  # librosa-compatible name
