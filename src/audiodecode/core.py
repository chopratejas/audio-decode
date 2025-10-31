"""
Main AudioDecoder class - high-level API for audio decoding.

PERFORMANCE OPTIMIZATIONS:
- Lazy imports to reduce startup overhead
- Singleton backend registry to avoid recreating backends
- Direct decode path with minimal object creation
- Zero-copy numpy array handling
"""

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from audiodecode.backends.base import AudioBackend, AudioData
from audiodecode.registry import get_backend_for_file
from audiodecode.cache import get_cache


class AudioDecoder:
    """
    High-level audio decoder with automatic backend selection.

    Examples:
        >>> # Basic usage
        >>> audio = AudioDecoder("audio.mp3").decode()

        >>> # Resample and convert to mono
        >>> audio = AudioDecoder("audio.mp3", target_sr=16000, mono=True).decode()

        >>> # Get file info without decoding
        >>> info = AudioDecoder("audio.mp3").info()
    """

    def __init__(
        self,
        filepath: str | Path,
        target_sr: int | None = None,
        mono: bool = False,
        output_format: Literal["numpy", "torch", "jax", "bytes"] = "numpy",
    ):
        """
        Initialize AudioDecoder.

        Args:
            filepath: Path to audio file
            target_sr: Target sample rate (None = keep original)
            mono: Convert to mono if True
            output_format: Output format ("numpy", "torch", "jax", "bytes")
        """
        self.filepath = Path(filepath)
        self.target_sr = target_sr
        self.mono = mono
        self.output_format = output_format

        if not self.filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {self.filepath}")

        # Select appropriate backend
        self.backend = self._select_backend()

    def _select_backend(self) -> AudioBackend:
        """
        Select the best backend for the file format.

        Uses singleton registry to avoid recreating backends.

        Returns:
            Cached backend instance

        Raises:
            ValueError: If no backend supports this format
        """
        return get_backend_for_file(self.filepath)

    def decode(
        self,
        offset: float = 0.0,
        duration: float | None = None,
        use_cache: bool = True,
    ) -> npt.NDArray[np.float32]:
        """
        Decode the audio file.

        Args:
            offset: Start reading after this time (in seconds)
            duration: Only load this much audio (in seconds)
            use_cache: Use cached data if available (default: True)

        Returns:
            Decoded audio as numpy array (or torch/jax tensor if specified)
        """
        # Try to get from cache first
        if use_cache:
            cache = get_cache()
            cached_audio = cache.get(
                self.filepath,
                self.target_sr,
                self.mono,
                offset,
                duration
            )

            if cached_audio is not None:
                # Cache hit! Return cached data
                audio_data = AudioData(
                    data=cached_audio,
                    sample_rate=self.target_sr or 0,  # Will be overwritten
                    channels=1 if self.mono else (2 if cached_audio.ndim > 1 else 1)
                )
                return self._convert_output(audio_data)

        # Cache miss - decode using backend
        audio_data = self.backend.decode(
            self.filepath,
            target_sr=self.target_sr,
            mono=self.mono,
            offset=offset,
            duration=duration,
        )

        # Store in cache
        if use_cache:
            cache = get_cache()
            cache.put(
                self.filepath,
                self.target_sr,
                self.mono,
                offset,
                duration,
                audio_data.data
            )

        # Convert to requested output format
        return self._convert_output(audio_data)

    def info(self) -> dict[str, any]:
        """
        Get file metadata without decoding.

        Returns:
            Dictionary with file metadata
        """
        info = self.backend.get_info(self.filepath)
        return {
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "duration": info.duration,
            "samples": info.samples,
            "format": info.format,
        }

    def _convert_output(self, audio_data: AudioData) -> npt.NDArray[np.float32]:
        """Convert AudioData to requested output format."""
        if self.output_format == "numpy":
            return audio_data.data

        elif self.output_format == "torch":
            try:
                import torch
                return torch.from_numpy(audio_data.data)
            except ImportError:
                raise ImportError(
                    "PyTorch not installed. Install with: pip install audiodecode[torch]"
                )

        elif self.output_format == "jax":
            try:
                import jax.numpy as jnp
                return jnp.array(audio_data.data)
            except ImportError:
                raise ImportError(
                    "JAX not installed. Install with: pip install audiodecode[jax]"
                )

        elif self.output_format == "bytes":
            # Convert float32 to int16 PCM bytes
            int16_data = (audio_data.data * 32767).astype(np.int16)
            return int16_data.tobytes()

        else:
            raise ValueError(
                f"Unknown output format: {self.output_format}. "
                f"Must be one of: numpy, torch, jax, bytes"
            )
