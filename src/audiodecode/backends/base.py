"""
Abstract base class for audio decoding backends.

All backends must implement this interface to be compatible with AudioDecoder.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class AudioInfo:
    """Metadata about an audio file."""

    sample_rate: int
    channels: int
    duration: float  # in seconds
    samples: int
    format: str  # e.g., "mp3", "flac", "wav"


@dataclass
class AudioData:
    """Decoded audio data with metadata."""

    data: npt.NDArray[np.float32]  # Shape: (samples,) for mono, (samples, channels) for stereo
    sample_rate: int
    channels: int

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.data.shape[0] / self.sample_rate

    @property
    def samples(self) -> int:
        """Number of samples."""
        return self.data.shape[0]


class AudioBackend(ABC):
    """
    Abstract base class for audio decoding backends.

    All backends must implement these methods to provide consistent
    decoding behavior across different underlying libraries.
    """

    @abstractmethod
    def supports_format(self, filepath: Path) -> bool:
        """
        Check if this backend supports the given file format.

        Args:
            filepath: Path to the audio file

        Returns:
            True if this backend can decode the file, False otherwise
        """
        pass

    @abstractmethod
    def get_info(self, filepath: Path) -> AudioInfo:
        """
        Extract metadata from an audio file without decoding.

        Args:
            filepath: Path to the audio file

        Returns:
            AudioInfo with file metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or corrupted
        """
        pass

    @abstractmethod
    def decode(
        self,
        filepath: Path,
        target_sr: int | None = None,
        mono: bool = False,
        offset: float = 0.0,
        duration: float | None = None,
    ) -> AudioData:
        """
        Decode an audio file to PCM data.

        Args:
            filepath: Path to the audio file
            target_sr: Target sample rate (None = keep original)
            mono: Convert to mono if True
            offset: Start reading after this time (in seconds)
            duration: Only load this much audio (in seconds)

        Returns:
            AudioData with decoded PCM data as float32 numpy array

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or corrupted
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging/debugging."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
