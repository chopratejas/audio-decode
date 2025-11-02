"""
PyTorch Dataset for efficient audio loading using AudioDecode backend.

This module provides AudioDataset, a drop-in replacement for custom PyTorch
datasets that handles audio loading, resampling, and optional augmentations
with minimal overhead.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from audiodecode import AudioDecoder


class AudioDataset(Dataset):
    """
    PyTorch Dataset for efficient audio loading.

    Uses AudioDecode backend for fast audio decoding with support for
    labels, metadata, and optional transformations.

    Args:
        files: List of audio file paths
        labels: Optional list of labels (one per file)
        target_sr: Target sample rate for resampling (None = no resampling)
        mono: Convert to mono if True
        duration: Fixed duration to extract (None = full file)
        offset: Offset in seconds to start reading
        transform: Optional callable to transform audio (e.g., augmentations)
        feature_extractor: Optional callable to extract features from audio
        return_path: If True, return (audio, label, path) instead of (audio, label)

    Example:
        >>> files = ["audio1.wav", "audio2.mp3"]
        >>> labels = [0, 1]
        >>> dataset = AudioDataset(files, labels, target_sr=16000)
        >>> audio, label = dataset[0]
        >>> print(audio.shape, label)
    """

    def __init__(
        self,
        files: Union[List[str], List[Path]],
        labels: Optional[Union[List[int], List[str], NDArray]] = None,
        target_sr: Optional[int] = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0,
        transform: Optional[Callable[[NDArray[np.float32]], NDArray[np.float32]]] = None,
        feature_extractor: Optional[Callable[[NDArray[np.float32]], NDArray]] = None,
        return_path: bool = False,
    ):
        self.files = [Path(f) for f in files]
        self.labels = labels
        self.target_sr = target_sr
        self.mono = mono
        self.duration = duration
        self.offset = offset
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.return_path = return_path

        # Validate inputs
        if labels is not None:
            if len(labels) != len(files):
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match number of files ({len(files)})"
                )
            # Convert labels to list if numpy array
            if isinstance(labels, np.ndarray):
                self.labels = labels.tolist()

        # Cache for file validation (optional)
        self._validated = False

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, Path]]:
        """
        Load and return a single audio sample.

        Args:
            idx: Index of the sample to load

        Returns:
            If return_path=False: (audio_tensor, label)
            If return_path=True: (audio_tensor, label, file_path)
        """
        file_path = self.files[idx]

        # Load audio using AudioDecode backend
        decoder = AudioDecoder(str(file_path), target_sr=self.target_sr, mono=self.mono)
        audio = decoder.decode(offset=self.offset, duration=self.duration)

        # Apply transformations (e.g., augmentations)
        if self.transform is not None:
            audio = self.transform(audio)

        # Extract features if specified
        if self.feature_extractor is not None:
            audio = self.feature_extractor(audio)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)

        # Get label
        label = self.labels[idx] if self.labels is not None else -1

        # Return with or without path
        if self.return_path:
            return audio_tensor, label, file_path
        return audio_tensor, label

    def get_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a file without loading the full audio.

        Args:
            idx: Index of the file

        Returns:
            Dictionary with file info (sample_rate, channels, duration, etc.)
        """
        file_path = self.files[idx]
        decoder = AudioDecoder(str(file_path))
        return decoder.info()

    def validate_files(self) -> List[int]:
        """
        Validate all files in the dataset.

        Returns:
            List of indices for files that failed validation
        """
        invalid_indices = []
        for idx, file_path in enumerate(self.files):
            if not file_path.exists():
                invalid_indices.append(idx)
                continue
            try:
                # Try to get info (lightweight operation)
                decoder = AudioDecoder(str(file_path))
                decoder.info()
            except Exception:
                invalid_indices.append(idx)

        self._validated = True
        return invalid_indices


class AudioDatasetWithCache(AudioDataset):
    """
    AudioDataset with built-in caching for repeated access.

    Useful when:
    - Dataset is small enough to fit in memory
    - Multiple epochs with augmentation
    - Repeated access to same files

    Args:
        Same as AudioDataset, plus:
        cache_decoded: Cache decoded audio in memory
        max_cache_size: Maximum number of files to cache (None = unlimited)

    Example:
        >>> dataset = AudioDatasetWithCache(
        ...     files, labels,
        ...     target_sr=16000,
        ...     cache_decoded=True,
        ...     max_cache_size=100
        ... )
    """

    def __init__(
        self,
        files: Union[List[str], List[Path]],
        labels: Optional[Union[List[int], List[str], NDArray]] = None,
        cache_decoded: bool = True,
        max_cache_size: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(files, labels, **kwargs)
        self.cache_decoded = cache_decoded
        self.max_cache_size = max_cache_size
        self._cache: Dict[int, NDArray[np.float32]] = {}

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, Path]]:
        """
        Load audio with caching support.
        """
        # Check cache first (only cache decoded audio, not transformed)
        if self.cache_decoded and idx in self._cache:
            audio = self._cache[idx].copy()
        else:
            file_path = self.files[idx]
            decoder = AudioDecoder(str(file_path), target_sr=self.target_sr, mono=self.mono)
            audio = decoder.decode(offset=self.offset, duration=self.duration)

            # Add to cache if enabled
            if self.cache_decoded:
                # Manage cache size
                if self.max_cache_size is not None and len(self._cache) >= self.max_cache_size:
                    # Simple FIFO eviction (could use LRU)
                    oldest_idx = next(iter(self._cache))
                    del self._cache[oldest_idx]

                self._cache[idx] = audio

        # Apply transformations (always fresh, not cached)
        if self.transform is not None:
            audio = self.transform(audio)

        # Extract features if specified
        if self.feature_extractor is not None:
            audio = self.feature_extractor(audio)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)

        # Get label
        label = self.labels[idx] if self.labels is not None else -1

        # Return with or without path
        if self.return_path:
            return audio_tensor, label, self.files[idx]
        return audio_tensor, label

    def clear_cache(self) -> None:
        """Clear the audio cache."""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Return number of cached items."""
        return len(self._cache)
