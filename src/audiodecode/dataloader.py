"""
Auto-tuned PyTorch DataLoader for audio ML pipelines.

This module provides AudioDataLoader, which automatically tunes DataLoader
parameters (num_workers, prefetch_factor, etc.) based on system capabilities
to maximize GPU utilization without manual configuration.
"""

import os
import platform
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Union

import psutil
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from audiodecode.dataset import AudioDataset, AudioDatasetWithCache


class AudioDataLoader:
    """
    Auto-tuned DataLoader for audio ML pipelines.

    Automatically configures optimal DataLoader parameters based on:
    - CPU core count
    - Available RAM
    - Batch size
    - File size estimates
    - GPU availability

    This eliminates the need for manual tuning of num_workers, prefetch_factor,
    pin_memory, and other DataLoader parameters.

    Args:
        files: List of audio file paths
        labels: Optional list of labels
        batch_size: Batch size for training
        target_sr: Target sample rate
        mono: Convert to mono if True
        duration: Fixed duration to extract
        offset: Offset in seconds to start reading
        transform: Optional transformation function
        feature_extractor: Optional feature extraction function
        shuffle: Shuffle dataset
        drop_last: Drop last incomplete batch
        device: Target device ('cpu', 'cuda', or torch.device)
        num_workers: Override auto-tuned num_workers (None = auto)
        prefetch_factor: Override auto-tuned prefetch_factor (None = auto)
        persistent_workers: Keep workers alive between epochs (None = auto)
        cache_decoded: Enable caching of decoded audio
        max_cache_size: Maximum cache size

    Example:
        >>> loader = AudioDataLoader(
        ...     files=audio_files,
        ...     labels=labels,
        ...     batch_size=32,
        ...     target_sr=16000,
        ...     device='cuda'
        ... )
        >>> for batch, labels in loader:
        ...     outputs = model(batch)
    """

    def __init__(
        self,
        files: Union[List[str], List[Path]],
        labels: Optional[List[Any]] = None,
        batch_size: int = 32,
        target_sr: Optional[int] = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0,
        transform: Optional[Callable[[NDArray], NDArray]] = None,
        feature_extractor: Optional[Callable[[NDArray], NDArray]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: Optional[bool] = None,
        cache_decoded: bool = False,
        max_cache_size: Optional[int] = None,
    ):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.target_sr = target_sr
        self.device = self._parse_device(device)

        # Auto-tune parameters if not provided
        self.num_workers = num_workers if num_workers is not None else self._auto_tune_workers()
        self.prefetch_factor = (
            prefetch_factor
            if prefetch_factor is not None
            else self._auto_tune_prefetch(batch_size)
        )
        self.persistent_workers = (
            persistent_workers
            if persistent_workers is not None
            else (self.num_workers > 0)
        )
        self.pin_memory = self.device.type == "cuda"

        # Create dataset
        if cache_decoded:
            self.dataset = AudioDatasetWithCache(
                files=files,
                labels=labels,
                target_sr=target_sr,
                mono=mono,
                duration=duration,
                offset=offset,
                transform=transform,
                feature_extractor=feature_extractor,
                cache_decoded=cache_decoded,
                max_cache_size=max_cache_size,
            )
        else:
            self.dataset = AudioDataset(
                files=files,
                labels=labels,
                target_sr=target_sr,
                mono=mono,
                duration=duration,
                offset=offset,
                transform=transform,
                feature_extractor=feature_extractor,
            )

        # Create DataLoader with optimized settings
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=drop_last,
        )

    def _parse_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Parse device string/object into torch.device."""
        if device is None:
            # Auto-detect: use CUDA if available
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            return torch.device(device)
        return device

    def _auto_tune_workers(self) -> int:
        """
        Automatically determine optimal number of DataLoader workers.

        Strategy:
        - CPU cores: Use 1/2 to 3/4 of available cores
        - Memory: Ensure enough RAM for workers
        - Platform: Different defaults for macOS/Linux/Windows
        - Conservative: Avoid oversubscription

        Returns:
            Optimal number of workers
        """
        cpu_count = os.cpu_count() or 4

        # Get available memory (in GB)
        available_ram = psutil.virtual_memory().available / (1024**3)

        # Platform-specific tuning
        system = platform.system()

        if system == "Darwin":  # macOS
            # macOS: More conservative due to different threading model
            # Core Audio backend is already efficient
            max_workers = min(4, cpu_count // 2)
        elif system == "Linux":
            # Linux: Can handle more workers efficiently
            # FFmpeg subprocess overhead benefits from parallelism
            max_workers = min(8, cpu_count * 3 // 4)
        else:  # Windows
            # Windows: Conservative due to process spawning overhead
            max_workers = min(4, cpu_count // 2)

        # Memory constraint: ~500MB per worker (conservative estimate)
        ram_limited_workers = int(available_ram / 0.5)

        # Take minimum of CPU-limited and RAM-limited
        num_workers = max(2, min(max_workers, ram_limited_workers))

        return num_workers

    def _auto_tune_prefetch(self, batch_size: int) -> int:
        """
        Automatically determine optimal prefetch_factor.

        Strategy:
        - Larger batches: Lower prefetch_factor (already large memory footprint)
        - Smaller batches: Higher prefetch_factor (keep pipeline full)
        - Memory-aware: Don't prefetch too much

        Args:
            batch_size: The batch size being used

        Returns:
            Optimal prefetch_factor
        """
        # Get available memory
        available_ram = psutil.virtual_memory().available / (1024**3)

        # Heuristic based on batch size and memory
        if batch_size >= 64:
            # Large batches: prefetch less
            prefetch = 2
        elif batch_size >= 32:
            # Medium batches: moderate prefetch
            prefetch = 3
        else:
            # Small batches: prefetch more
            prefetch = 4

        # If low memory, reduce prefetch
        if available_ram < 4.0:
            prefetch = 2

        return prefetch

    def __iter__(self) -> Iterator:
        """Return iterator over batches."""
        return iter(self.loader)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)

    def get_config(self) -> dict:
        """
        Get the auto-tuned configuration.

        Returns:
            Dictionary with DataLoader configuration
        """
        return {
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "num_files": len(self.files),
            "num_batches": len(self.loader),
        }

    def print_config(self) -> None:
        """Print the auto-tuned configuration."""
        config = self.get_config()
        print("AudioDataLoader Configuration:")
        print(f"  Device: {config['device']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Num workers: {config['num_workers']}")
        print(f"  Prefetch factor: {config['prefetch_factor']}")
        print(f"  Persistent workers: {config['persistent_workers']}")
        print(f"  Pin memory: {config['pin_memory']}")
        print(f"  Total files: {config['num_files']}")
        print(f"  Total batches: {config['num_batches']}")


def create_train_val_loaders(
    train_files: List[Union[str, Path]],
    train_labels: List[Any],
    val_files: List[Union[str, Path]],
    val_labels: List[Any],
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    **kwargs: Any,
) -> tuple[AudioDataLoader, AudioDataLoader]:
    """
    Convenience function to create train and validation loaders.

    Args:
        train_files: Training audio files
        train_labels: Training labels
        val_files: Validation audio files
        val_labels: Validation labels
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        **kwargs: Additional arguments passed to AudioDataLoader

    Returns:
        (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = create_train_val_loaders(
        ...     train_files, train_labels,
        ...     val_files, val_labels,
        ...     batch_size=32,
        ...     target_sr=16000,
        ...     device='cuda'
        ... )
    """
    if val_batch_size is None:
        val_batch_size = batch_size

    # Training loader with shuffling
    train_loader = AudioDataLoader(
        files=train_files,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Validation loader without shuffling
    val_loader = AudioDataLoader(
        files=val_files,
        labels=val_labels,
        batch_size=val_batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, val_loader
