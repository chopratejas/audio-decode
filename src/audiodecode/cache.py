"""
LRU cache for decoded audio data.

This provides memory-efficient caching similar to librosa's behavior,
where frequently accessed files are kept in memory.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
import hashlib
import numpy as np


class AudioCache:
    """
    LRU cache for decoded audio data.

    Caches decoded audio in memory to avoid re-decoding the same file.
    Uses file path + decode parameters as cache key.
    """

    def __init__(self, maxsize: int = 128):
        """
        Initialize audio cache.

        Args:
            maxsize: Maximum number of cached audio files (default: 128)
                    Set to None for unlimited cache
        """
        self._cache = {}
        self._maxsize = maxsize
        self._access_order = []  # For LRU eviction

    def _make_key(
        self,
        filepath: Path,
        target_sr: Optional[int],
        mono: bool,
        offset: float,
        duration: Optional[float]
    ) -> str:
        """Create cache key from decode parameters."""
        # Use file path + modification time + parameters
        try:
            mtime = filepath.stat().st_mtime
        except:
            mtime = 0

        key_parts = [
            str(filepath),
            str(mtime),
            str(target_sr),
            str(mono),
            str(offset),
            str(duration),
        ]

        # Hash to keep keys short
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        filepath: Path,
        target_sr: Optional[int],
        mono: bool,
        offset: float,
        duration: Optional[float]
    ) -> Optional[np.ndarray]:
        """
        Get cached audio data if available.

        Returns:
            Cached audio array or None if not in cache
        """
        key = self._make_key(filepath, target_sr, mono, offset, duration)

        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key].copy()  # Return copy to prevent mutation

        return None

    def put(
        self,
        filepath: Path,
        target_sr: Optional[int],
        mono: bool,
        offset: float,
        duration: Optional[float],
        audio: np.ndarray
    ) -> None:
        """
        Cache decoded audio data.

        Args:
            filepath: Path to audio file
            target_sr: Target sample rate used
            mono: Whether mono conversion was applied
            offset: Offset used for decoding
            duration: Duration used for decoding
            audio: Decoded audio array to cache
        """
        key = self._make_key(filepath, target_sr, mono, offset, duration)

        # Evict oldest if at capacity
        if self._maxsize is not None and len(self._cache) >= self._maxsize:
            if key not in self._cache:  # Don't evict if updating
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]

        # Store copy to prevent external mutation
        self._cache[key] = audio.copy()

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._access_order.clear()

    def info(self) -> dict:
        """Get cache statistics."""
        total_size_mb = sum(
            arr.nbytes / (1024 * 1024)
            for arr in self._cache.values()
        )

        return {
            "entries": len(self._cache),
            "maxsize": self._maxsize,
            "total_memory_mb": total_size_mb,
        }


# Global cache instance (like librosa)
_global_cache = AudioCache(maxsize=128)


def get_cache() -> AudioCache:
    """Get the global audio cache instance."""
    return _global_cache


def set_cache_size(maxsize: int) -> None:
    """
    Set the maximum cache size.

    Args:
        maxsize: Maximum number of files to cache (None for unlimited)
    """
    global _global_cache
    _global_cache = AudioCache(maxsize=maxsize)


def clear_cache() -> None:
    """Clear the global audio cache."""
    _global_cache.clear()
