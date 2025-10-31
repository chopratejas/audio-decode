"""
Backend registry - singleton pattern for backend instances.

This avoids recreating backend objects and re-importing modules on every decode.
"""

from pathlib import Path
from typing import Dict, Type

from audiodecode.backends.base import AudioBackend


class BackendRegistry:
    """
    Singleton registry for audio backends.

    Backends are created once and reused for all subsequent decodes.
    This eliminates the overhead of:
    - Re-importing backend modules
    - Re-instantiating backend objects
    - Re-initializing backend state
    """

    _instance: "BackendRegistry | None" = None
    _backends: Dict[str, AudioBackend] = {}
    _backend_classes_loaded: bool = False

    def __new__(cls) -> "BackendRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_backend(self, filepath: Path) -> AudioBackend:
        """
        Get the best backend for the given file format.

        Args:
            filepath: Path to audio file

        Returns:
            Cached backend instance

        Raises:
            ValueError: If no backend supports this format
        """
        # Lazy load backend classes only once
        if not self._backend_classes_loaded:
            self._load_backend_classes()

        # Try each backend in priority order
        for backend_name, backend in self._backends.items():
            if backend.supports_format(filepath):
                return backend

        # No backend supports this format
        supported_formats = set()
        for backend in self._backends.values():
            if hasattr(backend, "SUPPORTED_FORMATS"):
                supported_formats.update(backend.SUPPORTED_FORMATS)

        raise ValueError(
            f"No backend supports format '{filepath.suffix}'. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )

    def _load_backend_classes(self) -> None:
        """
        Lazy load and instantiate all available backends.

        This happens once per process lifetime.
        """
        # Import backends lazily
        from audiodecode.backends.soundfile_backend import SoundfileBackend
        from audiodecode.backends.pyav_backend import PyAVBackend

        # Create singleton instances
        # Priority order: soundfile (lossless formats), then PyAV (compressed formats)
        self._backends = {
            "soundfile": SoundfileBackend(),
            "pyav": PyAVBackend(),
        }

        self._backend_classes_loaded = True

    def register_backend(self, name: str, backend: AudioBackend) -> None:
        """
        Register a custom backend.

        Args:
            name: Backend name
            backend: Backend instance
        """
        self._backends[name] = backend


# Global singleton instance
_registry = BackendRegistry()


def get_backend_for_file(filepath: Path) -> AudioBackend:
    """
    Get the appropriate backend for a file.

    Args:
        filepath: Path to audio file

    Returns:
        Cached backend instance
    """
    return _registry.get_backend(filepath)
