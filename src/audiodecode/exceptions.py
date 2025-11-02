"""
Custom exceptions for AudioDecode with actionable error messages.
"""

from typing import Optional, List


class AudioDecodeError(Exception):
    """Base exception for all AudioDecode errors."""
    pass


class DependencyError(AudioDecodeError):
    """Raised when an optional dependency is missing."""

    def __init__(self, feature: str, package: str, extras: Optional[str] = None):
        self.feature = feature
        self.package = package
        self.extras = extras

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        msg = f"AudioDecodeError: {self.feature} requires {self.package}\n\n"
        msg += f"This feature requires additional dependencies that aren't installed.\n\n"
        msg += "Install options:\n"

        if self.extras:
            msg += f"  pip:     pip install 'audiodecode[{self.extras}]'\n"
            msg += f"  uv:      uv pip install 'audiodecode[{self.extras}]'\n"
            msg += f"  poetry:  poetry add 'audiodecode[{self.extras}]'\n"
            msg += f"  conda:   conda install -c conda-forge {self.package}\n"
        else:
            msg += f"  pip:     pip install {self.package}\n"
            msg += f"  uv:      uv pip install {self.package}\n"
            msg += f"  poetry:  poetry add {self.package}\n"
            msg += f"  conda:   conda install -c conda-forge {self.package}\n"

        msg += "\nDocumentation: https://github.com/audiodecode/audiodecode#installation"

        return msg


class FileFormatError(AudioDecodeError):
    """Raised when an audio format is unsupported or corrupted."""

    def __init__(self, filepath: str, format_ext: Optional[str] = None, details: Optional[str] = None):
        self.filepath = filepath
        self.format_ext = format_ext
        self.details = details

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        msg = "AudioDecodeError: Unsupported or corrupted audio format\n\n"
        msg += f"File: {self.filepath}\n"

        if self.format_ext:
            msg += f"Format: {self.format_ext}\n\n"
            msg += "Supported formats:\n"
            msg += "  Lossless:    .wav, .flac\n"
            msg += "  Compressed:  .mp3, .aac, .m4a, .ogg\n\n"
            msg += "Suggestions:\n"
            msg += f"  1. Convert the file:\n"
            msg += f"     ffmpeg -i {self.filepath} -ar 16000 output.mp3\n\n"

        if self.details:
            msg += f"Details: {self.details}\n\n"

        msg += "  2. Check if file is corrupted:\n"
        msg += f"     ffprobe {self.filepath}\n\n"
        msg += f"Need {self.format_ext} support? Open an issue:\n"
        msg += "https://github.com/audiodecode/audiodecode/issues/new"

        return msg


class FileAccessError(AudioDecodeError):
    """Raised when a file cannot be accessed."""

    def __init__(self, filepath: str, reason: str, current_dir: Optional[str] = None):
        self.filepath = filepath
        self.reason = reason
        self.current_dir = current_dir

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        msg = "AudioDecodeError: Audio file not found or inaccessible\n\n"
        msg += f"File: {self.filepath}\n"
        msg += f"Reason: {self.reason}\n\n"
        msg += "Troubleshooting:\n"
        msg += "  1. Check the path is correct\n"
        msg += f"  2. Check file permissions: ls -la {self.filepath}\n"

        if self.current_dir:
            msg += f"  3. Current directory: {self.current_dir}\n"
            msg += "     (Tip: Use absolute paths to avoid confusion)\n"

        msg += "\nSupported formats: .wav, .mp3, .flac, .m4a, .ogg, .aac"

        return msg


class MemoryError(AudioDecodeError):
    """Raised when memory limits are exceeded."""

    def __init__(self, operation: str, size: Optional[int] = None, limit: Optional[int] = None):
        self.operation = operation
        self.size = size
        self.limit = limit

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        msg = f"AudioDecodeError: Memory limit exceeded during {self.operation}\n\n"

        if self.size and self.limit:
            msg += f"Requested: {self._format_bytes(self.size)}\n"
            msg += f"Limit: {self._format_bytes(self.limit)}\n\n"

        msg += "Suggestions:\n"
        msg += "  1. Process file in chunks (use offset/duration parameters)\n"
        msg += "  2. Use a lower sample rate (e.g., sr=16000 instead of 48000)\n"
        msg += "  3. Disable caching: use_cache=False\n"
        msg += "  4. Increase system memory or use a machine with more RAM\n"

        return msg

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if num_bytes < 1024.0:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.1f} TB"


class ValidationError(AudioDecodeError):
    """Raised when input validation fails."""

    def __init__(self, parameter: str, value: any, expected: str, suggestion: Optional[str] = None):
        self.parameter = parameter
        self.value = value
        self.expected = expected
        self.suggestion = suggestion

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        msg = f"AudioDecodeError: Invalid parameter value\n\n"
        msg += f"Parameter: {self.parameter}\n"
        msg += f"Value: {self.value}\n"
        msg += f"Expected: {self.expected}\n\n"

        if self.suggestion:
            msg += f"Suggestion: {self.suggestion}\n"

        return msg


class InferenceError(AudioDecodeError):
    """Raised when transcription fails."""

    def __init__(self, reason: str, suggestions: Optional[List[str]] = None):
        self.reason = reason
        self.suggestions = suggestions or []

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        msg = f"AudioDecodeError: Transcription failed\n\n"
        msg += f"Reason: {self.reason}\n\n"

        if self.suggestions:
            msg += "Suggestions:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"  {i}. {suggestion}\n"

        return msg


class CacheError(AudioDecodeError):
    """Raised when cache operations fail."""
    pass


class ModelLoadError(AudioDecodeError):
    """Raised when model loading fails."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        msg = f"AudioDecodeError: Failed to load Whisper model '{self.model_name}'\n\n"
        msg += f"Reason: {self.reason}\n\n"
        msg += "Available models:\n"
        msg += "  Fast:     tiny, base\n"
        msg += "  Balanced: small, medium\n"
        msg += "  Accurate: large-v3, distil-large-v3\n\n"
        msg += "Suggestions:\n"
        msg += "  1. Check internet connection (models download on first use)\n"
        msg += "  2. Try a different model size\n"
        msg += "  3. Check disk space for model cache\n"

        return msg
