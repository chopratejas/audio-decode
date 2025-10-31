"""
PyAV backend for MP3, AAC, M4A, OGG, and Opus decoding.

This backend uses PyAV (FFmpeg bindings) for native audio decoding
without subprocess overhead. It's ideal for compressed formats like MP3 and AAC.
"""

from pathlib import Path

import av
import numpy as np

try:
    import soxr
    HAS_SOXR = True
except ImportError:
    HAS_SOXR = False
    # Fallback to scipy for resampling
    from scipy import signal as scipy_signal

from audiodecode.backends.base import AudioBackend, AudioData, AudioInfo


class PyAVBackend(AudioBackend):
    """
    Audio backend using PyAV (FFmpeg bindings).

    Supports: MP3, AAC, M4A, OGG (Vorbis/Opus), Opus
    Does not support: WAV, FLAC (use soundfile backend for these)

    Features:
    - Native FFmpeg decoding (no subprocess)
    - Zero-copy frame conversion where possible
    - Efficient seeking and partial decoding
    - High-quality resampling with soxr (or scipy fallback)
    - Efficient stereo-to-mono conversion
    """

    # Formats best handled by FFmpeg
    SUPPORTED_FORMATS = {".mp3", ".aac", ".m4a", ".ogg", ".opus"}

    def supports_format(self, filepath: Path) -> bool:
        """Check if this backend supports the file format."""
        return filepath.suffix.lower() in self.SUPPORTED_FORMATS

    def get_info(self, filepath: Path) -> AudioInfo:
        """
        Extract audio file metadata without decoding.

        Args:
            filepath: Path to audio file

        Returns:
            AudioInfo with file metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or corrupted
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        if not self.supports_format(filepath):
            raise ValueError(f"PyAV backend does not support format: {filepath.suffix}")

        try:
            container = av.open(str(filepath))

            # Get first audio stream
            if not container.streams.audio:
                raise ValueError(f"No audio stream found in file: {filepath}")

            stream = container.streams.audio[0]

            # Extract metadata
            sample_rate = stream.sample_rate
            channels = stream.channels

            # Calculate duration
            # Try to get duration from stream first
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration) / av.time_base
            else:
                # Fallback: estimate from frames (slow)
                duration = 0.0
                for frame in container.decode(stream):
                    duration += float(frame.samples) / sample_rate

            # Calculate total samples
            samples = int(duration * sample_rate)

            container.close()

            return AudioInfo(
                sample_rate=sample_rate,
                channels=channels,
                duration=duration,
                samples=samples,
                format=filepath.suffix[1:].lower(),  # Remove leading dot
            )

        except (av.AVError, OSError) as e:
            raise ValueError(f"Failed to read audio file info: {e}") from e

    def decode(
        self,
        filepath: Path,
        target_sr: int | None = None,
        mono: bool = False,
        offset: float = 0.0,
        duration: float | None = None,
    ) -> AudioData:
        """
        Decode audio file to PCM data.

        Args:
            filepath: Path to audio file
            target_sr: Target sample rate (None = keep original)
            mono: Convert to mono if True
            offset: Start reading after this time (in seconds)
            duration: Only load this much audio (in seconds)

        Returns:
            AudioData with decoded PCM as float32 numpy array

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or corrupted
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        if not self.supports_format(filepath):
            raise ValueError(f"PyAV backend does not support format: {filepath.suffix}")

        try:
            container = av.open(str(filepath))

            # Get first audio stream
            if not container.streams.audio:
                raise ValueError(f"No audio stream found in file: {filepath}")

            stream = container.streams.audio[0]
            original_sr = stream.sample_rate
            original_channels = stream.channels

            # FAST PATH: No offset (skip seeking)
            # Note: PyAV seeking is not frame-accurate for compressed formats like MP3
            # We seek to approximately the right position, then decode from there
            if offset > 0.0:
                # Seek to offset position (backward to ensure we don't skip content)
                # Convert seconds to AV_TIME_BASE units (microseconds)
                seek_target = int(offset * av.time_base)
                container.seek(seek_target, backward=True)

            # Decode frames and accumulate audio data
            frames_data = []
            total_samples = 0
            target_samples = int(duration * original_sr) if duration is not None else None

            for frame in container.decode(stream):
                # Convert frame to numpy array (zero-copy where possible)
                # PyAV returns planar format, need to handle properly
                array = frame.to_ndarray()

                # Handle different array formats
                # frame.to_ndarray() returns shape:
                # - (channels, samples) for planar formats
                # - (samples, channels) for packed formats
                # We want (samples, channels)

                if array.ndim == 1:
                    # Mono audio
                    samples = array.reshape(-1, 1)
                elif array.shape[0] == original_channels and array.shape[1] > original_channels:
                    # Planar format: (channels, samples) -> (samples, channels)
                    samples = array.T
                else:
                    # Already in correct format: (samples, channels)
                    samples = array

                frames_data.append(samples)
                total_samples += samples.shape[0]

                # Stop if we've reached target duration
                if target_samples is not None and total_samples >= target_samples:
                    break

            container.close()

            if not frames_data:
                # No audio data decoded
                return AudioData(
                    data=np.array([], dtype=np.float32),
                    sample_rate=original_sr,
                    channels=original_channels,
                )

            # Concatenate all frames
            audio = np.vstack(frames_data)

            # Trim to exact duration if specified
            if target_samples is not None:
                audio = audio[:target_samples]

            # Convert to float32 and normalize
            # PyAV typically returns int16 or int32, need to normalize to [-1.0, 1.0]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Ensure we have correct shape: (samples, channels)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            channels = audio.shape[1] if audio.ndim > 1 else 1

            # Convert to mono if requested
            if mono and channels > 1:
                audio = audio.mean(axis=1, keepdims=False).astype(np.float32)
                channels = 1

            # Resample if target_sr is specified and different from original
            if target_sr is not None and target_sr != original_sr:
                audio = self._resample(audio, original_sr, target_sr)
                sample_rate = target_sr
            else:
                sample_rate = original_sr

            # Ensure mono audio is 1D
            if channels == 1 and audio.ndim == 2:
                audio = audio.squeeze()

            return AudioData(
                data=audio,
                sample_rate=sample_rate,
                channels=channels,
            )

        except (av.AVError, OSError) as e:
            raise ValueError(f"Failed to decode audio file: {e}") from e

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses soxr if available (high quality), otherwise falls back to scipy.

        Args:
            audio: Audio data (1D or 2D)
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio as float32
        """
        if orig_sr == target_sr:
            return audio

        if HAS_SOXR:
            # Use soxr for high-quality resampling
            resampled = soxr.resample(
                audio,
                orig_sr,
                target_sr,
                quality="HQ",  # High quality
            )
            return resampled.astype(np.float32)
        else:
            # Fallback to scipy (slower but widely available)
            # Calculate number of output samples
            num_samples = int(len(audio) * target_sr / orig_sr)

            if audio.ndim == 1:
                # Mono
                resampled = scipy_signal.resample(audio, num_samples)
            else:
                # Stereo - resample each channel
                resampled = np.zeros((num_samples, audio.shape[1]), dtype=np.float32)
                for ch in range(audio.shape[1]):
                    resampled[:, ch] = scipy_signal.resample(audio[:, ch], num_samples)

            return resampled.astype(np.float32)

    @property
    def name(self) -> str:
        """Backend name."""
        return "pyav"

    def __repr__(self) -> str:
        return f"PyAVBackend(name='{self.name}', formats={self.SUPPORTED_FORMATS})"
