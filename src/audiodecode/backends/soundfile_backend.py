"""
Soundfile backend for WAV, FLAC, and OGG decoding.

This backend uses the soundfile library (libsndfile) for native,
zero-copy audio decoding without subprocess overhead.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import soxr
    HAS_SOXR = True
except ImportError:
    HAS_SOXR = False
    # Fallback to scipy for resampling
    from scipy import signal as scipy_signal

from audiodecode.backends.base import AudioBackend, AudioData, AudioInfo


class SoundfileBackend(AudioBackend):
    """
    Audio backend using soundfile (libsndfile).

    Supports: WAV, FLAC, OGG (Vorbis)
    Does not support: MP3, AAC, Opus (use PyAV backend for these)

    Features:
    - Zero-copy decoding where possible
    - Native C library performance (no subprocess)
    - High-quality resampling with soxr (or scipy fallback)
    - Efficient stereo-to-mono conversion
    """

    # Formats supported by libsndfile
    SUPPORTED_FORMATS = {".wav", ".flac", ".ogg"}

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
            raise ValueError(f"Soundfile backend does not support format: {filepath.suffix}")

        try:
            info = sf.info(str(filepath))

            return AudioInfo(
                sample_rate=info.samplerate,
                channels=info.channels,
                duration=info.duration,
                samples=info.frames,
                format=info.format.lower() if hasattr(info, "format") else filepath.suffix[1:],
            )
        except RuntimeError as e:
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
        if not self.supports_format(filepath):
            raise ValueError(f"Soundfile backend does not support format: {filepath.suffix}")

        try:
            # FAST PATH: No offset/duration and no resampling
            if offset == 0.0 and duration is None and target_sr is None:
                # Direct read - minimal overhead
                # Skip filepath.exists() check - soundfile will fail anyway if missing
                audio, sr = sf.read(
                    str(filepath),
                    dtype="float32",
                    always_2d=False,
                )

                # Determine channels from shape
                if audio.ndim == 1:
                    channels = 1
                else:
                    channels = audio.shape[1]

                # Convert to mono if requested
                if mono and channels > 1:
                    audio = audio.mean(axis=1).astype(np.float32)
                    channels = 1

                return AudioData(
                    data=audio,
                    sample_rate=sr,
                    channels=channels,
                )

            # SLOW PATH: Need offset/duration or resampling
            # Check file exists for better error messages
            if not filepath.exists():
                raise FileNotFoundError(f"Audio file not found: {filepath}")

            # Get file info to calculate frame positions
            info = sf.info(str(filepath))
            original_sr = info.samplerate

            # Calculate start and stop frames
            start_frame = int(offset * original_sr) if offset > 0 else 0

            if duration is not None:
                num_frames = int(duration * original_sr)
            else:
                num_frames = -1  # Read until end

            # Read audio data
            audio, sr = sf.read(
                str(filepath),
                start=start_frame,
                frames=num_frames,
                dtype="float32",
                always_2d=False,  # Return 1D for mono
            )

            # Ensure we have the right shape
            # soundfile returns (samples,) for mono, (samples, channels) for multi-channel
            if audio.ndim == 1:
                channels = 1
            else:
                channels = audio.shape[1]

            # Convert to mono if requested
            if mono and channels > 1:
                # Average across channels
                audio = audio.mean(axis=1).astype(np.float32)
                channels = 1

            # Resample if target_sr is specified and different from original
            if target_sr is not None and target_sr != sr:
                audio = self._resample(audio, sr, target_sr)
                sr = target_sr

            # Ensure mono audio is 1D
            if channels == 1 and audio.ndim == 2:
                audio = audio.squeeze()

            return AudioData(
                data=audio,
                sample_rate=sr,
                channels=channels,
            )

        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except RuntimeError as e:
            # Check if it's a file not found error from soundfile
            if "Error opening" in str(e) and "System error" in str(e):
                raise FileNotFoundError(f"Audio file not found: {filepath}") from e
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
        return "soundfile"

    def __repr__(self) -> str:
        return f"SoundfileBackend(name='{self.name}', formats={self.SUPPORTED_FORMATS})"
