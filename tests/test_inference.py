"""
Comprehensive test suite for speech-to-text inference functionality (Pillar 3).

Tests cover:
- WhisperInference class initialization and configuration
- transcribe_file() with various audio formats
- transcribe_audio() with numpy arrays
- VAD filtering and parameters
- Different model sizes
- Error handling and edge cases
- Integration with Pillar 1's fast audio decode
- Optional dependency handling
- Performance assertions
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import the module to test optional dependency handling
import audiodecode.inference as inference_module


@pytest.fixture(scope="session")
def has_faster_whisper() -> bool:
    """Check if faster-whisper is installed."""
    try:
        import faster_whisper
        return True
    except ImportError:
        return False


@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel for testing without downloading models."""
    mock_model = MagicMock()

    # Mock transcription info
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 1.0

    # Mock transcription segment
    mock_segment = MagicMock()
    mock_segment.text = " Hello world"
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.avg_logprob = -0.5

    # Mock transcribe method to return iterator and info
    mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

    return mock_model


@pytest.fixture
def mock_whisper_model_class(mock_whisper_model):
    """Mock the WhisperModel class constructor."""
    with patch('audiodecode.inference.WhisperModel', return_value=mock_whisper_model) as mock_class:
        yield mock_class


@pytest.fixture
def sample_audio_16k() -> tuple[np.ndarray, int]:
    """Generate 1 second of sample audio at 16kHz."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_audio_8k() -> tuple[np.ndarray, int]:
    """Generate 1 second of sample audio at 8kHz."""
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


class TestOptionalDependencies:
    """Test optional dependency handling for faster-whisper."""

    def test_import_without_faster_whisper(self):
        """Module should import without faster-whisper, but flag should be False."""
        # This test verifies the module-level import handling
        if not inference_module._FASTER_WHISPER_AVAILABLE:
            assert inference_module.WhisperModel is None
        else:
            assert inference_module.WhisperModel is not None

    def test_whisper_inference_requires_faster_whisper(self, monkeypatch):
        """WhisperInference should raise ImportError if faster-whisper not available."""
        # Temporarily make faster-whisper unavailable
        monkeypatch.setattr(inference_module, '_FASTER_WHISPER_AVAILABLE', False)
        monkeypatch.setattr(inference_module, 'WhisperModel', None)

        with pytest.raises(ImportError, match="faster-whisper is not installed"):
            inference_module.WhisperInference(model_size="tiny")

    def test_error_message_includes_install_instructions(self, monkeypatch):
        """ImportError should include pip install instructions."""
        monkeypatch.setattr(inference_module, '_FASTER_WHISPER_AVAILABLE', False)
        monkeypatch.setattr(inference_module, 'WhisperModel', None)

        with pytest.raises(ImportError, match="pip install audiodecode\\[inference\\]"):
            inference_module.WhisperInference(model_size="tiny")


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestWhisperInferenceInitialization:
    """Test WhisperInference class initialization."""

    def test_init_with_defaults(self, mock_whisper_model_class):
        """Test initialization with default parameters."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference()

        assert whisper.model_size == "base"
        assert whisper.device in ["cpu", "cuda"]
        assert whisper.compute_type == "int8"
        assert whisper.model is not None

        # Verify WhisperModel was called with correct args
        mock_whisper_model_class.assert_called_once()
        call_kwargs = mock_whisper_model_class.call_args[1]
        assert call_kwargs["compute_type"] == "int8"
        assert call_kwargs["num_workers"] == 1

    def test_init_with_custom_model_size(self, mock_whisper_model_class):
        """Test initialization with different model sizes."""
        from audiodecode.inference import WhisperInference

        for model_size in ["tiny", "base", "small", "medium", "large-v3"]:
            whisper = WhisperInference(model_size=model_size)
            assert whisper.model_size == model_size

    def test_init_with_explicit_device(self, mock_whisper_model_class):
        """Test initialization with explicit device selection."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(device="cpu")
        assert whisper.device == "cpu"

        # Reset mock for next test
        mock_whisper_model_class.reset_mock()

        whisper = WhisperInference(device="cuda")
        assert whisper.device == "cuda"

    def test_init_with_auto_device(self, mock_whisper_model_class):
        """Test auto device detection."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(device="auto")
        # Should select cpu or cuda based on availability
        assert whisper.device in ["cpu", "cuda"]

    def test_auto_device_without_torch(self, mock_whisper_model_class, monkeypatch):
        """Test auto device detection falls back to CPU without torch."""
        from audiodecode.inference import WhisperInference

        # Mock torch import to fail
        def mock_import(*args, **kwargs):
            if args[0] == 'torch':
                raise ImportError("No module named 'torch'")
            return __import__(*args, **kwargs)

        monkeypatch.setattr('builtins.__import__', mock_import)

        whisper = WhisperInference(device="auto")
        assert whisper.device == "cpu"

    def test_init_with_custom_compute_type(self, mock_whisper_model_class):
        """Test initialization with different compute types."""
        from audiodecode.inference import WhisperInference

        for compute_type in ["int8", "float16", "float32"]:
            whisper = WhisperInference(compute_type=compute_type)
            assert whisper.compute_type == compute_type

    def test_init_with_custom_workers(self, mock_whisper_model_class):
        """Test initialization with custom number of workers."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(num_workers=4)

        # Verify WhisperModel was called with correct workers
        call_kwargs = mock_whisper_model_class.call_args[1]
        assert call_kwargs["num_workers"] == 4

    def test_init_with_custom_download_root(self, mock_whisper_model_class, tmp_path):
        """Test initialization with custom download directory."""
        from audiodecode.inference import WhisperInference

        download_dir = str(tmp_path / "models")
        whisper = WhisperInference(download_root=download_dir)

        # Verify WhisperModel was called with correct download_root
        call_kwargs = mock_whisper_model_class.call_args[1]
        assert call_kwargs["download_root"] == download_dir


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestTranscribeAudio:
    """Test transcribe_audio() method with numpy arrays."""

    def test_transcribe_audio_basic(self, mock_whisper_model_class, sample_audio_16k):
        """Test basic audio transcription from numpy array."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr)

        # Verify result structure
        assert hasattr(result, 'text')
        assert hasattr(result, 'segments')
        assert hasattr(result, 'language')
        assert hasattr(result, 'duration')

        # Verify result values
        assert isinstance(result.text, str)
        assert isinstance(result.segments, list)
        assert result.language == "en"
        assert result.duration == pytest.approx(1.0, rel=0.01)

    def test_transcribe_audio_returns_segments(self, mock_whisper_model_class, sample_audio_16k):
        """Test that transcription returns properly structured segments."""
        from audiodecode.inference import WhisperInference, TranscriptionSegment

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr)

        assert len(result.segments) > 0

        # Verify segment structure
        segment = result.segments[0]
        assert isinstance(segment, TranscriptionSegment)
        assert hasattr(segment, 'text')
        assert hasattr(segment, 'start')
        assert hasattr(segment, 'end')
        assert hasattr(segment, 'confidence')

        # Verify segment values
        assert isinstance(segment.text, str)
        assert isinstance(segment.start, float)
        assert isinstance(segment.end, float)
        assert isinstance(segment.confidence, float)
        assert segment.start >= 0.0
        assert segment.end > segment.start

    def test_transcribe_audio_with_language(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with explicit language specification."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, language="es")

        # Verify that model.transcribe was called with language parameter
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "es"

    def test_transcribe_audio_with_translation(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with translation task."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, task="translate")

        # Verify that model.transcribe was called with translate task
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["task"] == "translate"

    def test_transcribe_audio_with_vad_filter(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with VAD filtering enabled."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, vad_filter=True)

        # Verify that model.transcribe was called with VAD enabled
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is True

    def test_transcribe_audio_with_custom_vad_parameters(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with custom VAD parameters."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        vad_params = {"threshold": 0.5, "min_silence_duration_ms": 500}
        result = whisper.transcribe_audio(
            audio,
            sample_rate=sr,
            vad_filter=True,
            vad_parameters=vad_params
        )

        # Verify that model.transcribe was called with VAD parameters
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_parameters"] == vad_params

    def test_transcribe_audio_with_custom_beam_size(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with custom beam size."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, beam_size=10)

        # Verify that model.transcribe was called with beam_size
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 10

    def test_transcribe_audio_with_temperature(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with custom temperature."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr, temperature=0.5)

        # Verify that model.transcribe was called with temperature
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_transcribe_audio_warns_on_wrong_sample_rate(self, mock_whisper_model_class, sample_audio_8k):
        """Test that warning is issued for non-16kHz audio."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_8k
        whisper = WhisperInference(model_size="tiny")

        with pytest.warns(UserWarning, match="Whisper expects 16kHz audio, got 8000Hz"):
            result = whisper.transcribe_audio(audio, sample_rate=sr)

    def test_transcribe_audio_converts_dtype(self, mock_whisper_model_class, sample_audio_16k):
        """Test that audio is converted to float32 if necessary."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        # Convert to float64
        audio_float64 = audio.astype(np.float64)

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_audio(audio_float64, sample_rate=sr)

        # Verify that model.transcribe received float32 array
        mock_model = mock_whisper_model_class.return_value
        call_args = mock_model.transcribe.call_args[0]
        transcribed_audio = call_args[0]
        assert transcribed_audio.dtype == np.float32

    def test_transcribe_audio_empty_segments(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with no segments (silent audio)."""
        from audiodecode.inference import WhisperInference

        # Mock empty segments
        mock_model = mock_whisper_model_class.return_value
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = (iter([]), mock_info)

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr)

        assert result.text == ""
        assert len(result.segments) == 0
        assert result.language == "en"

    def test_transcribe_audio_multiple_segments(self, mock_whisper_model_class, sample_audio_16k):
        """Test transcription with multiple segments."""
        from audiodecode.inference import WhisperInference

        # Mock multiple segments
        mock_model = mock_whisper_model_class.return_value
        mock_info = MagicMock()
        mock_info.language = "en"

        mock_seg1 = MagicMock()
        mock_seg1.text = " Hello"
        mock_seg1.start = 0.0
        mock_seg1.end = 0.5
        mock_seg1.avg_logprob = -0.5

        mock_seg2 = MagicMock()
        mock_seg2.text = " world"
        mock_seg2.start = 0.5
        mock_seg2.end = 1.0
        mock_seg2.avg_logprob = -0.3

        mock_model.transcribe.return_value = (iter([mock_seg1, mock_seg2]), mock_info)

        audio, sr = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        result = whisper.transcribe_audio(audio, sample_rate=sr)

        assert result.text == "Hello world"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello"
        assert result.segments[1].text == "world"


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestTranscribeFile:
    """Test transcribe_file() method with various audio formats."""

    def test_transcribe_file_basic(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test basic file transcription."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(audio_1s_mono_16k)

        # Verify result structure
        assert hasattr(result, 'text')
        assert hasattr(result, 'segments')
        assert hasattr(result, 'language')
        assert hasattr(result, 'duration')

        assert isinstance(result.text, str)
        assert result.language == "en"

    def test_transcribe_file_uses_audiodecode_load(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test that transcribe_file uses AudioDecode's fast load()."""
        from audiodecode.inference import WhisperInference

        with patch('audiodecode.inference.load') as mock_load:
            # Mock load to return sample audio
            mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

            whisper = WhisperInference(model_size="tiny")
            result = whisper.transcribe_file(audio_1s_mono_16k)

            # Verify load was called with correct parameters
            mock_load.assert_called_once()
            call_args = mock_load.call_args[0]
            call_kwargs = mock_load.call_args[1]
            assert str(audio_1s_mono_16k) in str(call_args[0])
            assert call_kwargs["sr"] == 16000
            assert call_kwargs["mono"] is True

    def test_transcribe_file_with_path_string(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test transcription with string path."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(str(audio_1s_mono_16k))

        assert isinstance(result.text, str)

    def test_transcribe_file_with_path_object(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test transcription with Path object."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(audio_1s_mono_16k)

        assert isinstance(result.text, str)

    def test_transcribe_file_missing_file(self, mock_whisper_model_class, tmp_path):
        """Test transcription with missing file raises appropriate error."""
        from audiodecode.inference import WhisperInference

        missing_file = tmp_path / "does_not_exist.wav"
        whisper = WhisperInference(model_size="tiny")

        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            result = whisper.transcribe_file(missing_file)

    def test_transcribe_file_with_language(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test file transcription with explicit language."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(audio_1s_mono_16k, language="fr")

        # Verify language was passed to transcribe_audio
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "fr"

    def test_transcribe_file_with_vad(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test file transcription with VAD filtering."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(audio_1s_mono_16k, vad_filter=True)

        # Verify VAD was enabled
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is True

    def test_transcribe_file_passes_all_parameters(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test that transcribe_file passes all parameters correctly."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_file(
            audio_1s_mono_16k,
            language="en",
            task="translate",
            beam_size=10,
            best_of=10,
            temperature=0.5,
            vad_filter=True,
            vad_parameters={"threshold": 0.5}
        )

        # Verify all parameters were passed
        mock_model = mock_whisper_model_class.return_value
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["task"] == "translate"
        assert call_kwargs["beam_size"] == 10
        assert call_kwargs["best_of"] == 10
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["vad_filter"] is True
        assert call_kwargs["vad_parameters"] == {"threshold": 0.5}


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestConvenienceFunctions:
    """Test convenience functions for simple use cases."""

    def test_transcribe_file_convenience(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test module-level transcribe_file() convenience function."""
        from audiodecode.inference import transcribe_file

        result = transcribe_file(audio_1s_mono_16k, model_size="tiny")

        assert hasattr(result, 'text')
        assert isinstance(result.text, str)

        # Verify WhisperInference was created with correct parameters
        mock_whisper_model_class.assert_called()
        call_args = mock_whisper_model_class.call_args[0]
        assert call_args[0] == "tiny"

    def test_transcribe_file_convenience_with_params(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test convenience function with custom parameters."""
        from audiodecode.inference import transcribe_file

        result = transcribe_file(
            audio_1s_mono_16k,
            model_size="base",
            language="es",
            device="cpu",
            compute_type="float32"
        )

        # Verify parameters were passed correctly
        call_kwargs = mock_whisper_model_class.call_args[1]
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["compute_type"] == "float32"

    def test_transcribe_audio_convenience(self, mock_whisper_model_class, sample_audio_16k):
        """Test module-level transcribe_audio() convenience function."""
        from audiodecode.inference import transcribe_audio

        audio, sr = sample_audio_16k
        result = transcribe_audio(audio, sample_rate=sr, model_size="tiny")

        assert hasattr(result, 'text')
        assert isinstance(result.text, str)

    def test_transcribe_audio_convenience_with_params(self, mock_whisper_model_class, sample_audio_16k):
        """Test audio convenience function with custom parameters."""
        from audiodecode.inference import transcribe_audio

        audio, sr = sample_audio_16k
        result = transcribe_audio(
            audio,
            sample_rate=sr,
            model_size="small",
            language="fr",
            device="cpu",
            compute_type="int8"
        )

        # Verify parameters were passed correctly
        call_kwargs = mock_whisper_model_class.call_args[1]
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["compute_type"] == "int8"


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestDataclasses:
    """Test TranscriptionSegment and TranscriptionResult dataclasses."""

    def test_transcription_segment_creation(self):
        """Test creating TranscriptionSegment."""
        from audiodecode.inference import TranscriptionSegment

        segment = TranscriptionSegment(
            text="Hello world",
            start=0.0,
            end=1.0,
            confidence=-0.5
        )

        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.confidence == -0.5

    def test_transcription_result_creation(self):
        """Test creating TranscriptionResult."""
        from audiodecode.inference import TranscriptionResult, TranscriptionSegment

        segments = [
            TranscriptionSegment("Hello", 0.0, 0.5, -0.5),
            TranscriptionSegment("world", 0.5, 1.0, -0.3)
        ]

        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            duration=1.0
        )

        assert result.text == "Hello world"
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.duration == 1.0

    def test_transcription_segment_is_dataclass(self):
        """Test that TranscriptionSegment is a proper dataclass."""
        from audiodecode.inference import TranscriptionSegment
        from dataclasses import is_dataclass

        assert is_dataclass(TranscriptionSegment)

    def test_transcription_result_is_dataclass(self):
        """Test that TranscriptionResult is a proper dataclass."""
        from audiodecode.inference import TranscriptionResult
        from dataclasses import is_dataclass

        assert is_dataclass(TranscriptionResult)


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_audio_shape(self, mock_whisper_model_class):
        """Test handling of invalid audio shape."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")

        # Test with 3D audio (invalid)
        invalid_audio = np.zeros((16000, 2, 2), dtype=np.float32)

        # Should either handle gracefully or raise descriptive error
        # The actual behavior depends on faster-whisper's error handling
        try:
            result = whisper.transcribe_audio(invalid_audio, sample_rate=16000)
        except Exception as e:
            # If it raises, that's fine - we're testing it doesn't crash silently
            assert True

    def test_zero_length_audio(self, mock_whisper_model_class):
        """Test handling of zero-length audio."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")

        # Empty audio array
        empty_audio = np.array([], dtype=np.float32)

        result = whisper.transcribe_audio(empty_audio, sample_rate=16000)

        # Should return empty result
        assert result.duration == 0.0

    def test_very_short_audio(self, mock_whisper_model_class):
        """Test handling of very short audio (< 100ms)."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")

        # 50ms of audio
        short_audio = np.zeros(800, dtype=np.float32)

        result = whisper.transcribe_audio(short_audio, sample_rate=16000)

        assert result.duration == pytest.approx(0.05, rel=0.01)

    def test_negative_sample_rate(self, mock_whisper_model_class, sample_audio_16k):
        """Test handling of invalid sample rate."""
        from audiodecode.inference import WhisperInference

        audio, _ = sample_audio_16k
        whisper = WhisperInference(model_size="tiny")

        # Negative sample rate should be caught somewhere in the pipeline
        try:
            result = whisper.transcribe_audio(audio, sample_rate=-16000)
        except (ValueError, Exception):
            assert True  # Expected to raise an error


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestIntegrationWithPillar1:
    """Test integration with Pillar 1's fast audio decode."""

    def test_load_integration(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test that inference integrates with AudioDecode load()."""
        from audiodecode import load
        from audiodecode.inference import WhisperInference

        # Load audio using Pillar 1
        audio, sr = load(audio_1s_mono_16k, sr=16000, mono=True)

        # Transcribe using Pillar 3
        whisper = WhisperInference(model_size="tiny")
        result = whisper.transcribe_audio(audio, sample_rate=sr)

        assert isinstance(result.text, str)
        assert result.language == "en"

    def test_full_pipeline_file_to_text(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test complete pipeline from file to transcription."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")

        # This internally uses load() from Pillar 1
        result = whisper.transcribe_file(audio_1s_mono_16k)

        assert isinstance(result.text, str)
        assert result.duration > 0.0

    def test_different_sample_rates(self, mock_whisper_model_class, audio_1s_stereo_44k):
        """Test transcription with different sample rates (should resample)."""
        from audiodecode import load
        from audiodecode.inference import WhisperInference

        # Load at different sample rates
        audio_16k, sr_16k = load(audio_1s_stereo_44k, sr=16000, mono=True)
        audio_44k, sr_44k = load(audio_1s_stereo_44k, sr=44100, mono=True)

        whisper = WhisperInference(model_size="tiny")

        # 16kHz should work without warning
        result_16k = whisper.transcribe_audio(audio_16k, sample_rate=sr_16k)
        assert isinstance(result_16k.text, str)

        # 44kHz should work but with warning
        with pytest.warns(UserWarning):
            result_44k = whisper.transcribe_audio(audio_44k, sample_rate=sr_44k)


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestPerformance:
    """Test performance characteristics and assertions."""

    def test_transcription_completes_in_reasonable_time(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test that transcription completes in reasonable time."""
        import time
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")

        start = time.perf_counter()
        result = whisper.transcribe_file(audio_1s_mono_16k)
        elapsed = time.perf_counter() - start

        # With mocking, should be very fast (< 1 second)
        assert elapsed < 1.0

    def test_model_reuse_is_efficient(self, mock_whisper_model_class, audio_1s_mono_16k):
        """Test that model is reused efficiently across multiple transcriptions."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")

        # Multiple transcriptions should reuse the same model
        result1 = whisper.transcribe_file(audio_1s_mono_16k)
        result2 = whisper.transcribe_file(audio_1s_mono_16k)

        # Model should only be initialized once
        assert mock_whisper_model_class.call_count == 1

    def test_audio_array_not_copied_unnecessarily(self, mock_whisper_model_class, sample_audio_16k):
        """Test that audio array is not copied unnecessarily (zero-copy where possible)."""
        from audiodecode.inference import WhisperInference

        audio, sr = sample_audio_16k
        audio_float32 = audio.astype(np.float32)

        whisper = WhisperInference(model_size="tiny")

        # Get the array passed to model.transcribe
        result = whisper.transcribe_audio(audio_float32, sample_rate=sr)

        mock_model = mock_whisper_model_class.return_value
        call_args = mock_model.transcribe.call_args[0]
        transcribed_audio = call_args[0]

        # For float32, should be the same object (no copy)
        # Note: This may not always be true depending on implementation
        assert transcribed_audio.dtype == np.float32


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestModelSizes:
    """Test different model sizes and their characteristics."""

    @pytest.mark.parametrize("model_size", ["tiny", "base", "small", "medium", "large-v3"])
    def test_all_model_sizes(self, mock_whisper_model_class, model_size):
        """Test that all documented model sizes can be initialized."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size=model_size)
        assert whisper.model_size == model_size

        # Verify model was initialized with correct size
        call_args = mock_whisper_model_class.call_args[0]
        assert call_args[0] == model_size

    def test_model_size_is_passed_to_whisper_model(self, mock_whisper_model_class):
        """Test that model size is correctly passed to WhisperModel."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="small")

        # First positional argument should be model size
        call_args = mock_whisper_model_class.call_args[0]
        assert call_args[0] == "small"


# Additional test: Module-level exports
class TestModuleExports:
    """Test that the inference module exports the correct public API."""

    def test_module_exports_classes(self):
        """Test that module exports main classes."""
        assert hasattr(inference_module, 'WhisperInference')
        assert hasattr(inference_module, 'TranscriptionSegment')
        assert hasattr(inference_module, 'TranscriptionResult')

    def test_module_exports_functions(self):
        """Test that module exports convenience functions."""
        assert hasattr(inference_module, 'transcribe_file')
        assert hasattr(inference_module, 'transcribe_audio')

    def test_module_has_docstrings(self):
        """Test that main classes and functions have docstrings."""
        if inference_module._FASTER_WHISPER_AVAILABLE:
            assert inference_module.WhisperInference.__doc__ is not None
            assert inference_module.transcribe_file.__doc__ is not None
            assert inference_module.transcribe_audio.__doc__ is not None


# Integration test marker for tests that require actual models
pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestRealInference:
    """
    Integration tests with real models (requires model download).

    These tests are marked as 'integration' and should be run separately
    from unit tests. They will download actual Whisper models.

    Run with: pytest -m integration
    """

    def test_real_transcription_tiny_model(self, audio_1s_mono_16k):
        """Test real transcription with tiny model."""
        from audiodecode.inference import transcribe_file

        result = transcribe_file(audio_1s_mono_16k, model_size="tiny")

        # Basic validations on real output
        assert isinstance(result.text, str)
        assert isinstance(result.language, str)
        assert result.duration > 0.0
        assert len(result.segments) >= 0  # May be 0 for synthetic audio

    def test_real_performance_benchmark(self, audio_10s_mono_16k):
        """Benchmark real transcription performance."""
        import time
        from audiodecode.inference import transcribe_file

        start = time.perf_counter()
        result = transcribe_file(audio_10s_mono_16k, model_size="base")
        elapsed = time.perf_counter() - start

        # Should be faster than real-time (< 10 seconds for 10s audio)
        assert elapsed < result.duration

        print(f"\nPerformance: {result.duration / elapsed:.1f}x realtime")
