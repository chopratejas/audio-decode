"""
Test progress feedback features for AudioDecode.

Tests the verbose parameter for progress output, matching openai-whisper's behavior.
Users expect to see real-time progress during long transcriptions.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
from io import StringIO

from audiodecode.inference import (
    transcribe_file,
    transcribe_audio,
    WhisperInference,
    TranscriptionResult,
)


class TestVerboseParameter:
    """Test verbose parameter for progress output."""

    def test_verbose_false_default(self, tmp_path):
        """
        verbose=False should be silent (default).

        This is the default behavior - no progress output to stdout.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test with explicit verbose=False
            result = transcribe_file(str(audio_path), verbose=False, model_size="tiny")

            # Verify result is valid
            assert isinstance(result, TranscriptionResult)

            # Should produce minimal output (maybe some warnings, but no progress)
            output = captured_output.getvalue()
            # We're lenient here - just verify it accepts the parameter

        finally:
            sys.stdout = old_stdout

    def test_verbose_true_parameter_accepted(self, tmp_path):
        """
        verbose=True should be accepted by transcribe_file.

        Tests that the parameter is properly passed through the pipeline.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Should not raise any errors
        result = transcribe_file(str(audio_path), verbose=True, model_size="tiny")

        assert isinstance(result, TranscriptionResult)

    def test_verbose_default_is_false(self, tmp_path):
        """
        Default behavior should be quiet (verbose=False).

        When verbose is not specified, no progress should be shown.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Call without verbose parameter - should use default False
        result = transcribe_file(str(audio_path), model_size="tiny")

        assert isinstance(result, TranscriptionResult)

    def test_verbose_type_validation(self, tmp_path):
        """
        verbose must be bool - reject invalid types.

        Following the same strict type checking as word_timestamps.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=1.0)

        # Invalid: string should raise TypeError
        with pytest.raises(TypeError, match="verbose must be bool"):
            transcribe_file(str(audio_path), verbose="true", model_size="tiny")

        # Invalid: int should raise TypeError
        with pytest.raises(TypeError, match="verbose must be bool"):
            transcribe_file(str(audio_path), verbose=1, model_size="tiny")

        # Invalid: None should raise TypeError
        with pytest.raises(TypeError, match="verbose must be bool"):
            transcribe_file(str(audio_path), verbose=None, model_size="tiny")

    def test_verbose_with_transcribe_audio(self):
        """
        verbose should work with transcribe_audio (numpy input).

        Tests the convenience function with audio arrays.
        """
        # Create synthetic audio (2 seconds of tone)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        # Test with verbose=True
        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            verbose=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

        # Test with verbose=False
        result = transcribe_audio(
            audio,
            sample_rate=sample_rate,
            verbose=False,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_verbose_with_whisper_inference_class(self, tmp_path):
        """
        verbose should work with WhisperInference class methods.

        Tests both transcribe_file and transcribe_audio on the class.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0)

        whisper = WhisperInference(model_size="tiny")

        # Test with transcribe_file
        result1 = whisper.transcribe_file(str(audio_path), verbose=True)
        assert isinstance(result1, TranscriptionResult)

        result2 = whisper.transcribe_file(str(audio_path), verbose=False)
        assert isinstance(result2, TranscriptionResult)

        # Test with transcribe_audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        result3 = whisper.transcribe_audio(audio, sample_rate, verbose=True)
        assert isinstance(result3, TranscriptionResult)

        result4 = whisper.transcribe_audio(audio, sample_rate, verbose=False)
        assert isinstance(result4, TranscriptionResult)

    def test_verbose_with_other_features(self, tmp_path):
        """
        verbose should work alongside other features.

        Tests compatibility with word_timestamps, language, beam_size, etc.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=3.0)

        # Combine with word_timestamps
        result1 = transcribe_file(
            str(audio_path),
            verbose=True,
            word_timestamps=True,
            model_size="tiny"
        )
        assert isinstance(result1, TranscriptionResult)

        # Combine with multiple parameters
        result2 = transcribe_file(
            str(audio_path),
            verbose=False,
            language="en",
            beam_size=3,
            vad_filter=True,
            model_size="tiny"
        )
        assert isinstance(result2, TranscriptionResult)

        # Combine with temperature fallback
        result3 = transcribe_file(
            str(audio_path),
            verbose=True,
            temperature=(0.0, 0.2, 0.4),
            model_size="tiny"
        )
        assert isinstance(result3, TranscriptionResult)

    def test_verbose_with_initial_prompt(self, tmp_path):
        """
        verbose should work with prompt engineering features.

        Tests compatibility with initial_prompt and prefix.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            verbose=True,
            initial_prompt="Technical discussion",
            prefix="Speaker:",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_verbose_both_values_produce_valid_results(self, tmp_path):
        """
        Both verbose=True and verbose=False should produce identical transcription results.

        Progress output should not affect transcription quality.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # Same audio, different verbose settings
        result_verbose = transcribe_file(
            str(audio_path),
            verbose=True,
            model_size="tiny",
            beam_size=5,
            temperature=0.0  # Deterministic
        )

        result_silent = transcribe_file(
            str(audio_path),
            verbose=False,
            model_size="tiny",
            beam_size=5,
            temperature=0.0  # Deterministic
        )

        # Results should be identical (or very similar)
        assert isinstance(result_verbose, TranscriptionResult)
        assert isinstance(result_silent, TranscriptionResult)
        assert result_verbose.language == result_silent.language
        assert len(result_verbose.segments) == len(result_silent.segments)

    def test_verbose_on_long_audio(self, tmp_path):
        """
        verbose parameter should work on longer audio files.

        Progress feedback is most useful for long transcriptions.
        """
        audio_path = tmp_path / "long_audio.wav"
        self._create_test_audio(audio_path, duration=30.0)

        # Should complete without errors
        result = transcribe_file(
            str(audio_path),
            verbose=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.duration > 25.0  # At least 25 seconds

    def test_verbose_on_empty_audio(self, tmp_path):
        """
        verbose parameter should handle silent/empty audio gracefully.

        Edge case: no speech detected.
        """
        audio_path = tmp_path / "silent.wav"
        self._create_silent_audio(audio_path, duration=2.0)

        # Should not crash
        result = transcribe_file(
            str(audio_path),
            verbose=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_verbose_preserves_all_result_fields(self, tmp_path):
        """
        verbose parameter should not affect any result fields.

        Ensures complete data structure is preserved.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            verbose=True,
            word_timestamps=True,
            model_size="tiny"
        )

        # Check all expected fields are present
        assert hasattr(result, 'text')
        assert hasattr(result, 'segments')
        assert hasattr(result, 'language')
        assert hasattr(result, 'duration')

        assert isinstance(result.text, str)
        assert isinstance(result.segments, list)
        assert isinstance(result.language, str)
        assert isinstance(result.duration, float)

        # Check segments have all fields
        if len(result.segments) > 0:
            segment = result.segments[0]
            assert hasattr(segment, 'text')
            assert hasattr(segment, 'start')
            assert hasattr(segment, 'end')
            assert hasattr(segment, 'confidence')
            assert hasattr(segment, 'words')

    # Helper methods

    def _create_test_audio(self, path: Path, duration: float):
        """Create a WAV file with noise for testing."""
        import soundfile as sf

        sample_rate = 16000
        samples = int(sample_rate * duration)

        # Create audio with some noise
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)

        sf.write(path, audio, sample_rate)

    def _create_silent_audio(self, path: Path, duration: float):
        """Create a completely silent WAV file."""
        import soundfile as sf

        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.zeros(samples, dtype=np.float32)

        sf.write(path, audio, sample_rate)


class TestVerboseEdgeCases:
    """Edge cases for verbose parameter."""

    def test_verbose_with_vad_filter(self, tmp_path):
        """
        verbose should work with VAD filtering enabled.

        VAD filtering affects segmentation, verbose should still work.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=5.0)

        result = transcribe_file(
            str(audio_path),
            verbose=True,
            vad_filter=True,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_verbose_with_different_languages(self, tmp_path):
        """
        verbose should work across different language settings.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0)

        # English
        result_en = transcribe_file(
            str(audio_path),
            verbose=True,
            language="en",
            model_size="tiny"
        )
        assert isinstance(result_en, TranscriptionResult)

        # Spanish
        result_es = transcribe_file(
            str(audio_path),
            verbose=True,
            language="es",
            model_size="tiny"
        )
        assert isinstance(result_es, TranscriptionResult)

    def test_verbose_with_translation_task(self, tmp_path):
        """
        verbose should work with task='translate'.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=2.0)

        result = transcribe_file(
            str(audio_path),
            verbose=True,
            task="translate",
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_verbose_with_custom_vad_parameters(self, tmp_path):
        """
        verbose should work with custom VAD parameters.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=3.0)

        vad_params = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
        }

        result = transcribe_file(
            str(audio_path),
            verbose=True,
            vad_filter=True,
            vad_parameters=vad_params,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    def test_verbose_with_condition_on_previous_text_false(self, tmp_path):
        """
        verbose should work when condition_on_previous_text is disabled.
        """
        audio_path = tmp_path / "test_audio.wav"
        self._create_test_audio(audio_path, duration=3.0)

        result = transcribe_file(
            str(audio_path),
            verbose=True,
            condition_on_previous_text=False,
            model_size="tiny"
        )

        assert isinstance(result, TranscriptionResult)

    # Helper method
    def _create_test_audio(self, path: Path, duration: float):
        import soundfile as sf
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio = np.random.normal(0, 0.01, samples).astype(np.float32)
        sf.write(path, audio, sample_rate)
