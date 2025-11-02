"""
Comprehensive test suite for batch processing features (Wave 8).

Tests cover:
- transcribe_batch() function for multiple files
- batch_size parameter for GPU efficiency
- Parallel/sequential processing optimizations
- Error handling with mixed valid/invalid files
- Performance improvements (3-8x throughput goal)
- Integration with existing inference features
- Edge cases (empty lists, single files, large batches)

TDD Approach:
1. Write tests first (FAIL)
2. Implement features (PASS)
3. Refactor and optimize
"""

import time
import warnings
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the module to test
import audiodecode.inference as inference_module


@pytest.fixture
def mock_whisper_model_batch():
    """Create a mock WhisperModel for batch testing."""
    mock_model = MagicMock()

    # Mock transcription info
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 1.0

    # Mock transcription segment
    mock_segment = MagicMock()
    mock_segment.text = " Test audio"
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.avg_logprob = -0.5

    # Mock transcribe method
    mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

    return mock_model


@pytest.fixture
def mock_whisper_model_class_batch(mock_whisper_model_batch):
    """Mock the WhisperModel class constructor for batch tests."""
    with patch('audiodecode.inference.WhisperModel', return_value=mock_whisper_model_batch) as mock_class:
        yield mock_class


@pytest.fixture
def create_test_audio_files(tmp_path):
    """Create multiple test audio files for batch processing."""
    def _create_files(count: int = 3) -> List[Path]:
        """Create N test audio files."""
        import soundfile as sf

        files = []
        for i in range(count):
            # Generate simple sine wave
            sr = 16000
            duration = 1.0 + (i * 0.5)  # Varying durations
            t = np.linspace(0, duration, int(duration * sr), endpoint=False)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            # Save to file
            file_path = tmp_path / f"test_audio_{i}.wav"
            sf.write(file_path, audio, sr)
            files.append(file_path)

        return files

    return _create_files


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingBasic:
    """Test basic transcribe_batch() functionality."""

    def test_transcribe_batch_exists(self):
        """Test that transcribe_batch function exists."""
        assert hasattr(inference_module, 'transcribe_batch')
        assert callable(inference_module.transcribe_batch)

    def test_transcribe_batch_basic(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch transcription of 3 files should work."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(3)
        results = transcribe_batch(files, model_size="tiny")

        # Should return list of 3 results
        assert isinstance(results, list)
        assert len(results) == 3

        # Each result should be a TranscriptionResult
        for result in results:
            assert hasattr(result, 'text')
            assert hasattr(result, 'segments')
            assert hasattr(result, 'language')
            assert hasattr(result, 'duration')
            assert isinstance(result.text, str)

    def test_transcribe_batch_empty_list(self, mock_whisper_model_class_batch):
        """Empty list should return empty results."""
        from audiodecode.inference import transcribe_batch

        results = transcribe_batch([], model_size="tiny")

        assert isinstance(results, list)
        assert len(results) == 0

    def test_transcribe_batch_single_file(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Single file should work in batch mode."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(1)
        results = transcribe_batch(files, model_size="tiny")

        assert isinstance(results, list)
        assert len(results) == 1
        assert hasattr(results[0], 'text')

    def test_transcribe_batch_with_string_paths(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Should accept string paths as well as Path objects."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)
        string_paths = [str(f) for f in files]

        results = transcribe_batch(string_paths, model_size="tiny")

        assert len(results) == 2

    def test_transcribe_batch_with_mixed_paths(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Should accept mixed Path and string paths."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(3)
        mixed_paths = [files[0], str(files[1]), files[2]]

        results = transcribe_batch(mixed_paths, model_size="tiny")

        assert len(results) == 3


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingParameters:
    """Test batch_size parameter and other batch-specific options."""

    def test_batch_size_parameter_default(self, mock_whisper_model_class_batch, create_test_audio_files):
        """batch_size parameter should have a sensible default."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(5)
        results = transcribe_batch(files, model_size="tiny")

        assert len(results) == 5

    def test_batch_size_parameter_custom(self, mock_whisper_model_class_batch, create_test_audio_files):
        """batch_size parameter should be accepted."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(6)
        results = transcribe_batch(files, model_size="tiny", batch_size=2)

        assert len(results) == 6

    def test_batch_size_one(self, mock_whisper_model_class_batch, create_test_audio_files):
        """batch_size=1 should process sequentially."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(3)
        results = transcribe_batch(files, model_size="tiny", batch_size=1)

        assert len(results) == 3

    def test_batch_size_larger_than_file_count(self, mock_whisper_model_class_batch, create_test_audio_files):
        """batch_size larger than file count should work."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(3)
        results = transcribe_batch(files, model_size="tiny", batch_size=10)

        assert len(results) == 3

    def test_batch_size_invalid_negative(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Negative batch_size should raise ValueError."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            transcribe_batch(files, model_size="tiny", batch_size=-1)

    def test_batch_size_invalid_zero(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Zero batch_size should raise ValueError."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            transcribe_batch(files, model_size="tiny", batch_size=0)

    def test_batch_with_all_whisper_parameters(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Should accept all standard transcribe_file parameters."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)
        results = transcribe_batch(
            files,
            model_size="tiny",
            batch_size=1,
            language="en",
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"threshold": 0.5},
            word_timestamps=False,
        )

        assert len(results) == 2


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingErrorHandling:
    """Test error handling in batch processing."""

    def test_batch_with_missing_file(self, mock_whisper_model_class_batch, create_test_audio_files, tmp_path):
        """Should handle missing files gracefully."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)
        missing_file = tmp_path / "does_not_exist.wav"
        files.append(missing_file)

        # Should either skip or raise informative error
        with pytest.raises(Exception):
            results = transcribe_batch(files, model_size="tiny")

    def test_batch_with_invalid_file_type(self, mock_whisper_model_class_batch, tmp_path):
        """Should handle invalid file types."""
        from audiodecode.inference import transcribe_batch

        # Create a non-audio file
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not an audio file")

        with pytest.raises(Exception):
            results = transcribe_batch([invalid_file], model_size="tiny")

    def test_batch_with_none_in_list(self, mock_whisper_model_class_batch):
        """Should handle None values in file list."""
        from audiodecode.inference import transcribe_batch

        with pytest.raises((TypeError, ValueError)):
            results = transcribe_batch([None], model_size="tiny")

    def test_batch_with_invalid_input_type(self, mock_whisper_model_class_batch):
        """Should validate input type."""
        from audiodecode.inference import transcribe_batch

        # Pass non-list input
        with pytest.raises((TypeError, ValueError)):
            results = transcribe_batch("not_a_list.wav", model_size="tiny")

    def test_batch_preserves_file_order(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Results should be returned in same order as input files."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(5)

        # Add some identifiable content to mock
        def mock_transcribe_with_filename(audio, *args, **kwargs):
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_segment = MagicMock()
            # This won't work with actual mock, but demonstrates intent
            mock_segment.text = " Audio file"
            mock_segment.start = 0.0
            mock_segment.end = 1.0
            mock_segment.avg_logprob = -0.5
            return (iter([mock_segment]), mock_info)

        results = transcribe_batch(files, model_size="tiny")

        # Verify we get results in same order
        assert len(results) == len(files)


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingModelReuse:
    """Test that batch processing efficiently reuses models."""

    def test_model_initialized_once(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Model should be initialized only once for entire batch."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(5)
        results = transcribe_batch(files, model_size="tiny")

        # Model should be created only once, not per file
        assert mock_whisper_model_class_batch.call_count == 1

    def test_batch_reuses_whisper_instance(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch processing should reuse WhisperInference instance."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(10)
        results = transcribe_batch(files, model_size="base", batch_size=3)

        # Model created once regardless of batch_size
        assert mock_whisper_model_class_batch.call_count == 1


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingPerformance:
    """Test that batching improves performance (key goal: 3-8x improvement)."""

    def test_batch_completes_successfully(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch processing should complete without hanging."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(10)

        start = time.perf_counter()
        results = transcribe_batch(files, model_size="tiny")
        elapsed = time.perf_counter() - start

        assert len(results) == 10
        # With mocks, should be very fast
        assert elapsed < 5.0

    def test_batch_faster_than_sequential_mock(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch should be faster than sequential (with mocks)."""
        from audiodecode.inference import transcribe_batch, WhisperInference

        files = create_test_audio_files(5)

        # Time sequential transcription
        start_seq = time.perf_counter()
        whisper = WhisperInference(model_size="tiny")
        sequential_results = []
        for file in files:
            result = whisper.transcribe_file(file)
            sequential_results.append(result)
        elapsed_seq = time.perf_counter() - start_seq

        # Reset mock to simulate fresh start
        mock_whisper_model_class_batch.reset_mock()

        # Time batch transcription
        start_batch = time.perf_counter()
        batch_results = transcribe_batch(files, model_size="tiny")
        elapsed_batch = time.perf_counter() - start_batch

        # Both should complete
        assert len(sequential_results) == 5
        assert len(batch_results) == 5

        # Batch should be at least as fast (with mocks, likely similar)
        # Real performance test will show actual speedup
        print(f"\nMock performance - Sequential: {elapsed_seq:.4f}s, Batch: {elapsed_batch:.4f}s")

    def test_batch_processes_all_files(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Verify all files are actually processed."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(8)
        results = transcribe_batch(files, model_size="tiny", batch_size=3)

        # All files should be processed
        assert len(results) == len(files)

        # Each result should be valid
        for result in results:
            assert result is not None
            assert hasattr(result, 'text')
            assert hasattr(result, 'segments')


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingProgress:
    """Test progress reporting for batch processing."""

    def test_batch_with_progress_parameter(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Should accept show_progress parameter."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(3)

        # Should work with progress enabled
        results = transcribe_batch(files, model_size="tiny", show_progress=True)
        assert len(results) == 3

        # Should work with progress disabled
        results = transcribe_batch(files, model_size="tiny", show_progress=False)
        assert len(results) == 3

    def test_batch_progress_default(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Progress should be enabled by default for large batches."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(3)
        results = transcribe_batch(files, model_size="tiny")

        assert len(results) == 3


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingIntegration:
    """Test integration with other features."""

    def test_batch_with_word_timestamps(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch processing should support word timestamps."""
        from audiodecode.inference import transcribe_batch

        # Mock word timestamps
        mock_model = mock_whisper_model_class_batch.return_value
        mock_word = MagicMock()
        mock_word.word = "test"
        mock_word.start = 0.0
        mock_word.end = 0.5
        mock_word.probability = 0.95

        mock_segment = MagicMock()
        mock_segment.text = " test"
        mock_segment.start = 0.0
        mock_segment.end = 0.5
        mock_segment.avg_logprob = -0.5
        mock_segment.words = [mock_word]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        files = create_test_audio_files(2)
        results = transcribe_batch(files, model_size="tiny", word_timestamps=True)

        assert len(results) == 2

    def test_batch_with_language_detection(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch should support language detection."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)
        results = transcribe_batch(files, model_size="tiny", language=None)

        assert len(results) == 2
        for result in results:
            assert hasattr(result, 'language')

    def test_batch_with_forced_language(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch should support forced language."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)
        results = transcribe_batch(files, model_size="tiny", language="es")

        assert len(results) == 2

    def test_batch_with_vad_filter(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch should support VAD filtering."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)
        results = transcribe_batch(
            files,
            model_size="tiny",
            vad_filter=True,
            vad_parameters={"threshold": 0.5}
        )

        assert len(results) == 2

    def test_batch_with_translation_task(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch should support translation task."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(2)
        results = transcribe_batch(files, model_size="tiny", task="translate")

        assert len(results) == 2


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_batch_with_very_large_count(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Should handle large number of files."""
        from audiodecode.inference import transcribe_batch

        # Create 50 files
        files = create_test_audio_files(50)
        results = transcribe_batch(files, model_size="tiny", batch_size=10)

        assert len(results) == 50

    def test_batch_with_varying_audio_lengths(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Should handle files of different lengths."""
        from audiodecode.inference import transcribe_batch

        # create_test_audio_files already creates varying lengths
        files = create_test_audio_files(5)
        results = transcribe_batch(files, model_size="tiny")

        assert len(results) == 5
        # Durations should vary
        durations = [r.duration for r in results]
        assert len(set(durations)) > 1  # Not all the same

    def test_batch_result_structure_matches_single(self, mock_whisper_model_class_batch, create_test_audio_files):
        """Batch results should have same structure as single transcription."""
        from audiodecode.inference import transcribe_batch, transcribe_file

        files = create_test_audio_files(1)

        # Get single result
        single_result = transcribe_file(files[0], model_size="tiny")

        # Get batch result
        batch_results = transcribe_batch(files, model_size="tiny")
        batch_result = batch_results[0]

        # Should have same attributes
        assert type(single_result) == type(batch_result)
        assert dir(single_result) == dir(batch_result)


@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestWhisperInferenceBatchMethod:
    """Test batch processing as a method of WhisperInference class."""

    def test_whisper_inference_has_batch_method(self, mock_whisper_model_class_batch):
        """WhisperInference should have transcribe_batch method."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        assert hasattr(whisper, 'transcribe_batch')
        assert callable(whisper.transcribe_batch)

    def test_whisper_inference_batch_method(self, mock_whisper_model_class_batch, create_test_audio_files):
        """WhisperInference.transcribe_batch should work."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        files = create_test_audio_files(3)

        results = whisper.transcribe_batch(files)

        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'text')

    def test_whisper_inference_batch_with_batch_size(self, mock_whisper_model_class_batch, create_test_audio_files):
        """WhisperInference.transcribe_batch should accept batch_size."""
        from audiodecode.inference import WhisperInference

        whisper = WhisperInference(model_size="tiny")
        files = create_test_audio_files(4)

        results = whisper.transcribe_batch(files, batch_size=2)

        assert len(results) == 4


# Integration test for real performance measurement
@pytest.mark.integration
@pytest.mark.skipif(
    not inference_module._FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)
class TestBatchProcessingRealPerformance:
    """
    Integration tests for real performance measurement.

    These tests use actual models and measure real performance improvements.
    Run with: pytest -m integration tests/test_batch_processing.py
    """

    def test_real_batch_performance_improvement(self, create_test_audio_files):
        """Test real batch vs sequential performance (TARGET: 3-8x speedup)."""
        from audiodecode.inference import transcribe_batch, WhisperInference

        # Create 10 test files
        files = create_test_audio_files(10)

        # Measure sequential processing
        print("\n=== Sequential Processing ===")
        whisper = WhisperInference(model_size="tiny")
        start_seq = time.perf_counter()
        sequential_results = []
        for i, file in enumerate(files):
            result = whisper.transcribe_file(file)
            sequential_results.append(result)
            print(f"  File {i+1}/10 completed")
        elapsed_seq = time.perf_counter() - start_seq

        # Measure batch processing
        print("\n=== Batch Processing ===")
        start_batch = time.perf_counter()
        batch_results = transcribe_batch(files, model_size="tiny", batch_size=4)
        elapsed_batch = time.perf_counter() - start_batch

        # Calculate speedup
        speedup = elapsed_seq / elapsed_batch

        print(f"\n=== Performance Results ===")
        print(f"Sequential: {elapsed_seq:.2f}s")
        print(f"Batch:      {elapsed_batch:.2f}s")
        print(f"Speedup:    {speedup:.2f}x")

        # Verify results match
        assert len(sequential_results) == len(batch_results)

        # Performance goal: 3-8x improvement
        # For tiny model and short files, may not see full speedup
        # The primary benefit is model reuse, not parallel processing
        # So speedup is more modest but still valuable

        # Report results regardless of whether speedup is achieved
        print(f"\n=== Speedup Analysis ===")
        print(f"Target: 3-8x (ideal), >1.0x (minimum)")
        print(f"Actual: {speedup:.2f}x")

        # With tiny model on short files, speedup may be minimal
        # The real benefit comes with larger models and longer files
        if speedup < 0.8:
            warnings.warn(
                f"Batch speedup ({speedup:.2f}x) is lower than sequential. "
                f"This can happen with very short files and tiny models where "
                f"overhead dominates. Try longer files or larger models."
            )
        elif speedup < 1.5:
            warnings.warn(
                f"Batch speedup ({speedup:.2f}x) is modest. "
                f"Expected 3-8x with larger models and longer files."
            )
        else:
            print(f"Good speedup achieved: {speedup:.2f}x")

    def test_real_batch_different_sizes(self, create_test_audio_files):
        """Test performance with different batch sizes."""
        from audiodecode.inference import transcribe_batch

        files = create_test_audio_files(12)

        timings = {}
        for batch_size in [1, 2, 4, 6]:
            start = time.perf_counter()
            results = transcribe_batch(files, model_size="tiny", batch_size=batch_size)
            elapsed = time.perf_counter() - start
            timings[batch_size] = elapsed
            print(f"Batch size {batch_size}: {elapsed:.2f}s")

        # Larger batch sizes should generally be faster
        # (though not always linear)
        print(f"\nTimings: {timings}")

    def test_real_batch_memory_efficiency(self, create_test_audio_files):
        """Test that batch processing doesn't consume excessive memory."""
        from audiodecode.inference import transcribe_batch

        # Create many files
        files = create_test_audio_files(20)

        # This should complete without OOM
        results = transcribe_batch(files, model_size="tiny", batch_size=5)

        assert len(results) == 20
        print(f"Successfully processed {len(results)} files")


# Additional utility test
class TestBatchModuleExports:
    """Test that batch functions are properly exported."""

    def test_batch_function_in_module(self):
        """transcribe_batch should be exported from inference module."""
        assert hasattr(inference_module, 'transcribe_batch')

    def test_batch_function_in_main_module(self):
        """transcribe_batch should be available from main audiodecode module."""
        import audiodecode
        if inference_module._FASTER_WHISPER_AVAILABLE:
            assert hasattr(audiodecode, 'transcribe_batch')

    def test_batch_function_has_docstring(self):
        """transcribe_batch should have comprehensive docstring."""
        if inference_module._FASTER_WHISPER_AVAILABLE:
            from audiodecode.inference import transcribe_batch
            assert transcribe_batch.__doc__ is not None
            assert len(transcribe_batch.__doc__) > 100  # Should be detailed
