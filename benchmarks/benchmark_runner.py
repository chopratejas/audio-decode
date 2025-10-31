"""
Comprehensive benchmark runner for AudioDecode vs librosa.

This module provides a BenchmarkRunner class that:
- Measures decode time, memory usage, and CPU utilization
- Compares AudioDecode performance against librosa baseline
- Supports multiple file formats and sizes
- Generates JSON results for analysis and CI/CD integration
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
import psutil


@dataclass
class BenchmarkResult:
    """Single benchmark measurement result."""

    library: str  # "audiodecode" or "librosa"
    file_path: str
    file_size_mb: float
    duration_seconds: float
    sample_rate: int
    channels: int

    # Performance metrics
    decode_time_seconds: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float

    # Configuration
    target_sr: int | None
    mono: bool
    backend: str | None  # For audiodecode: "soundfile" or "pyav"

    # Result validation
    output_shape: tuple[int, ...]
    output_dtype: str
    success: bool
    error_message: str | None = None


@dataclass
class ComparisonResult:
    """Comparison between audiodecode and librosa for same file."""

    file_path: str
    audiodecode: BenchmarkResult
    librosa_result: BenchmarkResult

    # Speedup metrics (positive = audiodecode faster)
    speedup_factor: float  # librosa_time / audiodecode_time
    memory_improvement_mb: float  # librosa_mem - audiodecode_mem (positive = less memory used)
    cpu_improvement_percent: float  # librosa_cpu - audiodecode_cpu

    # Pass/fail for regression tests
    faster_than_librosa: bool
    less_memory_than_librosa: bool


class BenchmarkRunner:
    """
    Run comprehensive benchmarks comparing AudioDecode vs librosa.

    Usage:
        >>> runner = BenchmarkRunner(results_dir="benchmarks/results")
        >>> results = runner.run_benchmarks(
        ...     audio_files=["audio.mp3", "audio.wav"],
        ...     backends=["soundfile", "pyav"]
        ... )
        >>> runner.save_results(results, "benchmark_results.json")
    """

    def __init__(
        self,
        results_dir: str | Path = "benchmarks/results",
        baseline_dir: str | Path = "benchmarks/baseline",
    ):
        """
        Initialize BenchmarkRunner.

        Args:
            results_dir: Directory to save benchmark results
            baseline_dir: Directory containing baseline measurements
        """
        self.results_dir = Path(results_dir)
        self.baseline_dir = Path(baseline_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        # Get current process for memory monitoring
        self.process = psutil.Process()

    def _get_file_info(self, file_path: Path) -> dict[str, Any]:
        """Get basic file information."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Use librosa to get audio metadata (fast, doesn't decode)
        duration = librosa.get_duration(path=str(file_path))

        # Get sample rate and channels without full decode
        import soundfile as sf
        with sf.SoundFile(str(file_path)) as f:
            sample_rate = f.samplerate
            channels = f.channels

        return {
            "file_size_mb": file_size_mb,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": channels,
        }

    def benchmark_librosa(
        self,
        file_path: Path,
        target_sr: int | None = None,
        mono: bool = False,
    ) -> BenchmarkResult:
        """
        Benchmark librosa.load() performance.

        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (None = keep original)
            mono: Convert to mono if True

        Returns:
            BenchmarkResult with librosa performance metrics
        """
        try:
            # Get file info
            file_info = self._get_file_info(file_path)

            # Measure memory before
            self.process.memory_info()  # Prime the cache
            mem_before = self.process.memory_info().rss / (1024 * 1024)

            # Start CPU monitoring
            cpu_percent_start = self.process.cpu_percent()

            # Benchmark decode
            start_time = time.perf_counter()
            audio, sr = librosa.load(
                str(file_path),
                sr=target_sr,
                mono=mono,
            )
            decode_time = time.perf_counter() - start_time

            # Measure memory after
            mem_after = self.process.memory_info().rss / (1024 * 1024)
            memory_delta = mem_after - mem_before

            # Get CPU usage (averaged over decode period)
            cpu_percent = self.process.cpu_percent()

            # Get peak memory (approximate)
            # Note: For true peak memory, use memory_profiler in test_performance.py
            memory_peak = mem_after

            return BenchmarkResult(
                library="librosa",
                file_path=str(file_path),
                file_size_mb=file_info["file_size_mb"],
                duration_seconds=file_info["duration_seconds"],
                sample_rate=file_info["sample_rate"],
                channels=file_info["channels"],
                decode_time_seconds=decode_time,
                memory_peak_mb=memory_peak,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_percent,
                target_sr=target_sr,
                mono=mono,
                backend=None,
                output_shape=audio.shape,
                output_dtype=str(audio.dtype),
                success=True,
            )

        except Exception as e:
            return BenchmarkResult(
                library="librosa",
                file_path=str(file_path),
                file_size_mb=0.0,
                duration_seconds=0.0,
                sample_rate=0,
                channels=0,
                decode_time_seconds=0.0,
                memory_peak_mb=0.0,
                memory_delta_mb=0.0,
                cpu_percent=0.0,
                target_sr=target_sr,
                mono=mono,
                backend=None,
                output_shape=(0,),
                output_dtype="",
                success=False,
                error_message=str(e),
            )

    def benchmark_audiodecode(
        self,
        file_path: Path,
        target_sr: int | None = None,
        mono: bool = False,
        backend: Literal["soundfile", "pyav"] | None = None,
    ) -> BenchmarkResult:
        """
        Benchmark AudioDecode performance.

        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (None = keep original)
            mono: Convert to mono if True
            backend: Force specific backend (None = auto-select)

        Returns:
            BenchmarkResult with AudioDecode performance metrics
        """
        try:
            # Import AudioDecode
            from audiodecode import AudioDecoder

            # Get file info
            file_info = self._get_file_info(file_path)

            # Measure memory before
            self.process.memory_info()  # Prime the cache
            mem_before = self.process.memory_info().rss / (1024 * 1024)

            # Start CPU monitoring
            cpu_percent_start = self.process.cpu_percent()

            # Benchmark decode
            start_time = time.perf_counter()
            decoder = AudioDecoder(
                file_path,
                target_sr=target_sr,
                mono=mono,
            )
            audio = decoder.decode()
            decode_time = time.perf_counter() - start_time

            # Measure memory after
            mem_after = self.process.memory_info().rss / (1024 * 1024)
            memory_delta = mem_after - mem_before

            # Get CPU usage
            cpu_percent = self.process.cpu_percent()

            # Get peak memory
            memory_peak = mem_after

            # Detect which backend was used
            backend_name = backend or type(decoder.backend).__name__

            return BenchmarkResult(
                library="audiodecode",
                file_path=str(file_path),
                file_size_mb=file_info["file_size_mb"],
                duration_seconds=file_info["duration_seconds"],
                sample_rate=file_info["sample_rate"],
                channels=file_info["channels"],
                decode_time_seconds=decode_time,
                memory_peak_mb=memory_peak,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_percent,
                target_sr=target_sr,
                mono=mono,
                backend=backend_name,
                output_shape=audio.shape,
                output_dtype=str(audio.dtype),
                success=True,
            )

        except Exception as e:
            return BenchmarkResult(
                library="audiodecode",
                file_path=str(file_path),
                file_size_mb=0.0,
                duration_seconds=0.0,
                sample_rate=0,
                channels=0,
                decode_time_seconds=0.0,
                memory_peak_mb=0.0,
                memory_delta_mb=0.0,
                cpu_percent=0.0,
                target_sr=target_sr,
                mono=mono,
                backend=backend,
                output_shape=(0,),
                output_dtype="",
                success=False,
                error_message=str(e),
            )

    def compare(
        self,
        file_path: Path,
        target_sr: int | None = None,
        mono: bool = False,
        backend: Literal["soundfile", "pyav"] | None = None,
    ) -> ComparisonResult:
        """
        Compare AudioDecode vs librosa for a single file.

        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (None = keep original)
            mono: Convert to mono if True
            backend: Force specific backend for AudioDecode

        Returns:
            ComparisonResult with performance comparison
        """
        # Run benchmarks
        audiodecode_result = self.benchmark_audiodecode(file_path, target_sr, mono, backend)
        librosa_result = self.benchmark_librosa(file_path, target_sr, mono)

        # Calculate comparison metrics
        if audiodecode_result.success and librosa_result.success:
            speedup = librosa_result.decode_time_seconds / audiodecode_result.decode_time_seconds
            memory_improvement = librosa_result.memory_delta_mb - audiodecode_result.memory_delta_mb
            cpu_improvement = librosa_result.cpu_percent - audiodecode_result.cpu_percent

            faster = speedup > 1.0
            less_memory = memory_improvement > 0
        else:
            speedup = 0.0
            memory_improvement = 0.0
            cpu_improvement = 0.0
            faster = False
            less_memory = False

        return ComparisonResult(
            file_path=str(file_path),
            audiodecode=audiodecode_result,
            librosa_result=librosa_result,
            speedup_factor=speedup,
            memory_improvement_mb=memory_improvement,
            cpu_improvement_percent=cpu_improvement,
            faster_than_librosa=faster,
            less_memory_than_librosa=less_memory,
        )

    def run_benchmarks(
        self,
        audio_files: list[str | Path],
        backends: list[Literal["soundfile", "pyav"]] | None = None,
        target_srs: list[int | None] | None = None,
        mono_configs: list[bool] | None = None,
    ) -> list[ComparisonResult]:
        """
        Run comprehensive benchmarks across multiple files and configurations.

        Args:
            audio_files: List of audio files to benchmark
            backends: List of backends to test (None = auto-select only)
            target_srs: List of target sample rates to test (None = [None] = original SR)
            mono_configs: List of mono settings to test (None = [False, True])

        Returns:
            List of ComparisonResults
        """
        if backends is None:
            backends = [None]  # Auto-select backend
        if target_srs is None:
            target_srs = [None]  # Keep original sample rate
        if mono_configs is None:
            mono_configs = [False, True]

        results = []

        for file_path in audio_files:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            for backend in backends:
                for target_sr in target_srs:
                    for mono in mono_configs:
                        print(f"Benchmarking: {file_path.name} "
                              f"(backend={backend}, sr={target_sr}, mono={mono})")

                        result = self.compare(file_path, target_sr, mono, backend)
                        results.append(result)

        return results

    def save_results(
        self,
        results: list[ComparisonResult],
        filename: str = "benchmark_results.json",
    ) -> Path:
        """
        Save benchmark results to JSON file.

        Args:
            results: List of ComparisonResults
            filename: Output filename

        Returns:
            Path to saved results file
        """
        output_path = self.results_dir / filename

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "file_path": result.file_path,
                "audiodecode": asdict(result.audiodecode),
                "librosa": asdict(result.librosa_result),
                "comparison": {
                    "speedup_factor": result.speedup_factor,
                    "memory_improvement_mb": result.memory_improvement_mb,
                    "cpu_improvement_percent": result.cpu_improvement_percent,
                    "faster_than_librosa": result.faster_than_librosa,
                    "less_memory_than_librosa": result.less_memory_than_librosa,
                },
            })

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def save_baseline(
        self,
        results: list[ComparisonResult],
        filename: str = "baseline.json",
    ) -> Path:
        """
        Save results as new baseline for regression testing.

        Args:
            results: List of ComparisonResults
            filename: Output filename

        Returns:
            Path to saved baseline file
        """
        output_path = self.baseline_dir / filename

        # Convert to baseline format (just the audiodecode results)
        baseline_data = {}
        for result in results:
            if result.audiodecode.success:
                key = f"{Path(result.file_path).name}_{result.audiodecode.target_sr}_{result.audiodecode.mono}"
                baseline_data[key] = {
                    "decode_time_seconds": result.audiodecode.decode_time_seconds,
                    "memory_delta_mb": result.audiodecode.memory_delta_mb,
                    "speedup_vs_librosa": result.speedup_factor,
                }

        with open(output_path, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print(f"Baseline saved to: {output_path}")
        return output_path

    def load_baseline(self, filename: str = "baseline.json") -> dict[str, Any]:
        """
        Load baseline measurements.

        Args:
            filename: Baseline filename

        Returns:
            Dictionary with baseline data
        """
        baseline_path = self.baseline_dir / filename

        if not baseline_path.exists():
            return {}

        with open(baseline_path) as f:
            return json.load(f)

    def print_summary(self, results: list[ComparisonResult]) -> None:
        """
        Print human-readable summary of benchmark results.

        Args:
            results: List of ComparisonResults
        """
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        successful_results = [r for r in results if r.audiodecode.success and r.librosa_result.success]

        if not successful_results:
            print("No successful benchmark runs.")
            return

        # Overall statistics
        avg_speedup = np.mean([r.speedup_factor for r in successful_results])
        avg_memory = np.mean([r.memory_improvement_mb for r in successful_results])
        faster_count = sum(r.faster_than_librosa for r in successful_results)

        print(f"\nTotal benchmarks: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(results) - len(successful_results)}")
        print(f"\nAudioDecode faster than librosa: {faster_count}/{len(successful_results)} tests")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Average memory improvement: {avg_memory:.1f} MB")

        # Detailed results
        print("\n" + "-" * 80)
        print("DETAILED RESULTS")
        print("-" * 80)

        for result in successful_results:
            file_name = Path(result.file_path).name
            print(f"\n{file_name}")
            print(f"  Config: sr={result.audiodecode.target_sr}, mono={result.audiodecode.mono}")
            print(f"  Speedup: {result.speedup_factor:.2f}x {'✓' if result.faster_than_librosa else '✗'}")
            print(f"  AudioDecode: {result.audiodecode.decode_time_seconds:.3f}s, "
                  f"{result.audiodecode.memory_delta_mb:.1f}MB")
            print(f"  Librosa:     {result.librosa_result.decode_time_seconds:.3f}s, "
                  f"{result.librosa_result.memory_delta_mb:.1f}MB")


if __name__ == "__main__":
    # Example usage
    import sys

    runner = BenchmarkRunner()

    # Get audio files from fixtures
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "audio"
    audio_files = list(fixtures_dir.glob("*.wav")) + list(fixtures_dir.glob("*.mp3"))

    if not audio_files:
        print("No audio files found in fixtures/audio/")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files to benchmark\n")

    # Run benchmarks
    results = runner.run_benchmarks(
        audio_files=audio_files,
        backends=[None],  # Auto-select backend
        target_srs=[None, 16000],  # Original SR and 16kHz
        mono_configs=[False, True],
    )

    # Print summary
    runner.print_summary(results)

    # Save results
    runner.save_results(results)
    runner.save_baseline(results)
