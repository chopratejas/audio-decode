"""
Benchmarking suite for AudioDecode performance testing.

This package provides:
- BenchmarkRunner: Comprehensive performance comparison against librosa
- pytest-benchmark integration for statistical rigor and regression detection
- Baseline storage and comparison for continuous performance monitoring
"""

from benchmarks.benchmark_runner import (
    BenchmarkResult,
    BenchmarkRunner,
    ComparisonResult,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "ComparisonResult",
]
