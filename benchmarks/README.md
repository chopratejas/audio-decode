# AudioDecode Benchmarks

Comprehensive benchmark infrastructure to continuously compare AudioDecode performance against librosa.

## Overview

This benchmark suite provides:
- **Statistical rigor** via pytest-benchmark (min, max, mean, stddev)
- **Regression detection** - tests FAIL if AudioDecode is slower than librosa
- **Baseline tracking** - compare against saved performance baselines
- **Multiple backends** - separate tests for soundfile (WAV/FLAC) and PyAV (MP3/M4A)
- **Memory profiling** - detect memory usage regressions
- **Quality validation** - ensure performance improvements don't sacrifice decode accuracy

## Directory Structure

```
benchmarks/
├── __init__.py              # Package exports
├── benchmark_runner.py      # Main benchmark orchestrator class
├── test_performance.py      # pytest-benchmark regression tests
├── baseline/                # Stored baseline results for comparison
│   └── baseline.json
├── results/                 # Benchmark run results
│   └── benchmark_results.json
└── README.md               # This file
```

## Quick Start

### 1. Run Benchmark Tests

Run all benchmarks with statistical analysis:

```bash
# Run all benchmark tests
pytest benchmarks/test_performance.py --benchmark-only

# Run with more rounds for better statistics
pytest benchmarks/test_performance.py --benchmark-only --benchmark-min-rounds=10

# Run only WAV tests
pytest benchmarks/test_performance.py --benchmark-only -k "wav"

# Run only MP3 tests
pytest benchmarks/test_performance.py --benchmark-only -k "mp3"
```

### 2. Run Regression Tests

These tests FAIL if AudioDecode is slower than librosa:

```bash
# Run regression tests (not benchmarks)
pytest benchmarks/test_performance.py -k "faster_than_librosa"

# Run all tests including regression checks
pytest benchmarks/test_performance.py
```

### 3. Generate Baseline

Save current performance as baseline for future comparisons:

```bash
# Run benchmarks and save baseline
pytest benchmarks/test_performance.py --benchmark-only --benchmark-save=baseline

# Compare current run to saved baseline
pytest benchmarks/test_performance.py --benchmark-only --benchmark-compare=baseline
```

### 4. Run Comprehensive Benchmark Suite

Use the BenchmarkRunner class for detailed analysis:

```bash
# Run the standalone benchmark runner
cd benchmarks
python benchmark_runner.py

# Or use in Python
python -c "
from pathlib import Path
from benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
fixtures = Path('fixtures/audio')
audio_files = list(fixtures.glob('*.wav')) + list(fixtures.glob('*.mp3'))

results = runner.run_benchmarks(audio_files)
runner.print_summary(results)
runner.save_results(results)
runner.save_baseline(results)
"
```

## Test Categories

### Performance Benchmarks (`pytest-benchmark`)

These tests measure performance with statistical rigor:

- `test_decode_wav_audiodecode` - Benchmark AudioDecode WAV decoding
- `test_decode_wav_librosa` - Benchmark librosa WAV decoding (baseline)
- `test_decode_mp3_audiodecode` - Benchmark AudioDecode MP3 decoding
- `test_decode_mp3_librosa` - Benchmark librosa MP3 decoding (baseline)
- `test_decode_*_resample_*` - Benchmarks with resampling to 16kHz

Results include:
- Min/Max/Mean/Median times
- Standard deviation
- Outlier detection
- Statistical comparison between runs

### Regression Tests (FAIL on slowdown)

These tests enforce performance requirements:

- `test_wav_faster_than_librosa` - **FAILS** if AudioDecode WAV decode is slower than librosa
- `test_mp3_faster_than_librosa` - **FAILS** if AudioDecode MP3 decode is slower than librosa
- `test_memory_usage_wav` - **FAILS** if memory usage exceeds threshold
- `test_compare_to_baseline_wav` - **FAILS** if slower than saved baseline

### Quality Validation Tests

These tests ensure correctness:

- `test_output_matches_librosa_wav` - Validates bit-accurate output for WAV
- `test_output_matches_librosa_mp3` - Validates similar output for MP3

## Performance Metrics

Each benchmark measures:

1. **Decode Time** - Wall clock time (seconds) using `time.perf_counter()`
2. **Memory Usage** - Peak RSS memory (MB) via `psutil`
3. **CPU Utilization** - CPU percentage during decode
4. **Output Validation** - Shape, dtype, and accuracy checks

## Baseline Management

### Creating a New Baseline

```bash
# Run benchmarks and save as baseline
pytest benchmarks/test_performance.py --benchmark-only --benchmark-save=baseline

# Or use BenchmarkRunner
python -c "
from benchmarks import BenchmarkRunner
from pathlib import Path

runner = BenchmarkRunner()
results = runner.run_benchmarks(['fixtures/audio/test.wav'])
runner.save_baseline(results)
"
```

### Comparing to Baseline

```bash
# Compare current performance to saved baseline
pytest benchmarks/test_performance.py --benchmark-compare=baseline

# This will show:
# - Performance changes (faster/slower)
# - Statistical significance
# - Percentage differences
```

### Updating Baseline After Improvement

When you intentionally improve performance:

```bash
# 1. Verify improvement
pytest benchmarks/test_performance.py --benchmark-compare=baseline

# 2. Update baseline if satisfied
pytest benchmarks/test_performance.py --benchmark-only --benchmark-save=baseline
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Benchmarks

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run regression tests
        run: |
          pytest benchmarks/test_performance.py -v

      - name: Run benchmarks
        run: |
          pytest benchmarks/test_performance.py --benchmark-only --benchmark-json=output.json

      - name: Compare to baseline
        run: |
          pytest benchmarks/test_performance.py --benchmark-compare=baseline || true
```

## Configuration

### Performance Thresholds

Edit `test_performance.py` to adjust thresholds:

```python
# Must be at least as fast as librosa
SPEEDUP_THRESHOLD = 1.0

# Must not use >50MB more memory than librosa
MEMORY_THRESHOLD_MB = 50.0

# Allow 10% regression tolerance vs baseline
REGRESSION_TOLERANCE = 10.0
```

### Test Fixtures

Add audio files to `fixtures/audio/`:

```bash
fixtures/audio/
├── test_16khz_mono.wav
├── test_44khz_stereo.wav
├── test_128kbps.mp3
└── test_320kbps.mp3
```

Tests automatically discover all `.wav`, `.flac`, `.mp3`, and `.m4a` files.

## Understanding Results

### pytest-benchmark Output

```
-------------------------- benchmark 'mp3-decode': 2 tests --------------------------
Name (time in ms)                    Min      Max     Mean  StdDev   Median     IQR
------------------------------------------------------------------------------------
test_decode_mp3_audiodecode       12.34    15.67    13.45    0.89    13.21    1.23
test_decode_mp3_librosa           45.67    52.34    48.12    2.45    47.89    3.12
------------------------------------------------------------------------------------
```

- **Min**: Fastest run (best case)
- **Max**: Slowest run (worst case)
- **Mean**: Average across all runs
- **Median**: Middle value (less affected by outliers)
- **StdDev**: Consistency (lower = more consistent)
- **IQR**: Interquartile range (spread of middle 50%)

### Regression Test Output

**PASS** (AudioDecode faster):
```
✓ MP3 decode speedup: 3.58x (AudioDecode: 13.45ms, librosa: 48.12ms)
```

**FAIL** (AudioDecode slower):
```
FAILED: AudioDecode is SLOWER than librosa for MP3 files!
  AudioDecode: 52.34s
  Librosa: 48.12s
  Speedup: 0.92x (threshold: 1.00x)
  File: test.mp3
```

## Troubleshooting

### No audio files found

```bash
# Create test audio files
mkdir -p fixtures/audio
# Add .wav and .mp3 files to fixtures/audio/
```

### AudioDecode not yet implemented

Tests will skip with:
```
SKIPPED: AudioDecode not yet implemented
```

This is expected during development. Tests will run once the implementation is complete.

### Memory measurements unstable

Memory profiling can be affected by:
- Other processes
- Python garbage collection
- OS memory management

Run multiple times or increase sample size:
```bash
pytest benchmarks/test_performance.py -k "memory" --count=5
```

### Baseline comparison fails

```bash
# Reset baseline if needed
rm benchmarks/baseline/baseline.json
pytest benchmarks/test_performance.py --benchmark-only --benchmark-save=baseline
```

## Advanced Usage

### Custom Benchmark Configurations

```python
from benchmarks import BenchmarkRunner
from pathlib import Path

runner = BenchmarkRunner()

# Test specific configurations
results = runner.run_benchmarks(
    audio_files=['test.wav'],
    backends=['soundfile', 'pyav'],  # Test both backends
    target_srs=[None, 16000, 22050],  # Test multiple sample rates
    mono_configs=[False, True],  # Test stereo and mono
)

# Analyze results
for result in results:
    if result.faster_than_librosa:
        print(f"✓ {result.file_path}: {result.speedup_factor:.2f}x faster")
    else:
        print(f"✗ {result.file_path}: {result.speedup_factor:.2f}x slower")
```

### Memory Profiling with memory_profiler

For detailed line-by-line memory profiling:

```bash
# Install memory_profiler
pip install memory_profiler

# Profile a specific function
python -m memory_profiler benchmarks/benchmark_runner.py
```

### Generating Reports

```python
from benchmarks import BenchmarkRunner
from pathlib import Path
import json

runner = BenchmarkRunner()
results = runner.run_benchmarks(['test.wav', 'test.mp3'])

# Save detailed JSON results
runner.save_results(results, 'latest_benchmarks.json')

# Generate comparison report
with open('benchmarks/results/latest_benchmarks.json') as f:
    data = json.load(f)
    for item in data:
        comp = item['comparison']
        print(f"{item['file_path']}: {comp['speedup_factor']:.2f}x faster")
```

## Best Practices

1. **Run on Idle System**: Close other applications for consistent results
2. **Warm Up**: First run may be slower due to disk caching
3. **Multiple Runs**: Use `--benchmark-min-rounds=10` for statistical significance
4. **Control Variables**: Test one change at a time
5. **Save Baselines**: Always save baselines before major changes
6. **Track Trends**: Compare multiple baseline versions over time
7. **Test Real Files**: Use representative audio files from your use case

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [psutil documentation](https://psutil.readthedocs.io/)
- [memory_profiler documentation](https://pypi.org/project/memory-profiler/)
