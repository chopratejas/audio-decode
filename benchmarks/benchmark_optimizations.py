"""
Benchmark script to test AudioDecode optimizations.

Tests different configurations to measure optimization impact:
1. Baseline: batch_size=24 (old default), vad_filter=True, language=None
2. Opt 1: batch_size=16 (new default)
3. Opt 2: batch_size=16 + vad_filter=False
4. Opt 3: batch_size=16 + vad_filter=False + language="en"
5. Opt 4 (auto): batch_size=16 + vad_filter="auto" + language="en"
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audiodecode import WhisperInference
import torch


def benchmark_config(name, **config):
    """Benchmark a specific configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    print(f"Configuration: {config}")

    # Create model with specified config
    model_config = {
        'model_size': 'base',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'compute_type': 'float16' if torch.cuda.is_available() else 'int8',
    }

    # Override batch_size if specified
    if 'batch_size' in config:
        model_config['batch_size'] = config.pop('batch_size')

    # Create model
    whisper = WhisperInference(**model_config)

    # Transcribe with specified config
    audio_file = "audio.mp3"

    start = time.perf_counter()
    result = whisper.transcribe_file(audio_file, **config)
    elapsed = time.perf_counter() - start

    # Calculate stats
    duration = result.duration
    rtf = duration / elapsed
    words = len(result.text.split())
    segments = len(result.segments)

    print(f"\nResults:")
    print(f"  Time:          {elapsed:.2f}s")
    print(f"  RTF:           {rtf:.1f}x realtime")
    print(f"  Audio:         {duration:.1f}s")
    print(f"  Segments:      {segments}")
    print(f"  Words:         {words}")
    print(f"  Language:      {result.language}")

    return elapsed, rtf


def main():
    """Run all benchmarks."""
    print("="*80)
    print("AudioDecode Optimization Benchmarks")
    print("="*80)
    print()

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("Running on CPU")
    print()

    results = {}

    # Baseline: Old defaults (batch_size=24)
    print("\n" + "🔵 BASELINE (Old Defaults)".center(80))
    results['baseline'] = benchmark_config(
        "Baseline: batch_size=24, vad_filter=True, no language",
        batch_size=24,
        vad_filter=True,
        language=None,
        word_timestamps=False
    )

    # Optimization 1: New batch_size=16
    print("\n" + "🟡 OPTIMIZATION 1: batch_size=16".center(80))
    results['opt1_batch'] = benchmark_config(
        "Optimization 1: batch_size=16",
        batch_size=16,
        vad_filter=True,
        language=None,
        word_timestamps=False
    )

    # Optimization 2: Specify language (skip language detection)
    print("\n" + "🟢 OPTIMIZATION 2: + language='en'".center(80))
    results['opt2_lang'] = benchmark_config(
        "Optimization 2: batch_size=16 + language (skip detection)",
        batch_size=16,
        vad_filter=True,
        language="en",
        word_timestamps=False
    )

    # Optimization 3: VAD auto mode (for 6.7min audio, VAD will be used)
    print("\n" + "🟣 OPTIMIZATION 3: vad_filter='auto'".center(80))
    results['opt3_auto'] = benchmark_config(
        "Optimization 3: batch_size=16 + VAD auto + language",
        batch_size=16,
        vad_filter="auto",  # NEW! Smart mode (will use VAD for >60s audio)
        language="en",
        word_timestamps=False
    )

    # Note: VAD cannot be fully disabled with BatchedInferencePipeline
    # It's required for audio chunking. VAD auto mode is the compromise.

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    baseline_time, baseline_rtf = results['baseline']

    print(f"\n{'Configuration':<50} {'Time':>10} {'Speedup':>10} {'RTF':>10}")
    print("-" * 80)

    for name, (time_val, rtf_val) in results.items():
        speedup = baseline_time / time_val
        config_name = name.replace('_', ' ').title()
        speedup_str = f"{speedup:.2f}x"
        if speedup > 1.0:
            speedup_str = f"✅ {speedup_str}"

        print(f"{config_name:<50} {time_val:>9.2f}s {speedup_str:>10} {rtf_val:>9.1f}x")

    # Calculate cumulative gains
    print("\n" + "="*80)
    print("CUMULATIVE OPTIMIZATION GAINS")
    print("="*80)

    opt1_gain = (baseline_time - results['opt1_batch'][0]) / baseline_time * 100
    opt2_gain = (baseline_time - results['opt2_lang'][0]) / baseline_time * 100
    opt3_gain = (baseline_time - results['opt3_auto'][0]) / baseline_time * 100

    print(f"\nBaseline:                              {baseline_time:.2f}s (100%)")
    print(f"+ Optimization 1 (batch_size=16):      {results['opt1_batch'][0]:.2f}s ({opt1_gain:+.1f}%)")
    print(f"+ Optimization 2 (language='en'):      {results['opt2_lang'][0]:.2f}s ({opt2_gain:+.1f}%)")
    print(f"+ Optimization 3 (VAD auto mode):      {results['opt3_auto'][0]:.2f}s ({opt3_gain:+.1f}%)")

    print(f"\n🎯 Best Speedup: {baseline_time / min(results['opt2_lang'][0], results['opt3_auto'][0]):.2f}x faster")
    print(f"   ({max(opt2_gain, opt3_gain):.1f}% performance improvement)")

    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
