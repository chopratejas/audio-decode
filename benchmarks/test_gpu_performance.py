#!/usr/bin/env python3
"""
GPU Performance Test for AudioDecode

Run this on a Linux system with NVIDIA GPU to measure actual speedup.
"""

import sys
import time
from pathlib import Path

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("✗ No GPU available")
            return False
    except ImportError:
        print("✗ PyTorch not installed (required for GPU detection)")
        return False


def benchmark_device(audio_file: str, device: str):
    """Benchmark on specific device."""
    from audiodecode import WhisperInference

    print(f"\n{'='*70}")
    print(f"  Benchmarking on {device.upper()}")
    print(f"{'='*70}")

    # Create model
    print(f"\n📦 Loading model...")
    start_load = time.time()
    whisper = WhisperInference(model_size="base", device=device)
    load_time = time.time() - start_load

    print(f"   Device: {whisper.device}")
    print(f"   Compute type: {whisper.compute_type}")
    print(f"   Batch size: {whisper.batch_size}")
    print(f"   Batched inference: {whisper._is_batched}")
    print(f"   Load time: {load_time:.2f}s")

    # Transcribe
    print(f"\n🎤 Transcribing...")
    start_transcribe = time.time()
    result = whisper.transcribe_file(audio_file)
    transcribe_time = time.time() - start_transcribe

    rtf = result.duration / transcribe_time if transcribe_time > 0 else 0

    print(f"   Duration: {result.duration:.1f}s")
    print(f"   Transcribe time: {transcribe_time:.2f}s")
    print(f"   Total time: {load_time + transcribe_time:.2f}s")
    print(f"   RTF: {rtf:.1f}x")
    print(f"   Words: {len(result.text.split())}")
    print(f"   Segments: {len(result.segments)}")

    return {
        'device': device,
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': load_time + transcribe_time,
        'rtf': rtf,
        'duration': result.duration,
    }


def main():
    """Run GPU vs CPU benchmark."""
    print("="*70)
    print("  AudioDecode GPU Performance Test")
    print("="*70)

    # Check GPU
    has_gpu = check_gpu()

    # Get audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Try to find a test file
        test_files = [
            "uzuPm5R_d8c.mp3",
            "test.mp3",
            "audio.mp3",
        ]
        audio_file = None
        for f in test_files:
            if Path(f).exists():
                audio_file = f
                break

        if not audio_file:
            print("\n❌ No audio file found!")
            print("Usage: python test_gpu_performance.py <audio_file>")
            sys.exit(1)

    print(f"\n📁 Audio file: {audio_file}")

    # Benchmark CPU
    cpu_result = benchmark_device(audio_file, "cpu")

    # Benchmark GPU if available
    if has_gpu:
        gpu_result = benchmark_device(audio_file, "cuda")

        # Compare
        print(f"\n{'='*70}")
        print("  COMPARISON: GPU vs CPU")
        print(f"{'='*70}")

        speedup = cpu_result['transcribe_time'] / gpu_result['transcribe_time']
        total_speedup = cpu_result['total_time'] / gpu_result['total_time']

        print(f"\n{'Metric':<25} {'CPU':<15} {'GPU':<15} {'Speedup':<15}")
        print("-"*70)
        print(f"{'Load Time':<25} {cpu_result['load_time']:.2f}s{'':<10} "
              f"{gpu_result['load_time']:.2f}s{'':<10} "
              f"{cpu_result['load_time']/gpu_result['load_time']:.2f}x")
        print(f"{'Transcribe Time':<25} {cpu_result['transcribe_time']:.2f}s{'':<10} "
              f"{gpu_result['transcribe_time']:.2f}s{'':<10} "
              f"{speedup:.2f}x")
        print(f"{'Total Time':<25} {cpu_result['total_time']:.2f}s{'':<10} "
              f"{gpu_result['total_time']:.2f}s{'':<10} "
              f"{total_speedup:.2f}x")
        print(f"{'RTF':<25} {cpu_result['rtf']:.1f}x{'':<11} "
              f"{gpu_result['rtf']:.1f}x{'':<11} "
              f"{gpu_result['rtf']/cpu_result['rtf']:.2f}x better")

        print(f"\n🚀 GPU Speedup Summary:")
        print(f"   Transcription: {speedup:.2f}x faster")
        print(f"   Total Pipeline: {total_speedup:.2f}x faster")
        print(f"   RTF Improvement: {gpu_result['rtf']/cpu_result['rtf']:.2f}x")

        # Estimate vs OpenAI Whisper
        # CPU is 1.8x faster than OpenAI, so GPU would be:
        vs_openai = 1.8 * speedup
        print(f"\n💡 Estimated vs OpenAI Whisper:")
        print(f"   CPU: 1.8x faster (measured)")
        print(f"   GPU: ~{vs_openai:.1f}x faster (projected)")

    else:
        print("\n⚠️  No GPU available for comparison")
        print("   Install on Linux with NVIDIA GPU for best performance!")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
