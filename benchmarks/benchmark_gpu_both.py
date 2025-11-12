#!/usr/bin/env python3
"""
GPU Benchmark: AudioDecode vs OpenAI Whisper - BOTH ON GPU!
This is the real comparison - optimized vs baseline, both using A10G GPU
"""

import time
import sys
from pathlib import Path

def benchmark_openai_whisper_gpu(audio_file: str):
    """OpenAI Whisper on GPU."""
    import whisper
    import torch

    print("\n" + "="*80)
    print("  OpenAI Whisper on GPU (Baseline)")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📦 Loading model on {device}...")

    start = time.time()
    model = whisper.load_model("base", device=device)
    load_time = time.time() - start
    print(f"   ✓ Loaded in {load_time:.2f}s")

    print("\n🎤 Transcribing...")
    start = time.time()
    result = model.transcribe(audio_file)
    transcribe_time = time.time() - start

    duration = result.get('duration', result['segments'][-1]['end'] if result['segments'] else 0)
    rtf = duration / transcribe_time if transcribe_time > 0 else 0

    print(f"   ✓ Complete!")
    print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"   Time: {transcribe_time:.2f}s")
    print(f"   RTF: {rtf:.1f}x")

    return {
        'name': 'OpenAI Whisper (GPU)',
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': load_time + transcribe_time,
        'rtf': rtf,
        'duration': duration,
        'words': len(result['text'].split())
    }


def benchmark_audiodecode_gpu(audio_file: str):
    """AudioDecode on GPU."""
    from audiodecode import WhisperInference

    print("\n" + "="*80)
    print("  AudioDecode on GPU (Optimized)")
    print("="*80)

    print("\n📦 Loading model on GPU...")
    start = time.time()
    whisper = WhisperInference(model_size="base", device="cuda")
    load_time = time.time() - start
    print(f"   ✓ Loaded in {load_time:.2f}s")
    print(f"   Device: {whisper.device}")
    print(f"   Compute type: {whisper.compute_type}")

    print("\n🎤 Transcribing...")
    start = time.time()
    result = whisper.transcribe_file(audio_file, word_timestamps=True)
    transcribe_time = time.time() - start

    rtf = result.duration / transcribe_time if transcribe_time > 0 else 0

    print(f"   ✓ Complete!")
    print(f"   Duration: {result.duration:.1f}s ({result.duration/60:.1f} min)")
    print(f"   Time: {transcribe_time:.2f}s")
    print(f"   RTF: {rtf:.1f}x")

    return {
        'name': 'AudioDecode (GPU)',
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': load_time + transcribe_time,
        'rtf': rtf,
        'duration': result.duration,
        'words': len(result.text.split()),
        'word_timestamps': sum(len(s.words) for s in result.segments if s.words)
    }


def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"

    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    print("="*80)
    print("  A10G GPU BENCHMARK: AudioDecode vs OpenAI Whisper")
    print("  Both running on GPU for fair comparison!")
    print("="*80)
    print(f"\n📁 Audio: {audio_file}")

    # Benchmark both on GPU
    openai_result = benchmark_openai_whisper_gpu(audio_file)
    audiodecode_result = benchmark_audiodecode_gpu(audio_file)

    # Comparison
    print("\n" + "="*80)
    print("  🏆 GPU vs GPU COMPARISON")
    print("="*80)

    speedup = openai_result['transcribe_time'] / audiodecode_result['transcribe_time']

    print(f"\n{'Metric':<25} {'OpenAI GPU':<20} {'AudioDecode GPU':<20} {'Speedup':<15}")
    print("-"*85)
    print(f"{'Transcribe Time':<25} {openai_result['transcribe_time']:.2f}s{'':<15} "
          f"{audiodecode_result['transcribe_time']:.2f}s{'':<15} "
          f"**{speedup:.2f}x FASTER**")
    print(f"{'RTF':<25} {openai_result['rtf']:.1f}x{'':<16} "
          f"{audiodecode_result['rtf']:.1f}x")

    print("\n" + "="*80)
    print(f"  🎉 AudioDecode is {speedup:.2f}x FASTER on A10G GPU!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
