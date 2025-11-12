"""
Final benchmark: Optimized AudioDecode vs OpenAI Whisper baseline.

This compares:
1. OpenAI Whisper (GPU) - The baseline everyone uses
2. AudioDecode optimized (GPU, batch_size=16, vad_filter="auto", language="en")
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import whisper as openai_whisper
from audiodecode import WhisperInference


def benchmark_openai_whisper(audio_file):
    """Benchmark OpenAI Whisper (baseline)."""
    print("\n" + "="*80)
    print("🔵 BASELINE: OpenAI Whisper (GPU)")
    print("="*80)

    # Load model
    print("Loading OpenAI Whisper model...")
    start_load = time.perf_counter()
    model = openai_whisper.load_model("base", device="cuda")
    load_time = time.perf_counter() - start_load
    print(f"  Model load time: {load_time:.2f}s")

    # Transcribe
    print("Transcribing...")
    start_transcribe = time.perf_counter()
    result = model.transcribe(str(audio_file))
    transcribe_time = time.perf_counter() - start_transcribe

    total_time = load_time + transcribe_time

    # Calculate stats
    # Estimate duration from result
    if result['segments']:
        duration = result['segments'][-1]['end']
    else:
        duration = 0

    rtf = duration / transcribe_time if transcribe_time > 0 else 0
    words = len(result['text'].split())
    segments = len(result['segments'])

    print(f"\nResults:")
    print(f"  Model load:    {load_time:.2f}s")
    print(f"  Transcribe:    {transcribe_time:.2f}s")
    print(f"  Total:         {total_time:.2f}s")
    print(f"  RTF:           {rtf:.1f}x realtime")
    print(f"  Audio:         {duration:.1f}s")
    print(f"  Segments:      {segments}")
    print(f"  Words:         {words}")
    print(f"  Language:      {result.get('language', 'unknown')}")

    return {
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': total_time,
        'rtf': rtf,
        'duration': duration,
        'segments': segments,
        'words': words,
        'language': result.get('language', 'unknown')
    }


def benchmark_audiodecode_optimized(audio_file):
    """Benchmark AudioDecode with all optimizations."""
    print("\n" + "="*80)
    print("🟢 OPTIMIZED: AudioDecode (GPU, all optimizations)")
    print("="*80)

    # Load model with optimizations
    print("Loading AudioDecode model...")
    start_load = time.perf_counter()
    whisper = WhisperInference(
        model_size="base",
        device="cuda",
        compute_type="float16",
        batch_size=16  # OPTIMIZED! (was 24)
    )
    load_time = time.perf_counter() - start_load
    print(f"  Model load time: {load_time:.2f}s")

    # Transcribe with optimizations
    print("Transcribing with optimizations...")
    print("  - batch_size=16 (optimized from 24)")
    print("  - vad_filter='auto' (smart mode)")
    print("  - language='en' (skip detection)")

    start_transcribe = time.perf_counter()
    result = whisper.transcribe_file(
        str(audio_file),
        language="en",          # OPTIMIZED: Skip language detection
        vad_filter="auto",      # OPTIMIZED: Smart VAD mode
        word_timestamps=False   # Default (no overhead)
    )
    transcribe_time = time.perf_counter() - start_transcribe

    total_time = load_time + transcribe_time

    # Calculate stats
    duration = result.duration
    rtf = duration / transcribe_time if transcribe_time > 0 else 0
    words = len(result.text.split())
    segments = len(result.segments)

    print(f"\nResults:")
    print(f"  Model load:    {load_time:.2f}s")
    print(f"  Transcribe:    {transcribe_time:.2f}s")
    print(f"  Total:         {total_time:.2f}s")
    print(f"  RTF:           {rtf:.1f}x realtime")
    print(f"  Audio:         {duration:.1f}s")
    print(f"  Segments:      {segments}")
    print(f"  Words:         {words}")
    print(f"  Language:      {result.language}")

    return {
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'total_time': total_time,
        'rtf': rtf,
        'duration': duration,
        'segments': segments,
        'words': words,
        'language': result.language
    }


def main():
    """Run comparison benchmark."""
    print("="*80)
    print("AudioDecode vs OpenAI Whisper - Final Comparison")
    print("="*80)
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ GPU not available! This benchmark requires CUDA.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    audio_file = Path("audio.mp3")
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return

    # Benchmark OpenAI Whisper (baseline)
    openai_results = benchmark_openai_whisper(audio_file)

    # Benchmark AudioDecode (optimized)
    audiodecode_results = benchmark_audiodecode_optimized(audio_file)

    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Metric':<25} {'OpenAI Whisper':>18} {'AudioDecode':>18} {'Improvement':>15}")
    print("-" * 80)

    # Model loading
    load_speedup = openai_results['load_time'] / audiodecode_results['load_time']
    print(f"{'Model Load Time':<25} {openai_results['load_time']:>17.2f}s "
          f"{audiodecode_results['load_time']:>17.2f}s "
          f"{load_speedup:>14.2f}x faster")

    # Transcription
    transcribe_speedup = openai_results['transcribe_time'] / audiodecode_results['transcribe_time']
    print(f"{'Transcription Time':<25} {openai_results['transcribe_time']:>17.2f}s "
          f"{audiodecode_results['transcribe_time']:>17.2f}s "
          f"{transcribe_speedup:>14.2f}x faster")

    # Total
    total_speedup = openai_results['total_time'] / audiodecode_results['total_time']
    print(f"{'Total Time':<25} {openai_results['total_time']:>17.2f}s "
          f"{audiodecode_results['total_time']:>17.2f}s "
          f"{total_speedup:>14.2f}x faster")

    print("-" * 80)

    # RTF
    print(f"{'Realtime Factor (RTF)':<25} {openai_results['rtf']:>17.1f}x "
          f"{audiodecode_results['rtf']:>17.1f}x "
          f"{audiodecode_results['rtf'] / openai_results['rtf']:>14.2f}x better")

    # Quality metrics
    print(f"\n{'Quality Metrics':<25} {'OpenAI':>18} {'AudioDecode':>18} {'Match':>15}")
    print("-" * 80)
    print(f"{'Segments':<25} {openai_results['segments']:>18} "
          f"{audiodecode_results['segments']:>18} "
          f"{'✅' if abs(openai_results['segments'] - audiodecode_results['segments']) <= 5 else '⚠️':>15}")
    print(f"{'Words':<25} {openai_results['words']:>18} "
          f"{audiodecode_results['words']:>18} "
          f"{'✅' if abs(openai_results['words'] - audiodecode_results['words']) <= 10 else '⚠️':>15}")
    print(f"{'Language':<25} {openai_results['language']:>18} "
          f"{audiodecode_results['language']:>18} "
          f"{'✅' if openai_results['language'] == audiodecode_results['language'] else '⚠️':>15}")

    # Final verdict
    print("\n" + "="*80)
    print("🎯 FINAL VERDICT")
    print("="*80)
    print()
    print(f"AudioDecode (optimized) is {total_speedup:.2f}x FASTER than OpenAI Whisper!")
    print(f"  • Model loading: {load_speedup:.2f}x faster")
    print(f"  • Transcription: {transcribe_speedup:.2f}x faster")
    print(f"  • Total pipeline: {total_speedup:.2f}x faster")
    print(f"  • Realtime factor: {audiodecode_results['rtf']:.1f}x vs {openai_results['rtf']:.1f}x")
    print()

    improvement_pct = (total_speedup - 1) * 100
    print(f"⚡ Performance gain: {improvement_pct:.1f}% faster than baseline!")
    print()

    # Time savings
    time_saved = openai_results['total_time'] - audiodecode_results['total_time']
    print(f"💰 Time saved per transcription: {time_saved:.2f}s")
    print()

    # Scalability
    hours_per_1000 = {
        'openai': (openai_results['total_time'] * 1000) / 3600,
        'audiodecode': (audiodecode_results['total_time'] * 1000) / 3600
    }
    print(f"📈 For 1000 files (6.7min each):")
    print(f"  • OpenAI Whisper: {hours_per_1000['openai']:.1f} hours")
    print(f"  • AudioDecode: {hours_per_1000['audiodecode']:.1f} hours")
    print(f"  • Time saved: {hours_per_1000['openai'] - hours_per_1000['audiodecode']:.1f} hours")
    print()

    print("="*80)
    print("✅ Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
