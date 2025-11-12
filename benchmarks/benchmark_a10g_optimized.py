#!/usr/bin/env python3
"""
A10G GPU Benchmark - OPTIMIZED SETTINGS
AudioDecode with optimal batch_size=16 vs OpenAI Whisper
"""
import time
import sys
from pathlib import Path

audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"
if not Path(audio_file).exists():
    print(f"File not found: {audio_file}")
    sys.exit(1)

print("="*80)
print("  A10G GPU BENCHMARK - OPTIMIZED AudioDecode")
print("="*80)
print(f"\nAudio: {audio_file}\n")

# OpenAI Whisper on GPU (baseline)
print("="*80)
print("BASELINE: OpenAI Whisper GPU")
print("="*80)
import whisper

print("\nLoading model...")
t0 = time.time()
model = whisper.load_model("base", device="cuda")
load_openai = time.time() - t0
print(f"Loaded in {load_openai:.2f}s")

print("\nTranscribing...")
t0 = time.time()
result_openai = model.transcribe(audio_file)
transcribe_openai = time.time() - t0
duration = result_openai.get('duration', result_openai['segments'][-1]['end'])
rtf_openai = duration / transcribe_openai
print(f"Done in {transcribe_openai:.2f}s (RTF: {rtf_openai:.1f}x)")

# AudioDecode OPTIMIZED
print("\n" + "="*80)
print("AUDIODECODE: OPTIMIZED GPU (batch_size=16)")
print("="*80)
from audiodecode import WhisperInference

print("\nLoading model...")
t0 = time.time()
whisper_ad = WhisperInference(
    model_size="base",
    device="cuda",
    compute_type="float16",
    batch_size=16  # OPTIMAL
)
load_ad = time.time() - t0
print(f"Loaded in {load_ad:.2f}s")

print("\nTranscribing...")
t0 = time.time()
result_ad = whisper_ad.transcribe_file(audio_file, word_timestamps=True)
transcribe_ad = time.time() - t0
rtf_ad = result_ad.duration / transcribe_ad
word_ts = sum(len(s.words) for s in result_ad.segments if s.words)
print(f"Done in {transcribe_ad:.2f}s (RTF: {rtf_ad:.1f}x)")

# RESULTS
print("\n" + "="*80)
print("  🏆 OPTIMIZED RESULTS")
print("="*80)

speedup = transcribe_openai / transcribe_ad
total_speedup = (load_openai + transcribe_openai) / (load_ad + transcribe_ad)

print(f"\n{'Metric':<30} {'OpenAI GPU':<20} {'AudioDecode GPU*':<20} {'Speedup'}")
print("-"*85)
print(f"{'Model Load':<30} {load_openai:.2f}s{'':<15} {load_ad:.2f}s")
print(f"{'Transcription':<30} {transcribe_openai:.2f}s{'':<15} {transcribe_ad:.2f}s{'':<15} **{speedup:.2f}x**")
print(f"{'Total Pipeline':<30} {load_openai+transcribe_openai:.2f}s{'':<15} {load_ad+transcribe_ad:.2f}s{'':<15} **{total_speedup:.2f}x**")
print(f"{'RTF (realtime factor)':<30} {rtf_openai:.1f}x{'':<16} {rtf_ad:.1f}x")
print(f"{'Word Timestamps':<30} 0{'':<20} {word_ts} ✨")

print("\n" + "="*80)
print("  🎉 KEY FINDINGS")
print("="*80)
print(f"\n  ✓ AudioDecode (optimized) is **{speedup:.2f}x FASTER** than OpenAI Whisper")
print(f"  ✓ Achieves {rtf_ad:.1f}x realtime factor on A10G")
print(f"  ✓ Optimal config: batch_size=16, compute_type=float16")
print(f"  ✓ Bonus: {word_ts} word-level timestamps")
print(f"\n  * With optimized batch_size=16 (vs default batch_size=24)")

# Save
with open("A10G_OPTIMIZED_RESULTS.md", "w") as f:
    f.write("# A10G GPU Benchmark - Optimized Results\n\n")
    f.write(f"**Audio:** {duration/60:.1f} minutes\n")
    f.write(f"**Optimal Config:** batch_size=16, float16\n\n")
    f.write("## Performance Comparison\n\n")
    f.write("| System | Transcribe Time | RTF | Speedup |\n")
    f.write("|--------|----------------|-----|----------|\n")
    f.write(f"| OpenAI Whisper (GPU) | {transcribe_openai:.2f}s | {rtf_openai:.1f}x | baseline |\n")
    f.write(f"| AudioDecode (GPU, optimized) | {transcribe_ad:.2f}s | {rtf_ad:.1f}x | **{speedup:.2f}x** |\n")
    f.write(f"\n**AudioDecode with optimal settings is {speedup:.2f}x faster than OpenAI Whisper on A10G GPU!**\n")

print(f"\n  💾 Results saved to A10G_OPTIMIZED_RESULTS.md\n")
print("="*80 + "\n")
