#!/usr/bin/env python3
"""
A10G GPU Optimization Benchmarks
Test different configurations to find optimal settings
"""
import time
import sys
from pathlib import Path

audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"
if not Path(audio_file).exists():
    print(f"File not found: {audio_file}")
    sys.exit(1)

from audiodecode import WhisperInference

print("="*80)
print("  A10G GPU OPTIMIZATION BENCHMARKS")
print("="*80)
print(f"\nAudio: {audio_file}\n")

# Test configurations
configs = [
    # (batch_size, compute_type, description)
    (8, "float16", "Small batch + FP16"),
    (16, "float16", "Medium batch + FP16"),
    (24, "float16", "Large batch + FP16 (default)"),
    (32, "float16", "XL batch + FP16"),
    (16, "int8", "Medium batch + INT8"),
]

results = []

for batch_size, compute_type, desc in configs:
    print(f"\n{'='*80}")
    print(f"  Testing: {desc}")
    print(f"  batch_size={batch_size}, compute_type={compute_type}")
    print(f"{'='*80}")

    try:
        # Load model
        t0 = time.time()
        whisper = WhisperInference(
            model_size="base",
            device="cuda",
            compute_type=compute_type,
            batch_size=batch_size
        )
        load_time = time.time() - t0

        # Transcribe
        t0 = time.time()
        result = whisper.transcribe_file(audio_file)
        transcribe_time = time.time() - t0

        rtf = result.duration / transcribe_time

        print(f"\n  ✓ Load: {load_time:.2f}s")
        print(f"  ✓ Transcribe: {transcribe_time:.2f}s")
        print(f"  ✓ RTF: {rtf:.1f}x")
        print(f"  ✓ Words: {len(result.text.split())}")

        results.append({
            'config': desc,
            'batch_size': batch_size,
            'compute_type': compute_type,
            'load_time': load_time,
            'transcribe_time': transcribe_time,
            'total_time': load_time + transcribe_time,
            'rtf': rtf,
            'words': len(result.text.split())
        })

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        continue

# Results
print("\n" + "="*80)
print("  OPTIMIZATION RESULTS")
print("="*80)

print(f"\n{'Configuration':<30} {'Load':<10} {'Transcribe':<12} {'RTF':<10} {'Rank'}")
print("-"*80)

# Sort by transcribe time (fastest first)
results_sorted = sorted(results, key=lambda x: x['transcribe_time'])

for i, r in enumerate(results_sorted, 1):
    marker = " ⭐" if i == 1 else ""
    print(f"{r['config']:<30} {r['load_time']:.2f}s{'':<5} "
          f"{r['transcribe_time']:.2f}s{'':<6} "
          f"{r['rtf']:.1f}x{'':<5} #{i}{marker}")

# Save detailed results
with open("A10G_OPTIMIZATION_RESULTS.md", "w") as f:
    f.write("# A10G GPU Optimization Results\n\n")
    f.write(f"Audio: {Path(audio_file).name}\n\n")
    f.write("## Results\n\n")
    f.write("| Configuration | Batch | Compute | Load Time | Transcribe | RTF | Rank |\n")
    f.write("|--------------|-------|---------|-----------|------------|-----|------|\n")

    for i, r in enumerate(results_sorted, 1):
        marker = " ⭐" if i == 1 else ""
        f.write(f"| {r['config']} | {r['batch_size']} | {r['compute_type']} | "
                f"{r['load_time']:.2f}s | {r['transcribe_time']:.2f}s | "
                f"{r['rtf']:.1f}x | #{i}{marker} |\n")

    best = results_sorted[0]
    f.write(f"\n## Optimal Configuration\n\n")
    f.write(f"**Winner:** {best['config']}\n\n")
    f.write(f"- Batch size: {best['batch_size']}\n")
    f.write(f"- Compute type: {best['compute_type']}\n")
    f.write(f"- Transcribe time: {best['transcribe_time']:.2f}s\n")
    f.write(f"- RTF: {best['rtf']:.1f}x\n")

best = results_sorted[0]
print(f"\n🏆 WINNER: {best['config']}")
print(f"   Batch size: {best['batch_size']}")
print(f"   Compute type: {best['compute_type']}")
print(f"   Time: {best['transcribe_time']:.2f}s (RTF: {best['rtf']:.1f}x)")

print(f"\n💾 Detailed results saved to A10G_OPTIMIZATION_RESULTS.md")
print("\n" + "="*80 + "\n")
