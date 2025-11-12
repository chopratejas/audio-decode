# A10G GPU Optimization Results

Audio: audio.mp3

## Results

| Configuration | Batch | Compute | Load Time | Transcribe | RTF | Rank |
|--------------|-------|---------|-----------|------------|-----|------|
| Medium batch + FP16 | 16 | float16 | 0.20s | 3.74s | 108.3x | #1 ⭐ |
| XL batch + FP16 | 32 | float16 | 0.18s | 3.82s | 106.2x | #2 |
| Large batch + FP16 (default) | 24 | float16 | 0.19s | 3.83s | 105.8x | #3 |
| Medium batch + INT8 | 16 | int8 | 1.25s | 5.02s | 80.7x | #4 |
| Small batch + FP16 | 8 | float16 | 0.57s | 7.48s | 54.2x | #5 |

## Optimal Configuration

**Winner:** Medium batch + FP16

- Batch size: 16
- Compute type: float16
- Transcribe time: 3.74s
- RTF: 108.3x
