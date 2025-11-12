# AudioDecode vs OpenAI Whisper - CPU Benchmark

**Platform:** A10G Machine (CPU-only)
**Date:** 2025-11-12 19:08:37
**Audio:** audio.mp3
**Duration:** 405.4s (6.8 min)

## Results

| Metric | OpenAI Whisper | AudioDecode | Speedup |
|--------|----------------|-------------|----------|
| Load Time | 1.27s | 2.76s | **0.46x** |
| Transcribe | 43.35s | 306.71s | **0.14x** |
| Total | 44.62s | 309.47s | **0.14x** |
| RTF | 9.2x | 1.3x | 0.14x |

**AudioDecode is 0.14x faster than OpenAI Whisper on CPU!**
