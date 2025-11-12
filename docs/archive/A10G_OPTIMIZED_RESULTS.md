# A10G GPU Benchmark - Optimized Results

**Audio:** 6.7 minutes
**Optimal Config:** batch_size=16, float16

## Performance Comparison

| System | Transcribe Time | RTF | Speedup |
|--------|----------------|-----|----------|
| OpenAI Whisper (GPU) | 10.75s | 37.2x | baseline |
| AudioDecode (GPU, optimized) | 7.70s | 52.6x | **1.39x** |

**AudioDecode with optimal settings is 1.39x faster than OpenAI Whisper on A10G GPU!**
