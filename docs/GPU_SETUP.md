# GPU Setup Guide

Quick guide to running AudioDecode with GPU acceleration.

## Prerequisites

- NVIDIA GPU with CUDA support (compute capability 7.0+)
- CUDA Toolkit 11.8+ or 12.x
- Python 3.10+

## Installation

```bash
# Install AudioDecode with GPU dependencies
pip install -e ".[inference]"

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Cloud GPU Options

If you don't have a local GPU, these cloud providers offer affordable GPU instances:

- **Google Colab**: Free T4 GPU, easy to use
- **RunPod**: RTX 4090 from $0.34/hour
- **Lambda Labs**: A100 from $1.10/hour
- **Vast.ai**: RTX 3090 from $0.20/hour

## Running Benchmarks

```bash
# Basic benchmark comparing with OpenAI Whisper
cd benchmarks
python benchmark_vs_openai_whisper.py

# Full benchmark suite
python benchmark_runner.py
```

## Performance

Validated performance on NVIDIA A10G (6.7min audio):

- **AudioDecode GPU**: 9.26s (43.8x real-time factor)
- **OpenAI Whisper GPU**: 22.58s (17.7x real-time factor)
- **Speedup**: 2.4x faster

Optimal configuration:
- `batch_size=16`
- `compute_type="float16"`

## Troubleshooting

### CUDA not found
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of memory
Reduce batch size in your code:
```python
from audiodecode import WhisperInference

whisper = WhisperInference(
    model_size='base',
    batch_size=8  # Reduce from default 16
)
```

### Slow model download
Models download from HuggingFace. If slow, use a mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Expected Performance

| GPU | OpenAI Whisper | AudioDecode | Speedup |
|-----|----------------|-------------|---------|
| A10G | 22.6s | 9.3s | 2.4x |
| T4 | ~5s | ~2s | ~2-3x |
| RTX 4090 | ~3s | ~1s | ~3-4x |
| A100 | ~2s | ~0.7s | ~3-4x |

*Times for transcribing 6.7 minutes of audio*
