# GPU Setup Guide for AudioDecode Benchmarking

## Quick Setup Instructions

### Step 1: Add Git Remote (if needed)

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/audiodecode.git

# Push the commit
git push -u origin main
```

### Step 2: Choose GPU Instance

**Recommended Cloud Providers:**

1. **RunPod** (Cheapest) - https://runpod.io
   - NVIDIA RTX 4090: $0.34/hour
   - NVIDIA A100 (40GB): $1.89/hour
   - NVIDIA A6000: $0.79/hour

2. **Lambda Labs** - https://lambdalabs.com
   - NVIDIA A100 (40GB): $1.10/hour
   - NVIDIA A10: $0.60/hour

3. **Vast.ai** (Community GPUs) - https://vast.ai
   - RTX 3090: $0.20-0.40/hour
   - RTX 4090: $0.30-0.50/hour

4. **Google Colab Pro** - https://colab.research.google.com
   - T4 GPU: $10/month (easiest for testing)
   - A100: $50/month

**Recommended for Benchmarking**: RTX 4090 or A100 (40GB)

---

## Option 1: RunPod Setup (Recommended)

### 1. Sign up and add credits
- Go to https://runpod.io
- Add $10-20 in credits

### 2. Deploy Pod
```
Template: PyTorch 2.0+
GPU: RTX 4090 or A100
Volume: 20GB (optional)
```

### 3. Connect and Clone
```bash
# SSH into the pod (RunPod provides SSH command)
ssh root@your-pod-ip

# Clone repository
git clone https://github.com/YOUR_USERNAME/audiodecode.git
cd audiodecode
```

### 4. Install Dependencies
```bash
# Install system dependencies
apt-get update
apt-get install -y ffmpeg libsndfile1

# Install Python dependencies
pip install uv
uv pip install -e ".[dev,inference]"

# Install OpenAI Whisper for comparison
pip install openai-whisper
```

### 5. Verify GPU
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Should output: CUDA available: True, GPU: NVIDIA RTX 4090 (or A100)
```

### 6. Run Benchmark
```bash
# Run GPU benchmark
python benchmark_vs_openai_whisper.py

# Results will be saved to BENCHMARK_VS_OPENAI_WHISPER.md
```

---

## Option 2: Google Colab Setup (Easiest)

### 1. Open Colab Notebook
https://colab.research.google.com

### 2. Enable GPU
- Runtime → Change runtime type → GPU (T4, A100, or V100)

### 3. Run Setup
```python
# Install dependencies
!apt-get update
!apt-get install -y ffmpeg libsndfile1

# Clone repository
!git clone https://github.com/YOUR_USERNAME/audiodecode.git
%cd audiodecode

# Install Python packages
!pip install -e ".[dev,inference]"
!pip install openai-whisper
```

### 4. Run Benchmark
```python
!python benchmark_vs_openai_whisper.py
```

### 5. Download Results
```python
from google.colab import files
files.download('BENCHMARK_VS_OPENAI_WHISPER.md')
```

---

## Option 3: Lambda Labs Setup

### 1. Create Instance
```bash
# Launch instance via Lambda Labs dashboard
# Choose: 1x A100 (40GB)
```

### 2. SSH and Setup
```bash
ssh ubuntu@your-instance-ip

# Clone and setup
git clone https://github.com/YOUR_USERNAME/audiodecode.git
cd audiodecode

# Install dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
pip install uv
uv pip install -e ".[dev,inference]"
pip install openai-whisper
```

### 3. Run Benchmark
```bash
python benchmark_vs_openai_whisper.py
```

---

## Expected GPU Benchmark Results

### Estimated Performance (6.7min audio):

| Platform | OpenAI Whisper | AudioDecode | Speedup |
|----------|----------------|-------------|---------|
| **Mac CPU** | 14.17s | 8.00s | **1.77x** |
| **Linux CPU** | 47.64s | 7.94s | **6.00x** |
| **T4 GPU** | ~5s | ~1.5s (est) | **~3x** |
| **RTX 4090** | ~3s | ~0.8s (est) | **~4x** |
| **A100** | ~2s | ~0.5s (est) | **~4x** |

**Note**: GPU estimates based on faster-whisper benchmarks. Actual results TBD.

---

## Troubleshooting

### CUDA Not Found
```bash
# Check CUDA version
nvcc --version

# If not found, install CUDA toolkit
# (Usually pre-installed on GPU instances)
```

### Out of Memory
```bash
# Reduce batch size in benchmark
# Edit benchmark_vs_openai_whisper.py:
# Change batch_size=24 to batch_size=16 or 8
```

### Slow Download
```bash
# Models download from HuggingFace
# If slow, set HF mirror:
export HF_ENDPOINT=https://hf-mirror.com
```

---

## After Benchmarking

### 1. Copy Results
```bash
# Save the benchmark results
cat BENCHMARK_VS_OPENAI_WHISPER.md > gpu_benchmark_results.md

# If remote, SCP to local:
scp root@your-pod-ip:~/audiodecode/BENCHMARK_VS_OPENAI_WHISPER.md .
```

### 2. Update Documentation
- Add GPU results to PLATFORM_BENCHMARK_COMPARISON.md
- Update README with GPU performance
- Update CRITICAL_GAPS_ANALYSIS.md (mark GPU gap as FIXED)

### 3. Cleanup
```bash
# Stop GPU instance to avoid charges
# (RunPod/Lambda: Stop via dashboard)
```

---

## Cost Estimate

**Total Cost for Comprehensive GPU Benchmarking**:

| Task | Time | Cost (RTX 4090) | Cost (A100) |
|------|------|-----------------|-------------|
| Setup | 10min | $0.06 | $0.32 |
| Single benchmark | 2min | $0.01 | $0.06 |
| Multiple runs (5x) | 10min | $0.06 | $0.32 |
| Testing/debugging | 20min | $0.11 | $0.63 |
| **Total** | **~40min** | **~$0.24** | **~$1.33** |

**Recommendation**: Use RTX 4090 (cheaper, similar performance to A100 for inference)

---

## Quick Command Reference

```bash
# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Run benchmark
python benchmark_vs_openai_whisper.py

# Check results
cat BENCHMARK_VS_OPENAI_WHISPER.md | grep "RESULT:"

# Monitor GPU usage during benchmark
watch -n 1 nvidia-smi
```

---

## Next Steps After GPU Benchmarks

1. ✅ **Update README** with GPU performance numbers
2. ✅ **Update CRITICAL_GAPS_ANALYSIS.md** - Mark GPU gap as FIXED
3. ✅ **Update design partner emails** with GPU numbers
4. ✅ **Create comparison table** (CPU vs GPU)
5. ✅ **Blog post**: "AudioDecode: 4x Faster on GPU, 6x Faster on CPU"

---

## Sample GPU Benchmark Output

```
==========================================================================================
  AUDIODECODE vs OPENAI-WHISPER: GPU BENCHMARK
==========================================================================================

Device: NVIDIA RTX 4090
CUDA Version: 12.1

OpenAI Whisper (GPU):
- Model Load: 0.8s
- Transcribe: 2.5s
- Total: 3.3s

AudioDecode (GPU):
- Model Load: 0.2s
- Transcribe: 0.6s
- Total: 0.8s

Result: AudioDecode is 4.1x FASTER on GPU!
```

---

*Ready to benchmark? Pick your GPU instance and follow the setup steps!*
