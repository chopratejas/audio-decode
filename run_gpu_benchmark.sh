#!/bin/bash
# Quick GPU benchmark runner
# Usage: ./run_gpu_benchmark.sh

set -e

echo "=================================="
echo "  AudioDecode GPU Benchmark"
echo "=================================="
echo ""

# Check for GPU
echo "🔍 Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "❌ nvidia-smi not found. Are you running on a GPU instance?"
    echo "   This script requires an NVIDIA GPU with CUDA support."
    exit 1
fi

# Check CUDA with Python
echo "🔍 Checking CUDA with Python..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "❌ PyTorch CUDA check failed. Installing torch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}
echo ""

# Install dependencies if needed
if ! python3 -c "import audiodecode" 2>/dev/null; then
    echo "📦 Installing AudioDecode..."
    pip install -e ".[dev,inference]"
fi

if ! python3 -c "import faster_whisper" 2>/dev/null; then
    echo "📦 Installing faster-whisper..."
    pip install faster-whisper
fi

if ! python3 -c "import whisper" 2>/dev/null; then
    echo "📦 Installing openai-whisper..."
    pip install openai-whisper
fi

echo ""
echo "🚀 Running GPU benchmark..."
echo ""

# Run the benchmark
python3 benchmark_vs_openai_whisper.py

echo ""
echo "✅ GPU benchmark complete!"
echo ""
echo "📊 Results saved to:"
echo "   - BENCHMARK_VS_OPENAI_WHISPER.md"
echo "   - Console output above"
echo ""
echo "Next steps:"
echo "1. Review results in BENCHMARK_VS_OPENAI_WHISPER.md"
echo "2. Update README.md with GPU performance numbers"
echo "3. Update CRITICAL_GAPS_ANALYSIS.md (mark GPU gap as FIXED)"
echo "4. Update PLATFORM_BENCHMARK_COMPARISON.md with GPU data"
echo ""
