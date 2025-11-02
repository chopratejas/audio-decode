#!/bin/bash
# Run AudioDecode STT benchmarks on Linux via Docker
# This script tests cold-start performance where AudioDecode shines most

set -e

echo "========================================"
echo "AudioDecode Linux Benchmark Runner"
echo "========================================"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker first."
    exit 1
fi

echo "Building Docker test image..."
docker build -f Dockerfile.test -t audiodecode-linux-bench .

echo ""
echo "Running benchmarks in Linux container..."
echo ""

# Run benchmarks and save results
docker run --rm \
    -v "$(pwd)/benchmarks:/app/benchmarks" \
    audiodecode-linux-bench \
    python benchmarks/benchmark_stt_real.py

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Results saved to: benchmarks/stt_benchmark_results.json"
echo "========================================"
