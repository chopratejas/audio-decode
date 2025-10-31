#!/bin/bash
# Comprehensive benchmark runner script
# Usage: ./benchmarks/run_benchmarks.sh [OPTIONS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
RUN_REGRESSION=true
RUN_BENCHMARKS=true
RUN_QUALITY=true
MIN_ROUNDS=5
SAVE_BASELINE=false
COMPARE_BASELINE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --regression-only)
      RUN_BENCHMARKS=false
      RUN_QUALITY=false
      shift
      ;;
    --benchmarks-only)
      RUN_REGRESSION=false
      RUN_QUALITY=false
      shift
      ;;
    --quality-only)
      RUN_REGRESSION=false
      RUN_BENCHMARKS=false
      shift
      ;;
    --min-rounds)
      MIN_ROUNDS="$2"
      shift 2
      ;;
    --save-baseline)
      SAVE_BASELINE=true
      shift
      ;;
    --compare-baseline)
      COMPARE_BASELINE=true
      shift
      ;;
    --help)
      echo "Benchmark Runner for AudioDecode"
      echo ""
      echo "Usage: ./benchmarks/run_benchmarks.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --regression-only      Run only regression tests (FAIL if slower)"
      echo "  --benchmarks-only      Run only statistical benchmarks"
      echo "  --quality-only         Run only quality validation tests"
      echo "  --min-rounds N         Set minimum benchmark rounds (default: 5)"
      echo "  --save-baseline        Save results as new baseline"
      echo "  --compare-baseline     Compare results to saved baseline"
      echo "  --help                 Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./benchmarks/run_benchmarks.sh"
      echo "  ./benchmarks/run_benchmarks.sh --regression-only"
      echo "  ./benchmarks/run_benchmarks.sh --benchmarks-only --min-rounds 20"
      echo "  ./benchmarks/run_benchmarks.sh --save-baseline"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AudioDecode Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check for test audio files
if [ ! -d "fixtures/audio" ] || [ -z "$(ls -A fixtures/audio 2>/dev/null)" ]; then
  echo -e "${YELLOW}⚠ No test audio files found in fixtures/audio/${NC}"
  echo "Generating test audio files..."
  python benchmarks/generate_test_audio.py
  echo ""
fi

# 1. Run Regression Tests
if [ "$RUN_REGRESSION" = true ]; then
  echo -e "${BLUE}1. Running Regression Tests (FAIL if slower than librosa)${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if pytest benchmarks/test_performance.py \
    -k "faster_than_librosa or memory_usage" \
    -v --tb=short --no-header; then
    echo -e "${GREEN}✓ All regression tests passed!${NC}"
  else
    echo -e "${RED}✗ Regression tests failed - performance degradation detected!${NC}"
    exit 1
  fi
  echo ""
fi

# 2. Run Statistical Benchmarks
if [ "$RUN_BENCHMARKS" = true ]; then
  echo -e "${BLUE}2. Running Statistical Benchmarks${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  BENCHMARK_ARGS="--benchmark-only --benchmark-min-rounds=$MIN_ROUNDS"

  if [ "$SAVE_BASELINE" = true ]; then
    BENCHMARK_ARGS="$BENCHMARK_ARGS --benchmark-save=baseline"
    echo -e "${YELLOW}Will save results as new baseline${NC}"
  fi

  if [ "$COMPARE_BASELINE" = true ]; then
    BENCHMARK_ARGS="$BENCHMARK_ARGS --benchmark-compare=baseline"
    echo -e "${YELLOW}Will compare to saved baseline${NC}"
  fi

  pytest benchmarks/test_performance.py $BENCHMARK_ARGS -v --no-header

  echo -e "${GREEN}✓ Benchmarks completed!${NC}"
  echo ""
fi

# 3. Run Quality Validation Tests
if [ "$RUN_QUALITY" = true ]; then
  echo -e "${BLUE}3. Running Quality Validation Tests${NC}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if pytest benchmarks/test_performance.py \
    -k "output_matches" \
    -v --tb=short --no-header; then
    echo -e "${GREEN}✓ Quality validation passed!${NC}"
  else
    echo -e "${RED}✗ Quality validation failed!${NC}"
    exit 1
  fi
  echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Benchmark suite completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Show results location
if [ -d "benchmarks/results" ] && [ -n "$(ls -A benchmarks/results 2>/dev/null)" ]; then
  echo "Results saved in: benchmarks/results/"
  ls -lh benchmarks/results/ | tail -n +2
fi

echo ""
echo "Next steps:"
echo "  • View detailed results: cat benchmarks/results/benchmark_results.json"
echo "  • Compare to baseline: pytest benchmarks/test_performance.py --benchmark-compare=baseline"
echo "  • Update baseline: pytest benchmarks/test_performance.py --benchmark-save=baseline"
