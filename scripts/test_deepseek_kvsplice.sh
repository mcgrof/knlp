#!/bin/bash
# Quick test script for KVSplice on DeepSeek models
#
# This script runs both benchmark and quality evaluation on DeepSeek-V2-Lite.
# Adjust parameters as needed for your hardware.
#
# Requirements:
#   - Python environment with transformers, torch, datasets, accelerate
#   - sentencepiece: pip install sentencepiece
#
# Usage:
#   Activate your Python environment first, then run this script:
#   $ source /path/to/your/venv/bin/activate
#   $ ./scripts/test_deepseek_kvsplice.sh

set -e

MODEL="deepseek-ai/DeepSeek-V2-Lite"
COMPRESSION_RATIO=0.5

echo "========================================================================"
echo "Testing KVSplice Plug-in on DeepSeek Models"
echo "========================================================================"
echo "Model: $MODEL"
echo "Compression ratio: $COMPRESSION_RATIO (2x KV cache compression)"
echo ""

# Check if model cache exists
echo "Note: First run will download the model (~27GB for Lite version)"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Run benchmark
echo ""
echo "========================================================================"
echo "1. Running Inference Benchmark"
echo "========================================================================"
echo ""

python scripts/benchmark_deepseek_kvsplice.py \
    --model "$MODEL" \
    --compression-ratio "$COMPRESSION_RATIO" \
    --batch-sizes 1,4,8 \
    --seq-len 512 \
    --prompt-len 64 \
    --trials 3

# Run quality evaluation
echo ""
echo "========================================================================"
echo "2. Running Quality Evaluation"
echo "========================================================================"
echo ""

python scripts/eval_deepseek_quality.py \
    --model "$MODEL" \
    --compression-ratio "$COMPRESSION_RATIO" \
    --dataset wikitext \
    --samples 500 \
    --max-length 512 \
    --batch-size 4

echo ""
echo "========================================================================"
echo "Testing Complete!"
echo "========================================================================"
echo ""
echo "Results show:"
echo "  - Throughput comparison (Original vs KVSplice)"
echo "  - Memory usage reduction from compressed KV cache"
echo "  - Perplexity degradation from untrained compression"
echo ""
echo "Next steps:"
echo "  - Try different compression ratios (0.3, 0.5, 0.7)"
echo "  - Test on larger models (DeepSeek-V2)"
echo "  - Fine-tune KVSplice layers to recover quality"
echo "========================================================================"
