#!/bin/bash
# Runs LeNet-5 tokenizer comparison: baseline vs PCA vs Spline-PCA
set -e

# Output directory
OUTPUT_BASE="test_matrix_results_lenet5_tokenizer"
mkdir -p "$OUTPUT_BASE"

echo "==================================================================="
echo "LeNet-5 Tokenizer Comparison: Baseline vs PCA vs Spline-PCA"
echo "==================================================================="
echo ""

# Import config
if [ ! -f config.py ]; then
    echo "Error: config.py not found. Run 'make defconfig-lenet5-tokenizer-comparison' first."
    exit 1
fi

# Test 1: Baseline (no tokenization)
echo "Test 1/3: Baseline (no tokenization, 784 dims)"
echo "-------------------------------------------------------------------"
export LENET5_TOKENIZER_METHOD="none"
export LENET5_ENABLE_TOKENIZER="n"
python3 lenet5/train.py \
    --output-dir "$OUTPUT_BASE/baseline" \
    --wandb-run-name "lenet5-baseline"
echo ""

# Test 2: PCA tokenization
echo "Test 2/3: PCA Tokenization (784→64 dims, spatial tiering)"
echo "-------------------------------------------------------------------"
export LENET5_TOKENIZER_METHOD="pca"
export LENET5_ENABLE_TOKENIZER="y"
python3 lenet5/train.py \
    --output-dir "$OUTPUT_BASE/pca" \
    --wandb-run-name "lenet5-pca"
echo ""

# Test 3: Spline-PCA tokenization
echo "Test 3/3: Spline-PCA Tokenization (spatial + temporal tiering)"
echo "-------------------------------------------------------------------"
export LENET5_TOKENIZER_METHOD="spline-pca"
export LENET5_ENABLE_TOKENIZER="y"
python3 lenet5/train.py \
    --output-dir "$OUTPUT_BASE/spline-pca" \
    --wandb-run-name "lenet5-spline-pca"
echo ""

echo "==================================================================="
echo "All tests complete! Results in: $OUTPUT_BASE/"
echo "Compare in W&B project: lenet5-tokenizer-comparison"
echo "==================================================================="
