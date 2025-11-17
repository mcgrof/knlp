#!/bin/bash
# Validate RA ablation steps with dry-run mode
# Catches architecture errors without GPU time
#
# Usage: ./scripts/validate_ablation_steps.sh

set -e

echo "============================================================"
echo "RA Ablation Steps Validation"
echo "============================================================"
echo "Testing R0, R1 ablation steps with dry-run mode"
echo "This validates architecture without using GPU"
echo "============================================================"
echo

FAILED_STEPS=()
PASSED_STEPS=()
TOTAL=0

# Test R0 and R1 steps
for step in R0 R1; do
    TOTAL=$((TOTAL + 1))
    echo -n "Step $step: "

    if python3 gpt2/train.py \
        --architecture unified-ra \
        --ablation-mode \
        --ra-step $step \
        --optimizer adamwspam \
        --dataset finewebedu \
        --dry-run \
        > /dev/null 2>&1; then
        echo "✓ PASS"
        PASSED_STEPS+=($step)
    else
        echo "✗ FAIL"
        FAILED_STEPS+=($step)
    fi
done

echo
echo "============================================================"
echo "Validation Summary"
echo "============================================================"
echo "Passed: ${#PASSED_STEPS[@]}/$TOTAL steps"
echo "Failed: ${#FAILED_STEPS[@]}/$TOTAL steps"

if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
    echo
    echo "Failed steps: ${FAILED_STEPS[@]}"
    echo
    echo "Run with --dry-run to see detailed errors:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  python3 gpt2/train.py --architecture unified-ra --ablation-mode --ra-step $step --optimizer adamwspam --dataset finewebedu --dry-run"
    done
    exit 1
else
    echo
    echo "✓ All ablation steps validated successfully"
    exit 0
fi
