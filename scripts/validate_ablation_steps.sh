#!/bin/bash
# Validate all RATIO ablation steps with dry-run mode
# Catches architecture errors without GPU time
#
# Usage: ./scripts/validate_ablation_steps.sh

set -e

echo "============================================================"
echo "RATIO Ablation Steps Validation"
echo "============================================================"
echo "Testing all 19 ablation steps (0-18) with dry-run mode"
echo "This validates architecture without using GPU"
echo "============================================================"
echo

FAILED_STEPS=()
PASSED_STEPS=()

for step in {0..18}; do
    echo -n "Step $step: "

    if python3 gpt2/train.py \
        --architecture unified-ra \
        --ra-step V$step \
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
echo "Passed: ${#PASSED_STEPS[@]}/19 steps"
echo "Failed: ${#FAILED_STEPS[@]}/19 steps"

if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
    echo
    echo "Failed steps: ${FAILED_STEPS[@]}"
    echo
    echo "Run with --dry-run to see detailed errors:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  python3 gpt2/train.py --architecture unified-ra --ra-step V$step --optimizer adamwspam --dataset finewebedu --dry-run"
    done
    exit 1
else
    echo
    echo "✓ All ablation steps validated successfully"
    exit 0
fi
