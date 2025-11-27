#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Validate MLA ablation steps with dry-run mode
# Tests all steps from CONFIG_RA_MLA_ABLATION_STEPS to catch architecture errors
# before committing expensive GPU resources.
#
# Usage: ./scripts/validate_mla_ablation_steps.sh
# Exit code: 0 if all pass, 1 if any fail

set -euo pipefail

# MLA ablation steps from gpt2-mla-ablation defconfig
STEPS=(MLA0 MLAKV0)

PASS_COUNT=0
FAIL_COUNT=0
FAILED_STEPS=()

echo "Validating ${#STEPS[@]} MLA ablation steps..."
echo ""

for STEP in "${STEPS[@]}"; do
    echo -n "Testing step $STEP... "

    if python3 gpt2/train_ra_mla.py \
        --ra-mla-ablation-step "$STEP" \
        --optimizer adamwspam \
        --dataset finewebedu \
        --dry-run \
        > /dev/null 2>&1; then
        echo "✓ PASS"
        ((PASS_COUNT++)) || true
    else
        echo "✗ FAIL"
        ((FAIL_COUNT++)) || true
        FAILED_STEPS+=("$STEP")
    fi
done

echo ""
echo "============================================================"
echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "============================================================"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed steps:"
    for STEP in "${FAILED_STEPS[@]}"; do
        echo "  - $STEP"
    done
    echo ""
    echo "To debug a failed step, run:"
    echo "  python3 gpt2/train_ra_mla.py \\"
    echo "    --ra-mla-ablation-step <STEP> \\"
    echo "    --optimizer adamwspam \\"
    echo "    --dataset finewebedu \\"
    echo "    --dry-run"
    echo ""
    exit 1
fi

exit 0
