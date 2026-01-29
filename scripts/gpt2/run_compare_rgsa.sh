#!/bin/bash
# run_compare_rgsa.sh - Run baseline vs RGSA comparison with matched configs
#
# Usage: ./scripts/gpt2/run_compare_rgsa.sh [--seed SEED] [--dry-run]
#
# This script:
# 1. Runs baseline GPT-2 and RGSA with identical hyperparameters
# 2. Sets WANDB_GROUP for easy comparison in W&B UI
# 3. Tags runs with model=baseline or model=rgsa
# 4. Saves results to timestamped directories

set -e

# Default seed
SEED=42
DRY_RUN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--seed SEED] [--dry-run]"
            exit 1
            ;;
    esac
done

# Timestamp for this comparison run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WANDB_GROUP="tinystories-rgsa-compare-${TIMESTAMP}"
OUTPUT_BASE="rgsa_compare_${TIMESTAMP}"

echo "========================================"
echo "RGSA Comparison Run"
echo "========================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Seed: ${SEED}"
echo "WANDB_GROUP: ${WANDB_GROUP}"
echo "Output base: ${OUTPUT_BASE}"
echo "========================================"

# Export W&B group for both runs
export WANDB_GROUP="${WANDB_GROUP}"

# Function to run a training config
run_config() {
    local config_name=$1
    local model_tag=$2
    local run_name="${model_tag}_seed${SEED}"
    local output_dir="${OUTPUT_BASE}/${model_tag}"

    echo ""
    echo "========================================"
    echo "Running: ${config_name} (${model_tag})"
    echo "Run name: ${run_name}"
    echo "Output: ${output_dir}"
    echo "========================================"

    # Load the defconfig (this also generates config.py)
    make defconfig-${config_name}

    # Update seed in .config if needed
    if grep -q "CONFIG_SEED=" .config 2>/dev/null; then
        sed -i "s/CONFIG_SEED=.*/CONFIG_SEED=${SEED}/" .config
    else
        echo "CONFIG_SEED=${SEED}" >> .config
    fi

    # Regenerate config.py after seed update
    python scripts/kconfig2py.py .config > config.py

    # Run training with specific run name
    if [ -n "${DRY_RUN}" ]; then
        echo "[DRY-RUN] Would run: make train"
        echo "[DRY-RUN] With WANDB_TAGS=${model_tag}"
    else
        # Set W&B tags for this run
        export WANDB_TAGS="${model_tag},seed${SEED},compare"

        # Run via makefile which handles the test matrix framework
        YES=1 make train 2>&1 | tee "${OUTPUT_BASE}/${model_tag}.log"
    fi

    echo ""
    echo "Completed: ${config_name}"
}

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Save run metadata
cat > "${OUTPUT_BASE}/metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "seed": ${SEED},
    "wandb_group": "${WANDB_GROUP}",
    "configs": ["gpt2-tinystories-baseline", "gpt2-tinystories-rgsa"]
}
EOF

echo "Metadata saved to ${OUTPUT_BASE}/metadata.json"

# Run baseline first
run_config "gpt2-tinystories-baseline" "baseline"

# Then run RGSA
run_config "gpt2-tinystories-rgsa" "rgsa"

echo ""
echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT_BASE}/"
echo "W&B Group: ${WANDB_GROUP}"
echo ""
echo "To view in W&B:"
echo "  1. Go to https://wandb.ai"
echo "  2. Filter by group: ${WANDB_GROUP}"
echo ""
echo "To generate analysis plots, run:"
echo "  python scripts/gpt2/analyze_rgsa_compare.py --group ${WANDB_GROUP}"
echo "========================================"
