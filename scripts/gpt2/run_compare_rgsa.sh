#!/bin/bash
# run_compare_rgsa.sh - Run multi-seed baseline vs RGSA comparison
#
# Usage:
#   ./scripts/gpt2/run_compare_rgsa.sh [--seeds "1 2 3"] [--dry-run]
#   ./scripts/gpt2/run_compare_rgsa.sh [--seeds "1 2 3"] [--ablations] [--dry-run]
#
# This script:
# 1. Runs baseline GPT-2 and RGSA for each seed
# 2. Optionally runs rgsa_dense and rgsa_random ablations
# 3. Sets WANDB_GROUP for easy comparison in W&B UI
# 4. Tags runs with model type and seed
# 5. Saves results to timestamped directories

set -e

# Defaults
SEEDS="1 2 3"
DRY_RUN=""
ABLATIONS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --seed)
            SEEDS="$2"
            shift 2
            ;;
        --ablations)
            ABLATIONS="yes"
            shift
            ;;
        --dry-run)
            DRY_RUN="yes"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--seeds \"1 2 3\"] [--ablations] [--dry-run]"
            exit 1
            ;;
    esac
done

# Timestamp for this comparison run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WANDB_GROUP="tinystories-rgsa-compare-${TIMESTAMP}"
OUTPUT_BASE="rgsa_compare_${TIMESTAMP}"

# Build list of configs to run
CONFIGS="gpt2-tinystories-baseline:baseline gpt2-tinystories-rgsa:rgsa"
if [ -n "${ABLATIONS}" ]; then
    CONFIGS="${CONFIGS} gpt2-tinystories-rgsa-dense:rgsa_dense"
    CONFIGS="${CONFIGS} gpt2-tinystories-rgsa-random:rgsa_random"
fi

# Count total runs
SEED_COUNT=$(echo ${SEEDS} | wc -w)
CONFIG_COUNT=$(echo ${CONFIGS} | wc -w)
TOTAL_RUNS=$((SEED_COUNT * CONFIG_COUNT))

echo "========================================"
echo "RGSA Multi-Seed Comparison Run"
echo "========================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Seeds: ${SEEDS}"
echo "Configs: $(echo ${CONFIGS} | tr ' ' '\n' | cut -d: -f2 | tr '\n' ' ')"
echo "Total runs: ${TOTAL_RUNS}"
echo "WANDB_GROUP: ${WANDB_GROUP}"
echo "Output base: ${OUTPUT_BASE}"
echo "========================================"

# Export W&B group for all runs
export WANDB_GROUP="${WANDB_GROUP}"

# Function to run a single training config with a specific seed
run_config() {
    local config_name=$1
    local model_tag=$2
    local seed=$3
    local run_name="${model_tag}_seed${seed}"
    local output_dir="${OUTPUT_BASE}/${model_tag}_seed${seed}"

    echo ""
    echo "========================================"
    echo "Running: ${config_name} (${model_tag}, seed=${seed})"
    echo "Run name: ${run_name}"
    echo "Output: ${output_dir}"
    echo "========================================"

    # Load the defconfig (this also generates config.py)
    make defconfig-${config_name}

    # Update seed in .config
    if grep -q "CONFIG_SEED=" .config 2>/dev/null; then
        sed -i "s/CONFIG_SEED=.*/CONFIG_SEED=${seed}/" .config
    else
        echo "CONFIG_SEED=${seed}" >> .config
    fi

    # Regenerate config.py after seed update
    python scripts/kconfig2py.py .config > config.py

    # Run training with specific run name
    if [ -n "${DRY_RUN}" ]; then
        echo "[DRY-RUN] Would run: make train"
        echo "[DRY-RUN] With WANDB_TAGS=${model_tag},seed${seed},compare"
    else
        # Set W&B tags for this run
        export WANDB_TAGS="${model_tag},seed${seed},compare"

        mkdir -p "${output_dir}"

        # Run via makefile which handles the test matrix framework
        YES=1 make train 2>&1 | tee "${OUTPUT_BASE}/${model_tag}_seed${seed}.log"
    fi

    echo ""
    echo "Completed: ${config_name} (seed=${seed})"
}

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Build configs array for metadata
CONFIG_NAMES=""
for config_pair in ${CONFIGS}; do
    config_name=$(echo ${config_pair} | cut -d: -f1)
    if [ -z "${CONFIG_NAMES}" ]; then
        CONFIG_NAMES="\"${config_name}\""
    else
        CONFIG_NAMES="${CONFIG_NAMES}, \"${config_name}\""
    fi
done

# Save run metadata
cat > "${OUTPUT_BASE}/metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "seeds": [$(echo ${SEEDS} | sed 's/ /, /g')],
    "wandb_group": "${WANDB_GROUP}",
    "configs": [${CONFIG_NAMES}],
    "ablations": $([ -n "${ABLATIONS}" ] && echo "true" || echo "false")
}
EOF

echo "Metadata saved to ${OUTPUT_BASE}/metadata.json"

# Run all configs for all seeds
RUN_NUM=0
for seed in ${SEEDS}; do
    for config_pair in ${CONFIGS}; do
        config_name=$(echo ${config_pair} | cut -d: -f1)
        model_tag=$(echo ${config_pair} | cut -d: -f2)
        RUN_NUM=$((RUN_NUM + 1))

        echo ""
        echo "========================================"
        echo "Run ${RUN_NUM}/${TOTAL_RUNS}"
        echo "========================================"

        run_config "${config_name}" "${model_tag}" "${seed}"
    done
done

echo ""
echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT_BASE}/"
echo "W&B Group: ${WANDB_GROUP}"
echo "Seeds: ${SEEDS}"
echo "Total runs completed: ${TOTAL_RUNS}"
echo ""
echo "To view in W&B:"
echo "  1. Go to https://wandb.ai"
echo "  2. Filter by group: ${WANDB_GROUP}"
echo ""
echo "To generate analysis plots, run:"
echo "  python scripts/gpt2/analyze_rgsa_compare.py --group ${WANDB_GROUP}"
echo "========================================"
