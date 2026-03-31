#!/bin/bash
set -euo pipefail

ARTIFACT_ROOT="/data/knlp-key-results/marin-ra-headscale/32b-ra28-defaultalpha-3seed-1hr-20260331T184750Z"
CONFIG_DIR="/data/knlp/fim/reciprocal_attention/configs"
TRAIN_SCRIPT="/data/knlp/fim/reciprocal_attention/llama150m_matched.py"

mkdir -p "$ARTIFACT_ROOT/jsonl" "$ARTIFACT_ROOT/resolved" "$ARTIFACT_ROOT/backend" "$ARTIFACT_ROOT/configs"

# Copy configs to durable sink
for SEED in 7 42 1337; do
    cp "$CONFIG_DIR/marin32b_ra28_1hr_s${SEED}.json" "$ARTIFACT_ROOT/configs/"
done

for SEED in 7 42 1337; do
    CONFIG="$CONFIG_DIR/marin32b_ra28_1hr_s${SEED}.json"
    RUN_NAME="marin32b-ra28-alpha0625-1hr-s${SEED}"
    OUT_DIR="/data/knlp/out/marin32b-ra28-1hr-s${SEED}"

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [RUN-START] seed=$SEED config=$CONFIG"

    cd /data/knlp
    torchrun --nproc_per_node=4 --master_port=29500 \
        "$TRAIN_SCRIPT" --config "$CONFIG"

    EXIT_CODE=$?
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [RUN-FINISH] seed=$SEED exit=$EXIT_CODE"

    # Copy artifacts to durable sink
    if [ -d "$OUT_DIR" ]; then
        cp "$OUT_DIR"/*.jsonl "$ARTIFACT_ROOT/jsonl/" 2>/dev/null || true
        cp "$OUT_DIR"/resolved_config*.json "$ARTIFACT_ROOT/resolved/" 2>/dev/null || true
        cp "$OUT_DIR"/backend*.json "$ARTIFACT_ROOT/backend/" 2>/dev/null || true
    fi

    # Clean up checkpoint bloat from scratch
    rm -rf "$OUT_DIR"/ckpt_* 2>/dev/null || true

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ARTIFACTS-SAVED] seed=$SEED -> $ARTIFACT_ROOT"
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [CAMPAIGN-COMPLETE] All 3 seeds finished."
echo "Artifact root: $ARTIFACT_ROOT"
