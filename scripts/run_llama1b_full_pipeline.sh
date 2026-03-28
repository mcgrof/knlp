#!/usr/bin/env bash
set -euo pipefail
#
# Full LLaMA-1B matched RA pipeline for 4xH100.
#
# Runs the complete experiment:
#   1. FIM collection (~15 min)
#   2. Derive top-4 selection
#   3. Baseline training (1 hr wall-clock)
#   4. RA-8 training (1 hr wall-clock)
#   5. HellaSwag + LAMBADA eval on both checkpoints
#
# Usage:
#   bash scripts/run_llama1b_full_pipeline.sh [NPROC]
#
# Results land in out/llama1b-matched/.
#

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

NPROC=${1:-${NPROC_PER_NODE:-4}}
export NPROC_PER_NODE=$NPROC
export WANDB_MODE=${WANDB_MODE:-online}

echo "=== LLaMA-1B Full Pipeline ==="
echo "Date: $(date -u)"
echo "GPUs: $NPROC"
echo "WANDB_MODE: $WANDB_MODE"
echo ""

RUNNER=scripts/run_llama1b_matched.sh
OUT_DIR=out/llama1b-matched
START_TIME=$(date +%s)

# Phase 1: FIM collection
echo "=== Phase 1/6: FIM collection ==="
bash $RUNNER full-fim
echo "FIM collection done at $(date -u)"

# Phase 2: Derive top-4 selection
echo "=== Phase 2/6: derive top-4 selection ==="
bash $RUNNER derive-top4
echo "Top-4 derivation done"

# Phase 3: Baseline training (1 hr)
echo "=== Phase 3/6: baseline training (1 hr) ==="
bash $RUNNER full-baseline
echo "Baseline done at $(date -u)"

# Phase 4: RA-8 surgical training (1 hr)
echo "=== Phase 4/6: RA-8 surgical training (1 hr) ==="
bash $RUNNER full-ra8
echo "RA-8 done at $(date -u)"

# Phase 5: Downstream evaluation
echo "=== Phase 5/6: eval baseline ==="
bash $RUNNER eval-baseline
echo "Baseline eval done at $(date -u)"

echo "=== Phase 6/6: eval RA-8 ==="
bash $RUNNER eval-ra8
echo "RA-8 eval done at $(date -u)"

END_TIME=$(date +%s)
TOTAL_SECS=$((END_TIME - START_TIME))

echo ""
echo "=== PIPELINE COMPLETE ==="
echo "Total wall-clock: ${TOTAL_SECS}s (~$((TOTAL_SECS / 60))m)"
echo "Results directory: $OUT_DIR"
echo ""

# Print summary
echo "=== Quick Summary ==="
for f in "$OUT_DIR"/llama1b-*.eval_results.json; do
  if [ -f "$f" ]; then
    echo "--- $(basename "$f") ---"
    python3 -c "
import json, sys
r = json.load(open('$f'))
print(f'  Steps: {r.get(\"completed_steps\", \"?\")}')
for task, res in r.get('tasks', {}).items():
    print(f'  {task}: accuracy={res.get(\"accuracy\", \"?\")}')
"
  fi
done

echo ""
echo "Backend parity files:"
ls -la "$OUT_DIR"/*.backend.json 2>/dev/null || echo "  (none found)"
echo ""
echo "Checkpoints:"
ls -lh "$OUT_DIR"/*.checkpoint.pt 2>/dev/null || echo "  (none found)"
