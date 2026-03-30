#!/usr/bin/env bash
set -euo pipefail

# Marin 32B RA head-count scaling screen
# Hardware: 4x H100 80GB, FSDP full-shard
# Sweep: baseline, RA-20, RA-24, RA-28, RA-32 @ 20 min each

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PY=${PYTHON:-python3}
RUNNER=fim/reciprocal_attention/llama150m_matched.py
CFG_DIR=fim/reciprocal_attention/configs
TORCHRUN=${TORCHRUN:-torchrun}
NPROC=${NPROC_PER_NODE:-4}

RESULT_ROOT=${RESULT_ROOT:-out/marin32b-headscale}
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

export WANDB_DISABLED=1

if command -v "$TORCHRUN" >/dev/null 2>&1; then
  TORCHRUN_CMD=("$TORCHRUN")
else
  TORCHRUN_CMD=("$PY" -m torch.distributed.run)
fi

run_variant() {
  local cfg=$1
  local label=$2
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [variant-start] $label cfg=$cfg"
  "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node "$NPROC" "$RUNNER" --config "$cfg"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [variant-done] $label"
}

write_partial_results() {
  local label=$1
  local summary_file="$RESULT_ROOT/partial_summary.jsonl"
  # Find the latest eval log line from the jsonl
  local jsonl_file
  jsonl_file=$(ls -t "$RESULT_ROOT"/marin32b-*.jsonl 2>/dev/null | head -1)
  if [ -n "$jsonl_file" ]; then
    local last_eval
    last_eval=$(grep '"event": "eval"' "$jsonl_file" | tail -1 || true)
    if [ -n "$last_eval" ]; then
      echo "{\"variant\": \"$label\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"last_eval\": $last_eval}" >> "$summary_file"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [partial-result] $label written to $summary_file"
    fi
  fi
  # Also grab the complete event from jsonl
  if [ -n "$jsonl_file" ]; then
    local complete
    complete=$(grep '"event": "complete"' "$jsonl_file" | tail -1 || true)
    if [ -n "$complete" ]; then
      echo "{\"variant\": \"$label\", \"complete\": $complete}" >> "$summary_file"
    fi
  fi
}

mode=${1:-sweep}

case "$mode" in
  baseline)
    run_variant "$CFG_DIR/marin32b_baseline_4xh100.json" "baseline"
    write_partial_results "baseline"
    ;;
  ra20)
    run_variant "$CFG_DIR/marin32b_ra20_4xh100.json" "RA-20"
    write_partial_results "RA-20"
    ;;
  ra24)
    run_variant "$CFG_DIR/marin32b_ra24_4xh100.json" "RA-24"
    write_partial_results "RA-24"
    ;;
  ra28)
    run_variant "$CFG_DIR/marin32b_ra28_4xh100.json" "RA-28"
    write_partial_results "RA-28"
    ;;
  ra32)
    run_variant "$CFG_DIR/marin32b_ra32_4xh100.json" "RA-32"
    write_partial_results "RA-32"
    ;;
  sweep)
    mkdir -p "$RESULT_ROOT"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [sweep-start] Marin 32B head-count screen: baseline RA-20 RA-24 RA-28 RA-32"

    echo "=== Variant 1/5: baseline ==="
    run_variant "$CFG_DIR/marin32b_baseline_4xh100.json" "baseline"
    write_partial_results "baseline"

    echo "=== Variant 2/5: RA-20 ==="
    run_variant "$CFG_DIR/marin32b_ra20_4xh100.json" "RA-20"
    write_partial_results "RA-20"

    echo "=== Variant 3/5: RA-24 ==="
    run_variant "$CFG_DIR/marin32b_ra24_4xh100.json" "RA-24"
    write_partial_results "RA-24"

    echo "=== Variant 4/5: RA-28 ==="
    run_variant "$CFG_DIR/marin32b_ra28_4xh100.json" "RA-28"
    write_partial_results "RA-28"

    echo "=== Variant 5/5: RA-32 ==="
    run_variant "$CFG_DIR/marin32b_ra32_4xh100.json" "RA-32"
    write_partial_results "RA-32"

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [sweep-done] All 5 variants complete."

    # Write final summary
    "$PY" -c "
import json, glob, math
from pathlib import Path

result_root = Path('$RESULT_ROOT')
summary = {'sweep': 'marin32b-ra-headscale', 'timestamp': '$TIMESTAMP', 'variants': {}}

for jsonl_path in sorted(result_root.glob('*.jsonl')):
    name = jsonl_path.stem
    last_eval = None
    complete = None
    for line in jsonl_path.read_text().strip().split('\n'):
        try:
            d = json.loads(line)
            if d.get('event') == 'eval':
                last_eval = d
            elif d.get('event') == 'complete':
                complete = d
        except json.JSONDecodeError:
            pass
    if last_eval:
        summary['variants'][name] = {
            'final_ppl': last_eval.get('perplexity'),
            'final_eval_loss': last_eval.get('eval_loss'),
            'completed_steps': complete.get('completed_steps') if complete else None,
            'elapsed_s': complete.get('stop_elapsed_s') if complete else None,
        }

(result_root / 'sweep_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
"
    ;;
  *)
    cat <<'EOF'
Usage: scripts/run_marin32b_headscale.sh <mode>

Marin-32B RA head-count scaling screen.
  Model: h=5120, L=64, H=40, KV=8, I=27648 (~32B params)
  Training: bs=1, ga=16, 4xGPU FSDP => effective batch=64
  Hardware: 4x H100 80GB
  Duration: 20 min per variant

Single variants:
  baseline    20-min baseline (no RA)
  ra20        20-min RA with 20 heads
  ra24        20-min RA with 24 heads
  ra28        20-min RA with 28 heads
  ra32        20-min RA with 32 heads

Full sweep:
  sweep       Run all 5 variants sequentially (~100 min total)

Environment:
  PYTHON=python3           python interpreter
  TORCHRUN=torchrun        torchrun binary
  NPROC_PER_NODE=4         GPUs per node for FSDP
  RESULT_ROOT=out/marin32b-headscale  output directory
EOF
    ;;
esac
