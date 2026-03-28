#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PY=${PYTHON:-python3}
RUNNER=fim/reciprocal_attention/llama150m_matched.py
CFG_DIR=fim/reciprocal_attention/configs
TORCHRUN=${TORCHRUN:-torchrun}
NPROC=${NPROC_PER_NODE:-4}

# wandb guardrails: honour WANDB_MODE / WANDB_DISABLED env vars so
# local reruns with production configs do not accidentally log online.
if [ "${WANDB_DISABLED:-}" = "1" ] || [ "${WANDB_DISABLED:-}" = "true" ]; then
  export WANDB_MODE=disabled
fi
if [ -n "${WANDB_MODE:-}" ]; then
  export WANDB_MODE
fi

if command -v "$TORCHRUN" >/dev/null 2>&1; then
  TORCHRUN_CMD=("$TORCHRUN")
else
  TORCHRUN_CMD=("$PY" -m torch.distributed.run)
fi

mode=${1:-help}

run_single() {
  local cfg=$1
  "$PY" "$RUNNER" --config "$cfg"
}

run_ddp() {
  local cfg=$1
  "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node "$NPROC" "$RUNNER" --config "$cfg"
}

derive_top4() {
  local input=${1:-configs/ra_surgical_llama150m.json}
  local output=${2:-configs/ra_surgical_llama150m_top4.json}
  "$PY" "$RUNNER" --derive-topk 4 \
    --input-selection "$input" \
    --output "$output"
}

case "$mode" in
  baseline)
    run_single "$CFG_DIR/llama150m_baseline_smoke.json"
    ;;
  fim)
    run_single "$CFG_DIR/llama150m_fim_smoke.json"
    ;;
  ra8)
    run_single "$CFG_DIR/llama150m_ra_surgical8_smoke.json"
    ;;
  ra4)
    run_single "$CFG_DIR/llama150m_ra_surgical4_smoke.json"
    ;;
  ddp-baseline)
    run_ddp "$CFG_DIR/llama150m_baseline_smoke.json"
    ;;
  ddp-ra8)
    run_ddp "$CFG_DIR/llama150m_ra_surgical8_smoke.json"
    ;;
  ddp-ra4)
    run_ddp "$CFG_DIR/llama150m_ra_surgical4_smoke.json"
    ;;
  ddp-fim)
    run_ddp "$CFG_DIR/llama150m_fim_smoke.json"
    ;;
  target-smoke-baseline)
    run_ddp "$CFG_DIR/llama150m_baseline_b200x4_smoke.json"
    ;;
  target-smoke-fim)
    run_ddp "$CFG_DIR/llama150m_fim_collection_b200x4_smoke.json"
    ;;
  target-smoke-ra8)
    run_ddp "$CFG_DIR/llama150m_ra_surgical8_b200x4_smoke.json"
    ;;
  target-smoke-ra4)
    run_ddp "$CFG_DIR/llama150m_ra_surgical4_b200x4_smoke.json"
    ;;
  target-smoke-all)
    run_ddp "$CFG_DIR/llama150m_baseline_b200x4_smoke.json"
    run_ddp "$CFG_DIR/llama150m_fim_collection_b200x4_smoke.json"
    run_ddp "$CFG_DIR/llama150m_ra_surgical8_b200x4_smoke.json"
    run_ddp "$CFG_DIR/llama150m_ra_surgical4_b200x4_smoke.json"
    ;;
  full-baseline)
    run_ddp "$CFG_DIR/llama150m_baseline_b200x4.json"
    ;;
  full-fim)
    run_ddp "$CFG_DIR/llama150m_fim_collection_b200x4.json"
    ;;
  full-ra8)
    run_ddp "$CFG_DIR/llama150m_ra_surgical8_b200x4.json"
    ;;
  full-ra4)
    run_ddp "$CFG_DIR/llama150m_ra_surgical4_b200x4.json"
    ;;
  derive-top4)
    derive_top4 "${2:-}" "${3:-}"
    ;;
  full-sequence)
    # Intended pipeline: FIM collection -> top4 derivation -> baseline -> RA8
    echo "=== Phase 1/4: FIM collection ==="
    run_ddp "$CFG_DIR/llama150m_fim_collection_b200x4.json"
    echo "=== Phase 2/4: derive top-4 selection ==="
    derive_top4
    echo "=== Phase 3/4: baseline (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama150m_baseline_b200x4.json"
    echo "=== Phase 4/4: RA-8 surgical (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama150m_ra_surgical8_b200x4.json"
    echo "=== full-sequence complete ==="
    ;;
  full-sequence-all)
    # Full pipeline including optional RA-4 variant
    echo "=== Phase 1/5: FIM collection ==="
    run_ddp "$CFG_DIR/llama150m_fim_collection_b200x4.json"
    echo "=== Phase 2/5: derive top-4 selection ==="
    derive_top4
    echo "=== Phase 3/5: baseline (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama150m_baseline_b200x4.json"
    echo "=== Phase 4/5: RA-8 surgical (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama150m_ra_surgical8_b200x4.json"
    echo "=== Phase 5/5: RA-4 surgical (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama150m_ra_surgical4_b200x4.json"
    echo "=== full-sequence-all complete ==="
    ;;
  all)
    run_single "$CFG_DIR/llama150m_baseline_smoke.json"
    run_single "$CFG_DIR/llama150m_fim_smoke.json"
    run_single "$CFG_DIR/llama150m_ra_surgical8_smoke.json"
    ;;
  *)
    cat <<'EOF'
Usage: scripts/run_llama150m_matched.sh <mode>

Smoke tests (single-process, CPU-safe):
  baseline          single-process baseline smoke
  fim               single-process FIM collection smoke
  ra8               single-process RA-8 smoke
  ra4               single-process RA-4 smoke
  all               run baseline + fim + ra8 smoke sequentially

DDP smoke (multi-GPU or CPU/gloo):
  ddp-baseline      DDP baseline smoke
  ddp-fim           DDP FIM collection smoke
  ddp-ra8           DDP RA-8 smoke
  ddp-ra4           DDP RA-4 smoke
  target-smoke-baseline  target-shape bf16 baseline smoke (seq_len=1024)
  target-smoke-fim       target-shape bf16 FIM collection smoke (seq_len=1024)
  target-smoke-ra8       target-shape bf16 RA-8 smoke (seq_len=1024)
  target-smoke-ra4       target-shape bf16 RA-4 smoke (seq_len=1024)
  target-smoke-all       baseline -> FIM -> RA-8 -> RA-4 target-shape smoke

Full production runs (DDP, wall-clock matched):
  full-baseline     1-hr baseline
  full-fim          FIM collection (~15 min)
  full-ra8          1-hr RA-8 surgical
  full-ra4          1-hr RA-4 surgical
  derive-top4 [in] [out]   trim 8-head selection to top-4

Full pipeline:
  full-sequence     FIM -> top4 -> baseline -> RA8
  full-sequence-all FIM -> top4 -> baseline -> RA8 -> RA4

Environment:
  PYTHON=python3           python interpreter
  TORCHRUN=torchrun        torchrun binary
  NPROC_PER_NODE=4         GPUs per node for DDP modes
  WANDB_MODE=offline       force wandb mode (offline/disabled/online)
  WANDB_DISABLED=1         disable wandb entirely
EOF
    ;;
esac
