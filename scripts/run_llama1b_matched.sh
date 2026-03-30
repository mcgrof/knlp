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


run_ddp_override() {
  local cfg=$1
  shift
  "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node "$NPROC" "$RUNNER" --config "$cfg" "$@"
}

derive_top4() {
  local input=${1:-configs/ra_surgical_llama1b.json}
  local output=${2:-configs/ra_surgical_llama1b_top4.json}
  "$PY" "$RUNNER" --derive-topk 4 \
    --input-selection "$input" \
    --output "$output"
}

EVAL_TASKS=${LLAMA1B_EVAL_TASKS:-hellaswag,winogrande}
EVAL_MAX_EXAMPLES=${LLAMA1B_EVAL_MAX_EXAMPLES:-1000}
EVAL_SMOKE_MAX_EXAMPLES=${LLAMA1B_EVAL_SMOKE_MAX_EXAMPLES:-32}

run_eval() {
  local ckpt=$1
  local out_dir=${2:-out/llama1b-matched}
  local max_examples=${3:-$EVAL_MAX_EXAMPLES}
  "$PY" "$RUNNER" --eval-checkpoint "$ckpt" \
    --eval-tasks "$EVAL_TASKS" \
    --eval-max-examples "$max_examples" \
    --output "$out_dir"
}

case "$mode" in
  baseline)
    run_single "$CFG_DIR/llama1b_baseline_smoke.json"
    ;;
  fim)
    run_single "$CFG_DIR/llama1b_fim_smoke.json"
    ;;
  ra8)
    run_single "$CFG_DIR/llama1b_ra_surgical8_smoke.json"
    ;;
  ra4)
    run_single "$CFG_DIR/llama1b_ra_surgical4_smoke.json"
    ;;
  ddp-baseline)
    run_ddp "$CFG_DIR/llama1b_baseline_smoke.json"
    ;;
  ddp-ra8)
    run_ddp "$CFG_DIR/llama1b_ra_surgical8_smoke.json"
    ;;
  ddp-ra4)
    run_ddp "$CFG_DIR/llama1b_ra_surgical4_smoke.json"
    ;;
  ddp-fim)
    run_ddp "$CFG_DIR/llama1b_fim_smoke.json"
    ;;
  target-smoke-baseline)
    run_ddp "$CFG_DIR/llama1b_baseline_4xh100_smoke.json"
    ;;
  target-smoke-fim)
    run_ddp "$CFG_DIR/llama1b_fim_collection_4xh100_smoke.json"
    ;;
  target-smoke-ra8)
    run_ddp "$CFG_DIR/llama1b_ra_surgical8_4xh100_smoke.json"
    ;;
  target-smoke-ra4)
    run_ddp "$CFG_DIR/llama1b_ra_surgical4_4xh100_smoke.json"
    ;;
  target-smoke-ra28)
    run_ddp "$CFG_DIR/llama1b_ra_surgical28_4xh100_smoke.json"
    ;;
  target-smoke-all)
    run_ddp "$CFG_DIR/llama1b_baseline_4xh100_smoke.json"
    run_ddp "$CFG_DIR/llama1b_fim_collection_4xh100_smoke.json"
    run_ddp "$CFG_DIR/llama1b_ra_surgical4_4xh100_smoke.json"
    run_ddp "$CFG_DIR/llama1b_ra_surgical8_4xh100_smoke.json"
    ;;
  full-baseline)
    run_ddp "$CFG_DIR/llama1b_baseline_4xh100.json"
    ;;
  full-fim)
    run_ddp "$CFG_DIR/llama1b_fim_collection_4xh100.json"
    ;;
  full-ra8)
    run_ddp "$CFG_DIR/llama1b_ra_surgical8_4xh100.json"
    ;;
  full-ra4)
    run_ddp "$CFG_DIR/llama1b_ra_surgical4_4xh100.json"
    ;;
  full-ra28)
    run_ddp "$CFG_DIR/llama1b_ra_surgical28_4xh100.json"
    ;;
  derive-top4)
    derive_top4 "${2:-}" "${3:-}"
    ;;
  eval-smoke-baseline)
    run_eval "out/llama1b-matched/llama1b-baseline-4xh100-smoke.checkpoint.pt" "${2:-out/llama1b-matched}" "$EVAL_SMOKE_MAX_EXAMPLES"
    ;;
  eval-baseline)
    run_eval "out/llama1b-matched/llama1b-baseline-4xh100-1hr.checkpoint.pt" "${2:-out/llama1b-matched}"
    ;;
  eval-ra8)
    run_eval "out/llama1b-matched/llama1b-ra-surgical8-4xh100-1hr.checkpoint.pt" "${2:-out/llama1b-matched}"
    ;;
  eval-ra4)
    run_eval "out/llama1b-matched/llama1b-ra-surgical4-4xh100-1hr.checkpoint.pt" "${2:-out/llama1b-matched}"
    ;;
  eval-ra28)
    run_eval "out/llama1b-matched/llama1b-ra-surgical28-4xh100-1hr.checkpoint.pt" "${2:-out/llama1b-matched}"
    ;;
  eval-all)
    echo "=== Eval: baseline ==="
    run_eval "out/llama1b-matched/llama1b-baseline-4xh100-1hr.checkpoint.pt" "out/llama1b-matched"
    echo "=== Eval: RA-8 ==="
    run_eval "out/llama1b-matched/llama1b-ra-surgical8-4xh100-1hr.checkpoint.pt" "out/llama1b-matched"
    echo "=== Eval complete ==="
    ;;
  readiness-check)
    # Quick local validation of the entire chain.
    # Runs CPU smokes (baseline, FIM, RA-8, RA-4) sequentially,
    # then verifies the eval codepath by checking that the harness
    # accepts --eval-checkpoint with a nonexistent path gracefully.
    echo "=== Readiness check: CPU smokes ==="
    echo "--- baseline ---"
    run_single "$CFG_DIR/llama1b_baseline_smoke.json"
    echo "--- FIM ---"
    run_single "$CFG_DIR/llama1b_fim_smoke.json"
    echo "--- RA-8 ---"
    run_single "$CFG_DIR/llama1b_ra_surgical8_smoke.json"
    echo "--- RA-4 ---"
    run_single "$CFG_DIR/llama1b_ra_surgical4_smoke.json"
    echo "--- derive-top4 ---"
    derive_top4
    echo "--- eval smoke (baseline, ${EVAL_SMOKE_MAX_EXAMPLES} examples) ---"
    run_eval "out/llama1b-matched/llama1b-baseline-smoke.checkpoint.pt" \
      "out/llama1b-matched" "$EVAL_SMOKE_MAX_EXAMPLES"
    echo "=== All readiness checks PASSED ==="
    ;;
  full-sequence)
    # Pipeline: FIM -> baseline -> RA-8
    echo "=== Phase 1/3: FIM collection ==="
    run_ddp "$CFG_DIR/llama1b_fim_collection_4xh100.json"
    echo "=== Phase 2/3: baseline (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama1b_baseline_4xh100.json"
    echo "=== Phase 3/3: RA-8 surgical (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama1b_ra_surgical8_4xh100.json"
    echo "=== full-sequence complete ==="
    ;;
  full-baseline-seed)
    # Usage: full-baseline-seed <seed>
    seed=${2:?seed required}
    run_ddp_override "$CFG_DIR/llama1b_baseline_4xh100.json" \
      --override "seed=$seed" "run_name=llama1b-baseline-4xh100-1hr-s${seed}"
    ;;
  full-ra28-seed)
    # Usage: full-ra28-seed <seed>
    seed=${2:?seed required}
    run_ddp_override "$CFG_DIR/llama1b_ra_surgical28_4xh100.json" \
      --override "seed=$seed" "run_name=llama1b-ra-surgical28-4xh100-1hr-s${seed}"
    ;;
  eval-baseline-seed)
    # Usage: eval-baseline-seed <seed>
    seed=${2:?seed required}
    run_eval "out/llama1b-matched/llama1b-baseline-4xh100-1hr-s${seed}.checkpoint.pt" "out/llama1b-matched"
    ;;
  eval-ra28-seed)
    # Usage: eval-ra28-seed <seed>
    seed=${2:?seed required}
    run_eval "out/llama1b-matched/llama1b-ra-surgical28-4xh100-1hr-s${seed}.checkpoint.pt" "out/llama1b-matched"
    ;;
  full-sequence-eval)
    # Full pipeline including eval
    echo "=== Phase 1/5: FIM collection ==="
    run_ddp "$CFG_DIR/llama1b_fim_collection_4xh100.json"
    echo "=== Phase 2/5: baseline (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama1b_baseline_4xh100.json"
    echo "=== Phase 3/5: RA-8 surgical (1 hr wall-clock) ==="
    run_ddp "$CFG_DIR/llama1b_ra_surgical8_4xh100.json"
    echo "=== Phase 4/5: eval baseline ==="
    run_eval "out/llama1b-matched/llama1b-baseline-4xh100-1hr.checkpoint.pt" "out/llama1b-matched"
    echo "=== Phase 5/5: eval RA-8 ==="
    run_eval "out/llama1b-matched/llama1b-ra-surgical8-4xh100-1hr.checkpoint.pt" "out/llama1b-matched"
    echo "=== full-sequence-eval complete ==="
    ;;
  *)
    cat <<'EOF'
Usage: scripts/run_llama1b_matched.sh <mode>

LLaMA-1B (1.175B params, TinyLlama architecture) matched RA lane.
  Model: h=2048, L=22, H=32, KV=4, I=5632, V=50304
  Training: bs=4, ga=8, 4xGPU => effective batch=128
  Hardware: 4xH100 80GB

Smoke tests (single-process, CPU-safe):
  baseline          single-process baseline smoke
  fim               single-process FIM collection smoke
  ra8               single-process RA-8 smoke
  ra4               single-process RA-4 smoke

DDP smoke (multi-GPU or CPU/gloo):
  ddp-baseline      DDP baseline smoke
  ddp-fim           DDP FIM collection smoke
  ddp-ra8           DDP RA-8 smoke
  ddp-ra4           DDP RA-4 smoke
  target-smoke-baseline  target-shape bf16 baseline smoke (seq_len=1024)
  target-smoke-fim       target-shape bf16 FIM collection smoke (seq_len=1024)
  target-smoke-ra8       target-shape bf16 RA-8 smoke (seq_len=1024)
  target-smoke-ra4       target-shape bf16 RA-4 smoke (seq_len=1024)
  target-smoke-all       all target-shape smokes sequentially

Full production runs (DDP, 4xH100, wall-clock matched):
  full-baseline     1-hr baseline
  full-fim          FIM collection (~15 min)
  full-ra8          1-hr RA-8 surgical
  full-ra28         1-hr RA-28 surgical (promoted headline arm)
  full-ra4          1-hr RA-4 surgical (negative-control trim)
  derive-top4 [in] [out]   trim 8-head selection to top-4

Seeded runs (for multi-seed campaigns):
  full-baseline-seed <seed>   1-hr baseline with explicit seed
  full-ra28-seed <seed>       1-hr RA-28 with explicit seed
  eval-baseline-seed <seed>   eval baseline checkpoint for given seed
  eval-ra28-seed <seed>       eval RA-28 checkpoint for given seed

Downstream evaluation (default: hellaswag,winogrande):
  eval-smoke-baseline  quick eval (32 examples) on baseline smoke ckpt
  eval-baseline     eval on baseline checkpoint
  eval-ra8          eval on RA-8 checkpoint
  eval-ra28         eval on RA-28 checkpoint
  eval-ra4          eval on RA-4 checkpoint
  eval-all          eval baseline + RA-8 (default headline comparison)

Readiness:
  readiness-check     CPU smokes (baseline+FIM+RA8+RA4) + derive-top4

Full pipeline (default: baseline vs RA-8):
  full-sequence       FIM -> baseline -> RA-8
  full-sequence-eval  FIM -> baseline -> RA-8 -> eval both

Environment:
  PYTHON=python3           python interpreter
  TORCHRUN=torchrun        torchrun binary
  NPROC_PER_NODE=4         GPUs per node for DDP modes
  WANDB_MODE=offline       force wandb mode (offline/disabled/online)
  WANDB_DISABLED=1         disable wandb entirely
  LLAMA1B_EVAL_TASKS=hellaswag,winogrande  eval task list
  LLAMA1B_EVAL_MAX_EXAMPLES=1000           examples per eval task
  LLAMA1B_EVAL_SMOKE_MAX_EXAMPLES=32       examples for smoke eval
EOF
    ;;
esac
