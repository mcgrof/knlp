#!/usr/bin/env bash
set -euo pipefail
#
# Layer-selector ablation: isolate layer selection while holding head selection
# fixed to eigenvalues.
#
# Arms:
#   baseline  — no RA
#   arm_a     — RA with layer_selector=fim_trace, head_selector=max_eigenvalue
#   arm_b     — RA with layer_selector=attn_layer_eigmax, head_selector=max_eigenvalue
#
# Phases:
#   fim-collect       — run FIM collection with eigmax scoring (generates summary)
#   gen-configs       — generate arm A and arm B selection JSONs from FIM summary
#   smoke             — 1 seed per arm, 8 steps, sanity check
#   real              — 3 seeds per arm, 1hr wall-clock each (9 total runs)
#
# Usage:
#   scripts/run_layer_selector_ablation.sh <phase>
#   scripts/run_layer_selector_ablation.sh smoke
#   scripts/run_layer_selector_ablation.sh real
#   scripts/run_layer_selector_ablation.sh full-pipeline
#

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PY=${PYTHON:-python3}
RUNNER=fim/reciprocal_attention/llama150m_matched.py
CFG_DIR=fim/reciprocal_attention/configs
GEN_SCRIPT=fim/reciprocal_attention/gen_layer_ablation_configs.py

FIM_SUMMARY=out/llama1b-matched/fim_summary_eigmax.json
SELECTION_DIR=configs

SEEDS=(1337 42 7)

run_single() {
  local cfg=$1
  shift
  "$PY" "$RUNNER" --config "$cfg" "$@"
}

run_override() {
  local cfg=$1
  shift
  "$PY" "$RUNNER" --config "$cfg" "$@"
}

phase=${1:-help}

case "$phase" in
  fim-collect-smoke)
    echo "=== FIM collection smoke (eigmax, 8 steps) ==="
    run_single "$CFG_DIR/llama1b_fim_collection_1xh100_eigmax_smoke.json"
    echo "=== FIM collection smoke: DONE ==="
    ;;

  fim-collect)
    echo "=== FIM collection (eigmax, ~15 min) ==="
    run_single "$CFG_DIR/llama1b_fim_collection_1xh100_eigmax.json"
    echo "=== FIM collection: DONE ==="
    echo "Summary at: $FIM_SUMMARY"
    ;;

  gen-configs)
    echo "=== Generating ablation selection configs ==="
    if [ ! -f "$FIM_SUMMARY" ]; then
      echo "ERROR: FIM summary not found at $FIM_SUMMARY"
      echo "Run: $0 fim-collect"
      exit 1
    fi
    "$PY" "$GEN_SCRIPT" \
      --fim-summary "$FIM_SUMMARY" \
      --top-heads 28 \
      --top-layers 5 \
      --output-dir "$SELECTION_DIR" \
      --prefix ra_ablation_llama1b
    echo "=== Selection configs generated ==="
    ;;

  smoke)
    echo "=== SMOKE PHASE: 1 seed per arm, 8 steps ==="
    echo ""
    echo "--- smoke: baseline ---"
    run_single "$CFG_DIR/llama1b_baseline_1xh100_smoke.json"
    echo ""
    echo "--- smoke: arm A (fim-trace layers) ---"
    run_single "$CFG_DIR/llama1b_ra_arm_a_fimtrace_1xh100_smoke.json"
    echo ""
    echo "--- smoke: arm B (eigmax layers) ---"
    run_single "$CFG_DIR/llama1b_ra_arm_b_eigmax_1xh100_smoke.json"
    echo ""
    echo "=== SMOKE PHASE: ALL PASS ==="
    ;;

  real)
    echo "=== REAL PHASE: 3 seeds × 3 arms = 9 runs, 1hr each ==="
    echo "Seeds: ${SEEDS[*]}"
    echo ""

    for seed in "${SEEDS[@]}"; do
      echo "--- baseline seed=$seed ---"
      run_override "$CFG_DIR/llama1b_baseline_1xh100.json" \
        --override "seed=$seed" \
        --override "run_name=\"llama1b-baseline-1xh100-1hr-s${seed}\""
    done

    for seed in "${SEEDS[@]}"; do
      echo "--- arm A (fim-trace) seed=$seed ---"
      run_override "$CFG_DIR/llama1b_ra_arm_a_fimtrace_1xh100.json" \
        --override "seed=$seed" \
        --override "run_name=\"llama1b-ra-arm-a-fimtrace-1xh100-1hr-s${seed}\""
    done

    for seed in "${SEEDS[@]}"; do
      echo "--- arm B (eigmax) seed=$seed ---"
      run_override "$CFG_DIR/llama1b_ra_arm_b_eigmax_1xh100.json" \
        --override "seed=$seed" \
        --override "run_name=\"llama1b-ra-arm-b-eigmax-1xh100-1hr-s${seed}\""
    done

    echo ""
    echo "=== REAL PHASE: ALL 9 RUNS COMPLETE ==="
    ;;

  full-pipeline)
    echo "=== FULL PIPELINE ==="
    echo ""
    "$0" fim-collect
    "$0" gen-configs
    "$0" smoke
    echo ""
    echo "Smoke passed. Starting real runs..."
    "$0" real
    echo ""
    echo "=== FULL PIPELINE COMPLETE ==="
    ;;

  *)
    cat <<EOF
Usage: scripts/run_layer_selector_ablation.sh <phase>

Layer-selector ablation (1xH100, LLaMA-1B matched harness)

Phases:
  fim-collect-smoke   FIM collection smoke (8 steps, sanity)
  fim-collect         FIM collection with eigmax scoring (~15 min)
  gen-configs         Generate arm A + arm B selection JSONs from FIM summary
  smoke               Smoke all 3 arms (baseline + arm A + arm B, 8 steps each)
  real                Full 3-seed run (9 runs × 1hr each)
  full-pipeline       fim-collect → gen-configs → smoke → real

Arms:
  baseline   no RA
  arm_a      RA layer_selector=fim_trace, head_selector=max_eigenvalue
  arm_b      RA layer_selector=attn_layer_eigmax, head_selector=max_eigenvalue
EOF
    ;;
esac
