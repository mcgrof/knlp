#!/usr/bin/env bash
set -euo pipefail
#
# GPT-2 paper ablation on 1xH100 using the matched reproducer lane.
#
# This script reflects the executed workflow:
#   1. smoke all three arms
#   2. run the clean seed-1337 three-arm comparison
#   3. if Arm A vs Arm B is still too close, run two extra A/B seeds
#
# Arms:
#   baseline   — no RA
#   arm_a      — RA with layer_selector=fim_trace
#   arm_b      — RA with layer_selector=attn_layer_eigmax
#
# No sdpa_gate. No gpt2-ra-sdpa-ablation.
#

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PY=${PYTHON:-python3}
RUNNER=fim/reciprocal_attention/gpt2_matched.py
CFG_DIR=fim/reciprocal_attention/configs

run_single() {
  local cfg=$1
  shift
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running: $cfg $*"
  "$PY" "$RUNNER" --config "$cfg" "$@"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done: $cfg"
}

phase=${1:-help}

case "$phase" in
  smoke)
    run_single "$CFG_DIR/gpt2_baseline_1xh100_smoke.json"
    run_single "$CFG_DIR/gpt2_arm_a_fimtrace_1xh100_smoke.json"
    run_single "$CFG_DIR/gpt2_arm_b_eigmax_1xh100_smoke.json"
    ;;

  seed1337)
    run_single "$CFG_DIR/gpt2_baseline_1xh100.json"
    run_single "$CFG_DIR/gpt2_arm_a_fimtrace_1xh100.json"
    run_single "$CFG_DIR/gpt2_arm_b_eigmax_1xh100.json"
    ;;

  extra-ab)
    run_single "$CFG_DIR/gpt2_arm_a_fimtrace_1xh100.json" \
      --override 'seed=42' \
      --override 'run_name="gpt2-arm-a-fimtrace-1xh100-s42"'
    run_single "$CFG_DIR/gpt2_arm_b_eigmax_1xh100.json" \
      --override 'seed=42' \
      --override 'run_name="gpt2-arm-b-eigmax-1xh100-s42"'
    run_single "$CFG_DIR/gpt2_arm_a_fimtrace_1xh100.json" \
      --override 'seed=7' \
      --override 'run_name="gpt2-arm-a-fimtrace-1xh100-s7"'
    run_single "$CFG_DIR/gpt2_arm_b_eigmax_1xh100.json" \
      --override 'seed=7' \
      --override 'run_name="gpt2-arm-b-eigmax-1xh100-s7"'
    ;;

  full)
    "$0" smoke
    "$0" seed1337
    "$0" extra-ab
    ;;

  *)
    cat <<EOF
Usage: scripts/run_gpt2_paper_ablation.sh <phase>

Phases:
  smoke      smoke all three arms
  seed1337   clean seed-1337 three-arm run
  extra-ab   two extra seeds for Arm A and Arm B only
  full       smoke + seed1337 + extra-ab
EOF
    ;;
esac
