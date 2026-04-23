#!/bin/bash
# Phase 1 orchestrator — runs on the pod after setup_h100_pod.sh completes.
#
# Run order, tuned to fail fast on the cheap steps and burn most time
# on the saturation sweep:
#   1. Calibration scale collection — all 5 models (~10-20 min total)
#   2. Qwen sanity — one eval, validates calibration pipeline (~5 min)
#   3. Llama saturation sweep — 6 configs x 18 B/T points (~60-90 min)
#   4. Llama calibration head-to-head — 6 configs x 6 evals (~3-5 hr)
#   5. Expanded saturation on Mistral / Phi-4 / DS-R1-Qwen if
#      step 3 succeeded and time remains.
#
# All steps have a `.done` marker under /workspace/results/.done/
# so re-runs are idempotent.

set -e
set -x
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=/root/.cache/huggingface

DONE=/workspace/results/.done
mkdir -p $DONE /workspace/results

run_once() {
  local tag="$1"
  shift
  if [ -f "$DONE/$tag" ]; then
    echo "skip $tag (done)"
    return 0
  fi
  echo "========= $tag ========="
  "$@"
  touch "$DONE/$tag"
}

# ---- 1. Calibration scales ----
for M in \
  meta-llama/Llama-3.1-8B-Instruct \
  mistralai/Mistral-7B-Instruct-v0.3 \
  microsoft/phi-4 \
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  Qwen/Qwen2.5-7B-Instruct
do
  slug="${M//\//__}"
  run_once "calib_${slug}" \
    python3.12 /workspace/phase1/collect_kv_scales.py \
      --model "$M" --n-tokens 2048
done

# ---- 2. Qwen sanity ----
run_once "qwen_sanity" \
  python3.12 /workspace/phase1/qwen_sanity.py

# ---- 3. Llama saturation sweep ----
run_once "sat_llama" \
  python3.12 /workspace/phase1/saturation_sweep.py \
    --model meta-llama/Llama-3.1-8B-Instruct

# ---- 4. Llama calibration head-to-head ----
run_once "h2h_llama" \
  python3.12 /workspace/phase1/calibration_head_to_head.py

# ---- 5. Expand saturation to other models ----
for M in \
  mistralai/Mistral-7B-Instruct-v0.3 \
  microsoft/phi-4 \
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
do
  slug="${M//\//__}"
  run_once "sat_${slug}" \
    python3.12 /workspace/phase1/saturation_sweep.py --model "$M"
done

echo "ALL PHASE 1 STEPS DONE"
