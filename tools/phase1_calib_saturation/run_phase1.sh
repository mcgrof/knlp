#!/bin/bash
# Phase 1 orchestrator — H100 pod.
#
# Runs the experiment sequence in order:
#   1. Qwen sanity — fastest, validates vLLM calibration path works
#   2. Llama saturation sweep — 5 configs x 18 B/T points
#   3. Llama calibration head-to-head — 5 configs x 4 evals
#   4. Mistral / Phi-4 / DS-R1-Qwen saturation sweeps
#
# .done markers under /workspace/results/.done/ make it idempotent.

set -e
set -x
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=/root/.cache/huggingface

DONE=/workspace/results/.done
mkdir -p $DONE /workspace/results

run_once() {
  local tag="$1"; shift
  if [ -f "$DONE/$tag" ]; then
    echo "skip $tag"
    return 0
  fi
  echo "========= $tag ========="
  "$@"
  touch "$DONE/$tag"
}

run_once "qwen_sanity" \
  python3.12 /workspace/phase1/qwen_sanity.py

run_once "sat_llama" \
  python3.12 /workspace/phase1/saturation_sweep.py \
    --model meta-llama/Llama-3.1-8B-Instruct

run_once "h2h_llama" \
  python3.12 /workspace/phase1/calibration_head_to_head.py

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
