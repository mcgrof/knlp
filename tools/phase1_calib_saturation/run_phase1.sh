#!/bin/bash
# Phase 1 orchestrator — H100 pod.
#
# IMPORTANT: loops over configs externally (one vLLM process per config)
# because vLLM v1's async engine-core does not reliably release GPU
# memory on del/shutdown within a single Python process.
#
# .done markers under /workspace/results/.done/ make it idempotent.

set -e
set -x
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=/root/.cache/huggingface

DONE=/workspace/results/.done
mkdir -p $DONE /workspace/results

CONFIGS="fp16 fp8_uncalib fp8_calib asym_uncalib asym_calib"
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "microsoft/phi-4"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

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

# ---- Qwen sanity (fast validation) ----
run_once "qwen_sanity" \
  python3.12 /workspace/phase1/qwen_sanity.py

# ---- Llama saturation sweep, one config per process ----
# Fresh file if tag not set
LLAMA_OUT=/workspace/results/saturation_meta-llama__Llama-3.1-8B-Instruct.jsonl
for cfg in $CONFIGS; do
  run_once "sat_llama_${cfg}" \
    python3.12 /workspace/phase1/saturation_sweep.py \
      --model meta-llama/Llama-3.1-8B-Instruct --config "$cfg"
done

# ---- Llama calibration head-to-head ----
# Kept as-is because calibration_head_to_head.py already shells out
# to lm-eval per config (subprocess isolation by design)
run_once "h2h_llama" \
  python3.12 /workspace/phase1/calibration_head_to_head.py

# ---- Remaining models: saturation only ----
for M in "${MODELS[@]:1}"; do
  slug="${M//\//__}"
  for cfg in $CONFIGS; do
    run_once "sat_${slug}_${cfg}" \
      python3.12 /workspace/phase1/saturation_sweep.py \
        --model "$M" --config "$cfg"
  done
done

echo "ALL PHASE 1 STEPS DONE"
