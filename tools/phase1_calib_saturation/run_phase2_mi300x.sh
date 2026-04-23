#!/bin/bash
# Phase 2 MI300X orchestrator.
set -e
set -x
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FP4BMM=0
export HF_HOME=/root/.cache/huggingface

DONE=/workspace/results/.done
mkdir -p $DONE /workspace/results

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "microsoft/phi-4"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)
CONFIGS="bf16 fp8_uncalib ptpc_fp8"

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

for M in "${MODELS[@]}"; do
  slug="${M//\//__}"
  for cfg in $CONFIGS; do
    run_once "sat_${slug}_${cfg}" \
      python3 /workspace/phase2/saturation_sweep_mi300x.py \
        --model "$M" --config "$cfg"
  done
done

echo "ALL PHASE 2 STEPS DONE"
