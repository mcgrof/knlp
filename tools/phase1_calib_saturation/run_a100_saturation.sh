#!/bin/bash
# A100 80GB saturation orchestrator — skip the 72B PPL run (A100's
# 80GB can't load 72B even at FP8); run only the saturation sweep
# across the 4 tolerant models × 5 configs × 6 batches × 3 context
# lengths for a cross-GPU comparison point on the paper's Hill fits.
set -e
set -x

source /root/.asym_env

DONE=/workspace/results/.done
mkdir -p "$DONE" /workspace/results

run_once() {
  local tag="$1"; shift
  if [ -f "$DONE/$tag" ]; then echo "skip $tag"; return 0; fi
  echo "========= $tag ========="
  "$@"
  touch "$DONE/$tag"
}

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "microsoft/phi-4"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)
CONFIGS="fp16 fp8_uncalib fp8_calib asym_uncalib asym_calib"

for M in "${MODELS[@]}"; do
  slug="${M//\//__}"
  for cfg in $CONFIGS; do
    run_once "sat_${slug}_${cfg}" \
      python3.12 /workspace/saturation_sweep.py --model "$M" --config "$cfg"
  done
done

echo "ALL A100 STEPS DONE"
