#!/bin/bash
# H200 battery orchestrator — runs two experiments:
#
#   Part A: Experiment 3 — Qwen2.5-72B PPL on WikiText-103 at T=2048
#           across 3 configs {fp16, fp8_sym, asym_k16_v8}
#
#   Part B: Saturation sweep — H200 as one more GPU data point for the
#           B x T decode throughput family across 5 configs x 4 models
set -e
set -x

source /root/.asym_env

DONE=/workspace/results/.done
mkdir -p "$DONE" /workspace/results

run_once() {
  local tag="$1"; shift
  if [ -f "$DONE/$tag" ]; then
    echo "skip $tag"; return 0
  fi
  echo "========= $tag ========="
  "$@"
  touch "$DONE/$tag"
}

# -------- Part A: Experiment 3 (Qwen2.5-72B PPL) --------
PPL_OUT=/workspace/results/qwen72b_h200_ppl.jsonl
for cfg in fp16 fp8_sym asym_k16_v8; do
  run_once "exp3_qwen72b_${cfg}" \
    python3.12 /workspace/qwen72b_h200_ppl.py \
      --config "$cfg" --chunk-len 2048 --n-chunks 128 \
      --mem-util 0.92 --out "$PPL_OUT"
done

# -------- Part B: Saturation sweep (same 4 models + 5 configs as H100) --------
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
      python3.12 /workspace/saturation_sweep.py \
        --model "$M" --config "$cfg"
  done
done

echo "ALL H200 STEPS DONE"
