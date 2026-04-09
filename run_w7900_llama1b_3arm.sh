#!/usr/bin/env bash
# Three-arm RA ablation on W7900: Llama 1B baseline, arm_a (FIM trace), arm_c (V2 JSD).
# Each arm capped at 3600s wall (max_time in config), runs sequentially on cuda:0.
source /home/mcgrof/envs/w7900-ml/bin/activate
cd /data/knlp
export CUDA_VISIBLE_DEVICES=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out/llama1b-matched-w7900
LOG=out/llama1b-matched-w7900/run_3arm.log
date -u +%Y-%m-%dT%H:%M:%SZ >> "$LOG"
echo "[llama1b-3arm] start" >> "$LOG"

RUNNER=fim/reciprocal_attention/llama150m_matched.py
CFG_DIR=fim/reciprocal_attention/configs

run_arm() {
  local cfg=$1
  local rn=$2
  echo "[llama1b-3arm] === $rn ===" >> "$LOG"
  date -u +%Y-%m-%dT%H:%M:%SZ >> "$LOG"
  python "$RUNNER" \
    --config "$cfg" \
    --override "run_name=\"$rn\"" \
    --override "tracking.out_dir=\"out/llama1b-matched-w7900\"" \
    --override "tracking.wandb=false" \
    --override "training.batch_size=2" \
    --override "training.gradient_accumulation_steps=64" \
    >> "$LOG" 2>&1
  echo "[llama1b-3arm] done $rn (exit $?)" >> "$LOG"
}

# Baseline already completed in previous run — skip
# run_arm "$CFG_DIR/llama1b_baseline_1xh100.json"         "llama1b-baseline-w7900"

# Fix: override selection_file with correct path from repo_root
run_arm_with_sel() {
  local cfg=$1
  local rn=$2
  local sel=$3
  echo "[llama1b-3arm] === $rn ===" >> "$LOG"
  date -u +%Y-%m-%dT%H:%M:%SZ >> "$LOG"
  python "$RUNNER" \
    --config "$cfg" \
    --override "run_name=\"$rn\"" \
    --override "tracking.out_dir=\"out/llama1b-matched-w7900\"" \
    --override "tracking.wandb=false" \
    --override "training.batch_size=2" \
    --override "training.gradient_accumulation_steps=64" \
    --override "ra.selection_file=\"$sel\"" \
    >> "$LOG" 2>&1
  echo "[llama1b-3arm] done $rn (exit $?)" >> "$LOG"
}

run_arm_with_sel "$CFG_DIR/llama1b_ra_arm_a_fimtrace_1xh100.json" \
  "llama1b-arm-a-fimtrace-w7900-v2" \
  "fim/reciprocal_attention/configs/ra_ablation_llama1b_arm_a_fimtrace.json"

run_arm_with_sel "$CFG_DIR/llama1b_arm_c_jsd_w7900.json" \
  "llama1b-arm-c-jsd-w7900-v2" \
  "fim/reciprocal_attention/configs/ra_ablation_llama1b_arm_c_jsd.json"

echo "[llama1b-3arm] all done" >> "$LOG"
date -u +%Y-%m-%dT%H:%M:%SZ >> "$LOG"
