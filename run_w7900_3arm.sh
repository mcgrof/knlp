#!/usr/bin/env bash
# Three-arm RA ablation on W7900: baseline, arm_a (FIM trace), arm_c (V2 JSD).
# Each arm capped at 3600s wall (max_time in config), runs sequentially on cuda:0.
source /home/mcgrof/envs/w7900-ml/bin/activate
set -u
cd /data/knlp
export CUDA_VISIBLE_DEVICES=0
mkdir -p out/gpt2-matched-w7900
LOG=out/gpt2-matched-w7900/run_3arm.log
date -u +%Y-%m-%dT%H:%M:%SZ >> "$LOG"
echo "[w7900-3arm] start" >> "$LOG"

run_arm() {
  local cfg=$1
  local rn=$2
  echo "[w7900-3arm] === $rn ===" >> "$LOG"
  date -u +%Y-%m-%dT%H:%M:%SZ >> "$LOG"
  python fim/reciprocal_attention/gpt2_matched.py \
    --config "$cfg" \
    --override "run_name=\"$rn\"" \
    --override "tracking.out_dir=\"out/gpt2-matched-w7900\"" \
    >> "$LOG" 2>&1
  echo "[w7900-3arm] done $rn (exit $?)" >> "$LOG"
}

run_arm "fim/reciprocal_attention/configs/gpt2_baseline_1xh100.json"        "gpt2-baseline-w7900"
run_arm "fim/reciprocal_attention/configs/gpt2_arm_a_fimtrace_1xh100.json"  "gpt2-arm-a-fimtrace-w7900"
run_arm "fim/reciprocal_attention/configs/gpt2_arm_c_jsd_w7900.json"        "gpt2-arm-c-jsd-w7900"

echo "[w7900-3arm] all done" >> "$LOG"
date -u +%Y-%m-%dT%H:%M:%SZ >> "$LOG"
