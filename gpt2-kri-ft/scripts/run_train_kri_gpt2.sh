#!/usr/bin/env bash
# Serious KRI fine-tune of GPT-2 small (124M) on FineWeb-Edu sample-10BT,
# streamed. seq_len=1024 to match GPT-2's max position embedding.
#
# Tune `--max_steps`, `--batch_size`, `--grad_accum` to fit the W7900's
# 48 GB. With grad checkpointing on, batch_size=2 + grad_accum=16
# (effective 32) is a good starting point at seq_len=1024.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PY="${PY:-$HOME/envs/w7900-ml/bin/python3}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

OUT="${OUT:-$HERE/runs/kri-gpt2-small}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
STREAMING="${STREAMING:-true}"
MAX_STEPS="${MAX_STEPS:-10000}"

"${PY}" -m src.train_kri \
  --init_model openai-community/gpt2 \
  --output_dir "${OUT}" \
  --mode mixed \
  --seq_len 1024 \
  --block_size 16 \
  --local_windows 64,128,256 \
  --topk_blocks 2,4,8,16 \
  --prefill_splits 256,384,512,768 \
  --sparse_prob 0.7 \
  --lr 1e-5 \
  --batch_size 2 \
  --grad_accum 16 \
  --max_steps "${MAX_STEPS}" \
  --warmup_steps 500 \
  --eval_every 500 \
  --eval_batches 32 \
  --save_every 1000 \
  --log_every 25 \
  --precision auto \
  --dataset_name "${DATASET}" \
  --dataset_config "${DATASET_CONFIG}" \
  --streaming "${STREAMING}" \
  --text_column text \
  --teacher_kl_alpha 0.1
