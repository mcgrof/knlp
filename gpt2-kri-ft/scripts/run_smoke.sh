#!/usr/bin/env bash
# Smoke pipeline:
#  1) Show GPU info (ROCm/HIP).
#  2) Run weight-loading, KRI mask, and round-trip tests.
#  3) Run a 200-step KRI fine-tune on TinyStories at seq_len=512.
#  4) Evaluate the smoke checkpoint vs vanilla GPT-2 under all policies.
#
# Expected wall time on a single AMD Radeon Pro W7900: a few minutes
# for tests, ~10-20 minutes for the 200-step train (depending on
# precision and grad_checkpointing), and ~2-5 minutes for eval.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PY="${PY:-$HOME/envs/w7900-ml/bin/python3}"
DEVICE="${HIP_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${DEVICE}"

OUT="${OUT:-$HERE/runs/smoke}"
mkdir -p "${OUT}"

echo "==[1/4] device info"
"${PY}" -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); [print('gpu', i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

echo
echo "==[2/4] tests"
"${PY}" tests/test_weight_loading.py
"${PY}" tests/test_kri_mask_causal.py
"${PY}" tests/test_export_equivalence.py

echo
echo "==[3/4] 200-step KRI fine-tune (TinyStories)"
"${PY}" -m src.train_kri \
  --init_model openai-community/gpt2 \
  --output_dir "${OUT}/train" \
  --mode mixed \
  --seq_len 512 \
  --block_size 16 \
  --local_windows 64,128 \
  --topk_blocks 2,4,8 \
  --prefill_splits 128,192,256,384 \
  --sparse_prob 0.7 \
  --lr 1e-5 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_steps 200 \
  --warmup_steps 20 \
  --eval_every 100 \
  --eval_batches 8 \
  --save_every 200 \
  --log_every 10 \
  --precision auto \
  --dataset_name roneneldan/TinyStories \
  --text_column text \
  --teacher_kl_alpha 0.0

echo
echo "==[4/4] smoke eval"
"${PY}" -m src.eval_pruned_ppl \
  --models "openai-community/gpt2,${OUT}/train/checkpoint_final.pt" \
  --seq_len 512 \
  --block_size 16 \
  --policies full,recent,sink_recent,kri \
  --retention_fracs 1.0,0.5,0.25,0.125 \
  --n_batches 8 \
  --batch_size 2 \
  --dataset_name roneneldan/TinyStories \
  --text_column text \
  --output "${OUT}/eval/pruned_ppl.jsonl"

echo
echo "smoke run complete: results under ${OUT}"
