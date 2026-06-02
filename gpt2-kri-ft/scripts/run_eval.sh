#!/usr/bin/env bash
# Evaluate vanilla GPT-2 vs the fine-tuned KRI model across policies
# and retention budgets. Optionally a dense-control checkpoint can be
# included by setting DENSE_CKPT.
#
# Outputs go to runs/eval/ as JSONL + CSV pairs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PY="${PY:-$HOME/envs/w7900-ml/bin/python3}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

KRI_CKPT="${KRI_CKPT:-$HERE/runs/kri-gpt2-small/checkpoint_final.pt}"
DENSE_CKPT="${DENSE_CKPT:-}"
OUT="${OUT:-$HERE/runs/eval}"
DATASET="${DATASET:-roneneldan/TinyStories}"
TEXT_COL="${TEXT_COL:-text}"
STREAMING="${STREAMING:-false}"
N_BATCHES="${N_BATCHES:-32}"
N_SYNTH="${N_SYNTH:-200}"

MODELS="openai-community/gpt2,${KRI_CKPT}"
if [[ -n "${DENSE_CKPT}" ]]; then
  MODELS="openai-community/gpt2,${DENSE_CKPT},${KRI_CKPT}"
fi
echo "models: ${MODELS}"

mkdir -p "${OUT}"

echo "==[1/2] pruned-PPL eval"
"${PY}" -m src.eval_pruned_ppl \
  --models "${MODELS}" \
  --seq_len 1024 \
  --block_size 16 \
  --policies full,recent,sink_recent,kri \
  --retention_fracs 1.0,0.5,0.25,0.125,0.0625 \
  --n_batches "${N_BATCHES}" \
  --batch_size 2 \
  --dataset_name "${DATASET}" \
  --text_column "${TEXT_COL}" \
  --streaming "${STREAMING}" \
  --output "${OUT}/pruned_ppl.jsonl"

echo
echo "==[2/2] synthetic retrieval eval"
"${PY}" -m src.eval_synthetic_retrieval \
  --models "${MODELS}" \
  --num_examples "${N_SYNTH}" \
  --seq_len 1024 \
  --block_size 16 \
  --policies full,recent,sink_recent,kri \
  --retention_fracs 1.0,0.5,0.25,0.125 \
  --output "${OUT}/synthetic_retrieval.jsonl"

echo
echo "eval complete: results in ${OUT}"
