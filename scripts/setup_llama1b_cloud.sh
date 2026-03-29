#!/usr/bin/env bash
set -euo pipefail
#
# Cloud setup for LLaMA-1B matched RA lane on 4xH100.
#
# Usage (run on the cloud pod after SSH):
#   bash scripts/setup_llama1b_cloud.sh
#
# Prerequisites:
#   - 4xH100 80GB RunPod pod (pytorch template)
#   - Repo synced to /data/knlp on the pod
#   - SSH key authorized
#

echo "=== LLaMA-1B Cloud Setup ==="
echo "Date: $(date -u)"
echo "Host: $(hostname)"

# Verify GPU availability
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 4 ]; then
  echo "WARNING: Expected 4 GPUs, found $GPU_COUNT"
fi

# Install dependencies
echo "=== Installing Dependencies ==="
pip install --quiet --upgrade datasets wandb transformers accelerate sentencepiece tiktoken tokenizers

# Verify critical imports
echo "=== Verifying Imports ==="
python3 -c "
import torch
import transformers
import datasets
import sentencepiece
import tiktoken
import tokenizers
print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'datasets: {datasets.__version__}')
print(f'sentencepiece: {sentencepiece.__version__}')
print(f'tiktoken: {tiktoken.__version__}')
print(f'tokenizers: {tokenizers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
"

# Verify data
echo "=== Data Check ==="
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_BIN="$REPO_ROOT/gpt2/data/finewebedu/train.bin"
VAL_BIN="$REPO_ROOT/gpt2/data/finewebedu/val.bin"

if [ ! -f "$TRAIN_BIN" ] || [ ! -f "$VAL_BIN" ]; then
  echo "ERROR: Missing data files"
  echo "  Expected: $TRAIN_BIN"
  echo "  Expected: $VAL_BIN"
  echo "Run data preparation first or rsync from source machine."
  exit 1
fi
echo "train.bin: $(stat --format='%s bytes' "$TRAIN_BIN")"
echo "val.bin: $(stat --format='%s bytes' "$VAL_BIN")"

# Disable wandb for setup verification
export WANDB_DISABLED=1
RUN_FIM_SMOKE=${RUN_FIM_SMOKE:-0}
export LLAMA1B_EVAL_TASKS=${LLAMA1B_EVAL_TASKS:-hellaswag,winogrande}
export LLAMA1B_EVAL_SMOKE_MAX_EXAMPLES=${LLAMA1B_EVAL_SMOKE_MAX_EXAMPLES:-32}

# Run target-shape smoke: baseline
echo "=== Smoke Test: baseline (target-shape, bf16, seq_len=1024) ==="
NPROC_PER_NODE=$GPU_COUNT scripts/run_llama1b_matched.sh target-smoke-baseline
echo "=== Baseline smoke: PASS ==="

# Validate downstream eval hook on the fresh smoke checkpoint.
echo "=== Eval Smoke: baseline checkpoint (${LLAMA1B_EVAL_TASKS}, max_examples=${LLAMA1B_EVAL_SMOKE_MAX_EXAMPLES}) ==="
scripts/run_llama1b_matched.sh eval-smoke-baseline
echo "=== Eval smoke: PASS ==="

# Run target-shape smoke: surgical RA top-4
echo "=== Smoke Test: RA-8 (surgical-8 headline arm) ==="
NPROC_PER_NODE=$GPU_COUNT scripts/run_llama1b_matched.sh target-smoke-ra8
echo "=== RA-8 smoke: PASS ==="

if [ "$RUN_FIM_SMOKE" = "1" ] || [ "$RUN_FIM_SMOKE" = "true" ]; then
  echo "=== Optional Smoke Test: FIM collection ==="
  NPROC_PER_NODE=$GPU_COUNT scripts/run_llama1b_matched.sh target-smoke-fim
  echo "=== FIM smoke: PASS ==="
else
  echo "=== Skipping target-shape FIM smoke ==="
  echo "Reason: baseline + surgical-RA top-4 are the required readiness gate for the matched comparison,"
  echo "and the 1B FIM collection is validated in the real pipeline rather than burning extra cloud time here."
fi

echo ""
echo "=== REQUIRED CLOUD READINESS PASSED ==="
echo "Cloud environment is ready for production runs."
echo ""
echo "To launch the full pipeline:"
echo "  NPROC_PER_NODE=$GPU_COUNT scripts/run_llama1b_matched.sh full-sequence-eval"
echo ""
echo "Or individual phases:"
echo "  NPROC_PER_NODE=$GPU_COUNT scripts/run_llama1b_matched.sh full-fim"
echo "  NPROC_PER_NODE=$GPU_COUNT scripts/run_llama1b_matched.sh full-baseline"
echo "  NPROC_PER_NODE=$GPU_COUNT scripts/run_llama1b_matched.sh full-ra8"
echo "  scripts/run_llama1b_matched.sh eval-all"
