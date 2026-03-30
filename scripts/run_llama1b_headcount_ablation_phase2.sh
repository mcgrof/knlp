#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
PY=${PYTHON:-python3}
RUNNER=fim/reciprocal_attention/llama150m_matched.py
CFG_DIR=fim/reciprocal_attention/configs
OUT_DIR=out/llama1b-headcount-ablation-phase2
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  LLaMA-1B RA Head-Count Ablation PHASE-2 — single A100 80GB"
echo "  Date: $(date -u)"
echo "  Host: $(hostname)"
echo "============================================================"

echo "=== Environment Manifest ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "GPU count: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
$PY - <<PYENV 2>&1 | tee "$OUT_DIR/environment_manifest.txt"
import platform, sys
import torch, transformers
print(f"python: {sys.version}")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
print(f"platform: {platform.platform()}")
PYENV

TRAIN_BIN="$REPO_ROOT/gpt2/data/finewebedu/train.bin"
VAL_BIN="$REPO_ROOT/gpt2/data/finewebedu/val.bin"
[ -f "$TRAIN_BIN" ] && [ -f "$VAL_BIN" ] || { echo "missing data"; exit 1; }
echo "Data OK: train.bin=$(stat --format=%s "$TRAIN_BIN") val.bin=$(stat --format=%s "$VAL_BIN")"

ARMS=(
  "ra16:$CFG_DIR/llama1b_headcount_ra16_a100_p2.json"
  "ra20:$CFG_DIR/llama1b_headcount_ra20_a100_p2.json"
  "ra24:$CFG_DIR/llama1b_headcount_ra24_a100_p2.json"
  "ra28:$CFG_DIR/llama1b_headcount_ra28_a100_p2.json"
  "ra32:$CFG_DIR/llama1b_headcount_ra32_a100_p2.json"
)
FAILED=0
TOTAL_START=$(date +%s)
for entry in "${ARMS[@]}"; do
  IFS=: read -r arm_name cfg_path <<< "$entry"
  echo ""
  echo "============================================================"
  echo "  ARM: $arm_name"
  echo "  Config: $cfg_path"
  echo "  Started: $(date -u)"
  echo "============================================================"
  ARM_START=$(date +%s)
  if WANDB_DISABLED=1 $PY "$RUNNER" --config "$cfg_path" 2>&1 | tee "$OUT_DIR/${arm_name}.log"; then
    echo "=== $arm_name PASSED ($(( $(date +%s) - ARM_START ))s) ==="
  else
    echo "=== $arm_name FAILED ($(( $(date +%s) - ARM_START ))s) ==="
    FAILED=$((FAILED+1))
  fi
done

echo ""
echo "============================================================"
echo "  ABLATION COMPLETE"
echo "  Total wall-clock: $(( $(date +%s) - TOTAL_START ))s"
echo "  Failed arms: $FAILED"
echo "============================================================"
exit $FAILED
