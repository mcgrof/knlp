#!/usr/bin/env bash
# smoke-xkv — run xKV long-context smoke inside knlp-rocm-bench.
#
# Exercises the xKV evaluation path (RULER niah_single_1 task)
# with a tiny workload to validate plumbing.  Expects:
#   - /data/xKV bind-mounted (contains evaluate/ code)
#   - A vLLM-compatible model already cached in HF_HOME
#
# Usage (from host):
#   ./container/run-bench.sh smoke-xkv
#   ./container/run-bench.sh smoke-xkv --model Qwen/Qwen2.5-0.5B
#
# Usage (inside container):
#   smoke-xkv
#   smoke-xkv --model Qwen/Qwen2.5-0.5B

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
XKV_DIR="${XKV_DIR:-/data/xKV}"
SMOKE_DIR="${SMOKE_DIR:-/results/smoke_xkv}"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --output) SMOKE_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ ! -d "$XKV_DIR/evaluate" ]; then
    echo "ERROR: $XKV_DIR/evaluate not found."
    echo "Bind-mount /data so that $XKV_DIR is accessible."
    exit 1
fi

mkdir -p "$SMOKE_DIR"

echo "=== xKV smoke test ==="
echo "  Model:  $MODEL"
echo "  xKV:    $XKV_DIR"
echo "  Output: $SMOKE_DIR"
echo ""

# xKV eval_acc.py needs its root on PYTHONPATH
export PYTHONPATH="${XKV_DIR}:${PYTHONPATH:-}"

# Run a minimal single-GPU eval with the RULER niah_single_1
# dataset at a short sequence length.  No --xKV flag means
# baseline mode (no patches applied).  This exercises:
#   - Dataset loading (HuggingFace or local)
#   - Tokenization and model forward pass
#   - Metric scoring and result serialization
python "$XKV_DIR/evaluate/eval_acc.py" \
    --model_name_or_path "$MODEL" \
    --datalen 4096 \
    --dataset_name "ruler/niah_single_1" \
    --num_samples 5 \
    --result_dir "$SMOKE_DIR" \
    2>&1 | tee "$SMOKE_DIR/smoke_xkv.log"

RC=${PIPESTATUS[0]}
if [ "$RC" -eq 0 ] && [ -s "$SMOKE_DIR/smoke_xkv.log" ]; then
    echo ""
    echo "SMOKE xKV: PASS"
else
    echo ""
    echo "SMOKE xKV: FAIL (exit code $RC)"
    exit 1
fi
