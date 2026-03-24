#!/usr/bin/env bash
# smoke-ib — run InfiniteBench smoke inside knlp-rocm-bench.
#
# Exercises the minimal HuggingFace InfiniteBench path with a
# single passkey sample.  Does NOT require a vLLM server —
# loads the model directly via HF Transformers.
#
# Usage (from host):
#   ./container/run-bench.sh smoke-ib
#   ./container/run-bench.sh smoke-ib --model Qwen/Qwen2.5-0.5B
#
# Usage (inside container):
#   smoke-ib
#   smoke-ib --model Qwen/Qwen2.5-0.5B

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
IB_DIR="${IB_DIR:-/data/InfiniteBench}"
SMOKE_DIR="${SMOKE_DIR:-/results/smoke_infinitebench}"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --output) SMOKE_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ ! -f "$IB_DIR/src/eval_hf_smoke.py" ]; then
    echo "ERROR: $IB_DIR/src/eval_hf_smoke.py not found."
    echo "Bind-mount /data so that $IB_DIR is accessible."
    exit 1
fi

mkdir -p "$SMOKE_DIR"

echo "=== InfiniteBench smoke test ==="
echo "  Model:  $MODEL"
echo "  IB dir: $IB_DIR"
echo "  Output: $SMOKE_DIR"
echo ""

# Run the minimal HF smoke runner: 1 passkey sample,
# model loaded directly (no vLLM server needed).
python "$IB_DIR/src/eval_hf_smoke.py" \
    --model "$MODEL" \
    --task passkey \
    --limit 1 \
    --output "$SMOKE_DIR/passkey_smoke.json" \
    2>&1 | tee "$SMOKE_DIR/smoke_ib.log"

RC=${PIPESTATUS[0]}
if [ "$RC" -eq 0 ] && [ -s "$SMOKE_DIR/passkey_smoke.json" ]; then
    echo ""
    echo "SMOKE InfiniteBench: PASS"
else
    echo ""
    echo "SMOKE InfiniteBench: FAIL (exit code $RC)"
    exit 1
fi
