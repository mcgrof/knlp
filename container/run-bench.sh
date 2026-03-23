#!/usr/bin/env bash
# run-bench.sh — launch the knlp-rocm-bench container on prune/W7900.
#
# Usage:
#   ./container/run-bench.sh              # interactive shell
#   ./container/run-bench.sh CMD [ARGS]   # run CMD inside container
#
# Examples:
#   ./container/run-bench.sh python3 -c "import vllm; print(vllm.__version__)"
#   ./container/run-bench.sh lm_eval --help
#   ./container/run-bench.sh python $VLLM_DIR/benchmarks/benchmark_latency.py --help

set -euo pipefail

IMAGE="${KNLP_BENCH_IMAGE:-knlp-rocm-bench:latest}"

# HuggingFace cache: share host cache to avoid re-downloading
# models inside the container.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Results directory: bind-mount a host directory for writing
# benchmark output that persists after the container exits.
RESULTS_DIR="${BENCH_RESULTS_DIR:-$(pwd)/bench_results}"
mkdir -p "$RESULTS_DIR"

exec docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --network=host \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    -e HF_HOME=/root/.cache/huggingface \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$RESULTS_DIR":/results \
    -v /data:/data \
    -w /workspace \
    "$IMAGE" \
    "$@"
