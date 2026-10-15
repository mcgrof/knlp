#!/usr/bin/env bash
# Same-model transport control. This is not a cross-model DroidSpeak test.

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-lmcache-mp-smoke}
LMCACHE_PORT=${LMCACHE_PORT:-5555}
LMCACHE_HTTP_PORT=${LMCACHE_HTTP_PORT:-7555}
VLLM_A_PORT=${VLLM_A_PORT:-8000}
VLLM_B_PORT=${VLLM_B_PORT:-8001}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
STORE_SETTLE_SECONDS=${STORE_SETTLE_SECONDS:-2}

if [[ "$GPU_A" == "$GPU_B" ]] && awk "BEGIN {exit !($GPU_MEMORY_UTILIZATION > 0.45)}"; then
    echo "same-GPU mode requires GPU_MEMORY_UTILIZATION <= 0.45" >&2
    exit 2
fi

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "output directory is not empty: $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

{
    printf 'model=%s\n' "$MODEL"
    printf 'lmcache_port=%s\n' "$LMCACHE_PORT"
    printf 'lmcache_http_port=%s\n' "$LMCACHE_HTTP_PORT"
    printf 'vllm_a_port=%s\n' "$VLLM_A_PORT"
    printf 'vllm_b_port=%s\n' "$VLLM_B_PORT"
    printf 'gpu_a=%s\n' "$GPU_A"
    printf 'gpu_b=%s\n' "$GPU_B"
    printf 'gpu_memory_utilization=%s\n' "$GPU_MEMORY_UTILIZATION"
    printf 'store_settle_seconds=%s\n' "$STORE_SETTLE_SECONDS"
} >"$OUTPUT_DIR/parameters.txt"

python - <<'PY' >"$OUTPUT_DIR/versions.txt"
import torch
import vllm
import lmcache

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("lmcache", lmcache.__version__)
print("cuda_available", torch.cuda.is_available())
PY
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader >"$OUTPUT_DIR/hardware.txt"

server_pid=
vllm_a_pid=
vllm_b_pid=

cleanup()
{
    for pid in "$vllm_b_pid" "$vllm_a_pid" "$server_pid"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
    done
}
trap cleanup EXIT INT TERM

wait_http()
{
    local url=$1
    local name=$2
    local tries=${3:-180}
    local count

    for ((count = 0; count < tries; count++)); do
        if curl --fail --silent --output /dev/null "$url"; then
            return 0
        fi
        sleep 1
    done
    echo "$name did not become ready: $url" >&2
    return 1
}

export PYTHONHASHSEED=0

lmcache server \
    --host 127.0.0.1 \
    --port "$LMCACHE_PORT" \
    --http-port "$LMCACHE_HTTP_PORT" \
    --l1-size-gb 8 \
    --eviction-policy LRU \
    >"$OUTPUT_DIR/lmcache.log" 2>&1 &
server_pid=$!
wait_http "http://127.0.0.1:$LMCACHE_HTTP_PORT/healthcheck" LMCache

connector=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":%s}}' "$LMCACHE_PORT")

CUDA_VISIBLE_DEVICES=$GPU_A vllm serve "$MODEL" \
    --port "$VLLM_A_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config "$connector" \
    >"$OUTPUT_DIR/vllm-a.log" 2>&1 &
vllm_a_pid=$!

CUDA_VISIBLE_DEVICES=$GPU_B vllm serve "$MODEL" \
    --port "$VLLM_B_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config "$connector" \
    >"$OUTPUT_DIR/vllm-b.log" 2>&1 &
vllm_b_pid=$!

wait_http "http://127.0.0.1:$VLLM_A_PORT/health" "vLLM A" 600
wait_http "http://127.0.0.1:$VLLM_B_PORT/health" "vLLM B" 600

python - "$OUTPUT_DIR/prompt.json" "$MODEL" <<'PY'
import json
import sys

path, model = sys.argv[1:]
prefix = "LMCache cross-instance prefix control. " * 1024
request = {
    "model": model,
    "prompt": prefix + "\nState the purpose of this control in one sentence:",
    "max_tokens": 8,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as output:
    json.dump(request, output)
PY

curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    --data-binary "@$OUTPUT_DIR/prompt.json" \
    "http://127.0.0.1:$VLLM_A_PORT/v1/completions" \
    >"$OUTPUT_DIR/response-a.json"

sleep "$STORE_SETTLE_SECONDS"

curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    --data-binary "@$OUTPUT_DIR/prompt.json" \
    "http://127.0.0.1:$VLLM_B_PORT/v1/completions" \
    >"$OUTPUT_DIR/response-b.json"

curl --fail --silent --show-error \
    "http://127.0.0.1:$LMCACHE_HTTP_PORT/status" \
    >"$OUTPUT_DIR/lmcache-status.json"

if ! grep -Ei \
    'Retrieved [1-9][0-9]* tokens|Found [1-9][0-9]* hit tokens|LMCache hit tokens: [1-9][0-9]*' \
    "$OUTPUT_DIR/vllm-b.log" "$OUTPUT_DIR/lmcache.log" \
    >"$OUTPUT_DIR/cache-hit-evidence.txt"; then
    echo "No positive cache-hit evidence found in vLLM B or server log" >&2
    exit 1
fi

echo "PASS: vLLM B reported an LMCache retrieval from the shared MP server"
