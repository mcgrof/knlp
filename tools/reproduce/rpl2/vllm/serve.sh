#!/bin/bash
# Start the LMCache MP server and vLLM, keep both until TERM, then stop
# both. With mode recompute, vLLM runs without LMCache.
#   serve.sh <arm-dir> <lmcache|recompute> <held> <odirect 0|1> <l2-dir>
# Settings come from the environment (run.sh sets them):
#   PY              Python with vLLM; LMCache comes from PYTHONPATH
#   MODEL           model name or path
#   L1_GB           LMCache L1 (host memory) size
#   MAX_MODEL_LEN   vLLM maximum model length
#   GPU_UTIL        vLLM GPU memory utilization
#   LMC_PORT VLLM_PORT
set -uo pipefail
ARM=$1 MODE=$2 HELD=$3 ODIRECT=$4 L2=$5
: "${PY:?}" "${MODEL:?}"
L1_GB=${L1_GB:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}
GPU_UTIL=${GPU_UTIL:-0.90}
LMC_PORT=${LMC_PORT:-6555}
VLLM_PORT=${VLLM_PORT:-8000}
mkdir -p "$ARM"
pids=()
stop() {
	local p alive
	for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
	for _ in $(seq 60); do
		alive=0
		for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
		[ $alive = 0 ] && break
		sleep 1
	done
	for p in "${pids[@]}"; do kill -KILL "$p" 2>/dev/null; done
	exit 0
}
trap stop TERM INT
kv=()
if [ "$MODE" = lmcache ]; then
	mkdir -p "$L2"
	adapter=$(printf '{"type":"fs_native","base_path":"%s","use_odirect":%s,"held_fd_capacity":%d}' \
		"$L2" "$([ "$ODIRECT" = 1 ] && echo true || echo false)" "$HELD")
	align=()
	# O_DIRECT needs L1 buffers aligned to the filesystem block size,
	# which NFS reports as its rsize
	[ "$ODIRECT" = 1 ] && align=(--l1-align-bytes "$(stat -f -c %s "$L2")")
	echo "$adapter ${align[*]}" >"$ARM/adapter.txt"
	"$PY" -c 'import sys; sys.argv[0] = "lmcache"; from lmcache.cli.main import main; main()' \
		server --host 127.0.0.1 --port "$LMC_PORT" --http-host 127.0.0.1 \
		--http-port $((LMC_PORT + 25)) --chunk-size 256 --l1-size-gb "$L1_GB" \
		--eviction-policy LRU "${align[@]}" --l2-adapter "$adapter" \
		>"$ARM/lmcache-server.log" 2>&1 &
	pids+=($!)
	for _ in $(seq 120); do
		ss -ltn | grep -q ":$LMC_PORT " && break
		kill -0 "${pids[0]}" 2>/dev/null || break
		sleep 1
	done
	kv=(--kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$LMC_PORT}}")
fi
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" --host 127.0.0.1 --port "$VLLM_PORT" \
	--seed 0 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 8 \
	--gpu-memory-utilization "$GPU_UTIL" --no-enable-prefix-caching \
	--enable-prompt-tokens-details "${kv[@]}" >"$ARM/vllm.log" 2>&1 &
pids+=($!)
while :; do
	for p in "${pids[@]}"; do
		kill -0 "$p" 2>/dev/null || { echo "child $p exited" >>"$ARM/serve.err"; stop; }
	done
	sleep 2
done
