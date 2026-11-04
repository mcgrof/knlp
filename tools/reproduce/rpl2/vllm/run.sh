#!/bin/bash
# vLLM through the LMCache multiprocess (MP) server, with the fs_native L2
# adapter on the filesystem under test: stock (held_fd_capacity 0)
# against held descriptors, end to end.
#
#   run.sh <out-dir>
#
# Each arm starts a fresh LMCache MP server and a fresh vLLM, so L1 starts
# empty. Arms:
#   recompute          vLLM without LMCache: the prefill cost being saved
#   populate-<io>      stores every prompt's KV in L2 once, per I/O mode
#   <io>-h<held>-r<k>  measured arms, in ABBA order per repeat
# Each measured arm runs one warm pass, then for each concurrency c an
# idle-c<c> phase and, when LOAD is set, a busy-c<c> phase with that
# load running. A phase sends requests for SECS seconds.
#
# Settings come from the environment (the Makefile sets them):
#   PY            Python with vLLM and CUDA PyTorch
#   LMCACHE_SRC   LMCache tree with its CUDA extensions built in place
#   DEPS          optional extra site directory for LMCache dependencies
#   L2_BASE       directory on the filesystem under test (created)
#   MODEL PROMPTS PREFIX_TOKENS L1_GB CAP IO SECS CONCURRENCY REPEATS
#   MAX_MODEL_LEN GPU_UTIL                                  run shape
#   WRAP          optional command prefix that runs a command as the
#                 owner where the mount is visible (the pNFS lab sets it)
#   LOAD          optional command run as
#                 LOAD <held> <start-epoch> <seconds> <record-prefix>
#                 during busy phases (the pNFS lab's extra clients)
#   MDS_CMD       optional command printing metadata-server counters
#   OWNER         when run as root with WRAP: the user who owns the records
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?out dir}
: "${PY:?Python with vLLM}" "${LMCACHE_SRC:?LMCache tree}" "${L2_BASE:?directory under test}"
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PROMPTS=${PROMPTS:-16}
PREFIX_TOKENS=${PREFIX_TOKENS:-8192}
L1_GB=${L1_GB:-8}
CAP=${CAP:-4096}
IO=${IO:-buffered odirect}
SECS=${SECS:-60}
CONCURRENCY=${CONCURRENCY:-1 4}
REPEATS=${REPEATS:-2}
LOAD=${LOAD:-}
MDS_CMD=${MDS_CMD:-}
read -r -a WRAPV <<<"${WRAP:-}"
[ -e "$OUT" ] && { echo "$OUT exists; runs are preserved" >&2; exit 1; }
mkdir -p "$OUT"
own() { [ -n "${OWNER:-}" ] && chown "$OWNER": "$@"; return 0; }
own "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
ENVV=(env HOME="${OWNER_HOME:-$HOME}" PATH="$(dirname "$PY"):/usr/sbin:/usr/bin:/sbin:/bin"
	PYTHONPATH="$LMCACHE_SRC${DEPS:+:$DEPS}" PY="$PY" MODEL="$MODEL" L1_GB="$L1_GB"
	MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}" GPU_UTIL="${GPU_UTIL:-0.90}"
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" LMCACHE_DISABLE_BANNER=1
	LMCACHE_TRACK_USAGE=false DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
	TOKENIZERS_PARALLELISM=false ${HF_HOME:+HF_HOME="$HF_HOME"}
	${HF_HUB_OFFLINE:+HF_HUB_OFFLINE="$HF_HUB_OFFLINE"})
as_owner() { "${WRAPV[@]}" "${ENVV[@]}" "$@"; }
log() { echo "== $(date -u +%T) $*"; }

counters() {
	{
		[ -n "$MDS_CMD" ] && { echo "# mds"; $MDS_CMD || true; }
		echo "# time $(date +%s.%N)"
	} >"$1"
}

gpu_idle() {
	command -v nvidia-smi >/dev/null || return 0
	for _ in $(seq 120); do
		[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
		sleep 1
	done
	echo "GPU still busy after the arm stopped" >&2
	return 1
}

ARM="" SERVE=""
start_arm() {
	local tag=$1 mode=$2 held=$3 odirect=$4 l2=$5
	ARM=$OUT/$tag
	mkdir -p "$ARM"
	own "$ARM"
	log "arm $tag: mode=$mode held=$held odirect=$odirect"
	as_owner bash "$here/serve.sh" "$ARM" "$mode" "$held" "$odirect" "$l2" >"$ARM/serve.out" 2>&1 &
	SERVE=$!
	for _ in $(seq 900); do
		if as_owner curl -sf -o /dev/null http://127.0.0.1:8000/health; then
			log "arm $tag ready"
			return 0
		fi
		if ! kill -0 "$SERVE" 2>/dev/null || [ -e "$ARM/serve.err" ]; then
			log "arm $tag failed to start"
			tail -30 "$ARM/vllm.log" "$ARM/lmcache-server.log" 2>/dev/null
			return 1
		fi
		sleep 1
	done
	log "arm $tag not ready after 900 s"
	return 1
}

stop_arm() {
	local p
	p=$(pgrep -f "bash $here/serve.sh $ARM " | head -1)
	[ -n "$p" ] && kill -TERM "$p" 2>/dev/null
	wait "$SERVE" 2>/dev/null
	gpu_idle
	log "arm stopped"
}

bench() {
	as_owner "$PY" "$here/bench.py" "$@" --prompts "$PROMPTS" \
		--prefix-tokens "$PREFIX_TOKENS" --model "$MODEL" --mount "$L2_BASE"
}

# phase <name> <concurrency> <pass-base> <held for the load, or none>
phase() {
	local name=$1 conc=$2 base=$3 load=$4 start rc=0 lp=""
	counters "$ARM/$name.before"
	start=$(($(date +%s) + 5))
	if [ "$load" != none ]; then
		$LOAD "$load" "$start" $((SECS + 10)) "$ARM/$name-load" &
		lp=$!
	fi
	bench run --out "$ARM/$name.json" --concurrency "$conc" --seconds "$SECS" \
		--pass-base "$base" --start-at $((start + 3)) || rc=1
	[ -n "$lp" ] && { wait "$lp" || rc=1; }
	counters "$ARM/$name.after"
	return $rc
}

measured_arm() {
	local tag=$1 held=$2 odirect=$3 l2=$4 c base=100 lheld=0
	[ "$held" -gt 0 ] && lheld=64
	start_arm "$tag" lmcache "$held" "$odirect" "$l2" || return 1
	bench run --out "$ARM/warm.json" --passes 1 --pass-base 10 || { stop_arm; return 1; }
	for c in $CONCURRENCY; do
		phase "idle-c$c" "$c" $base none
		[ -n "$LOAD" ] && phase "busy-c$c" "$c" $((base + 1000)) "$lheld"
		base=$((base + 100))
	done
	stop_arm
}

populate_arm() {
	local tag=$1 odirect=$2 l2=$3
	start_arm "$tag" lmcache 0 "$odirect" "$l2" || return 1
	bench populate --out "$ARM/populate.json" --pass-base 0 --l2 "$l2"
	local rc=$?
	stop_arm
	return $rc
}

log "model $MODEL prompts $PROMPTS x $PREFIX_TOKENS tokens, L1 $L1_GB GiB, held $CAP, io '$IO'"
log "kernel $(uname -r), LMCache $(git -C "$LMCACHE_SRC" rev-parse HEAD 2>/dev/null)"
as_owner "$PY" -c 'import vllm, torch; print("vLLM", vllm.__version__, "torch", torch.__version__)'
# as the owner, so the mount is resolved where WRAP makes it visible
as_owner findmnt -T "$L2_BASE" -o TARGET,SOURCE,FSTYPE,OPTIONS
gpu_idle || exit 1
B=$L2_BASE/rpl2-vllm-$(date -u +%Y%m%dT%H%M%SZ)
start_arm recompute recompute 0 0 none || exit 1
bench run --out "$ARM/warm.json" --passes 1 --pass-base 10
base=100
for c in $CONCURRENCY; do
	phase "idle-c$c" "$c" $base none
	base=$((base + 100))
done
stop_arm
for io in $IO; do
	od=0
	[ "$io" = odirect ] && od=1
	populate_arm "populate-$io" "$od" "$B/l2-$io" || exit 1
done
for rep in $(seq 1 "$REPEATS"); do
	# alternate which arm goes first between I/O modes and repeats, so
	# drift over the run does not favour one arm
	k=0
	for io in $IO; do
		od=0
		[ "$io" = odirect ] && od=1
		if [ $(((rep + k) % 2)) = 1 ]; then order="0 $CAP"; else order="$CAP 0"; fi
		k=$((k + 1))
		for held in $order; do
			measured_arm "$io-h$held-r$rep" "$held" "$od" "$B/l2-$io" || exit 1
		done
	done
done
python3 "$here/summarize.py" "$OUT" | tee "$OUT/summary.md"
log "END; remove $B to free its space"
