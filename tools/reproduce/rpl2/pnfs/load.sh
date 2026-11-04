#!/bin/bash
# Background load for the vLLM profile in the pNFS lab: the lab's extra
# clients c1..cN read the shared 256 KiB fixture, 16 objects per request,
# with the given held_fd_capacity, as other inference nodes sharing the
# metadata server would. Run as root after labctl.sh clients-up N.
#   load.sh <held> <start-epoch> <seconds> <record-prefix>
# LOAD_CLIENTS (default 8) sets N.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
held=${1:?held} start=${2:?start} secs=${3:?seconds} prefix=${4:?record prefix}
pids=()
rc=0
for k in $(seq 1 "${LOAD_CLIENTS:-8}"); do
	"$here/run.sh" "c$k" read --base "$RUN/c$k/mnt/rpl2-256k" --hot 16 --batch 16 \
		--held "$held" --start-at "$start" --seconds "$secs" --client "c$k" \
		--out "$prefix-c$k.json" >"$prefix-c$k.log" 2>&1 &
	pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
