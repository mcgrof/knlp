#!/bin/bash
# Many-client sweep: N lab clients c1..cN, each its own NFS client identity
# in its own namespace, read the same 16 hot objects (16 distinct objects
# per request) from one barrier time, with stock LMCache (held 0) and with
# held descriptors, in alternating order per N. The metadata server's op
# counters and the MDS and RonDB cgroup CPU are captured around each cell.
# Run as root after labctl.sh up, fixture.sh and labctl.sh clients-up <max N>.
#   sweep.sh <out-dir> [fixture: rpl2-256k|rpl2-8m] ["N list"] [seconds] [capacity]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
OUT=${1:?out dir}
SUB=${2:-rpl2-256k}
NS_LIST=${3:-1 2 4 8 16}
SECS=${4:-20}
CAP=${5:-64}
[ -e "$OUT" ] && { echo "$OUT exists; runs are preserved" >&2; exit 1; }
mkdir -p "$OUT"
chown "$OWNER": "$OUT"
L="$here/labctl.sh"
cg() { grep usage_usec "/sys/fs/cgroup/system.slice/$UNIT_PREFIX-$1.service/cpu.stat" | awk '{print $2}'; }
ops() { "$L" mdsns curl -s "http://$IP_MDS:9090/metrics" | grep -E '^pnfs_mds_nfs_op\{' || true; }
pos=0
for n in $NS_LIST; do
	pos=$((pos + 1))
	if [ $((pos % 2)) = 1 ]; then order=(0 "$CAP"); else order=("$CAP" 0); fi
	for held in "${order[@]}"; do
		cell=$OUT/n$n-held$held
		mkdir -p "$cell"
		chown "$OWNER": "$cell"
		ops >"$cell/mds-ops.before"
		echo "mds $(cg mds) rondb $(cg rondb)" >"$cell/cg.before"
		start=$(($(date +%s) + 3))
		pids=()
		for k in $(seq 1 "$n"); do
			"$here/run.sh" "c$k" read --base "$RUN/c$k/mnt/$SUB" --hot 16 --batch 16 \
				--held "$held" --start-at "$start" --seconds "$SECS" --client "c$k" \
				--out "$cell/c$k.json" >"$cell/c$k.log" 2>&1 &
			pids+=($!)
		done
		rc=0
		for p in "${pids[@]}"; do wait "$p" || rc=1; done
		echo "mds $(cg mds) rondb $(cg rondb)" >"$cell/cg.after"
		ops >"$cell/mds-ops.after"
		echo "$(date -Is) n=$n held=$held rc=$rc" | tee -a "$OUT/sweep.log"
		[ "$rc" = 0 ] || { echo "a reader failed in $cell" >&2; exit 1; }
	done
done
python3 "$here/summarize.py" sweep "$OUT"
