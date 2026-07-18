#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide bench: run the configured op x value-size x queue-depth matrix
# for each enabled initiator against the same target, one CSV row per
# cell.
#
# CPU accounting: busy jiffies are summed over all cores EXCEPT the
# target's reactor cores, which busy-poll at a constant 100% per core
# regardless of load. That isolates the initiator-side cost -- for SPDK
# the polling reactor core, for xNVMe the app thread PLUS kernel
# nvme-tcp softirq/kworker time that the app's own rusage would miss.
# Same measurement for both initiators = apples-to-apples.
#
# Single-cell mode: bench.sh <initiator> <op> <qd> <vsize>

. "$(dirname "$0")/lib.sh"

# Cell tolerance: a cell whose output cannot be parsed must land as an
# ERR row, not abort the matrix -- drop lib.sh's errexit, keep nounset.
set +e

SPDK_PERF="$SPDK_SRC/build/bin/spdk_nvme_perf"
XNVME_PERF="$KVTIDE_SRC/xnvme_kv_perf"
CORE=${CONFIG_KVTIDE_BENCH_INIT_CORE}
SECS=${SECS:-${CONFIG_KVTIDE_BENCH_SECS}}
TARGET_CORES=$(mask_to_cores "$TARGET_CORE_MASK")
CLK_TCK=$(getconf CLK_TCK)

case " $TARGET_CORES " in
*" $CORE "*)
	kvtide_die "initiator core $CORE overlaps target cores '$TARGET_CORES'" ;;
esac

# Sum busy jiffies across all cores except the target cores.
busy_jiffies() {
	awk -v excl="$TARGET_CORES" '
		BEGIN{split(excl,e," "); for(i in e) ex[e[i]]=1}
		/^cpu[0-9]+ /{
			n=substr($1,4)+0; if(n in ex) next;
			# user nice system idle iowait irq softirq steal ...
			busy=$2+$3+$4+$7+$8+$9;
			tot+=busy
		}
		END{print tot}' /proc/stat
}

run_cell() {
	local initiator=$1 op=$2 qd=$3 vsize=$4
	local b0 b1 out iops mbps lat p99 t0 t1 wall cores dev w
	b0=$(busy_jiffies)
	t0=$(date +%s.%N)
	if [ "$initiator" = spdk ]; then
		w=randwrite
		[ "$op" = retrieve ] && w=randread
		# Core list syntax, NOT a mask: SPDK parses a bare -c value
		# as hex, so -c 16 would mean 0x16 = cores 1,2,4 -- three
		# workers, one on a target reactor core. -c [N] is exact.
		out=$(sudo taskset -c "$CORE" "$SPDK_PERF" -r "$TRID" \
			-q "$qd" -o "$vsize" -w "$w" -t "$SECS" \
			-c "[$CORE]" 2>/dev/null)
		# Total line: Total : IOPS MiB/s Avg min max
		read -r iops mbps lat < <(echo "$out" | awk '/^Total /{print $3,$4,$5}')
		p99="na"
	else
		dev=$(kv_dev)
		out=$(sudo taskset -c "$CORE" "$XNVME_PERF" "$dev" "$op" \
			"$qd" "$vsize" "$SECS" 2>/dev/null | tail -1)
		iops=$(echo "$out" | grep -oE 'iops=[0-9.]+' | cut -d= -f2)
		mbps=$(echo "$out" | grep -oE 'MBps=[0-9.]+' | cut -d= -f2)
		lat=$(echo "$out"  | grep -oE 'p50=[0-9.]+'  | cut -d= -f2)
		p99=$(echo "$out"  | grep -oE 'p99=[0-9.]+'  | cut -d= -f2)
	fi
	t1=$(date +%s.%N)
	b1=$(busy_jiffies)
	wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
	cores=$(awk -v b0="$b0" -v b1="$b1" -v w="$wall" -v hz="$CLK_TCK" \
		'BEGIN{printf "%.3f", ((b1-b0)/hz)/w}')
	printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
		"$initiator" "$op" "$qd" "$vsize" "${iops:-ERR}" "${mbps:-ERR}" \
		"${lat:-ERR}" "${p99:-na}" "$cores"
}

# The latency column is p50 for xNVMe but the arithmetic mean for SPDK
# (spdk_nvme_perf's Total line has no percentiles); p99 is only present
# for xNVMe. Compare iops/MBps/cpu across initiators, latency shape
# within one. The jiffies window brackets the whole child process, so
# connect/teardown and xnvme_kv_perf's serial key prepopulation land in
# the CPU denominator -- ~1% at the default 8 s; raise SECS to shrink.

INITIATORS=""
want_spdk && INITIATORS="$INITIATORS spdk"
if want_xnvme; then
	INITIATORS="$INITIATORS xnvme"
	require_kv_dev >/dev/null
	test -x "$XNVME_PERF" || kvtide_die "xnvme_kv_perf not built"
fi
if want_spdk; then
	test -x "$SPDK_PERF" || kvtide_die "spdk_nvme_perf not built"
fi

if [ $# -eq 4 ]; then
	case "$1" in
	spdk|xnvme) ;;
	*) kvtide_die "initiator must be spdk or xnvme, not '$1'" ;;
	esac
	run_cell "$1" "$2" "$3" "$4"
	exit 0
elif [ $# -ne 0 ]; then
	kvtide_die "usage: bench.sh [<spdk|xnvme> <op> <qd> <vsize>]"
fi

mkdir -p "$KVTIDE_RESULTS"
CSV="$KVTIDE_RESULTS/kvtide-bench-$(date +%Y%m%d-%H%M%S).csv"
echo "initiator,op,qd,vsize,iops,MBps,lat_us,p99_us,init_cpu_cores" > "$CSV"
ln -sf "$(basename "$CSV")" "$KVTIDE_RESULTS/kvtide-bench-latest.csv"

kvtide_log "matrix: ops='${CONFIG_KVTIDE_BENCH_OPS}'" \
	"vsizes='${CONFIG_KVTIDE_BENCH_VSIZES}'" \
	"qds='${CONFIG_KVTIDE_BENCH_QDS}' secs=$SECS -> $CSV"

for op in ${CONFIG_KVTIDE_BENCH_OPS}; do
	for vsize in ${CONFIG_KVTIDE_BENCH_VSIZES}; do
		for qd in ${CONFIG_KVTIDE_BENCH_QDS}; do
			for initiator in $INITIATORS; do
				kvtide_log "cell: $initiator $op qd=$qd vsize=$vsize"
				run_cell "$initiator" "$op" "$qd" "$vsize" | tee -a "$CSV"
			done
		done
	done
done

kvtide_log "bench done: $CSV"
