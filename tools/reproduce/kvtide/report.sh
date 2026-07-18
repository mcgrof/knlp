#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide report: pretty-print the latest bench CSV; when both initiators
# are present, add the per-cell SPDK/xNVMe IOPS ratio.

. "$(dirname "$0")/lib.sh"

CSV="$KVTIDE_RESULTS/kvtide-bench-latest.csv"
test -e "$CSV" || kvtide_die "no bench CSV yet -- run 'make kvtide-bench'"

SUMMARY="$KVTIDE_RESULTS/kvtide-summary.txt"
RATIOS=$(mktemp)
trap 'rm -f "$RATIOS"' EXIT

awk -F, '
	NR == 1 { next }
	$1 == "spdk"  { s[$2 "," $3 "," $4] = $5 }
	$1 == "xnvme" { x[$2 "," $3 "," $4] = $5 }
	END {
		for (k in s) {
			if (!(k in x)) continue
			# Skip ERR/non-numeric cells: "ERR"+0 is 0, which
			# would render as a fake 0.00x ratio.
			if (s[k] !~ /^[0-9.]+$/ || x[k] !~ /^[0-9.]+$/) continue
			if (x[k] + 0 == 0) continue
			split(k, f, ",")
			printf "%-10s %-6s %-8s %.2fx\n", f[1], f[2], f[3], s[k] / x[k]
		}
	}' "$CSV" > "$RATIOS"

{
	echo "KVTide bench summary -- $(readlink -f "$CSV")"
	if [ -f "$(readlink -f "$CSV").meta" ]; then
		tr '\n' ' ' < "$(readlink -f "$CSV").meta"
		echo ""
	fi
	echo ""
	column -s, -t < "$CSV"
	if [ -s "$RATIOS" ]; then
		echo ""
		echo "A/B (spdk iops / xnvme iops):"
		printf '%-10s %-6s %-8s %s\n' "op" "qd" "vsize" "ratio"
		sort -k1,1 -k3,3n -k2,2n "$RATIOS"
	fi
} > "$SUMMARY"

cat "$SUMMARY"
kvtide_log "summary written to $SUMMARY"
