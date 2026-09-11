#!/bin/bash
# SPDX-License-Identifier: MIT

. "$(dirname "$0")/lib.sh"

LATEST="$KVTIDE_RESULTS/kvtide-pcie-latest"
[ -d "$LATEST" ] || \
	kvtide_die "no physical PCIe results; run make kvtide-pcie-bench"
python3 "$KVTIDE_DIR/pcie-report.py" "$LATEST/results.csv" \
	--output "$LATEST/summary.txt"
