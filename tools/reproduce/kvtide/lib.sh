#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide shared helpers. Sourced by every stage script. All policy comes
# from the knlp .config (Kconfig.kvtide); nothing is hard-coded here.

set -eu

KVTIDE_TOP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
KVTIDE_DIR="$KVTIDE_TOP/tools/reproduce/kvtide"

if [ ! -f "$KVTIDE_TOP/.config" ]; then
	echo "kvtide: no .config -- run 'make defconfig-kvtide-ab' first" >&2
	exit 1
fi

# .config is shell-syntax compatible (CONFIG_X=y, CONFIG_S="str", CONFIG_I=8)
# shellcheck disable=SC1091
. "$KVTIDE_TOP/.config"

if [ "${CONFIG_KVTIDE:-n}" != y ]; then
	echo "kvtide: CONFIG_KVTIDE is not enabled in .config" >&2
	echo "kvtide: run 'make defconfig-kvtide-ab' (or -spdk/-xnvme/-nixl)" >&2
	exit 1
fi

KVTIDE_SRC=${CONFIG_KVTIDE_SRC_DIR}
KVTIDE_RESULTS="$KVTIDE_SRC/results"
SPDK_SRC="$KVTIDE_SRC/spdk"
XNVME_SRC="$KVTIDE_SRC/xnvme"
XNVME_PREFIX="$KVTIDE_SRC/opt/xnvme"
NIXL_SRC="$KVTIDE_SRC/nixl"

TARGET_ADDR=${CONFIG_KVTIDE_TARGET_ADDR}
TARGET_PORT=${CONFIG_KVTIDE_TARGET_PORT}
TARGET_NQN=${CONFIG_KVTIDE_TARGET_NQN}
TARGET_CORE_MASK=${CONFIG_KVTIDE_TARGET_CORE_MASK}
TRID="trtype:TCP adrfam:IPv4 traddr:$TARGET_ADDR trsvcid:$TARGET_PORT subnqn:$TARGET_NQN"

kvtide_log() {
	echo "kvtide: $*"
}

kvtide_die() {
	echo "kvtide: ERROR: $*" >&2
	exit 1
}

want_spdk() {
	[ "${CONFIG_KVTIDE_SPDK:-n}" = y ]
}

want_xnvme() {
	[ "${CONFIG_KVTIDE_XNVME:-n}" = y ]
}

want_nixl() {
	[ "${CONFIG_KVTIDE_NIXL:-n}" = y ]
}

want_linux() {
	[ "${CONFIG_KVTIDE_LINUX:-n}" = y ]
}

# The nvme-tcp attach of the KV target is only needed for the xNVMe
# initiator and the NIXL integration test; SPDK connects userspace.
want_kernel_attach() {
	want_xnvme || want_nixl
}

xnvme_pkgconfig_dir() {
	find "$XNVME_PREFIX" -name xnvme.pc -printf '%h\n' 2>/dev/null | head -1
}

# Expand a core mask like 0x3 into "0 1" (target cores, excluded from
# initiator CPU accounting). Bounded per-bit test: bash arithmetic is
# signed 64-bit, so a right-shift loop would never terminate for masks
# with bit 63 set.
mask_to_cores() {
	local mask=$(( $1 )) i out=""
	for (( i = 0; i < 64; i++ )); do
		if [ $(( (mask >> i) & 1 )) -eq 1 ]; then
			out="$out $i"
		fi
	done
	echo "${out# }"
}

# Resolve the kernel char device of the attached KV namespace by
# subsystem NQN -- device numbering depends on attach order, so the NQN,
# not the number, is the reliable identifier.
kv_dev() {
	local c ctrl dev
	for c in /sys/class/nvme/nvme*; do
		[ -e "$c" ] || continue
		if [ "$(cat "$c/subsysnqn" 2>/dev/null)" = "$TARGET_NQN" ]; then
			ctrl=${c##*/}
			dev="/dev/ng${ctrl#nvme}n1"
			# With multipath the ng node is named after the
			# subsystem instance, which can diverge from the
			# controller instance -- only report nodes that exist.
			[ -e "$dev" ] || continue
			echo "$dev"
			return 0
		fi
	done
	return 1
}

require_kv_dev() {
	kv_dev || kvtide_die \
		"KV namespace for $TARGET_NQN not attached -- run 'make kvtide-target-up'"
}
