#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide target up: start the SPDK NVMe-oF/TCP KV target (kvmalloc KV
# bdev) on loopback and, when a kernel-side initiator is configured,
# attach it via nvme-tcp so a /dev/ngXnY KV char device appears.
#
# Deliberately never runs SPDK scripts/setup.sh: no PCI rebinding, no
# vfio/uio -- local NVMe devices stay kernel-owned. Only hugepages are
# reserved.

. "$(dirname "$0")/lib.sh"

RPC="sudo $SPDK_SRC/scripts/rpc.py"
NVMF_TGT="$SPDK_SRC/build/bin/nvmf_tgt"
LOG="$KVTIDE_RESULTS/nvmf_tgt.log"

test -x "$NVMF_TGT" || kvtide_die "nvmf_tgt not built -- run 'make kvtide-build'"
mkdir -p "$KVTIDE_RESULTS"

if $RPC spdk_get_version >/dev/null 2>&1; then
	kvtide_log "an SPDK app already answers on the RPC socket"
else
	kvtide_log "reserving ${CONFIG_KVTIDE_HUGEPAGES} x 2 MiB hugepages"
	sudo sh -c "echo ${CONFIG_KVTIDE_HUGEPAGES} > /proc/sys/vm/nr_hugepages"
	grep HugePages_Total /proc/meminfo

	kvtide_log "starting nvmf_tgt (cores $TARGET_CORE_MASK, log $LOG)"
	sudo sh -c "'$NVMF_TGT' -m $TARGET_CORE_MASK > '$LOG' 2>&1 &"

	up=0
	for _ in $(seq 1 30); do
		if $RPC spdk_get_version >/dev/null 2>&1; then
			up=1
			break
		fi
		sleep 1
	done
	[ "$up" = 1 ] || kvtide_die "nvmf_tgt did not come up, see $LOG"
fi

# Provision idempotently, step by step, so a partially provisioned
# target (a failed first run, or a bare nvmf_tgt someone else started)
# is completed rather than skipped or double-created.
$RPC nvmf_create_transport -t TCP -u "${CONFIG_KVTIDE_TARGET_INCAPSULE}" \
	2>/dev/null || true
if ! $RPC bdev_get_bdevs -b KVMalloc0 >/dev/null 2>&1; then
	$RPC bdev_kvmalloc_create -b KVMalloc0
fi
if ! $RPC nvmf_get_subsystems | grep -q "\"$TARGET_NQN\""; then
	$RPC nvmf_create_subsystem "$TARGET_NQN" -a \
		-s KVTIDE0000000000001 -d KVTide_KV_Controller
	$RPC nvmf_subsystem_add_ns "$TARGET_NQN" KVMalloc0
	$RPC nvmf_subsystem_add_listener "$TARGET_NQN" -t tcp \
		-a "$TARGET_ADDR" -s "$TARGET_PORT"
fi
kvtide_log "target serving $TARGET_NQN on $TARGET_ADDR:$TARGET_PORT"

if want_kernel_attach; then
	if kv_dev >/dev/null; then
		kvtide_log "KV namespace already attached: $(kv_dev)"
	else
		kvtide_log "attaching kernel nvme-tcp initiator"
		sudo modprobe nvme_tcp
		sudo nvme connect -t tcp -n "$TARGET_NQN" \
			-a "$TARGET_ADDR" -s "$TARGET_PORT"
		sleep 2
		require_kv_dev
		kvtide_log "KV namespace attached: $(kv_dev)"
	fi
fi
