#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide target down: detach the kernel nvme-tcp initiator (if attached)
# and stop nvmf_tgt. Note the process kill is by name: do not run this
# on a host with an unrelated nvmf_tgt you care about.

. "$(dirname "$0")/lib.sh"

if kv_dev >/dev/null 2>&1; then
	kvtide_log "disconnecting $TARGET_NQN"
	sudo nvme disconnect -n "$TARGET_NQN"
fi

if pgrep -x nvmf_tgt >/dev/null 2>&1; then
	kvtide_log "stopping nvmf_tgt"
	sudo pkill -x nvmf_tgt
	# Wait for exit so an immediate target-up cannot race the dying
	# process for hugepages or the RPC socket.
	for _ in $(seq 1 10); do
		pgrep -x nvmf_tgt >/dev/null 2>&1 || break
		sleep 1
	done
fi

kvtide_log "target down"
