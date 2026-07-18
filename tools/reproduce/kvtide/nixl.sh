#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide NIXL stage: run the XNVME_KV plugin's mock-device unit tests
# (no hardware needed), then the full-agent integration test against the
# KVTide target's KV namespace.

. "$(dirname "$0")/lib.sh"

want_nixl || kvtide_die "CONFIG_KVTIDE_NIXL is not enabled"
test -e "$NIXL_SRC/build" || kvtide_die "NIXL not built -- run 'make kvtide-build'"

UNIT=$(find "$NIXL_SRC/build" -name xnvme_kv_backend_test -type f | head -1)
INTEG=$(find "$NIXL_SRC/build" -name nixl_xnvme_kv_test -type f | head -1)

test -n "$UNIT" || kvtide_die "xnvme_kv_backend_test not found in the build"

kvtide_log "running XNVME_KV unit tests (mock device)"
"$UNIT"

if [ -n "$INTEG" ]; then
	DEV=$(require_kv_dev)
	NIXL_LIB=$(find "$NIXL_SRC/build" -name 'libnixl*.so*' | head -1)
	[ -n "$NIXL_LIB" ] || kvtide_die "libnixl not found in the build"
	kvtide_log "running XNVME_KV integration test against $DEV"
	sudo env NIXL_XNVME_KV_DEVICE="$DEV" \
		LD_LIBRARY_PATH="$(dirname "$NIXL_LIB")" \
		"$INTEG"
else
	kvtide_log "integration test binary not found, skipping"
fi

kvtide_log "nixl stage done"
