#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide fetch: bring the pinned sources into KVTIDE_SRC_DIR. Each
# component is prepared in a .tmp directory and only moved into place
# once complete, so an interrupted fetch is retried instead of being
# mistaken for a finished one. A component already in place is left
# alone (re-fetch after changing a pin: remove the directory first).

. "$(dirname "$0")/lib.sh"

mkdir -p "$KVTIDE_SRC"

# SPDK is always needed (it provides the target). The whole KV stack
# plus its base rides on a single Gerrit change ref, so no full clone
# is required: init an empty repo and fetch the ref, shallow when the
# server allows it.
if [ ! -e "$SPDK_SRC" ]; then
	TMP="$SPDK_SRC.tmp"
	rm -rf "$TMP"
	kvtide_log "fetching SPDK ${CONFIG_KVTIDE_SPDK_REF} from ${CONFIG_KVTIDE_SPDK_GIT}"
	mkdir -p "$TMP"
	git -C "$TMP" init -q
	if ! git -C "$TMP" fetch --depth=1 \
		"${CONFIG_KVTIDE_SPDK_GIT}" "${CONFIG_KVTIDE_SPDK_REF}"; then
		kvtide_log "shallow fetch refused, retrying full fetch"
		git -C "$TMP" fetch \
			"${CONFIG_KVTIDE_SPDK_GIT}" "${CONFIG_KVTIDE_SPDK_REF}"
	fi
	git -C "$TMP" checkout -q -B kvtide FETCH_HEAD
	git -C "$TMP" submodule update --init --depth 1
	if [ "${CONFIG_KVTIDE_SPDK_LOCAL_FIXES:-n}" = y ]; then
		kvtide_log "applying spdk-kv-stack-local-fixes.patch"
		git -C "$TMP" apply \
			"$KVTIDE_DIR/patches/spdk-kv-stack-local-fixes.patch"
	fi
	mv "$TMP" "$SPDK_SRC"
else
	kvtide_log "SPDK present at $SPDK_SRC, leaving as-is"
fi

# The xNVMe library is needed by the xNVMe initiator and by the NIXL
# XNVME_KV plugin.
if want_xnvme || want_nixl; then
	if [ ! -e "$XNVME_SRC" ]; then
		TMP="$XNVME_SRC.tmp"
		rm -rf "$TMP"
		kvtide_log "cloning xNVMe ${CONFIG_KVTIDE_XNVME_REF}"
		git clone "${CONFIG_KVTIDE_XNVME_GIT}" "$TMP"
		git -C "$TMP" checkout -q "${CONFIG_KVTIDE_XNVME_REF}"
		if [ -f "$TMP/.gitmodules" ]; then
			git -C "$TMP" submodule update --init --recursive
		fi
		mv "$TMP" "$XNVME_SRC"
	else
		kvtide_log "xNVMe present at $XNVME_SRC, leaving as-is"
	fi
fi

if want_nixl; then
	if [ ! -e "$NIXL_SRC" ]; then
		TMP="$NIXL_SRC.tmp"
		rm -rf "$TMP"
		kvtide_log "cloning NIXL ${CONFIG_KVTIDE_NIXL_REF}"
		git clone --depth 1 --branch "${CONFIG_KVTIDE_NIXL_REF}" \
			"${CONFIG_KVTIDE_NIXL_GIT}" "$TMP"
		mv "$TMP" "$NIXL_SRC"
	else
		kvtide_log "NIXL present at $NIXL_SRC, leaving as-is"
	fi
fi

kvtide_log "fetch done"
