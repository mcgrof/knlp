#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide fetch: bring the pinned sources into KVTIDE_SRC_DIR. Each
# component is prepared in a .tmp directory and only moved into place
# once complete, so an interrupted fetch is retried instead of being
# mistaken for a finished one. A component already in place is left
# alone in software-KV mode. Physical mode refreshes its configured
# upstream refs only when the harness-owned checkouts are clean.

. "$(dirname "$0")/lib.sh"

mkdir -p "$KVTIDE_SRC"

fetch_branch_or_ref() {
	local url=$1 ref=$2 dst=$3 tmp

	if [ -e "$dst" ]; then
		[ -d "$dst/.git" ] || kvtide_die "$dst is not a git checkout"
		[ -z "$(git -C "$dst" status --porcelain --untracked-files=no)" ] || \
			kvtide_die "refusing to refresh modified source at $dst"
		git -C "$dst" remote set-url origin "$url"
		git -C "$dst" fetch origin "$ref"
		git -C "$dst" checkout -q --detach FETCH_HEAD
		if [ -f "$dst/.gitmodules" ]; then
			git -C "$dst" submodule sync --recursive
			git -C "$dst" submodule update --init --recursive
		fi
		kvtide_log "resolved $ref to $(git -C "$dst" rev-parse HEAD)"
		return
	fi
	tmp="$dst.tmp"
	rm -rf "$tmp"
	mkdir -p "$tmp"
	git -C "$tmp" init -q
	git -C "$tmp" remote add origin "$url"
	if ! git -C "$tmp" fetch --depth=1 origin "$ref"; then
		kvtide_log "shallow fetch refused for $ref, retrying full fetch"
		git -C "$tmp" fetch origin "$ref"
	fi
	git -C "$tmp" checkout -q --detach FETCH_HEAD
	if [ -f "$tmp/.gitmodules" ]; then
		git -C "$tmp" submodule update --init --recursive --depth 1
	fi
	mv "$tmp" "$dst"
	kvtide_log "resolved $ref to $(git -C "$dst" rev-parse HEAD)"
}

if is_pcie; then
	if want_pcie_spdk; then
		fetch_branch_or_ref "${CONFIG_KVTIDE_PCIE_SPDK_GIT}" \
			"${CONFIG_KVTIDE_PCIE_SPDK_REF}" "$PCIE_SPDK_SRC"
	fi
	if want_pcie_upcie || want_pcie_linux; then
		fetch_branch_or_ref "${CONFIG_KVTIDE_PCIE_XNVME_GIT}" \
			"${CONFIG_KVTIDE_PCIE_XNVME_REF}" "$PCIE_XNVME_SRC"
	fi
	kvtide_log "physical PCIe source fetch done"
	exit 0
fi

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
