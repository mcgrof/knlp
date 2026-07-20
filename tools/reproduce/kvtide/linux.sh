#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide kernel-under-test stage: fetch, configure, build and install
# the configured Linux tree/branch, then gate the pipeline on actually
# running it. First run: build + install + a request for one reboot,
# exiting non-zero so make stops; after the reboot the gate passes and
# the pipeline continues on the kernel under test.
#
# Meant for disposable bench hosts (kdevops QEMU guests): this installs
# a kernel and updates the bootloader.

. "$(dirname "$0")/lib.sh"

if ! want_linux; then
	kvtide_log "no kernel under test configured, skipping"
	exit 0
fi

TREE=${CONFIG_KVTIDE_LINUX_TREE:-}
BRANCH=${CONFIG_KVTIDE_LINUX_BRANCH:-}
if [ -z "$TREE" ] || [ -z "$BRANCH" ]; then
	kvtide_die "KVTIDE_LINUX_TREE/BRANCH unset in .config -- install \
python3 kconfiglib and re-run make defconfig-kvtide-ab-linux (the \
defconfig omits them so olddefconfig can fill in defaults/overrides)"
fi

# Installing a kernel is always fine in a VM (the host is disposable);
# on bare metal it changes what the real machine boots, so require the
# explicit opt-in. A missing systemd-detect-virt reads as bare metal.
VIRT=$(systemd-detect-virt 2>/dev/null) || VIRT=none
if [ "${VIRT:-none}" = none ] && \
   [ "${CONFIG_KVTIDE_LINUX_ALLOW_BAREMETAL:-n}" != y ]; then
	kvtide_die "this host looks like bare metal and the kernel stage \
would install a kernel and update the bootloader -- use \
'make defconfig-kvtide-ab-linux-baremetal' (or set \
CONFIG_KVTIDE_LINUX_ALLOW_BAREMETAL=y) to allow that"
fi

LINUX_SRC="$KVTIDE_SRC/linux"

if [ ! -e "$LINUX_SRC" ]; then
	TMP="$LINUX_SRC.tmp"
	rm -rf "$TMP"
	kvtide_log "cloning $TREE ($BRANCH, shallow)"
	git clone --depth 1 --branch "$BRANCH" "$TREE" "$TMP"
	mv "$TMP" "$LINUX_SRC"
else
	kvtide_log "linux present at $LINUX_SRC, leaving as-is"
	kvtide_log "(re-fetch after changing the tree/branch: rm -rf it first)"
fi

cd "$LINUX_SRC"

# Configure once: base on the running kernel's config, drop the distro
# signing keys and heavy debug info for a fast bench build, then apply
# the pool knob.
if [ ! -f .config ]; then
	if [ -f "/boot/config-$(uname -r)" ]; then
		cp "/boot/config-$(uname -r)" .config
	elif [ -f /proc/config.gz ]; then
		zcat /proc/config.gz > .config
	else
		kvtide_die "no base kernel config found on this host"
	fi
	# The -kvtide localversion makes the gate unambiguous: a host may
	# already run a same-version kernel from an earlier build of the
	# same branch, which would otherwise falsely pass the release
	# comparison.
	./scripts/config --set-str SYSTEM_TRUSTED_KEYS "" \
		--set-str SYSTEM_REVOCATION_KEYS "" \
		--set-str LOCALVERSION "-kvtide" \
		--disable DEBUG_INFO_BTF \
		--disable DEBUG_INFO \
		--enable DEBUG_INFO_NONE
	if [ "${CONFIG_KVTIDE_LINUX_ENABLE_BLK_IOBUF_POOL:-n}" = y ]; then
		./scripts/config --enable BLK_IOBUF_POOL
	fi
	make olddefconfig
	if [ "${CONFIG_KVTIDE_LINUX_ENABLE_BLK_IOBUF_POOL:-n}" = y ] && \
	   ! grep -q '^CONFIG_BLK_IOBUF_POOL=y' .config; then
		kvtide_log "WARNING: CONFIG_BLK_IOBUF_POOL not available in $BRANCH"
	fi
fi

REL=$(make -s kernelrelease)
if [ "$(uname -r)" = "$REL" ]; then
	kvtide_log "running the kernel under test ($REL), gate passes"
	exit 0
fi

kvtide_log "building $REL"
make -j"$(nproc)"
kvtide_log "installing $REL"
sudo make modules_install install
if command -v update-grub >/dev/null 2>&1; then
	sudo update-grub
fi

cat <<EOF
kvtide: ==============================================================
kvtide: kernel under test installed: $REL
kvtide: currently running:           $(uname -r)
kvtide: reboot into $REL, then re-run 'make' -- the gate passes and
kvtide: the pipeline resumes (target-up, bench, report) on it.
kvtide: ==============================================================
EOF
exit 1
