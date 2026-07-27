#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide doctor: verify host prerequisites for the enabled components.
# Prints one PASS/FAIL line per check; exits non-zero on any FAIL.

. "$(dirname "$0")/lib.sh"

set +e
fails=0

check() {
	local desc=$1; shift
	if "$@" >/dev/null 2>&1; then
		printf 'PASS  %s\n' "$desc"
	else
		printf 'FAIL  %s\n' "$desc"
		fails=$(( fails + 1 ))
	fi
}

check "gcc"                  command -v gcc
check "g++"                  command -v g++
check "GNU make"             command -v make
check "git"                  command -v git
check "patch"                command -v patch
check "python3"              command -v python3
check "sudo"                 command -v sudo
check "taskset (util-linux)" command -v taskset

# knlp's config generation invokes bare python (python-is-python3 on
# Ubuntu; Debian usually ships the alias).
check "python (python-is-python3)" command -v python

# SPDK is always needed: it provides the target even in xnvme-only mode.
# The full canonical set is SPDK's scripts/pkgdep.sh; these are the ones
# that actually bit on a stock Ubuntu 24.04 server.
# meson/ninja are needed by EVERY SPDK build (the bundled DPDK submodule
# builds with meson), not just the xNVMe arm -- checking them only under
# want_xnvme let the spdk-only config pass doctor and then die at
# dpdkbuild with "meson: not found" (seen on stock Debian 13).
check "SPDK: meson (DPDK build)" command -v meson
check "SPDK: ninja (DPDK build)" command -v ninja
check "SPDK: python3-pyelftools" python3 -c 'import elftools'
check "SPDK: python3-tabulate"   python3 -c 'import tabulate'
check "SPDK: autotools (autoreconf)" command -v autoreconf
check "SPDK: libtool (libtoolize)"   command -v libtoolize
check "SPDK: libfuse3 dev"       pkg-config --exists fuse3
check "SPDK: ncurses dev"        pkg-config --exists ncurses
check "SPDK: libnuma (numa.h)"   test -e /usr/include/numa.h
check "SPDK: libssl dev"         pkg-config --exists openssl
check "SPDK: uuid dev"           pkg-config --exists uuid
check "SPDK: libaio (libaio.h)"  test -e /usr/include/libaio.h
# Debian installs nvme to /usr/sbin, outside a non-root PATH; the
# harness invokes it under sudo whose secure_path covers sbin.
check "nvme-cli"                 sh -c \
	'command -v nvme || test -x /usr/sbin/nvme'
check "column (bsdextrautils)"   command -v column

if want_xnvme; then
	check "pkg-config"           command -v pkg-config
	check "liburing dev"         pkg-config --exists liburing
	check "kernel nvme_tcp module" sh -c \
		'modinfo nvme_tcp 2>/dev/null || \
		 /usr/sbin/modinfo nvme_tcp 2>/dev/null || \
		 test -e "/lib/modules/$(uname -r)/kernel/drivers/nvme/host/nvme-tcp.ko" || \
		 test -d /sys/module/nvme_tcp'
fi

if want_nixl; then
	check "meson (nixl)"         command -v meson
	check "cmake"                command -v cmake
fi

if want_linux; then
	check "kernel build: bison"      command -v bison
	check "kernel build: flex"       command -v flex
	check "kernel build: bc"         command -v bc
	check "kernel build: libelf dev" sh -c \
		'test -e /usr/include/libelf.h || pkg-config --exists libelf'
	check "kernel build: zstd"       command -v zstd
fi

check "src dir parent writable" sh -c \
	"test -w \"\$(dirname \"$KVTIDE_SRC\")\" || test -w \"$KVTIDE_SRC\""
check "hugepage sysctl present" test -e /proc/sys/vm/nr_hugepages

if [ "$fails" -ne 0 ]; then
	echo "kvtide-doctor: $fails check(s) FAILED" >&2
	echo "kvtide-doctor: on Debian family, the kdevops knlp plugin" >&2
	echo "kvtide-doctor: provisions these (WORKFLOW_KNLP_KVTIDE=y)" >&2
	exit 1
fi
echo "kvtide-doctor: all checks passed"
