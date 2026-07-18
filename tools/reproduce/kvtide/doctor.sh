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

# SPDK is always needed: it provides the target even in xnvme-only mode.
check "SPDK: python3-pyelftools" python3 -c 'import elftools'
check "SPDK: libnuma (numa.h)"   test -e /usr/include/numa.h
check "SPDK: libssl dev"         pkg-config --exists openssl
check "SPDK: uuid dev"           pkg-config --exists uuid
check "SPDK: libaio (libaio.h)"  test -e /usr/include/libaio.h
check "nvme-cli"                 command -v nvme
check "column (bsdextrautils)"   command -v column

if want_xnvme; then
	check "meson"                command -v meson
	check "ninja"                command -v ninja
	check "pkg-config"           command -v pkg-config
	check "liburing dev"         pkg-config --exists liburing
	check "kernel nvme_tcp module" sh -c \
		'modinfo nvme_tcp || test -d /sys/module/nvme_tcp'
fi

if want_nixl; then
	check "meson (nixl)"         command -v meson
	check "cmake"                command -v cmake
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
