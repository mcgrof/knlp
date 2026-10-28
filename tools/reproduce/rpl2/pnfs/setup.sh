#!/bin/bash
# Fetch and build everything the pNFS lab needs into RPL2P_ROOT, as an
# unprivileged user: no root, no system packages, nothing outside the root.
#
#   RPL2P_ROOT=/path/on/disk tools/reproduce/rpl2/pnfs/setup.sh
#
# Builds, at the pins below and with every pin checked:
#   - the Lattice pNFS metadata server (github.com/PEAK-AIO/pnfs-lattice)
#     with its RonDB catalogue, RonDB 26.02.4 and ntirpc v5.8;
#   - LMCache with the native file connector's held_fd_capacity option,
#     only its CPU extensions (NO_GPU_EXT=1).
# The LMCache build needs a Python with torch. Give it in RPL2P_PYTHON, or
# leave it unset and setup creates $R/venv with uv (CPU torch from the
# PyTorch wheel index plus LMCache's build requirements).
# The helpers are copied to $R/bin; every later command runs from there.
# RPL2P_CLIENT_ONLY=1 builds only LMCache, for a host that is only a
# remote client of a lab elsewhere.
# Needs gcc >= 13, cmake, ninja, git, wget, and the development files for
# krb5, openssl, liburcu and libbpf, plus a libnsl runtime.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
R=${RPL2P_ROOT:?set RPL2P_ROOT to an owned directory on a disk filesystem}
LATTICE_COMMIT=${RPL2P_LATTICE_COMMIT:-1e7d1f19667844ca6babd44f394e3694f35cac14}
NTIRPC_TAG=v5.8
NTIRPC_COMMIT=f6241a85a06ee66787182f15f60fd4a6751ef569
RONDB_URL=https://repo.hops.works/master/rondb-26.02.4-linux-glibc2.28-x86_64.tar.gz
RONDB_SHA=dfe8ef1346769568b1dad3cf7f66ed3884acc47cb05ac23a59a3dc0797a0e626
LMCACHE_REPO=${RPL2P_LMCACHE_REPO:-https://github.com/mcgrof/LMCache.git}
LMCACHE_REF=${RPL2P_LMCACHE_REF:-2026-10-28-feature-fs-native-held-fd}
LMCACHE_COMMIT=${RPL2P_LMCACHE_COMMIT:-3e4b3839ce47655e2d0d33cad3f66d4bf920d767}
JOBS=${RPL2P_JOBS:-8}

case $(stat -f -c %T "$R" 2>/dev/null || echo missing) in
tmpfs | ramfs | missing)
	echo "RPL2P_ROOT must exist on a disk filesystem" >&2
	exit 1
	;;
esac
mkdir -p "$R/src" "$R/deps" "$R/prefix" "$R/bin" "$R/run" "$R/storage"

# --- Lattice, ntirpc, RonDB -------------------------------------------
if [ "${RPL2P_CLIENT_ONLY:-0}" != 1 ]; then
if [ ! -d "$R/src/pnfs-lattice" ]; then
	git clone -q https://github.com/PEAK-AIO/pnfs-lattice.git "$R/src/pnfs-lattice"
fi
git -C "$R/src/pnfs-lattice" checkout -q "$LATTICE_COMMIT"
[ "$(git -C "$R/src/pnfs-lattice" rev-parse HEAD)" = "$LATTICE_COMMIT" ]
[ -z "$(git -C "$R/src/pnfs-lattice" status --porcelain)" ] || {
	echo "Lattice tree is not clean" >&2
	exit 1
}
if [ ! -d "$R/deps/ntirpc" ]; then
	git clone -q --depth 1 -b "$NTIRPC_TAG" https://github.com/nfs-ganesha/ntirpc.git "$R/deps/ntirpc"
fi
[ "$(git -C "$R/deps/ntirpc" rev-parse HEAD)" = "$NTIRPC_COMMIT" ]
tar=$R/deps/$(basename "$RONDB_URL")
[ -f "$tar" ] || wget -q -O "$tar" "$RONDB_URL"
echo "$RONDB_SHA  $tar" | sha256sum -c --quiet
dir=$R/deps/$(basename "$RONDB_URL" .tar.gz)
[ -d "$dir" ] || tar -C "$R/deps" -xzf "$tar"
ln -sfn "$dir" "$R/rondb"
nsl=$(ls /usr/lib/x86_64-linux-gnu/libnsl.so* /usr/lib64/libnsl.so* 2>/dev/null | head -1 || true)
[ -n "$nsl" ] || { echo "no libnsl runtime found" >&2; exit 1; }
if [ ! -f "$R/prefix/lib/libntirpc.so.5.8" ]; then
	cmake -S "$R/deps/ntirpc" -B "$R/deps/ntirpc/build" -G Ninja \
		-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX="$R/prefix" -DUSE_GSS=ON -DTIRPC_EPOLL=ON \
		-DNSL_LIBRARY="$nsl" >"$R/deps/ntirpc-cmake.log"
	ninja -C "$R/deps/ntirpc/build" -j "$JOBS" >"$R/deps/ntirpc-build.log"
	ninja -C "$R/deps/ntirpc/build" install >"$R/deps/ntirpc-install.log"
fi
cmake -S "$R/src/pnfs-lattice" -B "$R/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
	-DENABLE_RONDB=ON -DRonDB_ROOT="$R/rondb" -DCMAKE_PREFIX_PATH="$R/prefix" \
	-DCMAKE_POLICY_VERSION_MINIMUM=3.5 >"$R/lattice-cmake.log"
grep -q "RonDB catalogue backend enabled" "$R/lattice-cmake.log" || {
	echo "RonDB backend was not enabled; refusing a silent backend change" >&2
	exit 1
}
ninja -C "$R/build" -j "$JOBS" >"$R/lattice-build.log"
(cd "$R/build" && ctest -j "$JOBS" --timeout 120 >"$R/lattice-ctest.log" 2>&1) || {
	tail -20 "$R/lattice-ctest.log" >&2
	exit 1
}
grep "tests passed" "$R/lattice-ctest.log"
fi

# --- LMCache ------------------------------------------------------------
if [ -z "${RPL2P_PYTHON:-}" ]; then
	command -v uv >/dev/null || { echo "set RPL2P_PYTHON or install uv" >&2; exit 1; }
	if [ ! -x "$R/venv/bin/python" ]; then
		uv venv -q --python 3.12 "$R/venv"
		uv pip install -q --python "$R/venv/bin/python" torch numpy --index-url https://download.pytorch.org/whl/cpu
	fi
	RPL2P_PYTHON=$R/venv/bin/python
fi
if [ ! -d "$R/src/lmcache" ]; then
	git clone -q --branch "$LMCACHE_REF" "$LMCACHE_REPO" "$R/src/lmcache"
fi
git -C "$R/src/lmcache" fetch -q origin "$LMCACHE_REF" || true
git -C "$R/src/lmcache" checkout -q "$LMCACHE_COMMIT"
[ "$(git -C "$R/src/lmcache" rev-parse HEAD)" = "$LMCACHE_COMMIT" ]
if [ -x "$R/venv/bin/python" ] && [ "$RPL2P_PYTHON" = "$R/venv/bin/python" ]; then
	uv pip install -q --python "$RPL2P_PYTHON" -r "$R/src/lmcache/requirements/build.txt"
fi
"$RPL2P_PYTHON" -c 'import torch' || { echo "$RPL2P_PYTHON has no torch" >&2; exit 1; }
(cd "$R/src/lmcache" && env NO_GPU_EXT=1 MAX_JOBS="$JOBS" SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0 \
	"$RPL2P_PYTHON" setup.py build_ext --inplace >"$R/lmcache-build.log" 2>&1) || {
	tail -20 "$R/lmcache-build.log" >&2
	exit 1
}
ls "$R/src/lmcache"/lmcache/lmcache_fs*.so >/dev/null
echo "$RPL2P_PYTHON" >"$R/python"

cp -a "$here/." "$R/bin/"
bins=("$R/src/lmcache"/lmcache/lmcache_fs*.so)
[ "${RPL2P_CLIENT_ONLY:-0}" = 1 ] ||
	bins+=("$R/build/src/mds/pnfs-mds" "$R/prefix/lib/libntirpc.so.5.8" "$R/rondb/lib/libndbclient.so")
sha256sum "${bins[@]}" "$R/bin/"*.sh "$R/bin/"*.py >"$R/setup-hashes.txt"
echo "setup complete: $R (hashes in $R/setup-hashes.txt)"
