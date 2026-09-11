#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Build the selected KVTide software-KV or physical-PCIe tools. Everything
# installs under KVTIDE_SRC_DIR; nothing touches system prefixes.

. "$(dirname "$0")/lib.sh"

JOBS=$(nproc)

if is_pcie; then
	if want_pcie_spdk; then
		kvtide_log "building standalone SPDK PCIe benchmark"
		( cd "$PCIE_SPDK_SRC" && \
		  ./configure --disable-tests --disable-unit-tests \
			--disable-examples && make -j"$JOBS" )
		test -x "$PCIE_SPDK_SRC/build/bin/spdk_nvme_perf" || \
			kvtide_die "standalone spdk_nvme_perf did not build"
	fi

	if want_pcie_upcie || want_pcie_linux; then
		MESON_SETUP=(meson setup)
		CUDA_OPT=disabled
		if want_pcie_cuda; then
			CUDA_OPT=enabled
			if ! command -v nvcc >/dev/null 2>&1; then
				[ -x /usr/local/cuda/bin/nvcc ] || \
					kvtide_die "CUDA is enabled but nvcc is unavailable"
				PATH=/usr/local/cuda/bin:$PATH
				export PATH
			fi
		fi
		if [ -f "$PCIE_XNVME_SRC/build-kvtide-pcie/meson-private/coredata.dat" ]; then
			MESON_SETUP+=(--reconfigure)
		fi
		kvtide_log "building xNVMe PCIe backends"
		( cd "$PCIE_XNVME_SRC" && \
			  "${MESON_SETUP[@]}" build-kvtide-pcie \
				-Dwith-liburing=enabled -Dwith-spdk=disabled \
				-Dwith-cuda="$CUDA_OPT" -Dwith-hip=disabled \
				-Dwith-libvfn=disabled -Dbe_upcie=true \
				-Dtests=false -Dexamples=false && \
			  meson compile -C build-kvtide-pcie )
		XNVMEPERF=$(pcie_xnvmeperf)
		[ -n "$XNVMEPERF" ] || kvtide_die "xnvmeperf did not build"
	fi

	if want_pcie_linux; then
		SRC="$KVTIDE_DIR/src/uring_nvm_perf.c"
		DST="$KVTIDE_SRC/uring_nvm_perf"
		if [ ! -x "$DST" ] || [ "$SRC" -nt "$DST" ]; then
			kvtide_log "building Linux fixed and premapped NVM benchmark"
			gcc -O2 -Wall -Wextra -Werror -pthread "$SRC" -luring -o "$DST"
		fi
	fi

	kvtide_log "physical PCIe build done"
	exit 0
fi

# SPDK: provides nvmf_tgt (the target) always, and spdk_nvme_perf (the
# userspace initiator) when the SPDK initiator is enabled. Flags match
# the validated build; tests/examples stay off for speed.
if [ ! -x "$SPDK_SRC/build/bin/nvmf_tgt" ]; then
	kvtide_log "building SPDK (this is the long one)"
	( cd "$SPDK_SRC" && \
	  ./configure --disable-tests --disable-unit-tests --disable-examples && \
	  make -j"$JOBS" )
else
	kvtide_log "SPDK already built"
fi
test -x "$SPDK_SRC/build/bin/nvmf_tgt" || kvtide_die "nvmf_tgt did not build"
if want_spdk; then
	test -x "$SPDK_SRC/build/bin/spdk_nvme_perf" || \
		kvtide_die "spdk_nvme_perf did not build"
fi

# The xNVMe library serves both the xNVMe initiator and the NIXL plugin.
if want_xnvme || want_nixl; then
	if [ ! -e "$XNVME_PREFIX" ]; then
		kvtide_log "building xNVMe"
		( cd "$XNVME_SRC" && \
		  meson setup build \
			-Dwith-liburing=enabled -Dwith-spdk=disabled \
			-Dwith-cuda=disabled -Dwith-hip=disabled \
			-Dwith-libvfn=disabled -Dprefix="$XNVME_PREFIX" && \
		  meson compile -C build && \
		  { meson install -C build || \
		    kvtide_log "xNVMe post-install extras failed (system" \
			"bash-completion dir); the prefix artifacts are" \
			"verified below"; } )
	else
		kvtide_log "xNVMe already installed at $XNVME_PREFIX"
	fi
	PKGDIR=$(xnvme_pkgconfig_dir)
	[ -n "$PKGDIR" ] || kvtide_die "xnvme.pc not found under $XNVME_PREFIX"
	if want_xnvme; then
		if [ ! -x "$KVTIDE_SRC/xnvme_kv_perf" ] || \
		   [ "$KVTIDE_DIR/src/xnvme_kv_perf.c" -nt "$KVTIDE_SRC/xnvme_kv_perf" ]; then
			kvtide_log "building xnvme_kv_perf"
			# rpath: the bench runs the tool under sudo, which
			# strips LD_LIBRARY_PATH, and libxnvme lives under the
			# harness's private prefix.
			# -luring -laio: when pkg-config resolves the static
			# libxnvme its private deps are not on the line.
			PKG_CONFIG_PATH="$PKGDIR" gcc -O2 \
				"$KVTIDE_DIR/src/xnvme_kv_perf.c" \
				$(PKG_CONFIG_PATH="$PKGDIR" pkg-config --cflags --libs xnvme) \
				-luring -laio \
				-Wl,-rpath,"$(dirname "$PKGDIR")" \
				-o "$KVTIDE_SRC/xnvme_kv_perf"
		else
			kvtide_log "xnvme_kv_perf up to date"
		fi
	fi
fi

if want_uring_fixed; then
	if [ ! -x "$KVTIDE_SRC/kv_uring_fixed" ] || \
	   [ "$KVTIDE_DIR/src/kv_uring_fixed.c" -nt "$KVTIDE_SRC/kv_uring_fixed" ]; then
		kvtide_log "building kv_uring_fixed"
		gcc -O2 "$KVTIDE_DIR/src/kv_uring_fixed.c" -luring \
			-o "$KVTIDE_SRC/kv_uring_fixed"
	else
		kvtide_log "kv_uring_fixed up to date"
	fi
fi

if want_nixl; then
	PKGDIR=$(xnvme_pkgconfig_dir)
	[ -n "$PKGDIR" ] || kvtide_die "NIXL plugin needs xNVMe installed first"
	kvtide_log "building NIXL with the XNVME_KV plugin"
	if [ ! -e "$NIXL_SRC/build" ]; then
		( cd "$NIXL_SRC" && \
		  PKG_CONFIG_PATH="$PKGDIR" meson setup build \
			--buildtype=debug -Dwerror=false \
			-Ddisable_plugins=UCX -Dbuild_tests=true )
	fi
	# ninja is a fast no-op when up to date and self-heals a build
	# that previously failed partway.
	ninja -C "$NIXL_SRC/build"
fi

kvtide_log "build done"
