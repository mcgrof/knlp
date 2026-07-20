#!/bin/bash
# SPDX-License-Identifier: MIT
#
# KVTide build: SPDK target + enabled initiators. Everything installs
# under KVTIDE_SRC_DIR; nothing touches system prefixes.

. "$(dirname "$0")/lib.sh"

JOBS=$(nproc)

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
