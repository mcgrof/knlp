#!/bin/bash
# Task-owned tracefs instance for client-side pNFS attribution. Uses
# /sys/kernel/tracing/instances/<prefix> only; global tracing is untouched.
#   trace.sh start            enable the event set, clear the buffer
#   trace.sh dump <file>      copy the buffer to <file>
#   trace.sh stop             disable and remove the instance
# Run as root.
set -euo pipefail
. "$(dirname "$0")/env.sh"
T=$TRACE_INSTANCE
EVENTS=(
	sunrpc/rpc_clnt_new sunrpc/rpc_request sunrpc/rpc_task_end
	nfs4/nfs4_open_file nfs4/nfs4_cached_open nfs4/nfs4_close
	nfs4/nfs4_layoutget nfs4/nfs4_layoutreturn nfs4/nfs4_layoutreturn_on_close
	nfs4/nfs4_getdeviceinfo nfs4/pnfs_update_layout nfs4/pnfs_ds_connect
	nfs4/nfs4_pnfs_read nfs4/ff_layout_read_error nfs4/nfs4_layouterror
	nfs4/nfs4_cb_layoutrecall_file
	nfs4/pnfs_mds_fallback_pg_init_read nfs4/pnfs_mds_fallback_read_done
	nfs4/pnfs_mds_fallback_read_pagelist nfs4/pnfs_mds_fallback_pg_init_write
	nfs/nfs_initiate_read nfs/nfs_readpage_done
	nfs4/nfs4_set_delegation nfs4/nfs4_delegreturn nfs4/nfs_delegation_need_return
)
case $1 in
start)
	mkdir -p "$T"
	echo 0 >"$T/tracing_on"
	echo 65536 >"$T/buffer_size_kb"
	for e in "${EVENTS[@]}"; do echo 1 >"$T/events/$e/enable"; done
	: >"$T/trace"
	# CLOCK_MONOTONIC timestamps, so measurement processes can split the
	# trace by their own time.monotonic_ns() phase windows.
	echo mono >"$T/trace_clock"
	echo 1 >"$T/tracing_on"
	;;
dump) cat "$T/trace" >"$2" ;;
stop)
	echo 0 >"$T/tracing_on"
	for e in "${EVENTS[@]}"; do echo 0 >"$T/events/$e/enable"; done
	rmdir "$T"
	;;
esac
