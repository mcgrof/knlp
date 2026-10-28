#!/bin/bash
# Lattice pnfs-mds inside the MDS network namespace, with a private mount
# namespace holding its NFSv3 mounts of each task-owned data server.
# Arguments: <ds_count> [extra mds.conf lines file]
set -euo pipefail
. "$(dirname "$0")/env.sh"
nds=$1
extra=${2:-}
D=$RUN/mds
mkdir -p "$D"
for i in $(seq 0 $((nds - 1))); do
	mkdir -p "$D/ds$i"
	mount -t nfs -o vers=3,proto=tcp,port=2049,mountport=$DS_MOUNTD_PORT,mountproto=tcp,nolock,hard \
		"$(ds_ip "$i"):$(ds_export "$i")" "$D/ds$i"
done
findmnt -t nfs -o TARGET,SOURCE,OPTIONS

cat >"$D/rondb.conf" <<EOF
connect_string = 127.0.0.1:1186
schema_name = pnfs_mds
EOF
{
	cat <<EOF
mds_id = 1
hostname = $IP_MDS
nfs_port = 2049
grpc_port = 50051
catalogue_backend = rondb
catalogue_backend_conf = $D/rondb.conf
worker_threads = $MDS_WORKERS
inline_enabled = false
lease_time_sec = 90
grace_period_sec = 0
ds_heartbeat_ms = 5000
cluster_size = 1
cluster_bind_addr = $IP_MDS
inode_cache_size = 32768
log_file = $D/pnfs-mds.log
ds_count = $nds
EOF
	for i in $(seq 0 $((nds - 1))); do
		echo "ds[$i] = $(ds_ip "$i"):$(ds_export "$i")"
	done
	cat <<EOF
ds_mount_path_fmt = $D/ds%u
ds_transport = tcp
EOF
	[ -n "$extra" ] && cat "$extra"
} >"$D/mds.conf"
cat "$D/mds.conf"
sha256sum "$D/mds.conf" "$MDS_BIN"

cleanup() {
	set +e
	for i in $(seq 0 $((nds - 1))); do umount -l "$D/ds$i"; done
}
trap cleanup EXIT
export LD_LIBRARY_PATH=$LIBPATH
"$MDS_BIN" "$D/mds.conf"
