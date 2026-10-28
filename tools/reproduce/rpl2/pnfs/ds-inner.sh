#!/bin/bash
# One NFSv3-only knfsd data server inside its own network namespace.
# Runs under `ip netns exec <prefix>-ds<i> unshare -m --propagation private`
# so the bind mounts below (private /run, /var/lib/nfs, /etc/exports and
# a per-netns nfsd filesystem) are invisible to the host. The host knfsd,
# its exports and its rpcbind registrations are never touched.
set -euo pipefail
. "$(dirname "$0")/env.sh"
i=$1
D=$RUN/ds$i
EXP=$(ds_export "$i")
mkdir -p "$D/run" "$D/varlibnfs/rpc_pipefs" "$D/varlibnfs/v4recovery" "$EXP"
chmod 0777 "$EXP"
mkdir -p "$EXP/data" && chmod 0777 "$EXP/data"

opts="rw,sync,no_subtree_check,no_root_squash,insecure,fsid=$((7700 + i))"
# Remote clients reach the DS through the host route (labctl.sh remote-up)
# with their own source address, so each is exported explicitly.
{
	echo "$EXP $NET.0/$NETMASK($opts)"
	for c in ${RPL2P_REMOTE_CLIENTS:-}; do echo "$EXP $c($opts)"; done
} >"$D/exports"
mount --bind "$D/run" /run
mount --bind "$D/varlibnfs" /var/lib/nfs
mount --bind "$D/exports" /etc/exports
mount -t nfsd nfsd /proc/fs/nfsd

pids=()
cleanup() {
	set +e
	rpc.nfsd 0
	exportfs -ua
	kill "${pids[@]}" 2>/dev/null
	wait
}
trap cleanup EXIT TERM INT

rpcbind -f -w &
pids+=($!)
for _ in $(seq 50); do [ -S /run/rpcbind.sock ] && break; sleep 0.1; done
exportfs -ra
rpc.mountd -F -p "$DS_MOUNTD_PORT" -N 4 &
pids+=($!)
rpc.nfsd -N 4 -N 4.0 -N 4.1 -N 4.2 -U "$DS_NFSD_THREADS"
echo "versions: $(cat /proc/fs/nfsd/versions)"
echo "threads: $(cat /proc/fs/nfsd/threads)"
exportfs -v
rpcinfo -p 127.0.0.1
echo READY
wait "${pids[1]}"
