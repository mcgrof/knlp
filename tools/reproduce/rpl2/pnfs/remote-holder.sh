#!/bin/bash
# Mount holder for remote-client.sh, run under systemd-run and unshare -m.
# Arguments: <mountpoint> <mds-address> <mount options> <pid file>
set -euo pipefail
mnt=$1 mds=$2 opts=$3 pidfile=$4
cleanup() { umount "$mnt" || umount -l "$mnt"; }
trap cleanup EXIT TERM INT
mount -t nfs4 -o "$opts" "$mds:/" "$mnt"
findmnt -o TARGET,SOURCE,OPTIONS "$mnt"
echo "$BASHPID" >"$pidfile"
echo READY
sleep infinity &
wait $!
