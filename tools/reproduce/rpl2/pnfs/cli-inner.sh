#!/bin/bash
# Client holder: mounts the MDS over NFSv4.2 in the client network
# namespace and a private mount namespace, then sleeps so reader
# processes can join both namespaces with nsenter -t <pid> -n -m.
# Argument: mount options (default below).
set -euo pipefail
. "$(dirname "$0")/env.sh"
opts=${1:-vers=4.2,sec=sys,hard,proto=tcp,rsize=1048576,wsize=1048576}
mkdir -p "$CLI_MNT"
cleanup() {
	set +e
	umount "$CLI_MNT" || umount -l "$CLI_MNT"
}
trap cleanup EXIT TERM INT
mount -t nfs4 -o "$opts" "$IP_MDS:/" "$CLI_MNT"
findmnt -o TARGET,SOURCE,OPTIONS "$CLI_MNT"
echo $$ >"$RUN/$CLI/holder.pid"
echo READY
sleep infinity &
wait $!
