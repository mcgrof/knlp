#!/bin/bash
# Seal the shared fixtures once per lab: <mount>/rpl2-8m (8 MiB objects,
# direct I/O) and <mount>/rpl2-256k (256 KiB objects; on NFS mounts with a
# 1 MiB block size these are read buffered, which isolates the metadata
# path). Run as root after labctl.sh up.
#   fixture.sh [size ...]      default: 8388608 262144
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
sizes=${*:-8388608 262144}
for s in $sizes; do
	sub=rpl2-$((s >> 10))k
	[ "$s" -ge 1048576 ] && sub=rpl2-$((s >> 20))m
	if "$here/labctl.sh" cli test -e "$CLI_MNT/$sub/RPL2-MANIFEST.json"; then
		echo "fixture $sub already sealed"
		continue
	fi
	"$here/run.sh" cli init --base "$CLI_MNT/$sub" --size "$s" --count "${RPL2P_OBJECTS:-64}" --seed "${RPL2P_SEED:-1}"
done
