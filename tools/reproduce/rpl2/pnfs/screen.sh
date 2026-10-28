#!/bin/bash
# Paired screen: stock LMCache (held_fd_capacity 0) against held
# descriptors, in alternating blocks, one client, one request at a time.
# Run as root after labctl.sh up and fixture.sh (or remote-client.sh up).
#   screen.sh <out-dir> [client] [blocks] [requests] [capacity]
# client: cli (default) or remote. Shapes: b1 = one 8 MiB object per
# request over 8 hot objects; b16 = sixteen distinct 8 MiB objects per
# request. Every 8th request is verified by digest.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
OUT=${1:?out dir}
who=${2:-cli}
blocks=${3:-3}
reqs=${4:-200}
cap=${5:-64}
[ -e "$OUT" ] && { echo "$OUT exists; runs are preserved" >&2; exit 1; }
mkdir -p "$OUT"
chown "$OWNER": "$OUT"
if [ "$who" = remote ]; then mnt=$RUN/rcli/mnt; else mnt=$CLI_MNT; fi
for block in $(seq 1 "$blocks"); do
	if [ $((block % 2)) = 1 ]; then order=(0 "$cap"); else order=("$cap" 0); fi
	for shape in b1 b16; do
		case $shape in b1) hot=8 batch=1 ;; b16) hot=16 batch=16 ;; esac
		for held in "${order[@]}"; do
			"$here/run.sh" "$who" read --base "$mnt/rpl2-8m" --hot "$hot" --batch "$batch" \
				--requests "$reqs" --held "$held" --block "$block" --client "$who" \
				--out "$OUT/$shape-held$held-k$block.json" | tee -a "$OUT/screen.log"
		done
	done
done
