#!/bin/bash
# Count the RPCs each arm sends: a fresh mount, then one first touch of
# each hot object and 100 repeat requests, traced by trace.sh and split
# by phase. Run as root on the client host after the lab and fixtures are
# up (for a remote client, after remote-client.sh up).
#   fingerprint.sh <out-dir> [client] [capacity]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
OUT=${1:?out dir}
who=${2:-cli}
cap=${3:-64}
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 1; }
mkdir -p "$OUT"
chown "$OWNER": "$OUT"
if [ "$who" = remote ]; then mnt=$RUN/rcli/mnt; else mnt=$CLI_MNT; fi
trap '[ -d "$TRACE_INSTANCE" ] && "$here/trace.sh" stop' EXIT
for held in 0 "$cap"; do
	"$here/trace.sh" start
	if [ "$who" = remote ]; then
		lab=$(cat "$RUN/rcli/lab-host")
		"$here/remote-client.sh" down >/dev/null
		"$here/remote-client.sh" up "$lab" >/dev/null
	else
		"$here/labctl.sh" remount >/dev/null
	fi
	"$here/run.sh" "$who" fingerprint --base "$mnt/rpl2-8m" --hot 8 --batch 1 --requests 100 \
		--held "$held" --client "$who" --out "$OUT/held$held.json"
	"$here/trace.sh" dump "$OUT/held$held.trace"
	"$here/trace.sh" stop
	python3 "$here/trace_summary.py" "$OUT/held$held.trace" "$NET." "$OUT/held$held.json" >"$OUT/held$held.summary.json"
done
python3 "$here/summarize.py" fingerprint "$OUT"
