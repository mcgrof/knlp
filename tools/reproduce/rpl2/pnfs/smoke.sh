#!/bin/bash
# One-command functional smoke for the pNFS lab. Run as root after
# setup.sh. Brings up one MDS and one DS, writes one 8 MiB object, reads
# it back with O_DIRECT through a fresh mount, records the RPC trace and
# server counters, tears everything down and writes receipt.json, which
# check_receipt.py then validates. Exit status 0 only on PASS.
#   smoke.sh <receipt-dir>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
O=${1:?receipt dir}
[ -e "$O" ] && { echo "$O exists" >&2; exit 1; }
mkdir -p "$O"
L="$here/labctl.sh"
extra=$RUN/mds-extra.conf
mkdir -p "$RUN"
echo "admin_allowed_hosts = $IP_MDS,127.0.0.1" >"$extra"
# On any failure: stop tracing, tear the lab down and still write a FAIL
# receipt, so a failed smoke leaves nothing running and is parseable.
finish() {
	rc=$?
	trap - EXIT
	if [ "$rc" != 0 ]; then
		echo "smoke step failed (rc=$rc); tearing down" >&2
		[ -d "$TRACE_INSTANCE" ] && "$here/trace.sh" stop
		"$L" down >"$O/down.log" 2>&1 || true
		python3 "$here/check_receipt.py" "$O" "$(ds_ip 0)" "$IP_MDS" || true
		exit "$rc"
	fi
}
trap finish EXIT
"$L" up 1 "$extra" >"$O/up.log" 2>&1
"$here/trace.sh" start
"$L" remount >/dev/null
"$L" dsns 0 cat /proc/net/rpc/nfsd | grep '^io' >"$O/ds-io.before"
"$L" cli python3 -c '
import hashlib, mmap, os, sys
p = sys.argv[1] + "/smoke-object"
data = os.urandom(8 << 20)
with open(p, "wb") as f:
    f.write(data); f.flush(); os.fsync(f.fileno())
want = hashlib.sha256(data).hexdigest()
buf = mmap.mmap(-1, 8 << 20)
fd = os.open(p, os.O_RDONLY | os.O_DIRECT)
got = 0
while got < len(buf):
    n = os.preadv(fd, [memoryview(buf)[got:]], got)
    if n <= 0: break
    got += n
os.close(fd)
print(want, hashlib.sha256(buf).hexdigest(), got)
' "$CLI_MNT" >"$O/client.txt"
"$L" dsns 0 cat /proc/net/rpc/nfsd | grep '^io' >"$O/ds-io.after"
"$L" mdsns curl -s "http://$IP_MDS:9090/metrics" | grep -E '^pnfs_mds_nfs_op\{' >"$O/mds-ops.txt" || true
"$here/trace.sh" dump "$O/trace.txt"
"$here/trace.sh" stop
python3 "$here/trace_summary.py" "$O/trace.txt" "$NET." >"$O/trace-summary.json"
"$L" down >"$O/down.log" 2>&1 || true
python3 "$here/check_receipt.py" "$O" "$(ds_ip 0)" "$IP_MDS"
