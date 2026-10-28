#!/usr/bin/env python3
"""Validate a smoke.sh receipt directory and write receipt.json.

PASS requires: the read digest equals the written digest with all bytes
read; the DS knfsd read counter rose by at least the object size; the
client sent READ RPCs to the DS and none to the MDS; no pNFS fallback
or Flex Files error event; no failed RPC to a lab peer; MDS READ and
WRITE op counters are zero; and teardown reported TEARDOWN CLEAN.
Usage: check_receipt.py <dir> <ds-ip> <mds-ip>
"""

import json
import sys
from pathlib import Path

d, ds_ip, mds_ip = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
size = 8 << 20
missing = []


def text(name):
    try:
        return (d / name).read_text()
    except OSError:
        missing.append(name)
        return ""


def io(name):
    t = text(name).split()
    return int(t[1]) if len(t) > 1 else 0


parts = text("client.txt").split()
want, got_sha, got = parts if len(parts) == 3 else ("", "-", "0")
ds_read = io("ds-io.after") - io("ds-io.before")
s = json.loads(
    text("trace-summary.json")
    or '{"requests_by_peer": {}, "events": {}, "failed_by_peer": {}}'
)
reqs = s["requests_by_peer"]
ds_reads = sum(
    v
    for p, r in reqs.items()
    if p.startswith(ds_ip + ":")
    for k, v in r.items()
    if k.endswith(" READ")
)
mds_reads = sum(
    v
    for p, r in reqs.items()
    if p.startswith(mds_ip + ":")
    for k, v in r.items()
    if k.endswith(" READ")
)
bad_events = {
    k: v
    for k, v in s["events"].items()
    if "fallback" in k or k.startswith("ff_layout_") or k == "nfs4_layouterror"
}
ops = {}
for line in text("mds-ops.txt").splitlines():
    k, v = line.rsplit(" ", 1)
    ops[k] = int(v)
mds_rw = ops.get('pnfs_mds_nfs_op{op="read"}', 0) + ops.get(
    'pnfs_mds_nfs_op{op="write"}', 0
)
teardown = "TEARDOWN CLEAN" in text("down.log")
checks = {
    "all_steps_ran": not missing,
    "digest_match": bool(want) and want == got_sha and int(got) == size,
    "ds_bytes_read": ds_read >= size,
    "ds_read_rpcs": ds_reads > 0,
    "no_mds_read_rpcs": mds_reads == 0,
    "no_fallback_or_ff_error": not bad_events,
    "no_failed_lab_rpc": not s["failed_by_peer"],
    "mds_proxy_ops_zero": mds_rw == 0,
    "teardown_clean": teardown,
}
receipt = {
    "result": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "ds_read_bytes": ds_read,
    "ds_read_rpcs": ds_reads,
    "bad_events": bad_events,
    "mds_read_write_ops": mds_rw,
    "missing_files": missing,
}
(d / "receipt.json").write_text(json.dumps(receipt, indent=1))
print(json.dumps(receipt, indent=1))
sys.exit(0 if receipt["result"] == "PASS" else 1)
