#!/usr/bin/env python3
"""Summarize a trace.sh dump for the task-owned pNFS clients only.

RPC client ids are bound to peers through sunrpc:rpc_clnt_new, so the
trace must start before the mount (labctl.sh remount). Requests from any
other client id on the host, such as unrelated NFS mounts, are counted
separately and excluded from the per-peer table.

Prints JSON: per peer (program/version/procedure counts and failed task
ends), pNFS events, fallback events and bytes requested per peer from
nfs_initiate_read.

Usage: trace_summary.py <trace> [subnet-prefix] [windows.json]
windows.json is a native_reader.py fingerprint record; with it, counts are split
into its "<phase> begin" .. "<phase> end" windows. The trace instance
uses the mono clock, so its timestamps are CLOCK_MONOTONIC seconds.
"""

import collections
import json
import re
import sys

NET = sys.argv[2] if len(sys.argv) > 2 else "10.247.0."
WIN = []
if len(sys.argv) > 3:
    marks = json.load(open(sys.argv[3]))["windows"]
    begins = {}
    for m in marks:
        ph, what = m["mark"].split()[1:3]
        if what == "begin":
            begins[ph] = m["t_ns"] / 1e9
        else:
            WIN.append((ph, begins[ph], m["t_ns"] / 1e9))
ts_re = re.compile(r"\s(\d+\.\d+):\s")
phase_req = collections.defaultdict(
    lambda: collections.defaultdict(collections.Counter)
)
phase_ev = collections.defaultdict(collections.Counter)


def phase_of(line):
    if not WIN:
        return None
    m = ts_re.search(line)
    if not m:
        return None
    t = float(m.group(1))
    for ph, lo, hi in WIN:
        if lo <= t <= hi:
            return ph
    return None


clnt_re = re.compile(
    r"rpc_clnt_new: client=([0-9a-f]+) peer=\[([^\]]+)\]:(\d+) program=(\S+)"
)
req_re = re.compile(r"rpc_request: task:([0-9a-f]+)@([0-9a-f]+) (\S+)v(\d+) (\S+)")
end_re = re.compile(r"rpc_task_end: task:([0-9a-f]+)@([0-9a-f]+) .* status=(-?\d+)")
ev_re = re.compile(
    r"(nfs4_\w+|pnfs_\w+|ff_layout_\w+|nfs_initiate_read|nfs_readpage_done): (.*)"
)

peers = {}
tasks = {}
per_peer = collections.defaultdict(collections.Counter)
failed = collections.defaultdict(collections.Counter)
events = collections.Counter()
event_examples = {}
foreign = collections.Counter()
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    m = clnt_re.search(line)
    if m:
        cid, addr, port, prog = m.groups()
        peers[cid] = f"{addr}:{port}/{prog}"
        continue
    m = req_re.search(line)
    if m:
        task, cid, prog, ver, proc = m.groups()
        key = f"{prog}v{ver} {proc}"
        if cid in peers and peers[cid].startswith(NET):
            per_peer[peers[cid]][key] += 1
            tasks[(task, cid)] = key
            ph = phase_of(line)
            if ph:
                phase_req[ph][peers[cid]][key] += 1
        else:
            foreign[key] += 1
        continue
    m = end_re.search(line)
    if m:
        task, cid, status = m.groups()
        if (task, cid) in tasks and int(status) < 0:
            failed[peers[cid]][f"{tasks[(task, cid)]} status={status}"] += 1
        continue
    m = ev_re.search(line)
    if m:
        name, rest = m.groups()
        events[name] += 1
        event_examples.setdefault(name, rest[:240])
        ph = phase_of(line)
        if ph:
            phase_ev[ph][name] += 1

print(
    json.dumps(
        {
            "peers": peers,
            "requests_by_peer": {k: dict(v) for k, v in per_peer.items()},
            "failed_by_peer": {k: dict(v) for k, v in failed.items()},
            "events": dict(events),
            "event_examples": event_examples,
            "foreign_requests": sum(foreign.values()),
            "phases": {
                ph: {
                    "requests_by_peer": {k: dict(v) for k, v in phase_req[ph].items()},
                    "events": dict(phase_ev[ph]),
                }
                for ph, _, _ in WIN
            },
        },
        indent=1,
        sort_keys=True,
    )
)
