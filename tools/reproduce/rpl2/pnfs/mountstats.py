#!/usr/bin/env python3
"""Per-operation NFS client counters for one mount, without root.

The kernel keeps, for every NFS mount, a count of each RPC procedure the
client sent and its summed round-trip time. They are in
/proc/self/mountstats, readable by any user. Taking a snapshot before
and after a phase gives the RPCs that phase sent, for example OPEN,
CLOSE, GETATTR and LAYOUTGET per object read. The counters are per
mount, not per process: anything else using the same mount at the same
time is counted too, so a delta is only exact on a mount the run has to
itself.

  mountstats.py <path>          print the mount's header and counters
"""

import json
import os
import sys


def _mount_point_of(path):
    path = os.path.realpath(path)
    while not os.path.ismount(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def parse(text):
    """Return {mount point: {"fstype", "header", "ops": {op: [fields]}}}."""
    mounts, cur, in_ops = {}, None, False
    for line in text.splitlines():
        if line.startswith("device "):
            parts = line.split()
            # device <src> mounted on <mnt> with fstype <type> [statvers=..]
            mnt = parts[4]
            fstype = parts[7] if len(parts) > 7 else ""
            cur = {"fstype": fstype, "header": [line.strip()], "ops": {}}
            mounts[mnt] = cur
            in_ops = False
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith("per-op statistics"):
            in_ops = True
            continue
        if in_ops and ":" in s:
            op, rest = s.split(":", 1)
            nums = rest.split()
            if nums and all(n.isdigit() for n in nums):
                cur["ops"][op] = [int(n) for n in nums]
                continue
        if not in_ops and s.split(":")[0] in ("opts", "caps", "nfsv4", "sec", "pnfs"):
            cur["header"].append(s)
    return mounts


def snapshot(path):
    """Counters of the NFS mount holding `path`, or None if it is not NFS."""
    mnt = _mount_point_of(path)
    try:
        text = open("/proc/self/mountstats").read()
    except OSError:
        return None
    m = parse(text).get(mnt)
    if not m or not m["fstype"].startswith("nfs"):
        return None
    return {"mount": mnt, **m}


def delta(before, after):
    """Calls and summed RTT (ms) per op between two snapshots, zeros dropped."""
    if not before or not after:
        return None
    out = {}
    for op, a in after["ops"].items():
        b = before["ops"].get(op, [0] * len(a))
        calls = a[0] - b[0]
        if calls:
            # fields: ops, transmissions, timeouts, bytes sent, bytes
            # received, queue ms, RTT ms, execute ms[, errors]
            rtt = a[6] - b[6] if len(a) > 6 else None
            out[op] = {"calls": calls, "rtt_ms": rtt}
    return out


def main():
    s = snapshot(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps(s, indent=1) if s else "not an NFS mount")


if __name__ == "__main__":
    main()
