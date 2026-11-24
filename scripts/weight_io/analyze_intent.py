#!/usr/bin/env python3
"""Intent sizes against NVMe command sizes, per traced load.

Usage: analyze_intent.py OUT_DIR [MODEL_DIRS_JSON] > intent_summary.json

The model directories come from each model's meta.json (written by
trace_intent.py) unless a JSON map of name to directory is given.

Intent is what the loader asked the kernel for, inside the load window:
read-family calls on the shard files with their requested sizes, the
shard mappings, the demand faults and page-cache fills on the shard
inodes (whole process; the shards are touched only by the load), and
the io_uring request sizes. Device is every NVMe command that started
inside the window, with its size.
"""

import collections, json, os, sys
from pathlib import Path

READS = {"read", "pread64", "readv", "preadv", "preadv2"}


def shard_inodes(model_dir):
    out = {}
    for p in Path(model_dir).glob("*.safetensors"):
        st = os.stat(p)
        out[st.st_ino] = p.name
    return out


def device(path, t0, t1):
    cmds, sizes, n, b = {}, collections.Counter(), 0, 0
    for line in open(path):
        e = json.loads(line)
        if (
            e.get("event_type") == "nvme_cmd"
            and t0 <= e["ts"] <= t1
            and e["op_name"] == "read"
        ):
            sizes[e["bytes"]] += 1
            n += 1
            b += e["bytes"]
    return {"cmds": n, "bytes": b, "sizes": dict(sizes.most_common())}


def intent(sc_path, flt_path, bt_path, inos, t0, t1):
    rsz, rbytes, maps, other = collections.Counter(), 0, [], collections.Counter()
    for line in open(sc_path):
        e = json.loads(line)
        if e.get("type") != "call" or e["ts_exit_ns"] < t0 or e["ts_enter_ns"] > t1:
            continue
        on_shard = e.get("ino") in inos
        if e["name"] in READS and on_shard:
            rsz[e.get("count", 0)] += 1
            rbytes += max(e.get("ret", 0), 0)
        elif e["name"] == "mmap" and on_shard:
            maps.append(e.get("count"))
        elif e["name"] in ("madvise", "fadvise64", "io_uring_enter", "mmap", "munmap"):
            other[e["name"]] += 1
    faults = {
        "faults": 0,
        "fault_bytes": 0,
        "major": 0,
        "major_bytes": 0,
        "fill_bytes": 0,
    }
    if Path(flt_path).exists():
        for line in open(flt_path):
            e = json.loads(line)
            if e.get("ino") not in inos:
                continue
            if e.get("table") == "faults":
                for k in ("faults", "fault_bytes", "major", "major_bytes"):
                    faults[k] += e.get(k, 0)
            elif e.get("table") == "pgcache_fill":
                faults["fill_bytes"] += e.get("fill_bytes", 0)
    req = collections.Counter()
    for line in Path(bt_path).read_text().splitlines():
        if line.startswith("@req_size["):
            k, v = line[len("@req_size[") :].split("]:")
            req[int(k)] += int(v)
    return {
        "read_calls": sum(rsz.values()),
        "read_bytes": rbytes,
        "read_sizes": dict(rsz.most_common()),
        "shard_mmaps": maps,
        "other_calls": dict(other),
        "faults": faults,
        "uring_requests": sum(req.values()),
        "uring_sizes": dict(req.most_common()),
    }


def main(top, dirs=None):
    dirs = json.loads(Path(dirs).read_text()) if dirs else {}
    res = {}
    for md in sorted(p for p in Path(top).iterdir() if p.is_dir()):
        mdir = dirs.get(md.name) or json.loads((md / "meta.json").read_text())["dir"]
        inos = shard_inodes(mdir)
        rows = []
        for r in json.loads((md / "loads.json").read_text()):
            row = {
                k: r.get(k)
                for k in (
                    "loader",
                    "rep",
                    "load_s",
                    "verified",
                    "bytes",
                    "tensors",
                    "cold",
                    "rc",
                )
            }
            row["mismatched"] = len(r.get("mismatched") or [])
            if "t0_mono_ns" in r:
                rd = md / f"{r['loader']}-rep{r['rep']}"
                t0, t1 = r["t0_mono_ns"], r["t1_mono_ns"]
                row["intent"] = intent(
                    rd / "syscalls.jsonl",
                    rd / "faults.jsonl",
                    rd / "bpftrace.out",
                    inos,
                    t0,
                    t1,
                )
                row["device"] = device(rd / "device.jsonl", t0, t1)
            else:
                row["error"] = r.get("error", "")[-500:]
            rows.append(row)
        res[md.name] = rows
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
