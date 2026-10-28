#!/usr/bin/env python3
"""Run a range replay under the NVMe tracepoint monitor and attribute commands.

The wrapper starts `nvme_tp_monitor` on the block device that holds the
target file, waits until it reports attachment, runs `replay_ranges.py`
with the given arguments, waits for completions to drain, stops the
tracer and joins its command stream to the replay's device sector ranges.

Attribution is by logical block address and time window, not by PID:
readahead, writeback and other tenants of the same device appear in the
trace, so every command inside a trial window is classified as

  target      overlaps a sector range of the selected tensor bytes
  file_other  inside the target file's extents but outside the selection
              (readahead or neighbouring tensors)
  other       elsewhere on the device (unattributed; never hidden)

Only read commands are attributed to the replay; writes are reported as
a separate count. Command size histograms are given both count-weighted
and byte-weighted. Latency uses the matched completion event when the
tracer provides one for the same hardware queue and command id.

The tracer and Python both use CLOCK_MONOTONIC; the clock anchor emitted
by the tracer is checked against Python's reading at start.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def start_tracer(tracer: str, disk: str, jsonl: str, dur: int):
    cmd = ["sudo", tracer, "--disk", disk, "--dur", str(dur), "--jsonl", jsonl]
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    t_py = None
    for _ in range(200):
        line = p.stdout.readline()
        if not line:
            break
        if "attached" in line:
            t_py = time.monotonic_ns()
            break
    if t_py is None:
        p.kill()
        raise RuntimeError("tracer did not report attachment")
    return p, t_py


def stop_tracer(p):
    # the tracer runs as root under sudo; send SIGINT to the process group leader via sudo
    subprocess.run(["sudo", "kill", "-INT", str(p.pid)], check=False)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "kill", "-TERM", str(p.pid)], check=False)
        p.wait(timeout=10)
    return p.stdout.read()


def load_events(path: str):
    """Load command events; attach each completion to its command in stream order.

    Completions carry only (hwq, cid), which the driver reuses, so a command
    is matched to the first later completion on the same queue and id.
    """
    cmds, meta = [], {}
    pending: dict[tuple, list] = {}
    n_cmp = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            t = e.get("event_type")
            if t == "nvme_cmd":
                cmds.append(e)
                pending.setdefault((e.get("hwq"), e.get("cid")), []).append(e)
            elif t == "nvme_cmp":
                n_cmp += 1
                q = pending.get((e.get("hwq"), e.get("cid")))
                if q:
                    c = q.pop(0)
                    c["lat_ns"] = e.get("lat_ns", e["ts"] - c["ts"])
                    c["status"] = e.get("status")
            else:
                meta[t] = e
    unmatched = sum(len(q) for q in pending.values())
    meta["_match"] = {"n_cmp": n_cmp, "unmatched_cmds": unmatched}
    return cmds, meta


def classify(cmds, target_lbas, file_lbas, windows):
    """Attribute read commands to windows and sector classes."""
    tl = sorted(target_lbas)
    fl = sorted(file_lbas)
    t_starts = [a for a, _ in tl]
    f_starts = [a for a, _ in fl]

    def overlaps(sorted_ranges, starts, slba, nlb):
        i = bisect.bisect_right(starts, slba + nlb - 1)
        for j in range(max(0, i - 2), i):
            a, n = sorted_ranges[j]
            if a < slba + nlb and slba < a + n:
                return True
        return False

    out = []
    for w in windows:
        t0, t1 = w["t_start_ns"], w["t_drain_end_ns"]
        acc = {
            k: {"cmds": 0, "bytes": 0, "sizes": {}}
            for k in ("target", "file_other", "other")
        }
        writes = {"cmds": 0, "bytes": 0}
        lats = {"target": []}
        bad_status = 0
        for c in cmds:
            ts = c["ts"]
            if ts < t0 or ts > t1:
                continue
            if c.get("op_name") != "read":
                if c.get("op_name") == "write":
                    writes["cmds"] += 1
                    writes["bytes"] += c["bytes"]
                continue
            slba, nlb = c["slba"], c["nlb"]
            if overlaps(tl, t_starts, slba, nlb):
                k = "target"
            elif overlaps(fl, f_starts, slba, nlb):
                k = "file_other"
            else:
                k = "other"
            a = acc[k]
            a["cmds"] += 1
            a["bytes"] += c["bytes"]
            a["sizes"][c["bytes"]] = a["sizes"].get(c["bytes"], 0) + 1
            if k == "target":
                if "lat_ns" in c:
                    lats["target"].append(c["lat_ns"])
                if c.get("status") not in (0, None):
                    bad_status += 1
        l = sorted(lats["target"])

        def pct(p):
            if not l:
                return None
            return l[min(len(l) - 1, int(round(p / 100 * (len(l) - 1))))]

        out.append(
            {
                "trial": w["trial"],
                "elapsed_ns": w["elapsed_ns"],
                "F_bytes": w["F_bytes"],
                "F_kind": w["F_kind"],
                "resident_pages_before": w["resident_pages_before"],
                "target_pages": w["target_pages"],
                "D_target_bytes": acc["target"]["bytes"],
                "D_target_cmds": acc["target"]["cmds"],
                "D_file_other_bytes": acc["file_other"]["bytes"],
                "D_file_other_cmds": acc["file_other"]["cmds"],
                "D_other_bytes": acc["other"]["bytes"],
                "D_other_cmds": acc["other"]["cmds"],
                "writes": writes,
                "target_bad_status": bad_status,
                "size_hist_target": dict(sorted(acc["target"]["sizes"].items())),
                "size_hist_file_other": dict(
                    sorted(acc["file_other"]["sizes"].items())
                ),
                "lat_ns_target": {
                    "n": len(l),
                    "p50": pct(50),
                    "p95": pct(95),
                    "p99": pct(99),
                    "max": l[-1] if l else None,
                },
                "bandwidth_target_MiBps": (
                    (acc["target"]["bytes"] / 2**20) / (w["elapsed_ns"] / 1e9)
                    if w["elapsed_ns"]
                    else None
                ),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tracer", required=True, help="path to nvme_tp_monitor")
    ap.add_argument("--disk", required=True, help="block device name, e.g. nvme0n1")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-seconds", type=int, default=3600)
    ap.add_argument(
        "--idle-seconds",
        type=float,
        default=2.0,
        help="background interval measured before the replay",
    )
    ap.add_argument(
        "replay_args",
        nargs=argparse.REMAINDER,
        help="-- then replay_ranges.py arguments (without --out)",
    )
    args = ap.parse_args()
    rd = Path(args.run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    trace = rd / "device.jsonl"
    replay_out = rd / "replay.json"
    rargs = [a for a in args.replay_args if a != "--"]

    p, t_attach = start_tracer(args.tracer, args.disk, str(trace), args.max_seconds)
    t_idle0 = time.monotonic_ns()
    time.sleep(args.idle_seconds)
    t_idle1 = time.monotonic_ns()
    here = Path(__file__).resolve().parent
    rc = subprocess.run(
        [
            sys.executable,
            str(here / "replay_ranges.py"),
            *rargs,
            "--out",
            str(replay_out),
        ]
    )
    time.sleep(1.0)
    tail = stop_tracer(p)
    (rd / "tracer.out").write_text(tail or "")
    if rc.returncode != 0:
        sys.exit(f"replay failed rc={rc.returncode}")

    replay = json.loads(replay_out.read_text())
    cmds, meta = load_events(str(trace))
    anchor = meta.get("clock_anchor", {})
    drops = meta.get("drops", {}).get("dropped")
    file_lbas = []
    for e in replay["extents"]:
        file_lbas.append(
            (
                replay["part_start_sectors"] + e["physical"] // 512,
                (e["length"] + 511) // 512,
            )
        )
    windows = replay["trials"]
    attributed = classify(
        cmds, [tuple(x) for x in replay["device_lbas"]], file_lbas, windows
    )
    idle = classify(
        cmds,
        [tuple(x) for x in replay["device_lbas"]],
        file_lbas,
        [
            {
                "trial": -1,
                "t_start_ns": t_idle0,
                "t_drain_end_ns": t_idle1,
                "elapsed_ns": t_idle1 - t_idle0,
                "F_bytes": 0,
                "F_kind": "idle",
                "resident_pages_before": None,
                "target_pages": None,
            }
        ],
    )[0]
    n_cmp = meta["_match"]["n_cmp"]
    summary = {
        "replay": {
            k: replay[k]
            for k in (
                "file",
                "file_size",
                "inode",
                "label",
                "repo",
                "unit",
                "arm",
                "cache",
                "qd",
                "chunk",
                "align",
                "n_ranges",
                "W_required_bytes",
            )
        },
        "tracer": {
            "capture_meta": meta.get("capture_meta"),
            "clock_anchor_monotonic_ns": anchor.get("monotonic_ns"),
            "python_monotonic_at_attach_ns": t_attach,
            "dropped": drops,
            "n_cmd_events": len(cmds),
            "n_cmp_events": n_cmp,
            "unmatched_cmds": meta["_match"]["unmatched_cmds"],
        },
        "idle_background": idle,
        "trials": attributed,
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=1))
    for t in attributed:
        print(
            f"trial {t['trial']}: F={t['F_bytes']} D_target={t['D_target_bytes']} ({t['D_target_cmds']} cmds) "
            f"file_other={t['D_file_other_bytes']} other={t['D_other_bytes']} writes={t['writes']['bytes']} "
            f"{t['elapsed_ns'] / 1e6:.3f} ms drops={drops}"
        )


if __name__ == "__main__":
    main()
