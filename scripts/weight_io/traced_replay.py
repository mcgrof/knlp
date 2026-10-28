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

With `--intent-tracer` (syscall_monitor) and `--fault-tracer`
(mmap_readamp) the child's own file calls are recorded as well: the
child is started stopped, both tracers attach to its process id, and it
is resumed once they report attachment. The summary then carries, per
trial window, the bytes the child asked the kernel for on the target
file through read-family calls and the mappings it created, and, for
the whole run, the demand-fault and page-cache-fill bytes of the target
file. That is the observed intent F, as opposed to the F the child
reports about itself.

The tracers and Python all use CLOCK_MONOTONIC; the clock anchor
emitted by the NVMe tracer is checked against Python's reading at start.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

READ_CALLS = ("read", "pread64", "readv", "preadv", "preadv2")


def start_tracer(cmd: list[str], ready: str):
    """Start a tracer under sudo and return (process, python monotonic ns
    when it reported `ready` on its output). The rest of its output is
    drained in the background so a long table dump at exit cannot block
    on a full pipe."""
    p = subprocess.Popen(
        ["sudo", *cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    t_py = None
    for _ in range(200):
        line = p.stdout.readline()
        if not line:
            break
        if ready in line:
            t_py = time.monotonic_ns()
            break
    if t_py is None:
        p.kill()
        raise RuntimeError(f"tracer {cmd[0]} did not report '{ready}'")
    p.drained = []

    def drain():
        for line in p.stdout:
            p.drained.append(line)

    p.drain_thread = threading.Thread(target=drain, daemon=True)
    p.drain_thread.start()
    return p, t_py


def stop_tracer(p, grace: float = 300.0):
    """Interrupt a tracer and wait for it to write its tables; a tracer that
    does not finish within the grace period is killed and reported."""
    # the tracer runs as root under sudo; send SIGINT to the process group leader via sudo
    subprocess.run(["sudo", "kill", "-INT", str(p.pid)], check=False)
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "kill", "-KILL", str(p.pid)], check=False)
        p.wait(timeout=10)
        p.drained.append(
            f"[traced_replay] tracer killed after {grace:.0f} s without exiting\n"
        )
    p.drain_thread.join(timeout=5)
    return "".join(p.drained)


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


def file_identity(path: str):
    st = os.stat(path)
    return f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}", st.st_ino


def load_intent(path: Path):
    """Return (meta, calls, loss) of a syscall_monitor capture."""
    meta, loss, calls = {}, {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            t = e.get("type")
            if t == "call":
                calls.append(e)
            elif t == "meta":
                meta = e
            elif t == "loss":
                loss = e
    return meta, calls, loss


def load_faults(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def intent_summary(calls, faults, targets: set, windows):
    """Observed intent on the target files: per window from the syscall
    capture, per run from the fault tracer."""
    on_target = [c for c in calls if (c.get("dev"), c.get("ino")) in targets]
    per_trial = []
    for w in windows:
        t0, t1 = w["t_start_ns"], w["t_drain_end_ns"]
        inw = [c for c in on_target if t0 <= c["ts_enter_ns"] <= t1]
        reads = [c for c in inw if c["name"] in READ_CALLS]
        hist: dict = {}
        by_name: dict = {}
        for c in reads:
            got = max(0, c["ret"])
            hist[c["count"]] = hist.get(c["count"], 0) + 1
            b = by_name.setdefault(c["name"], {"calls": 0, "bytes": 0})
            b["calls"] += 1
            b["bytes"] += got
        maps = [c for c in inw if c["name"] == "mmap" and c["ret"] >= 0]
        per_trial.append(
            {
                "trial": w["trial"],
                "read_calls": len(reads),
                "read_bytes_requested": sum(c["count"] for c in reads),
                "read_bytes_returned": sum(max(0, c["ret"]) for c in reads),
                "read_size_hist": dict(sorted(hist.items())),
                "read_by_name": by_name,
                "direct_io_calls": sum(
                    1 for c in reads if c.get("fflags", 0) & 0o40000
                ),
                "mmap_calls": len(maps),
                "mmap_bytes": sum(c["count"] for c in maps),
                "madvise_calls": sum(1 for c in inw if c["name"] == "madvise"),
                "fadvise_calls": sum(1 for c in inw if c["name"] == "fadvise64"),
                "open_calls": sum(
                    1 for c in inw if c["name"] in ("openat", "open", "openat2")
                ),
                "other_calls": {
                    n: sum(1 for c in inw if c["name"] == n)
                    for n in sorted({c["name"] for c in inw})
                },
            }
        )
    run = {
        "faults": 0,
        "minor_faults": 0,
        "major_faults": 0,
        "fault_bytes": 0,
        "major_bytes": 0,
        "pgcache_fill_bytes": 0,
        "unique_pages": 0,
        "readahead_prepared_pages": 0,
        "syscall_io_bytes": 0,
        "odirect_bytes": 0,
    }
    for r in faults:
        if (r.get("dev"), r.get("ino")) not in targets:
            continue
        t = r.get("table")
        if t == "faults":
            run["faults"] += r["faults"]
            run["minor_faults"] += r["minor"]
            run["major_faults"] += r["major"]
            run["fault_bytes"] += r["fault_bytes"]
            run["major_bytes"] += r["major_bytes"]
        elif t == "pgcache_fill":
            run["pgcache_fill_bytes"] += r["fill_bytes"]
        elif t == "pages":
            run["unique_pages"] += r["unique_pages"]
        elif t == "readahead":
            run["readahead_prepared_pages"] += r["ra_prepared_pages"]
        elif t == "syscall_io" and r.get("rw") == "R":
            run["syscall_io_bytes"] += r["bytes"]
        elif t == "odirect":
            run["odirect_bytes"] += r["dio_bytes"]
    return {"per_trial": per_trial, "run_faults": run, "target_calls": len(on_target)}


def readable_trace(trace: Path):
    """The tracer runs under sudo; under a restrictive umask (for example
    inside a systemd user service) its output is root-only. Reopen it for
    the owner of the run directory."""
    if trace.exists() and not os.access(trace, os.R_OK):
        subprocess.run(["sudo", "chmod", "644", str(trace)], check=True)


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
        "--child",
        default="replay_ranges.py",
        help="script in this directory to run under the tracer; it must write the window JSON to --out",
    )
    ap.add_argument(
        "--intent-tracer",
        default=None,
        help="path to syscall_monitor; records the child's file calls",
    )
    ap.add_argument(
        "--fault-tracer",
        default=None,
        help="path to mmap_readamp; records the child's demand faults",
    )
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="do not trace or run a child; rebuild summary.json from the "
        "run directory's replay.json and device.jsonl",
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
    intent_path = rd / "intent.jsonl"
    faults_path = rd / "faults.jsonl"
    replay_out = rd / "replay.json"
    window_path = rd / "window.json"
    rargs = [a for a in args.replay_args if a != "--"]
    intent_meta: dict = {}

    if args.summarize_only:
        if not (replay_out.exists() and trace.exists()):
            sys.exit("summarize-only needs replay.json and device.jsonl")
        win = json.loads(window_path.read_text()) if window_path.exists() else {}
        t_attach = win.get("python_monotonic_at_attach_ns")
        t_idle0 = win.get("idle_start_ns")
        t_idle1 = win.get("idle_end_ns")
        intent_meta = win.get("intent", {})
    else:
        p, t_attach = start_tracer(
            [
                args.tracer,
                "--disk",
                args.disk,
                "--dur",
                str(args.max_seconds),
                "--jsonl",
                str(trace),
            ],
            "attached",
        )
        t_idle0 = time.monotonic_ns()
        time.sleep(args.idle_seconds)
        t_idle1 = time.monotonic_ns()
        here = Path(__file__).resolve().parent
        cmd = [sys.executable, str(here / args.child), *rargs, "--out", str(replay_out)]
        intent_procs = []
        if args.intent_tracer or args.fault_tracer:
            # start the child stopped so the tracers can attach to its pid:
            # a shell stops itself, then execs the child in the same process
            # (Popen itself would block on a child stopped before its exec)
            child = subprocess.Popen(
                ["/bin/sh", "-c", 'kill -STOP $$; exec "$@"', "sh", *cmd]
            )
            if args.intent_tracer:
                ip, t_int = start_tracer(
                    [
                        args.intent_tracer,
                        "--pid",
                        str(child.pid),
                        "--jsonl-out",
                        str(intent_path),
                        "--duration",
                        str(args.max_seconds),
                        "--quiet",
                    ],
                    "attached",
                )
                intent_procs.append(("intent", ip, intent_path))
                intent_meta["intent_tracer"] = args.intent_tracer
                intent_meta["python_monotonic_at_intent_attach_ns"] = t_int
            if args.fault_tracer:
                fp, t_flt = start_tracer(
                    [
                        args.fault_tracer,
                        "--pid",
                        str(child.pid),
                        "--dur",
                        str(args.max_seconds),
                        "--jsonl",
                        str(faults_path),
                    ],
                    "tracing",
                )
                intent_procs.append(("faults", fp, faults_path))
                intent_meta["fault_tracer"] = args.fault_tracer
                intent_meta["python_monotonic_at_fault_attach_ns"] = t_flt
            intent_meta["child_pid"] = child.pid
            os.kill(child.pid, signal.SIGCONT)
            try:
                rc = child.wait()
            except BaseException:
                child.kill()
                for _, proc, _ in intent_procs:
                    stop_tracer(proc, grace=30)
                stop_tracer(p, grace=30)
                raise
        else:
            rc = subprocess.run(cmd).returncode
        time.sleep(1.0)
        for name, proc, out in intent_procs:
            tail = stop_tracer(proc)
            (rd / f"{name}.out").write_text(tail or "")
            readable_trace(out)
        tail = stop_tracer(p)
        (rd / "tracer.out").write_text(tail or "")
        readable_trace(trace)
        window_path.write_text(
            json.dumps(
                {
                    "python_monotonic_at_attach_ns": t_attach,
                    "idle_start_ns": t_idle0,
                    "idle_end_ns": t_idle1,
                    "intent": intent_meta,
                }
            )
        )
        if rc != 0:
            sys.exit(f"replay failed rc={rc}")

    readable_trace(trace)
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
    idle = None
    if t_idle0 is not None and t_idle1 is not None:
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

    # observed intent of the child on the target file(s)
    if intent_path.exists() or faults_path.exists():
        files = replay.get("files") or [replay["file"]]
        targets = set()
        for fpath in files:
            try:
                targets.add(file_identity(fpath))
            except OSError:
                pass
        calls, faults, imeta, iloss = [], [], {}, {}
        if intent_path.exists():
            readable_trace(intent_path)
            imeta, calls, iloss = load_intent(intent_path)
        if faults_path.exists():
            readable_trace(faults_path)
            faults = load_faults(faults_path)
        isum = intent_summary(calls, faults, targets, windows)
        isum["targets"] = sorted(targets)
        isum["capture"] = {
            "schema": imeta.get("schema"),
            "tool_commit": imeta.get("tool_commit"),
            "records": iloss.get("records"),
            "ringbuf_drops": iloss.get("ringbuf_drops"),
            "join_miss": iloss.get("join_miss"),
            "stash_fail": iloss.get("stash_fail"),
            "start_ns": imeta.get("start_ns"),
            "python_monotonic_at_intent_attach_ns": intent_meta.get(
                "python_monotonic_at_intent_attach_ns"
            ),
            "fault_rows": len(faults),
        }
        by_trial = {t["trial"]: t for t in isum["per_trial"]}
        for t in attributed:
            it = by_trial.get(t["trial"])
            if it:
                t["F_syscall_read_bytes"] = it["read_bytes_returned"]
                t["F_syscall_read_calls"] = it["read_calls"]
                t["F_mmap_bytes"] = it["mmap_bytes"]
        summary["intent"] = isum
    (rd / "summary.json").write_text(json.dumps(summary, indent=1))
    for t in attributed:
        extra = ""
        if "F_syscall_read_bytes" in t:
            extra = (
                f" F_syscall={t['F_syscall_read_bytes']} ({t['F_syscall_read_calls']} calls)"
                f" F_mmap={t['F_mmap_bytes']}"
            )
        print(
            f"trial {t['trial']}: F={t['F_bytes']}{extra} D_target={t['D_target_bytes']} ({t['D_target_cmds']} cmds) "
            f"file_other={t['D_file_other_bytes']} other={t['D_other_bytes']} writes={t['writes']['bytes']} "
            f"{t['elapsed_ns'] / 1e6:.3f} ms drops={drops}"
        )
    if "intent" in summary:
        rf = summary["intent"]["run_faults"]
        print(
            f"run faults on target: {rf['faults']} ({rf['fault_bytes']} B, major {rf['major_bytes']} B), "
            f"page-cache fill {rf['pgcache_fill_bytes']} B, unique pages {rf['unique_pages']}"
        )


if __name__ == "__main__":
    main()
