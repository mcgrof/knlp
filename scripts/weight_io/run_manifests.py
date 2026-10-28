#!/usr/bin/env python3
"""Write a manifest.json next to every traced run summary.

The manifest gathers, for one run, what a reader needs to assess and
reproduce it: the tool commit, the model repository revision and shard
hash, the replay arguments, the tracer configuration and capture loss,
host kernel and block-device geometry, the cache method, the timing
window and the output files with sizes and SHA-256 hashes. It is derived
from summary.json and replay.json plus the Stage A pin files; nothing is
re-measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

GEOMETRY = (
    "logical_block_size",
    "physical_block_size",
    "max_sectors_kb",
    "max_hw_sectors_kb",
    "max_segments",
    "read_ahead_kb",
    "scheduler",
    "nr_requests",
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def disk_geometry(disk: str) -> dict:
    out = {}
    base = Path("/sys/block") / disk / "queue"
    for k in GEOMETRY:
        p = base / k
        if p.exists():
            out[k] = p.read_text().strip()
    return out


def fs_of(path: str) -> dict:
    try:
        r = subprocess.run(
            ["findmnt", "-no", "FSTYPE,SOURCE,OPTIONS", "-T", path],
            capture_output=True,
            text=True,
            check=True,
        )
        fstype, source, options = r.stdout.split(None, 2)
        return {"fstype": fstype, "source": source, "options": options.strip()}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def evidence_class(run: str) -> str:
    if run.startswith("qual"):
        return "control (known single read)"
    if run.startswith("gpu/") and "/native-" in run:
        return "native model execution"
    if run.startswith("gpu/"):
        return "real GPU layer consumption, synthetic activations"
    return "controlled real-weight replay"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--pins-dir", required=True)
    ap.add_argument("--disk", required=True)
    ap.add_argument("--tools-commit", required=True)
    ap.add_argument("--tracer-commit", required=True)
    ap.add_argument("--force", action="store_true", help="rewrite existing manifests")
    args = ap.parse_args()
    root = Path(args.root)
    pins = {}
    for p in Path(args.pins_dir).glob("*.pin.json"):
        d = json.loads(p.read_text())
        pins[d["repo"]] = d
    host = {
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "disk": args.disk,
        "geometry": disk_geometry(args.disk),
    }
    n = 0
    for sp in sorted(root.glob("runs/**/summary.json")):
        rd = sp.parent
        mp = rd / "manifest.json"
        if mp.exists() and not args.force:
            continue
        s = json.loads(sp.read_text())
        rep = s.get("replay", {})
        run = str(rd.relative_to(root / "runs"))
        full = None
        if (rd / "replay.json").exists():
            full = json.loads((rd / "replay.json").read_text())
        trials = (full or {}).get("trials", [])
        repo = rep.get("repo")
        pin = pins.get(repo) if repo else None
        shard = Path(rep["file"]).name if rep.get("file") else None
        shard_pin = None
        if pin and shard:
            files = pin.get("files")
            if isinstance(files, dict):
                shard_pin = files.get(shard)
            elif isinstance(files, list):
                shard_pin = next(
                    (
                        f
                        for f in files
                        if f.get("rfilename") == shard or f.get("path") == shard
                    ),
                    None,
                )
        tr = s.get("tracer", {})
        outputs = {}
        for f in sorted(rd.iterdir()):
            if f.is_file() and f.name != "manifest.json":
                outputs[f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
        window = None
        if trials:
            t0 = min(t["t_start_ns"] for t in trials if "t_start_ns" in t)
            t1 = max(t.get("t_drain_end_ns", t.get("t_end_ns", 0)) for t in trials)
            window = {"monotonic_start_ns": t0, "monotonic_end_ns": t1}
        man = {
            "run": run,
            "evidence_class": evidence_class(run),
            "tools": {
                "repository": "knlp scripts/weight_io",
                "commit": args.tools_commit,
                "tracer": {"name": "nvme_tp_monitor", "commit": args.tracer_commit},
            },
            "model": {
                "repo": repo,
                "revision": pin.get("sha") if pin else None,
                "shard": shard,
                "shard_pin": shard_pin,
                "file": rep.get("file"),
                "file_size": rep.get("file_size"),
                "inode": rep.get("inode"),
            },
            "replay": {
                k: rep.get(k)
                for k in (
                    "label",
                    "unit",
                    "arm",
                    "cache",
                    "qd",
                    "chunk",
                    "chunk_align",
                    "merge",
                    "align",
                    "n_ranges",
                    "W_required_bytes",
                    "mode",
                    "read",
                    "layer",
                    "trials",
                )
                if k in rep
            },
            "cache_method": {
                "cold": "posix_fadvise(DONTNEED) on the file, residency verified by mincore before each trial",
                "warm": "buffered pre-read of the ranges, residency verified by mincore before each trial",
            }.get(rep.get("cache"), None),
            "tracer_config": {
                "capture_meta": tr.get("capture_meta"),
                "dropped": tr.get("dropped"),
                "n_cmd_events": tr.get("n_cmd_events"),
                "n_cmp_events": tr.get("n_cmp_events"),
                "unmatched_cmds": tr.get("unmatched_cmds"),
                "clock_anchor_monotonic_ns": tr.get("clock_anchor_monotonic_ns"),
            },
            "capture_loss": {
                "dropped_events": tr.get("dropped"),
                "unmatched_commands": tr.get("unmatched_cmds"),
            },
            "idle_background_window": s.get("idle_background"),
            "n_trials": len(s.get("trials", [])),
            "timing_window": window,
            "summary_written_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(sp.stat().st_mtime)
            ),
            "host": {
                **host,
                "filesystem": fs_of(rep["file"]) if rep.get("file") else None,
            },
            "outputs": outputs,
        }
        mp.write_text(json.dumps(man, indent=1))
        n += 1
    print(f"wrote {n} manifests under {root/'runs'}")


if __name__ == "__main__":
    main()
