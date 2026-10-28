#!/usr/bin/env python3
"""Aggregate traced run summaries into one metrics table.

Reads every runs/**/summary.json produced by traced_replay.py (including
GPU harness runs that use the same window format) and writes one CSV row
per trial with the byte counters, command counts, size histograms,
latency percentiles and evidence class. Raw trials are preserved; a
second CSV holds per-run medians for tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

COLS = [
    "run",
    "group",
    "repo",
    "unit",
    "arm",
    "cache",
    "qd",
    "chunk",
    "mode",
    "trial",
    "W_required",
    "F_bytes",
    "F_kind",
    "D_target_bytes",
    "D_target_cmds",
    "D_file_other_bytes",
    "D_file_other_cmds",
    "D_other_bytes",
    "D_other_cmds",
    "write_bytes_in_window",
    "resident_pages_before",
    "target_pages",
    "elapsed_ms",
    "read_ms",
    "copy_ms",
    "compute_ms",
    "H2D_bytes",
    "D_over_W",
    "F_over_W",
    "target_bw_MiBps",
    "lat_n",
    "lat_p50_us",
    "lat_p95_us",
    "lat_p99_us",
    "size_hist_target",
    "drops",
    "unmatched",
    "evidence_class",
]


def rows_for(summary_path: Path, root: Path):
    s = json.loads(summary_path.read_text())
    r = s["replay"]
    run = str(summary_path.parent.relative_to(root / "runs"))
    group = run.split("/")[0]
    replay = json.loads((summary_path.parent / "replay.json").read_text())
    tr_by = {t["trial"]: t for t in replay["trials"]}
    out = []
    for t in s["trials"]:
        rt = tr_by.get(t["trial"], {})
        W = r["W_required_bytes"]
        lat = t.get("lat_ns_target") or {}
        out.append(
            {
                "run": run,
                "group": group,
                "repo": r.get("repo"),
                "unit": r.get("unit"),
                "arm": r.get("arm") + ("-merged" if replay.get("merge") else ""),
                "cache": r.get("cache"),
                "qd": r.get("qd"),
                "chunk": r.get("chunk"),
                "mode": replay.get("mode", ""),
                "trial": t["trial"],
                "W_required": W,
                "F_bytes": t["F_bytes"],
                "F_kind": t["F_kind"],
                "D_target_bytes": t["D_target_bytes"],
                "D_target_cmds": t["D_target_cmds"],
                "D_file_other_bytes": t["D_file_other_bytes"],
                "D_file_other_cmds": t["D_file_other_cmds"],
                "D_other_bytes": t["D_other_bytes"],
                "D_other_cmds": t["D_other_cmds"],
                "write_bytes_in_window": t["writes"]["bytes"],
                "resident_pages_before": t.get("resident_pages_before"),
                "target_pages": t.get("target_pages"),
                "elapsed_ms": t["elapsed_ns"] / 1e6,
                "read_ms": rt.get("read_ns", 0) / 1e6 if "read_ns" in rt else "",
                "copy_ms": rt.get("copy_ns", 0) / 1e6 if "copy_ns" in rt else "",
                "compute_ms": (
                    rt.get("compute_ns", 0) / 1e6 if "compute_ns" in rt else ""
                ),
                "H2D_bytes": rt.get("H2D_bytes", ""),
                "D_over_W": (t["D_target_bytes"] / W) if W else "",
                "F_over_W": (t["F_bytes"] / W) if W else "",
                "target_bw_MiBps": t.get("bandwidth_target_MiBps"),
                "lat_n": lat.get("n"),
                "lat_p50_us": (lat.get("p50") or 0) / 1e3 if lat.get("p50") else "",
                "lat_p95_us": (lat.get("p95") or 0) / 1e3 if lat.get("p95") else "",
                "lat_p99_us": (lat.get("p99") or 0) / 1e3 if lat.get("p99") else "",
                "size_hist_target": json.dumps(t.get("size_hist_target")),
                "drops": s["tracer"].get("dropped"),
                "unmatched": s["tracer"].get("unmatched_cmds"),
                "evidence_class": (
                    "controlled real-weight replay" if group != "qual" else "control"
                ),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--out",
        required=True,
        help="metrics.csv path; a _medians.csv is written beside it",
    )
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for sp in sorted((root / "runs").glob("**/summary.json")):
        try:
            rows.extend(rows_for(sp, root))
        except Exception as e:  # keep going; report the broken run
            print(f"skip {sp}: {e}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    # medians per run
    byrun = {}
    for r in rows:
        byrun.setdefault(r["run"], []).append(r)
    med_cols = [
        "run",
        "group",
        "repo",
        "unit",
        "arm",
        "cache",
        "qd",
        "chunk",
        "mode",
        "n_trials",
        "W_required",
        "F_bytes_med",
        "D_target_bytes_med",
        "D_target_cmds_med",
        "D_file_other_bytes_med",
        "D_other_bytes_med",
        "write_bytes_med",
        "elapsed_ms_med",
        "elapsed_ms_min",
        "elapsed_ms_max",
        "read_ms_med",
        "copy_ms_med",
        "compute_ms_med",
        "D_over_W_med",
        "target_bw_MiBps_med",
        "lat_p50_us_med",
        "lat_p95_us_med",
        "drops_total",
        "unmatched_total",
    ]
    out2 = Path(args.out).with_name(Path(args.out).stem + "_medians.csv")

    def med(key, rs):
        v = [x[key] for x in rs if x[key] not in ("", None)]
        return statistics.median(v) if v else ""

    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=med_cols)
        w.writeheader()
        for run, rs in sorted(byrun.items()):
            r0 = rs[0]
            w.writerow(
                {
                    "run": run,
                    "group": r0["group"],
                    "repo": r0["repo"],
                    "unit": r0["unit"],
                    "arm": r0["arm"],
                    "cache": r0["cache"],
                    "qd": r0["qd"],
                    "chunk": r0["chunk"],
                    "mode": r0["mode"],
                    "n_trials": len(rs),
                    "W_required": r0["W_required"],
                    "F_bytes_med": med("F_bytes", rs),
                    "D_target_bytes_med": med("D_target_bytes", rs),
                    "D_target_cmds_med": med("D_target_cmds", rs),
                    "D_file_other_bytes_med": med("D_file_other_bytes", rs),
                    "D_other_bytes_med": med("D_other_bytes", rs),
                    "write_bytes_med": med("write_bytes_in_window", rs),
                    "elapsed_ms_med": med("elapsed_ms", rs),
                    "elapsed_ms_min": min(x["elapsed_ms"] for x in rs),
                    "elapsed_ms_max": max(x["elapsed_ms"] for x in rs),
                    "read_ms_med": med("read_ms", rs),
                    "copy_ms_med": med("copy_ms", rs),
                    "compute_ms_med": med("compute_ms", rs),
                    "D_over_W_med": med("D_over_W", rs),
                    "target_bw_MiBps_med": med("target_bw_MiBps", rs),
                    "lat_p50_us_med": med("lat_p50_us", rs),
                    "lat_p95_us_med": med("lat_p95_us", rs),
                    "drops_total": sum((x["drops"] or 0) for x in rs),
                    "unmatched_total": sum((x["unmatched"] or 0) for x in rs),
                }
            )
    print(f"{len(rows)} trial rows from {len(byrun)} runs -> {args.out}, {out2}")


if __name__ == "__main__":
    main()
