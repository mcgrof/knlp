#!/usr/bin/env python3
"""Summarize batch-probe runs into one table.

Reads the per-point run JSONs the matched-micro harness wrote
(<out-dir>/probe-<arm>-b<batch>/<arm>.json), takes the last logged
throughput reading of each, and writes probe_summary.json plus a
markdown table to the results directory. The speedup column is
relative to the first batch in the ladder, so saturation reads
directly off the table."""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--arms", required=True, help="space-separated arm names")
    ap.add_argument("--batches", required=True, help="space-separated batch sizes")
    a = ap.parse_args()

    rows = []
    for arm in a.arms.split():
        base = None
        for b in a.batches.split():
            path = os.path.join(a.out_dir, f"probe-{arm}-b{b}", f"{arm}.json")
            run = json.load(open(path))
            hist = [h for h in run["history"] if "tok_s" in h]
            tok_s = hist[-1]["tok_s"]
            if base is None:
                base = tok_s
            rows.append(
                dict(
                    arm=arm,
                    batch=int(b),
                    tok_s=round(tok_s, 1),
                    speedup=round(tok_s / base, 2),
                    wall_s=round(run["final"]["wall_s"], 1),
                    peak_mem_gb=round(run["final"].get("peak_mem_gb", 0.0), 2),
                    val_loss=round(run["final"]["val_loss"], 4),
                )
            )

    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "probe_summary.json"), "w") as f:
        json.dump(rows, f, indent=1)

    header = "| arm | batch | tok/s | speedup | wall s | peak GB | val loss |"
    rule = "|---|---|---|---|---|---|---|"
    lines = [header, rule]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['batch']} | {r['tok_s']:.0f} "
            f"| {r['speedup']:.2f}x | {r['wall_s']:.0f} "
            f"| {r['peak_mem_gb']:.2f} | {r['val_loss']:.4f} |"
        )
    md = "\n".join(lines) + "\n"
    with open(os.path.join(a.results_dir, "probe_summary.md"), "w") as f:
        f.write(md)
    print(md, end="")


if __name__ == "__main__":
    main()
