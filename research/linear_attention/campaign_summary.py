#!/usr/bin/env python3
"""Summarize campaign runs into one table.

Globs the per-run JSONs the matched-micro harness wrote
(<out-dir>/campaign-b<batch>-seed<seed>-<arm>/<arm>.json), reports
final validation loss, throughput, wall-clock, and checkpoint
presence per arm and seed, plus a per-arm mean across seeds, and
writes campaign_summary.json and a markdown table to the results
directory. Partial campaigns summarize whatever runs exist."""

import argparse
import glob
import json
import os
import re
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    a = ap.parse_args()

    pat = re.compile(r"campaign-b(\d+)-seed(\d+)-([a-z0-9]+)$")
    rows = []
    for d in sorted(glob.glob(os.path.join(a.out_dir, "campaign-b*-seed*-*"))):
        m = pat.search(os.path.basename(d))
        if not m:
            continue
        batch, seed, arm = int(m.group(1)), int(m.group(2)), m.group(3)
        path = os.path.join(d, f"{arm}.json")
        if not os.path.exists(path):
            continue
        run = json.load(open(path))
        hist = [h for h in run["history"] if "tok_s" in h]
        rows.append(
            dict(
                arm=arm,
                seed=seed,
                batch=batch,
                steps=hist[-1]["step"] + 1 if hist else 0,
                tok_s=round(hist[-1]["tok_s"], 1) if hist else 0.0,
                wall_h=round(run["final"]["wall_s"] / 3600, 2),
                peak_mem_gb=round(run["final"].get("peak_mem_gb", 0.0), 2),
                val_loss=round(run["final"]["val_loss"], 4),
                val_ppl=round(run["final"]["val_ppl"], 2),
                checkpoint=os.path.exists(os.path.join(d, f"{arm}.pt")),
            )
        )

    rows.sort(key=lambda r: (r["arm"], r["seed"]))
    means = []
    for arm in sorted({r["arm"] for r in rows}):
        losses = [r["val_loss"] for r in rows if r["arm"] == arm]
        means.append(
            dict(
                arm=arm,
                seeds=len(losses),
                mean_val_loss=round(statistics.mean(losses), 4),
                stdev_val_loss=(
                    round(statistics.stdev(losses), 4) if len(losses) > 1 else None
                ),
            )
        )

    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "campaign_summary.json"), "w") as f:
        json.dump(dict(runs=rows, per_arm=means), f, indent=1)

    lines = [
        "| arm | seed | batch | steps | tok/s | wall h | peak GB | val loss | val ppl | ckpt |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['seed']} | {r['batch']} | {r['steps']} "
            f"| {r['tok_s']:.0f} | {r['wall_h']:.2f} | {r['peak_mem_gb']:.2f} "
            f"| {r['val_loss']:.4f} | {r['val_ppl']:.2f} "
            f"| {'y' if r['checkpoint'] else 'MISSING'} |"
        )
    lines += ["", "| arm | seeds | mean val loss | stdev |", "|---|---|---|---|"]
    for m in means:
        stdev = f"{m['stdev_val_loss']:.4f}" if m["stdev_val_loss"] is not None else "-"
        lines.append(
            f"| {m['arm']} | {m['seeds']} | {m['mean_val_loss']:.4f} | {stdev} |"
        )
    md = "\n".join(lines) + "\n"
    with open(os.path.join(a.results_dir, "campaign_summary.md"), "w") as f:
        f.write(md)
    print(md, end="")


if __name__ == "__main__":
    main()
