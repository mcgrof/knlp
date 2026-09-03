#!/usr/bin/env python3
"""Summarize option-ranking scores into one table.

Globs <out-dir>/rank-eval/campaign-b*-seed*-<arm>/score.json (the
certified benchmark scorer's output) plus each run's manifest, and
writes rank_summary.json and a markdown table — accuracy with its
bootstrap CI, coverage, and skip counts per arm and seed, plus a
per-arm mean — to the results directory."""

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
    for d in sorted(glob.glob(os.path.join(a.out_dir, "rank-eval", "campaign-*"))):
        m = pat.search(os.path.basename(d))
        spath = os.path.join(d, "score.json")
        if not m or not os.path.exists(spath):
            continue
        s = json.load(open(spath))
        man = json.load(open(os.path.join(d, "manifest.json")))
        rows.append(
            dict(
                arm=m.group(3),
                seed=int(m.group(2)),
                accuracy=round(s["overall"], 4),
                ci95=[round(x, 4) for x in s.get("bootstrap_95ci", [])],
                coverage=round(s["coverage"], 4),
                n_queries=s["n_queries"],
                skipped=man.get("queries_skipped_overlength", 0),
                norm=man.get("norm"),
            )
        )

    rows.sort(key=lambda r: (r["arm"], r["seed"]))
    means = []
    for arm in sorted({r["arm"] for r in rows}):
        accs = [r["accuracy"] for r in rows if r["arm"] == arm]
        means.append(
            dict(
                arm=arm,
                seeds=len(accs),
                mean_accuracy=round(statistics.mean(accs), 4),
                stdev=round(statistics.stdev(accs), 4) if len(accs) > 1 else None,
            )
        )

    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "rank_summary.json"), "w") as f:
        json.dump(dict(runs=rows, per_arm=means), f, indent=1)

    lines = [
        "| arm | seed | accuracy | 95% CI | coverage | queries | skipped |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"{r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}" if r["ci95"] else "-"
        lines.append(
            f"| {r['arm']} | {r['seed']} | {r['accuracy']:.4f} | {ci} "
            f"| {r['coverage']:.3f} | {r['n_queries']} | {r['skipped']} |"
        )
    lines += ["", "| arm | seeds | mean accuracy | stdev |", "|---|---|---|---|"]
    for m in means:
        sd = f"{m['stdev']:.4f}" if m["stdev"] is not None else "-"
        lines.append(f"| {m['arm']} | {m['seeds']} | {m['mean_accuracy']:.4f} | {sd} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(a.results_dir, "rank_summary.md"), "w") as f:
        f.write(md)
    print(md, end="")


if __name__ == "__main__":
    main()
