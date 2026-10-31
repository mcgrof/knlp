# SPDX-License-Identifier: MIT
"""Apply the frozen decision rule to the confirmation and summarize cache damage."""

import argparse
from pathlib import Path

import numpy as np

from research.layerrope_beta.intercept.plan import (
    ANCHOR,
    AUTHOR_POINT,
    CONFIRM_SEEDS,
    confirm_rows,
    job,
    primary,
    row,
)
from research.mo_variants.stage_c_common import read, write

TOLERANCE = 0.1
MIN_GAIN = 0.01


def endpoint(root, r, seed):
    folder = root / "language/runs" / job(r, "final", seed)["id"]
    result = read(folder / "RESULT.json")
    if result["status"] != "complete" or result["completed_steps"] != 2000:
        raise ValueError("Incomplete confirmation endpoint: " + str(folder))
    quantized = read(folder / "QUANTIZATION.json")
    cache = {x["name"]: x for x in quantized["cache"]}
    other = {x["name"]: x for x in quantized["weight_activation_residual"]}
    return dict(
        primary=primary(result),
        clean=quantized["clean"]["ce"],
        k3v2_increment=cache["symmetric-K3V2"]["ce"] - cache["symmetric-K3V3"]["ce"],
        cache=cache,
        all_fp4_damage=other["all-FP4-weights"]["delta_ce"],
        training_seconds=sum(x["seconds"] for x in result["steps"]),
    )


def summarize(seeds):
    cases = seeds[0]["cache"]
    passing = [
        cases[name]["storage"]["ideal_total_bytes"]
        for name in cases
        if all(s["cache"][name]["delta_ce"] <= TOLERANCE for s in seeds)
    ]
    out = {
        key: float(np.mean([s[key] for s in seeds]))
        for key in (
            "primary",
            "clean",
            "k3v2_increment",
            "all_fp4_damage",
            "training_seconds",
        )
    }
    out["primary_by_seed"] = [s["primary"] for s in seeds]
    out["minimum_passing_bytes_per_token"] = min(passing) if passing else None
    return out


def contrast(rows, better, worse):
    pairs = [
        b - a
        for a, b in zip(rows[better]["primary_by_seed"], rows[worse]["primary_by_seed"])
    ]
    return dict(
        better=better,
        worse=worse,
        mean_gain=float(np.mean(pairs)),
        gains_by_seed=pairs,
        wins=sum(x > 0 for x in pairs),
        n=len(pairs),
    )


def analyze(root):
    root = Path(root)
    chosen = read(root / "SELECTED.json")["chosen"]
    rows = {}
    for r in confirm_rows(chosen):
        rows[r["id"]] = dict(
            arm=r["arm"],
            alpha_out=r["alpha_out"],
            beta=r["beta"],
            **summarize([endpoint(root, r, seed) for seed in CONFIRM_SEEDS]),
        )
    decisions = {}
    for arm in ("L", "ML"):
        anchor = row(arm, *ANCHOR)["id"]
        winner = chosen[arm]
        c = contrast(rows, winner, anchor)
        c["helps"] = (
            winner != anchor and c["wins"] == c["n"] and c["mean_gain"] >= MIN_GAIN
        )
        decisions[arm] = dict(
            winner_vs_anchor=c,
            author_point_vs_anchor=contrast(
                rows, row(arm, *AUTHOR_POINT)["id"], anchor
            ),
            winner_vs_mhc=contrast(rows, winner, "M"),
            anchor_vs_mhc=contrast(rows, anchor, "M"),
        )
    report = dict(
        rule=f"Winner beats a=0/beta=0 anchor in all {len(CONFIRM_SEEDS)} paired seeds with mean primary-CE gain >= {MIN_GAIN}",
        seeds=list(CONFIRM_SEEDS),
        chosen=chosen,
        rows=rows,
        decisions=decisions,
    )
    write(root / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    print(
        f"{'row':30} {'primary':>8} {'clean':>8} {'K3V2 inc':>8} {'FP4 all':>8} {'bytes':>6}"
    )
    for name, r in sorted(report["rows"].items(), key=lambda kv: kv[1]["primary"]):
        print(
            f"{name:30} {r['primary']:8.4f} {r['clean']:8.4f} "
            f"{r['k3v2_increment']:8.4f} {r['all_fp4_damage']:8.4f} "
            f"{r['minimum_passing_bytes_per_token'] or 0:6}"
        )
    for arm, d in report["decisions"].items():
        for name, c in d.items():
            print(
                f"{arm:3} {name:24} gain={c['mean_gain']:+.4f} wins={c['wins']}/{c['n']}"
                + (f" helps={c['helps']}" if "helps" in c else "")
            )
