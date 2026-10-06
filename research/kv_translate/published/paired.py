#!/usr/bin/env python3
"""Paired differences between a transferred arm and the receiver, with error bars.

Every arm is scored on the same examples, so the honest comparison is paired:
for each example, was the transferred cache right where the receiver's own
cache was right? Treating the two accuracies as independent overstates the
error several times over, and hides the one number that decides a transfer
claim, which is how many examples changed their answer and in which
direction.

Uncertainty comes from resampling examples with replacement. For a grouped
task the examples of every subject are pooled, which matches the way the
harness weights its aggregate by size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

BASELINE = "native"
RESAMPLES = 10000
SEED = 20260928


def load_outcomes(work, task, mode):
    """One arm's per-example results as {example key: 0.0 or 1.0}."""
    path = os.path.join(work, "outcomes", f"{task}.{mode}.json")
    with open(path) as f:
        by_leaf = json.load(f)
    flat = {}
    for leaf, rows in by_leaf.items():
        for doc_id, value in rows.items():
            flat[f"{leaf}/{doc_id}"] = float(value)
    return flat


def compare(base, arm, *, resamples=RESAMPLES, seed=SEED):
    """Paired statistics for one task. Both maps must cover the same examples."""
    import numpy as np

    if set(base) != set(arm):
        missing = len(set(base) ^ set(arm))
        raise ValueError(
            f"the two arms were not scored on the same examples; {missing} "
            "are in one and not the other"
        )
    if not base:
        raise ValueError("no examples to compare")
    keys = sorted(base)
    b = np.array([base[k] for k in keys])
    a = np.array([arm[k] for k in keys])
    n = len(keys)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    bs, as_ = b[idx].mean(1), a[idx].mean(1)
    diff = as_ - bs
    ratio = 100.0 * as_ / np.clip(bs, 1e-12, None)
    return {
        "examples": n,
        "baseline_accuracy": float(b.mean()),
        "arm_accuracy": float(a.mean()),
        "difference": float(a.mean() - b.mean()),
        "difference_interval_95": [
            float(np.quantile(diff, 0.025)),
            float(np.quantile(diff, 0.975)),
        ],
        # the bound a deficit is tested against: how much worse could it be
        "deficit_upper_bound_95_one_sided": float(-np.quantile(diff, 0.05)),
        "retention_percent": float(100.0 * a.mean() / max(b.mean(), 1e-12)),
        "retention_interval_95": [
            float(np.quantile(ratio, 0.025)),
            float(np.quantile(ratio, 0.975)),
        ],
        "both_right": int(((b == 1) & (a == 1)).sum()),
        "only_baseline_right": int(((b == 1) & (a != 1)).sum()),
        "only_arm_right": int(((b != 1) & (a == 1)).sum()),
        "neither_right": int(((b != 1) & (a != 1)).sum()),
        "_resampled_retention": ratio,
    }


def report(work, tasks, modes, *, baseline=BASELINE, resamples=RESAMPLES, seed=SEED):
    """Every arm against the baseline on every task, and the mean over tasks."""
    import numpy as np

    out = {"baseline": baseline, "resamples": resamples, "seed": seed, "arms": {}}
    for mode in modes:
        if mode == baseline:
            continue
        per_task, draws = {}, []
        for i, task in enumerate(tasks):
            r = compare(
                load_outcomes(work, task, baseline),
                load_outcomes(work, task, mode),
                resamples=resamples,
                seed=seed + i,
            )
            draws.append(r.pop("_resampled_retention"))
            per_task[task] = r
        mean_draws = np.mean(np.stack(draws), axis=0)
        out["arms"][mode] = {
            "tasks": per_task,
            "mean_retention_percent": float(
                np.mean([v["retention_percent"] for v in per_task.values()])
            ),
            "mean_retention_interval_95": [
                float(np.quantile(mean_draws, 0.025)),
                float(np.quantile(mean_draws, 0.975)),
            ],
            "lowest_task_retention_percent": float(
                min(v["retention_percent"] for v in per_task.values())
            ),
        }
    return out


def between(work, tasks, first, second, *, resamples=RESAMPLES, seed=SEED):
    """One transferred arm against the other, paired the same way."""
    out = {}
    for i, task in enumerate(tasks):
        r = compare(
            load_outcomes(work, task, first),
            load_outcomes(work, task, second),
            resamples=resamples,
            seed=seed + i,
        )
        r.pop("_resampled_retention")
        out[task] = r
    return {"baseline": first, "arm": second, "tasks": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--tasks", default="hellaswag,arc_challenge,winogrande,mmlu")
    ap.add_argument("--modes", default="direct,native,full_head,cache_bridge")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tasks, modes = args.tasks.split(","), args.modes.split(",")
    out = report(args.work, tasks, modes)
    if "full_head" in modes and "cache_bridge" in modes:
        out["between_arms"] = between(args.work, tasks, "full_head", "cache_bridge")
    path = args.out or os.path.join(args.work, "PAIRED.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print("PAIRED " + json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
