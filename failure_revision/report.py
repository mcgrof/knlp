"""Pilot analysis: success at the budget, paired item-clustered contrasts.

The item is the inferential unit.  Each arm's outcome for an item is the
mean over its repetitions; contrasts are paired per item and resampled
by whole items (10,000 bootstrap draws, fixed seed).  A paired sign-flip
permutation test supplies p-values, Holm-corrected over the three
comparisons of the summary arm against the diversity, planning and
critique arms.  An exact McNemar test on the first repetition is the
sparse-event sensitivity check.  The pilot sizes a confirmation; it is
not one.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

from .arms import ARMS
from .run import jdump, jload, load_screen, read_jsonl

BOOT = 10_000
SEED = 20260918


def load_runs(out, mode):
    runs = []
    for p in sorted(
        glob.glob(os.path.join(out, "shard*", f"pilot_{mode}", "runs.jsonl"))
    ):
        runs += read_jsonl(p)
    return runs


def outcome(r, frac=1.0):
    if not r["solved"]:
        return 0.0
    if r["mode"] == "time":
        return float(r["t_success"] <= frac * r["budget_seconds"])
    return float(r["tokens_at_success"] <= frac * r["budget_tokens"])


def item_means(runs, items, frac=1.0):
    acc = defaultdict(list)
    for r in runs:
        if r["item"] in items:
            acc[(r["item"], r["arm"])].append(outcome(r, frac))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired(m, items, a, b, rng):
    diffs = [m[(i, a)] - m[(i, b)] for i in items if (i, a) in m and (i, b) in m]
    n = len(diffs)
    if n == 0:
        return {"n_items": 0}
    est = sum(diffs) / n
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(BOOT))
    lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]
    obs = abs(est)
    hits = 0
    for _ in range(BOOT):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        hits += abs(s) >= obs - 1e-12
    return {
        "n_items": n,
        "delta": est,
        "ci95": [lo, hi],
        "p_signflip": (hits + 1) / (BOOT + 1),
        "n_nonzero_items": sum(d != 0 for d in diffs),
    }


def mcnemar_rep0(runs, items, a, b):
    y = {
        (r["item"], r["arm"]): outcome(r)
        for r in runs
        if r["rep"] == 0 and r["item"] in items
    }
    n10 = sum(1 for i in items if y.get((i, a)) == 1 and y.get((i, b)) == 0)
    n01 = sum(1 for i in items if y.get((i, a)) == 0 and y.get((i, b)) == 1)
    n = n10 + n01
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_exact": 1.0}
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n)
    return {"a_only": n10, "b_only": n01, "p_exact": p}


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for k, name in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[name]))
        out[name] = running
    return out


def collision(screen_rec, window=8):
    fails = [
        a
        for a in sorted(screen_rec["attempts"], key=lambda a: a["k"])
        if not a["correct"]
    ]
    last = fails[-window:]
    if len(last) < window or any(a["answer"] is None for a in last):
        return None
    cnt = defaultdict(int)
    for a in last:
        cnt[a["answer"]] += 1
    m = len(last)
    return sum(c * (c - 1) for c in cnt.values()) / (m * (m - 1))


def arm_table(runs, items, mode):
    rows = {}
    for arm in ARMS:
        rs = [r for r in runs if r["arm"] == arm and r["item"] in items]
        if not rs:
            continue
        m = item_means(rs, items)
        vals = [v for (i, a), v in m.items()]
        row = {
            "runs": len(rs),
            "items": len(vals),
            "solved_at_budget": sum(vals) / len(vals),
            "gen_tokens": sum(r["gen_tokens"] for r in rs),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
            "cached_tokens": sum(r["cached_tokens"] for r in rs),
            "wall_seconds": sum(r["elapsed"] for r in rs),
            "solves": sum(r["solved"] for r in rs),
            "truncated_outputs": sum(r["n_truncated_out"] for r in rs),
            "graded_calls": sum(r["n_graded"] for r in rs),
            "aux_calls": sum(r["n_aux"] for r in rs),
            "max_overrun_seconds": max(r["overrun"] for r in rs),
        }
        if mode == "time":
            for frac in (0.25, 0.5):
                mm = item_means(rs, items, frac)
                row[f"solved_at_{frac}C"] = sum(mm.values()) / len(mm)
            C = rs[0]["budget_seconds"]
            row["restricted_mean_solve_seconds"] = sum(
                (r["t_success"] if r["solved"] else C) for r in rs
            ) / len(rs)
        else:
            T = rs[0]["budget_tokens"]
            row["restricted_mean_solve_tokens"] = sum(
                (r["tokens_at_success"] if r["solved"] else T) for r in rs
            ) / len(rs)
        row["solves_per_million_gen_tokens"] = (
            1e6 * row["solves"] / max(1, row["gen_tokens"])
        )
        row["solves_per_gpu_hour"] = (
            3600 * row["solves"] / max(1e-9, row["wall_seconds"])
        )
        rows[arm] = row
    return rows


def analyze(out):
    sel = jload(os.path.join(out, "selection.json"))
    frozen = jload(os.path.join(out, "frozen.json"))
    screen = load_screen(out, "pilot")
    zeros, mods = set(sel["observed_zero"]), set(sel["moderate"])
    rng = random.Random(SEED)
    rep = {
        "frozen": frozen,
        "selection": {k: v for k, v in sel.items() if not isinstance(v, list)},
    }
    rep["selection"]["n_observed_zero_used"] = len(zeros)
    rep["selection"]["n_moderate"] = len(mods)
    for mode in ("time", "tokens"):
        runs = load_runs(out, mode)
        if not runs:
            continue
        sec = {
            "observed_zero": arm_table(runs, zeros, mode),
            "moderate": arm_table(runs, mods, mode),
        }
        m = item_means(runs, zeros)
        zi = sorted(zeros)
        sec["primary_B_minus_A"] = paired(m, zi, "B", "A", rng)
        sec["primary_B_minus_A_rep0_mcnemar"] = mcnemar_rep0(runs, zeros, "B", "A")
        comps = {}
        for other in ("C", "D", "R"):
            comps[f"B_minus_{other}"] = paired(m, zi, "B", other, rng)
        adj = holm({k: v["p_signflip"] for k, v in comps.items() if "p_signflip" in v})
        for k, v in comps.items():
            v["p_holm"] = adj.get(k)
        sec["secondary"] = comps
        sec["each_arm_minus_A"] = {
            f"{a}_minus_A": paired(m, zi, a, "A", rng) for a in ("C", "D", "R")
        }
        rep[mode] = sec
    # Descriptive answer-collision statistic over the screening failures.
    col = {i: collision(screen[i]) for i in zeros if i in screen}
    avail = {i: c for i, c in col.items() if c is not None}
    trig = {i for i, c in avail.items() if c >= 0.6}
    rep["answer_collision"] = {
        "items": len(col),
        "available": len(avail),
        "triggered_at_0.6": len(trig),
        "values": avail,
    }
    tr = load_runs(out, "time")
    if tr and trig:
        m = item_means(tr, zeros)
        for name, group in (("triggered", trig), ("not_triggered", set(avail) - trig)):
            d = [
                m[(i, "B")] - m[(i, "A")]
                for i in group
                if (i, "B") in m and (i, "A") in m
            ]
            rep["answer_collision"][f"B_minus_A_{name}"] = (
                {"n": len(d), "mean": sum(d) / len(d)} if d else {"n": 0}
            )
    costs = []
    for p in glob.glob(os.path.join(out, "shard*", "screen_cost.jsonl")):
        costs += read_jsonl(p)
    rep["screen_cost"] = costs
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = analyze(args.out)
    jdump(rep, os.path.join(args.out, "report.json"))
    slim = {k: v for k, v in rep.items() if k != "answer_collision"}
    print(json.dumps(slim, indent=1)[:20000])


if __name__ == "__main__":
    main()
