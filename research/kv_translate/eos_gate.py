# SPDX-License-Identifier: GPL-2.0
"""Apply the corrected-arm advancement criteria to a scored development run.

The inherited scorer measures everything needed and then decides `passes`
using the parent plan's conjunction, which includes the 512-token latency
gates and a comparison against a latency-matched native suffix. This stage is
explicitly not that: it tests 512-token quality and health, and the 512
latency failure is known and is not allowed to block the longer-context
question. So the measurements are reused and the decision is made here.

Three things the inherited decision also gets wrong for this stage. Prompt
specificity is computed for retrieval alone, while the requirement is that
each informative task clears its own wrong-prompt control. Health is judged by
a bootstrap tail on the healthy rate, while the requirement is an absolute
two-point limit on excess unhealthy outputs plus no repetition increase
against the matched anchor. And nothing checks that the benchmark is
informative at all, so an arm could be reported as passing or failing against
a native control that had collapsed.

Ending a sequence is not a pathology and is not counted as one.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

TASKS = ("retrieval", "cloze")
ABSOLUTE_POINTS = 0.05  # within five absolute points of native
UNHEALTHY_POINTS = 2.0 / 64.0  # at most two absolute points of excess
RECOVERY_POINT = 0.75
RECOVERY_FLOOR = 0.50
NATIVE_RETRIEVAL_MIN = 0.90
CLOZE_HEADROOM_MIN = 0.10
N_BOOT = 2000
SEED = 0


def load(rows_path):
    by = collections.defaultdict(dict)
    for line in open(rows_path):
        r = json.loads(line)
        by[(r["kind"], r["condition"])][r["doc"]] = r
    return by


def paired(by, kind, a, b, field="primary", docs=None):
    """Paired difference a-b over shared documents, with a bootstrap interval."""
    ra, rb = by.get((kind, a)), by.get((kind, b))
    if not ra or not rb:
        return None
    keys = sorted(set(ra) & set(rb)) if docs is None else sorted(docs)
    d = np.array([float(ra[k][field]) - float(rb[k][field]) for k in keys])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    bs = d[idx].mean(axis=1)
    return {
        "n": int(d.size),
        "point": float(d.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
    }


def rate(by, kind, cond, field="primary"):
    r = by.get((kind, cond))
    if not r:
        return None
    return float(np.mean([float(v[field]) for v in r.values()]))


def free_breakdown(by, cond):
    """Every free-generation outcome, reported separately.

    Stopping is listed but never counted as unhealthy: the arm under test is
    the one taught to stop, and scoring termination as a pathology would
    decide the experiment by definition.
    """
    r = by.get(("free", cond))
    if not r:
        return None
    n = len(r)
    out = {"n": n}
    for f in ("healthy", "empty", "too_short", "degenerate", "stopped", "cap_hit"):
        out[f] = float(np.mean([float(v.get(f, 0.0)) for v in r.values()]))
    out["repetition_mean"] = float(
        np.mean([float(v.get("repeat_4gram_frac", 0.0)) for v in r.values()])
    )
    out["unhealthy"] = 1.0 - out["healthy"]
    out["note"] = "stopped is reported, never counted as unhealthy"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--arms", required=True, help="comma separated arm names")
    ap.add_argument("--anchor", default="anchor77")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by = load(args.rows)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    nat = {t: rate(by, t, "target_native") for t in TASKS}
    emp = {t: rate(by, t, "empty_cache") for t in TASKS}
    informative = {
        "native_retrieval": nat["retrieval"],
        "native_retrieval_min": NATIVE_RETRIEVAL_MIN,
        "cloze_headroom": (
            None
            if nat["cloze"] is None or emp["cloze"] is None
            else nat["cloze"] - emp["cloze"]
        ),
        "cloze_headroom_min": CLOZE_HEADROOM_MIN,
    }
    informative["passed"] = bool(
        nat["retrieval"] is not None
        and nat["retrieval"] >= NATIVE_RETRIEVAL_MIN
        and informative["cloze_headroom"] is not None
        and informative["cloze_headroom"] >= CLOZE_HEADROOM_MIN
    )
    # Thresholds recomputed from the matched native actually measured here,
    # rather than carried over from the run that first stated them.
    thresholds = {
        t: {
            "native": nat[t],
            "within_five_points_at_least": (
                None if nat[t] is None else nat[t] - ABSOLUTE_POINTS
            ),
        }
        for t in TASKS
    }

    anchor_free = free_breakdown(by, args.anchor)
    native_free = free_breakdown(by, "target_native")

    results = {}
    for arm in arms:
        r = {"tasks": {}, "free": free_breakdown(by, arm)}
        for t in TASKS:
            got = rate(by, t, t and arm)
            gain = None
            if nat[t] is not None and emp[t] is not None and got is not None:
                denom = nat[t] - emp[t]
                gain = None if abs(denom) < 1e-12 else (got - emp[t]) / denom
            vs_wrong = paired(by, t, arm, f"{arm}|wrong")
            vs_empty = paired(by, t, arm, "empty_cache")
            vs_native = paired(by, t, arm, "target_native")
            rec_int = None
            if vs_empty and nat[t] is not None and emp[t] is not None:
                denom = nat[t] - emp[t]
                if abs(denom) > 1e-12:
                    rec_int = {k: vs_empty[k] / denom for k in ("point", "lo", "hi")}
            r["tasks"][t] = {
                "score": got,
                "absolute_points_behind_native": (
                    None if got is None or nat[t] is None else (nat[t] - got) * 100.0
                ),
                "within_five_points": bool(
                    got is not None
                    and nat[t] is not None
                    and got >= nat[t] - ABSOLUTE_POINTS
                ),
                "recovery": gain,
                "recovery_interval": rec_int,
                "recovers_enough": bool(
                    gain is not None
                    and gain >= RECOVERY_POINT
                    and rec_int is not None
                    and rec_int["lo"] > RECOVERY_FLOOR
                ),
                "vs_own_wrong_prompt": vs_wrong,
                # Required on each informative task, not on retrieval alone.
                "beats_own_wrong_prompt": bool(vs_wrong and vs_wrong["lo"] > 0.0),
                "vs_native": vs_native,
            }
        f, nf = r["free"], native_free
        excess = None if not f or not nf else f["unhealthy"] - nf["unhealthy"]
        rep_up = (
            None
            if not f or not anchor_free
            else f["repetition_mean"] - anchor_free["repetition_mean"]
        )
        r["health"] = {
            "excess_unhealthy_vs_native": excess,
            "excess_limit": UNHEALTHY_POINTS,
            "within_two_points": bool(
                excess is not None and excess <= UNHEALTHY_POINTS
            ),
            "repetition_vs_anchor": rep_up,
            "no_repetition_increase": bool(rep_up is not None and rep_up <= 0.0),
            "anchor": args.anchor,
        }
        r["health"]["passed"] = bool(
            r["health"]["within_two_points"] and r["health"]["no_repetition_increase"]
        )
        # Every requirement stands on its own. Averaging the two tasks would
        # let a passing one carry a failing one.
        reqs = {
            f"{t}_{k}": r["tasks"][t][k]
            for t in TASKS
            for k in ("within_five_points", "recovers_enough", "beats_own_wrong_prompt")
        }
        reqs["health"] = r["health"]["passed"]
        r["requirements"] = reqs
        r["failed_requirements"] = sorted(k for k, v in reqs.items() if not v)
        r["advances"] = bool(informative["passed"] and not r["failed_requirements"])
        results[arm] = r

    passing = [a for a in arms if results[a]["advances"]]
    if not informative["passed"]:
        verdict = "BENCHMARK_UNINFORMATIVE"
    elif not passing:
        gained = [
            a
            for a in arms
            if results[a]["tasks"]["retrieval"]["recovers_enough"]
            or results[a]["health"]["passed"]
        ]
        verdict = "MECHANISM_ONLY" if gained else "NO_ARM_ADVANCES"
    else:
        verdict = "ARM_ADVANCES"

    chosen = None
    if len(passing) > 1:

        def margin(a):
            return min(
                results[a]["tasks"][t]["score"] - (thresholds[t]["native"] or 0.0)
                for t in TASKS
            )

        best = max(margin(a) for a in passing)
        tied = [a for a in passing if abs(margin(a) - best) < 1e-12]
        if len(tied) > 1:
            hb = max(results[a]["free"]["healthy"] for a in tied)
            tied = [a for a in tied if abs(results[a]["free"]["healthy"] - hb) < 1e-12]
        chosen = "eos_on" if "eos_on" in tied else sorted(tied)[0]
    elif passing:
        chosen = passing[0]

    out = {
        "contract": "eos_repair_gate_v1",
        "scope": (
            "512-token quality and health. Not 512-token deployment: the known "
            "512 latency failure and the latency-matched native suffix "
            "comparison are reported by the scorer and deliberately do not "
            "gate advancement at this stage."
        ),
        "benchmark_informative": informative,
        "thresholds": thresholds,
        "native_free": native_free,
        "anchor_free": anchor_free,
        "arms": results,
        "passing_arms": passing,
        "selected": chosen,
        "selection_note": (
            "greatest minimum native-relative margin, then health, then the on "
            "arm; selection on reused development documents is exploratory and "
            "is not fresh confirmation"
        ),
        "verdict": verdict,
    }
    json.dump(
        open_out := out, open(args.out, "w"), indent=2, sort_keys=True, default=str
    )

    print(f"benchmark informative: {informative['passed']}")
    for a in arms:
        r = results[a]
        print(
            f"  {a:<10s} retr {r['tasks']['retrieval']['score']:.4f} "
            f"cloze {r['tasks']['cloze']['score']:.4f} "
            f"healthy {r['free']['healthy']:.3f} "
            f"advances={r['advances']} failed={r['failed_requirements']}"
        )
    print(f"verdict: {verdict}  selected: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
