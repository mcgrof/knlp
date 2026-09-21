# SPDX-License-Identifier: GPL-2.0
"""Recompute the task gate from measurements already taken.

The expensive half of the gate is the per-document scores; the statistics
over them are seconds of arithmetic. Keeping the two separable means a defect
in a threshold, a control or an interval can be corrected against the same
measurements instead of by spending the hardware a second time -- and it means
the correction is auditable, because the inputs are unchanged and on disk.

This exists because one such defect happened. The wrong-prompt control was
made per-arm during a review pass, but the edit that repointed the gate at the
per-arm control silently failed to apply, so the gate looked for a condition
name that no longer existed, received nothing, and recorded every arm as
failing a control that was never evaluated. The rows had the answer in them
the whole time.

Reads `rows.jsonl` and `smoke.json` from a completed run and writes
`smoke.rescored.json` beside them, leaving the originals untouched.

Env: none; reads JSON only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.run_h1_smoke import (  # noqa: E402
    DETERMINISTIC_POINTS,
    MIN_HEADROOM,
    RECOVERY_INTERVAL_FLOOR,
    RECOVERY_POINT,
)

WRONG = "|wrong"


def by_doc(rows, kind, cond, field="primary"):
    d = {}
    for r in rows:
        if r["kind"] == kind and r["condition"] == cond:
            d.setdefault(r["doc"], []).append(r[field])
    return d


def cluster_bootstrap(values_by_doc, n_boot, seed):
    rng = np.random.default_rng(seed)
    docs = list(values_by_doc)
    if not docs:
        return (float("nan"),) * 3
    point = float(np.mean([v for d in docs for v in values_by_doc[d]]))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(docs), len(docs))
        vals = [v for i in pick for v in values_by_doc[docs[i]]]
        if vals:
            boots.append(float(np.mean(vals)))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return point, float(lo), float(hi)


def paired_delta(rows, kind, a, b, n_boot, seed):
    da, db = by_doc(rows, kind, a), by_doc(rows, kind, b)
    docs = sorted(set(da) & set(db))
    if not docs:
        return None

    def m(sel):
        return float(np.mean([v for d in sel for v in da[d]])) - float(
            np.mean([v for d in sel for v in db[d]])
        )

    rng = np.random.default_rng(seed)
    boots = [
        m([docs[i] for i in rng.integers(0, len(docs), len(docs))])
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": m(docs), "lo": float(lo), "hi": float(hi), "n_docs": len(docs)}


def paired_recovery(rows, kind, cond, n_boot, seed):
    arm = by_doc(rows, kind, cond)
    nat = by_doc(rows, kind, "target_native")
    emp = by_doc(rows, kind, "empty_cache")
    docs = sorted(set(arm) & set(nat) & set(emp))
    if not docs:
        return None

    def ratio(sel):
        a = float(np.mean([v for d in sel for v in arm[d]]))
        n = float(np.mean([v for d in sel for v in nat[d]]))
        e = float(np.mean([v for d in sel for v in emp[d]]))
        g = n - e
        return None if abs(g) < 1e-9 else ((a - e) / g, n - a)

    base = ratio(docs)
    if base is None:
        return None
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        r = ratio([docs[i] for i in rng.integers(0, len(docs), len(docs))])
        if r is not None:
            boots.append(r[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2
    return {
        "point": base[0],
        "lo": float(lo),
        "hi": float(hi),
        "abs_points_behind_native": 100.0 * base[1],
        "n_docs": len(docs),
        "headroom": float(
            np.mean([v for d in docs for v in nat[d]])
            - np.mean([v for d in docs for v in emp[d]])
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.run_dir, "rows.jsonl"))]
    old = json.load(open(os.path.join(args.run_dir, "smoke.json")))
    first_token = old.get("first_token", {})
    window = old.get("latency_matched_window", {}).get("tokens", 0)

    conds = sorted({r["condition"] for r in rows})
    arms = sorted(
        c
        for c in conds
        if not c.endswith(WRONG)
        and c not in ("target_native", "empty_cache")
        and not c.startswith("recent_")
    )
    task_kinds = ["retrieval", "cloze"]
    kinds = task_kinds + ["free"]

    summary = {}
    for kind in kinds:
        present = sorted({r["condition"] for r in rows if r["kind"] == kind})
        summary[kind] = {}
        for c in present:
            p, lo, hi = cluster_bootstrap(by_doc(rows, kind, c), args.n_boot, args.seed)
            summary[kind][c] = {"point": p, "lo": lo, "hi": hi}

    recovery = {
        c: {k: paired_recovery(rows, k, c, args.n_boot, args.seed) for k in task_kinds}
        for c in conds
    }

    gates = {}
    for cond in arms:
        g = {"retrieval_point": summary["retrieval"][cond]["point"]}
        for name, ctrl in (
            ("empty_cache", "empty_cache"),
            ("wrong_prompt", cond + WRONG),
        ):
            d = paired_delta(rows, "retrieval", cond, ctrl, args.n_boot, args.seed)
            g["vs_" + name] = d
            g["beats_" + name] = bool(d is not None and d["lo"] > 0.0)
        g["prompt_specific"] = g["beats_empty_cache"] and g["beats_wrong_prompt"]

        ok_rec, ok_abs, measured = True, True, []
        for kind in task_kinds:
            rc = recovery[cond].get(kind)
            if rc is None or abs(rc.get("headroom", 0.0)) < MIN_HEADROOM:
                continue
            measured.append(kind)
            ok_rec = ok_rec and rc["point"] >= RECOVERY_POINT
            ok_rec = ok_rec and rc["lo"] > RECOVERY_INTERVAL_FLOOR
            ok_abs = ok_abs and rc["abs_points_behind_native"] <= DETERMINISTIC_POINTS
        g["kinds_with_headroom"] = measured
        g["recovers_incremental_gain"] = bool(measured) and ok_rec
        g["within_absolute_points"] = bool(measured) and ok_abs

        fr = summary["free"].get(cond, {}).get("point")
        nat = summary["free"].get("target_native", {}).get("point")
        if fr is None or nat is None or fr != fr:
            g["no_pathology_regression"] = None
        else:
            d = paired_delta(
                rows, "free", cond, "target_native", args.n_boot, args.seed
            )
            g["vs_native_pathology"] = d
            g["no_pathology_regression"] = bool(d is not None and d["lo"] > -0.10)
        g["pathology_measured"] = g["no_pathology_regression"] is not None

        if window:
            d = paired_delta(
                rows, "retrieval", cond, f"recent_{window}", args.n_boot, args.seed
            )
            g["vs_recent_window"] = d
            g["beats_latency_matched_native"] = bool(d is not None and d["lo"] > 0.0)
        else:
            g["vs_recent_window"] = None
            g["beats_latency_matched_native"] = None

        ft = first_token.get(cond, {})
        g["retains_h0_latency_gates"] = bool(
            ft.get("gate_p95_under_native") and ft.get("gate_p50_at_or_under_075")
        )
        g["passes"] = bool(
            g["prompt_specific"]
            and g["recovers_incremental_gain"]
            and g["within_absolute_points"]
            and g["no_pathology_regression"] is True
            and g["beats_latency_matched_native"] is True
            and g["retains_h0_latency_gates"]
        )
        gates[cond] = g

    out = dict(old)
    out.update(
        {
            "summary": summary,
            "recovery": recovery,
            "gates": gates,
            "passing_arms": [c for c, g in gates.items() if g["passes"]],
            "rescored": {
                "reason": (
                    "the wrong-prompt control was made per-arm but the gate "
                    "still named a condition that no longer existed, so every "
                    "arm was recorded as failing a control never evaluated"
                ),
                "n_rows": len(rows),
                "n_boot": args.n_boot,
                "seed": args.seed,
            },
        }
    )
    path = os.path.join(args.run_dir, "smoke.rescored.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    for c, g in sorted(gates.items()):
        print(
            f"  {c:<18s} prompt-specific {g['prompt_specific']} "
            f"recovers {g['recovers_incremental_gain']} "
            f"within-{DETERMINISTIC_POINTS:.0f}pts {g['within_absolute_points']} "
            f"no-pathology {g['no_pathology_regression']} "
            f"beats-recent-window {g['beats_latency_matched_native']} "
            f"latency {g['retains_h0_latency_gates']} "
            f"=> {'PASS' if g['passes'] else 'FAIL'}"
        )
    print(f"passing arms: {out['passing_arms'] or 'none'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
