# SPDX-License-Identifier: GPL-2.0
"""Compare two arms' per-update loss traces, separated by objective.

The schedule interleaves two losses that are not on the same scale: a
continuation objective over the full vocabulary and an answer objective over a
handful of tokens. A running mean over both, which is all the earlier rounds
kept, mixes them, so a spike in one is indistinguishable from a shift in the
mix. The update order is recorded, so they can be separated here.

The question this is for: when an arm's loss leaves the floor and does not
return, is that a property of the arm or of the run? Two paired runs of this
recipe have now diverged on opposite arms from the same seed and the same
order, which is the shape of an unstable optimisation rather than an effect of
the thing being ablated.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def summarise(trace, order, tag):
    trace = np.asarray(trace, dtype=float)
    kinds = np.array([k for k, _ in order])
    out = {"n_updates": int(trace.size)}
    for kind in ("obj", "cont"):
        m = kinds == kind
        if not m.any():
            continue
        v = trace[m]
        # Where does this objective sit early, and where does it end?
        out[kind] = {
            "n": int(v.size),
            "first_50": float(v[:50].mean()),
            "last_50": float(v[-50:].mean()),
            "median": float(np.median(v)),
            "max": float(v.max()),
            "argmax_update": int(np.arange(trace.size)[m][int(v.argmax())]),
            "ended_above_10x_its_floor": bool(
                v[-50:].mean() > 10.0 * max(np.percentile(v, 5), 1e-9)
            ),
        }
    # A spike is a sustained departure, not a single noisy update.
    w = 25
    if trace.size > 2 * w:
        roll = np.convolve(trace, np.ones(w) / w, mode="valid")
        floor = np.percentile(roll, 10)
        above = roll > 5.0 * max(floor, 1e-9)
        idx = np.flatnonzero(above)
        out["excursion"] = {
            "floor": float(floor),
            "first_update_above_5x_floor": int(idx[0]) if idx.size else None,
            "fraction_of_run_above": float(above.mean()),
            "recovered_by_end": bool(not above[-1]) if idx.size else True,
        }
    out["tag"] = tag
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=path.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        d = json.load(open(path))
        trace, order = d.get("loss_trace"), d.get("update_order")
        if not trace or not order:
            arms[name] = {"error": "no retained trace or order in this record"}
            continue
        arms[name] = summarise(trace, order, name)
        arms[name]["recorded"] = {
            k: d.get(k)
            for k in (
                "eos_arm",
                "sample_order_sha256",
                "denominator_total",
                "supervised_terms_total",
                "valid_supervised_tokens",
            )
        }

    names = [n for n in arms if "error" not in arms[n]]
    out = {
        "contract": "arm_trace_comparison_v1",
        "arms": arms,
        "shared_order": (
            len({arms[n]["recorded"]["sample_order_sha256"] for n in names}) == 1
            if names
            else None
        ),
        "shared_denominator": (
            len({arms[n]["recorded"]["denominator_total"] for n in names}) == 1
            if names
            else None
        ),
        "reading": (
            "An arm whose excursion does not recover by the end has diverged. "
            "If the arms share an order, a denominator and a token count, and "
            "differ in which of them recovers, the divergence is not "
            "attributable to the ablated term."
        ),
    }
    json.dump(out, open(args.out, "w"), indent=2, sort_keys=True, default=str)
    for n in names:
        a = arms[n]
        e = a.get("excursion", {})
        print(
            f"  {n:<8s} obj last50 {a.get('obj', {}).get('last_50', float('nan')):8.3f} "
            f"cont last50 {a.get('cont', {}).get('last_50', float('nan')):8.3f} "
            f"| excursion at {e.get('first_update_above_5x_floor')} "
            f"recovered={e.get('recovered_by_end')}"
        )
    print(
        "shared order:",
        out["shared_order"],
        "| shared denominator:",
        out["shared_denominator"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
