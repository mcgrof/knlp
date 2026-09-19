# SPDX-License-Identifier: GPL-2.0
"""Roll the Stage 0 cells up into one verdict, with multiplicity control.

A single cell is one (model, context length, context source, seed, tensor kind)
combination.  Reading sixteen runs one at a time invites picking the cell that
agrees with what you hoped, so this collapses them into one table, applies
Benjamini-Hochberg across the whole family, and states the promotion decision
in the terms the plan pre-registered: improvement of at least 0.10 Spearman
over the best cheap baseline, a paired bootstrap interval above zero, ordering
stable between four and eight probes, and the result repeating on both
held-out seeds.

It also prints the attribution, which is what makes a verdict useful rather
than merely binary: how much of any ranking power comes from the positional
weighting, how much from the second moment's matrix shape, and how much is
lost by averaging the metric across contexts instead of computing it on the
context being scored.

Run:
    python -m research.jacobian_kv.aggregate_stage0 --runs <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv.evaluate import benjamini_hochberg  # noqa: E402

DEPLOYABLE = (
    "jtfj_p8",
    "jtfj_pos_p8",
    "jtfj_prefill_p8",
    "jtfj_rank_p8",
    "pos_energy_p8",
)
ORACLE = ("jtfj_future_p8", "jtfj_oracle_p8")
CANDIDATES = DEPLOYABLE + ORACLE


def load_cells(runs_dir: str):
    cells = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        with open(path) as fh:
            s = json.load(fh)
        run = os.path.basename(os.path.dirname(path))
        for kind, cell in s.get("cells", {}).items():
            cells.append(
                {
                    "run": run,
                    "model": s["model"],
                    "ctx": s["ctx"],
                    "seed": s["seed"],
                    "source": s["context_source"],
                    "kind": kind,
                    "n": cell["n"],
                    "n_groups": cell.get("n_groups"),
                    "n_clusters": cell.get("n_clusters"),
                    "blocked_rho": cell.get("blocked_rho", {}),
                    "pooled_rho": cell.get("pooled_rho", {}),
                    "gates": {
                        c: g
                        for c, g in cell.get("gates", {}).items()
                        if c in CANDIDATES
                    },
                    "attribution": cell.get("attribution", {}),
                    "floor_kl_max": cell.get("floor_kl_max"),
                    "exact_kl_median": cell.get("exact_kl_median"),
                    "exact_kl_min": cell.get("exact_kl_min"),
                }
            )
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fdr-q", type=float, default=0.05)
    args = ap.parse_args()

    cells = load_cells(args.runs)
    if not cells:
        print(f"no summary.json under {args.runs}")
        return 1

    print(f"{len(cells)} cells from {len({c['run'] for c in cells})} runs\n")

    # ---- the gate family, with BH across every cell x candidate ----------
    family = []
    for c in cells:
        for cand, g in c["gates"].items():
            family.append((c, cand, g))
    # One-sided p-values, so an arm that is significantly *worse* cannot be
    # counted as a discovery in the family-wise correction.
    pvals = [g["p_value"] for _, _, g in family]
    reject, qvals = benjamini_hochberg(pvals, q=args.fdr_q)

    print("=" * 108)
    print("GATE FAMILY  (delta = Spearman improvement over the best cheap baseline)")
    print("=" * 108)
    hdr = f"{'run':44s} {'kind':4s} {'candidate':16s} {'delta':>8s} {'CI lo':>8s} {'CI hi':>8s} {'q':>7s} {'gate':>5s}"
    print(hdr)
    for (c, cand, g), rej, q in zip(family, reject, qvals):
        print(
            f"{c['run']:44s} {c['kind']:4s} {cand:16s} "
            f"{g['delta']:+8.4f} {g['ci_lo']:+8.4f} {g['ci_hi']:+8.4f} "
            f"{q:7.4f} {'PASS' if g['passed'] else 'fail':>5s}"
        )

    # ---- per-candidate roll-up -------------------------------------------
    print("\n" + "=" * 108)
    print("PER-CANDIDATE ROLL-UP  (a candidate is promoted only if every cell passes)")
    print("=" * 108)
    by_cand = defaultdict(list)
    for (c, cand, g), rej, q in zip(family, reject, qvals):
        by_cand[cand].append((c, g, rej, q))

    verdicts = {}
    for cand in CANDIDATES:
        entries = by_cand.get(cand, [])
        if not entries:
            continue
        deltas = [g["delta"] for _, g, _, _ in entries]
        n_pass = sum(1 for _, g, _, _ in entries if g["passed"])
        n_bh = sum(1 for _, _, r, _ in entries if r)
        seeds_pass = defaultdict(set)
        for c, g, _, _ in entries:
            if g["passed"]:
                seeds_pass[(c["model"], c["kind"], c["ctx"], c["source"])].add(
                    c["seed"]
                )
        repeats = sum(1 for v in seeds_pass.values() if len(v) >= 2)
        verdicts[cand] = {
            "cells": len(entries),
            "cells_passing_gate": n_pass,
            "cells_significant_after_bh": n_bh,
            "configs_passing_on_both_seeds": repeats,
            "delta_min": min(deltas),
            "delta_median": sorted(deltas)[len(deltas) // 2],
            "delta_max": max(deltas),
            "promoted": n_pass == len(entries) and repeats > 0,
        }
        v = verdicts[cand]
        print(
            f"{cand:16s} passed {n_pass:2d}/{len(entries):2d} cells, "
            f"BH-significant {n_bh:2d}, both-seed configs {repeats:2d}, "
            f"delta median {v['delta_median']:+.4f} "
            f"[{v['delta_min']:+.4f}, {v['delta_max']:+.4f}]  "
            f"=> {'PROMOTE' if v['promoted'] else 'STOP'}"
        )

    # ---- attribution ------------------------------------------------------
    print("\n" + "=" * 108)
    print("ATTRIBUTION  (mean over cells; CI is the range of per-cell point estimates)")
    print("=" * 108)
    attrib = defaultdict(list)
    for c in cells:
        for name, a in c["attribution"].items():
            attrib[(c["kind"], name)].append(a["delta"])
    for (kind, name), vals in sorted(attrib.items()):
        mean = sum(vals) / len(vals)
        print(
            f"  {kind}  {name:26s} mean {mean:+.4f}  "
            f"range [{min(vals):+.4f}, {max(vals):+.4f}]  n={len(vals)}"
        )

    # ---- the ranking table ------------------------------------------------
    print("\n" + "=" * 108)
    print(
        "BLOCKED SPEARMAN  (within matched-norm groups; size and layer held constant)"
    )
    print("=" * 108)
    by_kind = defaultdict(lambda: defaultdict(list))
    for c in cells:
        for m, v in c["blocked_rho"].items():
            if not math.isnan(v):
                by_kind[c["kind"]][m].append(v)
    for kind in sorted(by_kind):
        print(f"\n  kind={kind}")
        ranked = sorted(
            by_kind[kind].items(),
            key=lambda kv: -sum(kv[1]) / len(kv[1]),
        )
        for m, vals in ranked:
            mean = sum(vals) / len(vals)
            print(f"    {m:22s} {mean:+.4f}   [{min(vals):+.4f}, {max(vals):+.4f}]")

    # ---- numerical floor --------------------------------------------------
    floors = [c["floor_kl_max"] for c in cells if c.get("floor_kl_max") is not None]
    mins = [c["exact_kl_min"] for c in cells if c.get("exact_kl_min") is not None]
    medians = [c["exact_kl_median"] for c in cells if c.get("exact_kl_median")]
    if floors and mins and medians:
        worst = max(floors)
        smallest = min(mins)
        med = sorted(medians)[len(medians) // 2]
        print("\n" + "=" * 108)
        print("MEASUREMENT FLOOR  (an independent re-prefill of the same prompt)")
        print("=" * 108)
        print(f"  worst control divergence   {worst:.3e}")
        if worst == 0.0:
            print("    the re-prefill is bit-reproducible on this stack, so the")
            print("    control cannot bound the estimator's own resolution")
        # A negative divergence is impossible in exact arithmetic, so the most
        # negative value measured is a direct read of the float32 summation
        # error in the divergence itself -- a better floor than a control that
        # comes back bit-identical.
        resolution = abs(min(0.0, smallest))
        print(
            f"  most negative divergence   {smallest:.3e}  "
            f"(impossible in exact arithmetic; this is the float32 floor)"
        )
        print(f"  median measured damage     {med:.3e}", end="")
        if resolution > 0:
            print(f"  ({med / resolution:,.0f}x that floor)")
        else:
            print()
        if resolution > 0 and med < 100 * resolution:
            print("  WARNING: median damage is within 100x the floor")

    out = {
        "n_cells": len(cells),
        "fdr_q": args.fdr_q,
        "verdicts": verdicts,
        "attribution_mean": {
            f"{k}:{n}": sum(v) / len(v) for (k, n), v in attrib.items()
        },
        "blocked_rho_mean": {
            kind: {m: sum(v) / len(v) for m, v in ms.items()}
            for kind, ms in by_kind.items()
        },
        "cells": cells,
    }
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "stage0_rollup.json"), "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {os.path.join(args.out, 'stage0_rollup.json')}")

    promoted = [
        k for k, v in verdicts.items() if v["promoted"] and v["tier"] == "deployable"
    ]
    print(
        "\nSTAGE 0 DECISION: "
        + (
            f"promote {promoted}"
            if promoted
            else "no deployable candidate clears the pre-registered gate "
            "on any tensor kind -- stop"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
