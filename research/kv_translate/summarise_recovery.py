# SPDX-License-Identifier: GPL-2.0
"""Render recovery rows as a table, with the oracle rows marked as oracle.

Every table this lane has published needed a note afterwards explaining which
rows were unavailable at serving time. Marking them in the table instead means
the note cannot be dropped when the table is copied.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--baseline", default="mapped_unrepaired")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = json.load(open(args.conditions))
    defs = cfg["definitions"]
    rows = [json.loads(l) for l in open(args.rows) if l.strip()]
    errs = [r for r in rows if r.get("error")]

    by = collections.defaultdict(dict)
    for r in rows:
        if not r.get("error"):
            by[(r["kind"], r["condition"])][r["doc"]] = r

    docs = sorted({r["doc"] for r in rows if not r.get("error")})
    base = {
        k: np.array([by[(k, args.baseline)][d]["primary"] for d in docs])
        for k in ("retrieval", "cloze")
        if (k, args.baseline) in by
    }

    out = {"documents": len(docs), "failed_cells": len(errs), "conditions": {}}
    print(f"{len(docs)} documents, {len(errs)} failed cells\n")
    hdr = (
        f"{'condition':20s} {'':6s} {'retr':>6s} {'net':>5s} "
        f"{'cloze':>6s} {'drop':>6s} {'healthy':>8s} {'empty':>6s} {'degen':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cond in sorted({c for _, c in by}):
        spec = defs.get(cond, {})
        mark = "ORACLE" if _is_oracle(spec) else ""
        rec = {"oracle": bool(mark), "spec": spec}
        line = f"{cond:20s} {mark:6s}"
        for k in ("retrieval", "cloze"):
            if (k, cond) in by:
                v = np.array([by[(k, cond)][d]["primary"] for d in docs])
                rec[k] = float(v.mean())
                if k == "retrieval":
                    net = float(v.sum() - base[k].sum()) if k in base else float("nan")
                    rec["net_retrieval"] = net
                    line += f" {v.mean():6.3f} {net:+5.0f}"
                else:
                    drop = (
                        float(base[k].mean() - v.mean()) if k in base else float("nan")
                    )
                    rec["cloze_drop"] = drop
                    line += f" {v.mean():6.3f} {drop:+6.3f}"
            else:
                line += (
                    f" {'-':>6s} {'-':>5s}"
                    if k == "retrieval"
                    else f" {'-':>6s} {'-':>6s}"
                )
        if ("free", cond) in by:
            f = by[("free", cond)]
            for name, field in (
                ("healthy", "healthy"),
                ("empty", "empty"),
                ("degen", "degenerate"),
            ):
                v = float(np.mean([f[d][field] for d in docs if d in f]))
                rec[name] = v
                line += f" {v:8.3f}" if name == "healthy" else f" {v:6.3f}"
        else:
            line += f" {'-':>8s} {'-':>6s} {'-':>6s}"
        print(line)
        out["conditions"][cond] = rec
    if errs:
        print(f"\n{len(errs)} failed cells:")
        for e in errs[:6]:
            print(f"  {e['kind']}/{e['condition']}/{e['doc']}: {e['error'][:90]}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0


def _is_oracle(spec):
    return (
        spec.get("repair") in ("oracle", "native")
        or spec.get("selector") == "native_attn"
    )


if __name__ == "__main__":
    raise SystemExit(main())
