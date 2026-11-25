#!/usr/bin/env python3
"""Costs of the fdec cartridge cells, with bootstrap intervals.

``research/cartridges_cas/scripts/cas_cart_loss.py`` scores the cartridge
files of one document with one reader. It writes a JSON summary and a
``.npz`` file that holds the loss of every scored position. The fdec
cartridge test runs it twice per document: with the 16-bit reader, and
with the int8 reader (``W8=1``). Each run scores the 16-bit cartridge and
its quantized copies from ``cas_cart_quantize.py``.

This script reads the two result files of one document. It prints the
loss of every cell and its cost against the 16-bit reader with the 16-bit
cartridge. It also prints the interaction term: the cost of the int8
reader with the quantized cartridge minus the sum of the two costs alone.
The bricks compose when that term is zero within its interval.

Positions of one packed sequence share a context, so they are not
independent. Each interval is a percentile bootstrap that draws whole
packed sequences with replacement.

    python3 scripts/fdec_cart_cells.py loss_w0.json loss_w1.json

``--packs-out FILE`` saves the per-sequence sums that the bootstrap uses.
``--packs FILE`` reads such a file in place of the two result files, so a
result can be checked again without the per-position data.
"""

import argparse
import json
import os

import numpy as np


def read_packs(w0_path, w1_path, base, names):
    """Per-sequence loss sums of every cell from two cas_cart_loss runs."""
    runs = []
    for path, int8 in ((w0_path, False), (w1_path, True)):
        meta = json.load(open(path))
        if bool(meta.get("int8_weights")) != int8:
            raise SystemExit(
                f"{path}: int8_weights is {meta.get('int8_weights')}, "
                f"expected {int8}; give the 16-bit run first"
            )
        runs.append((meta, np.load(os.path.splitext(path)[0] + ".npz")))
    (m0, z0), (m1, z1) = runs
    for key in ("parquet", "limit", "seed", "n_packed"):
        if m0.get(key) != m1.get(key):
            raise SystemExit(f"the two runs differ in {key}")
    n_packed = int(m0["n_packed"])
    count = z0[f"{base}__n"]
    if len(count) % n_packed:
        raise SystemExit("positions are not a multiple of the packed sequences")
    cells = {}
    for reader, z in (("16-bit", z0), ("int8", z1)):
        for name in [base] + names:
            if f"{name}__ce" not in z.files:
                raise SystemExit(f"no cartridge named {name} in the {reader} run")
            if not np.array_equal(z[f"{name}__n"], count):
                raise SystemExit(f"{reader}/{name}: scored positions differ")
            ce = z[f"{name}__ce"].reshape(n_packed, -1).sum(1)
            cells[f"{reader}/{name}"] = ce.tolist()
    return {
        "document": os.path.basename(str(m0.get("parquet", ""))),
        "model": m0.get("model"),
        "base": base,
        "cartridges": names,
        "entries": count.reshape(n_packed, -1).sum(1).astype(float).tolist(),
        "positions": int((count > 0).sum()),
        "cells": cells,
    }


def costs(packs, idx):
    """Loss of each cell, its relative cost in percent, and interactions."""
    n = np.asarray(packs["entries"])[idx].sum()
    loss = {k: np.asarray(v)[idx].sum() / n for k, v in packs["cells"].items()}
    ref = loss[f"16-bit/{packs['base']}"]
    cost = {k: 100.0 * (v / ref - 1.0) for k, v in loss.items()}
    inter, inter_abs = {}, {}
    for name in packs["cartridges"]:
        both, w8 = f"int8/{name}", f"int8/{packs['base']}"
        kv = f"16-bit/{name}"
        inter[name] = cost[both] - cost[w8] - cost[kv]
        inter_abs[name] = loss[both] - loss[w8] - loss[kv] + ref
    return loss, cost, inter, inter_abs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="*", help="16-bit run JSON, int8 run JSON")
    ap.add_argument("--base", default="bf16", help="name of the 16-bit cartridge")
    ap.add_argument(
        "--cartridges",
        default="v8,k8v8",
        help="names of the quantized cartridges, as given in CARTS",
    )
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--packs", help="read per-sequence sums from this file")
    ap.add_argument("--packs-out", help="write per-sequence sums to this file")
    ap.add_argument("--json", help="write the table to this file")
    args = ap.parse_args()

    if args.packs:
        packs = json.load(open(args.packs))
    elif len(args.results) == 2:
        names = [x for x in args.cartridges.split(",") if x]
        packs = read_packs(args.results[0], args.results[1], args.base, names)
    else:
        raise SystemExit("give the two result files, or --packs FILE")
    if args.packs_out:
        with open(args.packs_out, "w") as f:
            json.dump(packs, f)

    n_packs = len(packs["entries"])
    loss, cost, inter, inter_abs = costs(packs, np.arange(n_packs))
    rng = np.random.default_rng(args.seed)
    draws_cost = {k: np.empty(args.draws) for k in cost}
    draws_inter = {k: np.empty(args.draws) for k in inter}
    for d in range(args.draws):
        _, c, i, _ = costs(packs, rng.integers(0, n_packs, n_packs))
        for k in c:
            draws_cost[k][d] = c[k]
        for k in i:
            draws_inter[k][d] = i[k]

    def interval(draws):
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return float(lo), float(hi)

    print(
        f"{packs['document']}: {n_packs} packed sequences, "
        f"{packs['positions']} positions, {args.draws} bootstrap draws"
    )
    print(f"{'reader/cartridge':<18} {'loss':>9} {'cost':>9}   95% interval")
    rows = []
    ref = f"16-bit/{packs['base']}"
    for key in packs["cells"]:
        if key == ref:
            print(f"{key:<18} {loss[key]:9.6f} reference")
            rows.append({"cell": key, "loss": float(loss[key])})
            continue
        lo, hi = interval(draws_cost[key])
        print(
            f"{key:<18} {loss[key]:9.6f} {cost[key]:+8.3f}%   "
            f"{lo:+.3f}% to {hi:+.3f}%"
        )
        rows.append(
            {
                "cell": key,
                "loss": float(loss[key]),
                "cost_pct": float(cost[key]),
                "ci95_pct": [lo, hi],
            }
        )
    interactions = []
    for name in inter:
        lo, hi = interval(draws_inter[name])
        zero = lo <= 0.0 <= hi
        print(
            f"int8 reader with {name} minus the sum of the two alone: "
            f"{inter[name]:+.3f}% of the loss, 95% interval {lo:+.3f}% to "
            f"{hi:+.3f}%, {'contains' if zero else 'excludes'} zero; "
            f"{inter_abs[name]:+.7f} nats"
        )
        interactions.append(
            {
                "cartridge": name,
                "interaction_pct": float(inter[name]),
                "ci95_pct": [lo, hi],
                "interaction_nats": float(inter_abs[name]),
                "interval_contains_zero": bool(zero),
            }
        )
    if args.json:
        out = {
            "document": packs["document"],
            "model": packs.get("model"),
            "packed_sequences": n_packs,
            "positions": packs["positions"],
            "draws": args.draws,
            "seed": args.seed,
            "cells": rows,
            "interactions": interactions,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
