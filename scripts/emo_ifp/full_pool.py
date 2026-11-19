#!/usr/bin/env python3
"""Write a pool file that keeps every expert (CPU).

A server that falls back to the full model for some requests can treat
the full model as one more pool: every routed expert, in index order,
plus the shared experts. This script writes that pool file from any
pool file of the same model, so pack_pool_layer.py --stacked can pack
the full model per layer and delta_pool.py can list what a switch from
a pool to the full model must read. Every pool is then a subset of the
full pool.

Usage:
    full_pool.py POOL.json NUM_EXPERTS OUT.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("num_experts", type=int)
    ap.add_argument("out")
    args = ap.parse_args()
    with open(args.pool) as f:
        p = json.load(f)
    out = {
        "method": "every expert",
        "routed_per_layer": None,
        "num_shared_experts": p["num_shared_experts"],
        "note": "all routed experts in index order, then the shared experts",
        "derived_from": args.pool,
        "layers": {},
    }
    for li, lp in p["layers"].items():
        shared = list(lp["shared"])
        routed = [e for e in range(args.num_experts) if e not in shared]
        out["layers"][li] = {"routed": routed, "shared": shared}
        out["routed_per_layer"] = len(routed)
    out["total_per_layer"] = out["routed_per_layer"] + len(shared)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{out['routed_per_layer']} routed + {len(shared)} shared per layer")


if __name__ == "__main__":
    main()
