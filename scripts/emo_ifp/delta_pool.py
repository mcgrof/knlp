#!/usr/bin/env python3
"""Write the delta pool file for a switch from pool A to pool B (CPU).

A switch from A to B needs only the routed experts of B that A lacks;
the rest of B can be copied from A's weights already in GPU memory. This
script writes a pool file whose per-layer routed list is exactly those
experts, in B's order, so pack_pool_layer.py --stacked packs them into
a delta file per layer. The shared experts are copied from B; the
stacked packing holds routed experts only. The delta's metadata names
both pools so a loader can rebuild B's order.

Usage:
    delta_pool.py POOL_A.json POOL_B.json OUT.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_a")
    ap.add_argument("pool_b")
    ap.add_argument("out")
    args = ap.parse_args()
    with open(args.pool_a) as f:
        a = json.load(f)
    with open(args.pool_b) as f:
        b = json.load(f)
    assert set(a["layers"]) == set(b["layers"])
    out = {k: v for k, v in b.items() if k != "layers"}
    out["note"] = "routed experts of POOL_B absent from POOL_A, in POOL_B order"
    out["delta_from"] = args.pool_a
    out["delta_to"] = args.pool_b
    out["layers"] = {}
    n = 0
    for li, lb in b["layers"].items():
        have = set(a["layers"][li]["routed"])
        routed = [e for e in lb["routed"] if e not in have]
        out["layers"][li] = {"routed": routed, "shared": lb["shared"]}
        n += len(routed)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{n} routed experts in the delta")


if __name__ == "__main__":
    main()
