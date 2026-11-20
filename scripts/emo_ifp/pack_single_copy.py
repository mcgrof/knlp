#!/usr/bin/env python3
"""Store each routed expert once, ordered so every pool is a few runs (CPU).

Today every pool is packed as its own per-layer file, and every switch
between two pools as its own delta file, so an expert kept by several
pools is stored several times on disk. This script writes the other
layout: one file per layer that holds every routed expert exactly once,
with each pool and each switch delta described as a list of byte ranges
into it. A pool then costs no extra disk space and no extra writes.

The order of experts in the file decides how many separate ranges a
pool needs. Each expert gets a membership mask over the given pools
(bit i set when pool i keeps it). Experts are sorted by the rank of
their mask in the reflected Gray code, ties by expert index, so that
adjacent groups differ in one pool and each pool's experts fall into a
few contiguous runs. The first pool given becomes the lowest bit,
whose membership changes most often along the code; the last pool given
is one run per layer, so list the most often loaded pool last.

Each layer file is raw bytes with no header: the gate/up weights (w13)
of every expert in file order, then the down weights (w2) in the same
order, so a pool load reads w13 runs into the pool's w13 tensor and w2
runs into its w2 tensor. The source is the packed full model
(pack_pool_layer.py --stacked on full_pool.py's pool), whose routed
tensors are in expert index order. LAYOUT.json records the order, the
offsets, every pool's and every ordered pool pair's runs, and a SHA-256
per expert slice.

Usage:
    pack_single_copy.py FULL_PACKED_DIR OUT_DIR NAME=POOL.json... \
        [--layers 16]
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import stacked_name  # noqa: E402


def gray_rank(nbits):
    """Map each mask to its position in the reflected Gray code."""
    return {i ^ (i >> 1): i for i in range(1 << nbits)}


def runs(order, members):
    """Contiguous runs (start index, count) of members in file order."""
    out = []
    for pos, e in enumerate(order):
        if e in members:
            if out and out[-1][0] + out[-1][1] == pos:
                out[-1][1] += 1
            else:
                out.append([pos, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_dir")
    ap.add_argument("out_dir")
    ap.add_argument("pools", nargs="+", help="NAME=POOL.json, lowest bit first")
    ap.add_argument("--layers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names, pools = [], []
    for spec in args.pools:
        name, path = spec.split("=", 1)
        with open(path) as f:
            pools.append(json.load(f))
        names.append(name)
    rank = gray_rank(len(pools))
    layout = {
        "note": "every routed expert once per layer; w13 region then w2 region",
        "order": "reflected Gray code of pool membership masks, ties by index",
        "pools": names,
        "layers": {},
    }
    for li in range(args.layers):
        src = os.path.join(args.full_dir, f"layer{li}.safetensors")
        with open(src + ".json") as f:
            ids = json.load(f)["routed_expert_ids"]
        with safe_open(src, "pt") as f:
            w13 = f.get_tensor(stacked_name(li, "w13"))
            w2 = f.get_tensor(stacked_name(li, "w2"))
        assert w13.shape[0] == w2.shape[0] == len(ids)
        row = {e: i for i, e in enumerate(ids)}
        sets = [set(p["layers"][str(li)]["routed"]) for p in pools]
        mask = {e: sum(1 << b for b, s in enumerate(sets) if e in s) for e in ids}
        order = sorted(ids, key=lambda e: (rank[mask[e]], e))
        b13 = w13[0].numel() * w13.element_size()
        b2 = w2[0].numel() * w2.element_size()
        out = os.path.join(args.out_dir, f"layer{li}.bin")
        hashes = {}
        with open(out, "wb") as f:
            for which, t, nb in (("w13", w13, b13), ("w2", w2, b2)):
                for e in order:
                    raw = t[row[e]].contiguous().view(torch.uint8).numpy().tobytes()
                    assert len(raw) == nb
                    f.write(raw)
                    hashes[f"{which}:{e}"] = hashlib.sha256(raw).hexdigest()
        lay = {
            "file": os.path.basename(out),
            "order": order,
            "w13_offset": 0,
            "w13_bytes": b13,
            "w2_offset": b13 * len(order),
            "w2_bytes": b2,
            "masks": {str(e): mask[e] for e in order},
            "pool_runs": {n: runs(order, s) for n, s in zip(names, sets)},
            "delta_runs": {
                f"{a}->{b}": runs(order, sb - sa)
                for a, sa in zip(names, sets)
                for b, sb in zip(names, sets)
                if a != b
            },
            "sha256": hashes,
        }
        layout["layers"][str(li)] = lay
        npool = {n: len(r) for n, r in lay["pool_runs"].items()}
        print(f"layer {li}: {os.path.getsize(out)} bytes, runs per pool {npool}")
    with open(os.path.join(args.out_dir, "LAYOUT.json"), "w") as f:
        json.dump(layout, f, indent=1)


if __name__ == "__main__":
    main()
