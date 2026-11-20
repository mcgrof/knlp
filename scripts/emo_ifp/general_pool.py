#!/usr/bin/env python3
"""Build a general expert pool from several domain pools (CPU).

Each domain pool from calibrate_pool.py keeps the routed experts its own
calibration text uses most, so a request from a domain with no pool runs
on another domain's specialists. A general pool of the same size keeps
experts that several domains use instead. This script builds one in
either of two ways; both write a pool file in calibrate_pool.py's format
that apply_pool, pool_matrix.py and pack_pool_layer.py accept.

    vote OUT.json POOL.json...
        Per layer, rank routed experts by how many of the given pools
        keep them, breaking ties by the sum of their average router
        probabilities over the pools' calibration runs, and keep the
        top --routed. No GPU is needed.

    mix-calibration OUT.json CALIBRATION.json...
        Concatenate calibration files into one, for calibrate_pool.py.
        Each file contributes all its texts, labelled by the file's
        directory name in "groups". The calibration average is per
        token, so a domain with longer texts weighs more unless
        calibrate_pool.py runs with --balance-groups.

Both print, per layer, how many kept experts each number of source pools
shares (vote only).

Usage:
    general_pool.py vote OUT.json POOL.json... [--routed 32]
    general_pool.py mix-calibration OUT.json CALIBRATION.json...
"""

import argparse
import collections
import json
import os


def vote(args):
    pools = []
    for p in args.inputs:
        with open(p) as f:
            pools.append(json.load(f))
    layers0 = pools[0]["layers"]
    for p in pools:
        assert set(p["layers"]) == set(layers0)
    out = {k: v for k, v in pools[0].items() if k not in ("layers", "runtime")}
    out.update(
        {
            "method": "vote over domain pools, ties by summed router probability",
            "routed_per_layer": args.routed,
            "total_per_layer": args.routed + len(layers0["0"]["shared"]),
            "note": "routed experts kept by the most source pools",
            "sources": [os.path.abspath(p) for p in args.inputs],
            "layers": {},
        }
    )
    for k in ("calibration_file", "calibration_count", "seconds"):
        out.pop(k, None)
    shares = collections.Counter()
    for li in sorted(layers0, key=int):
        votes = collections.Counter(e for p in pools for e in p["layers"][li]["routed"])
        mass = collections.defaultdict(float)
        for p in pools:
            for e, v in enumerate(p["layers"][li]["avg_probs"]):
                mass[e] += v
        shared = list(layers0[li]["shared"])
        assert all(p["layers"][li]["shared"] == shared for p in pools)
        cand = [e for e in mass if e not in shared]
        cand.sort(key=lambda e: (-votes[e], -mass[e], e))
        routed = sorted(cand[: args.routed])
        kept_votes = collections.Counter(votes[e] for e in routed)
        shares.update(kept_votes)
        out["layers"][li] = {
            "routed": routed,
            "shared": shared,
            "votes": {str(e): votes[e] for e in routed},
        }
        print(
            f"layer {li}: kept by n pools "
            + " ".join(f"{n}:{kept_votes[n]}" for n in range(len(pools), -1, -1))
        )
    n = len(layers0)
    print(
        "mean per layer: "
        + " ".join(f"{k}:{shares[k] / n:.1f}" for k in range(len(pools), -1, -1))
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


def mix(args):
    items, texts, groups, sources = [], [], [], []
    for p in args.inputs:
        with open(p) as f:
            c = json.load(f)
        assert len(c["items"]) == len(c["texts"])
        items += c["items"]
        texts += c["texts"]
        groups += [os.path.basename(os.path.dirname(os.path.abspath(p)))] * len(
            c["texts"]
        )
        sources.append({"file": os.path.abspath(p), "count": len(c["texts"])})
    with open(args.out, "w") as f:
        json.dump(
            {"items": items, "texts": texts, "groups": groups, "sources": sources},
            f,
            indent=1,
        )
    print(f"{len(texts)} calibration texts from {len(sources)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["vote", "mix-calibration"])
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--routed", type=int, default=32)
    args = ap.parse_args()
    (vote if args.mode == "vote" else mix)(args)


if __name__ == "__main__":
    main()
