#!/usr/bin/env python3
"""Score a random pool pick for domains with no pool (CPU).

A pool server must serve requests from domains it has no pool for. The
embedding-centroid selector (mixed_stream.py) sends such a request to
the nearest pool. This script asks whether that pick is any better than
a pool chosen at random, which would cost nothing to compute and could
be whichever pool is already resident.

Inputs are the prompt features from selector_features.py and the
pool-by-domain matrix from pool_matrix.py, which scores every pool and
the full model on every test question of every domain. Each case is a
domain with no pool, served by a set of candidate pools:

    unpooled domains   every domain without a pool, candidates = all
                       pools
    held-out pooled    each pooled domain treated as unseen: its own
                       pool is removed, candidates = the other pools,
                       centroids fitted on the other pooled domains only

For each case the script reports correct answers of the test questions
for the full model, each candidate pool, the selector pick (cosine-
nearest centroid of the mean input embedding, centroids from the first
--n-train training questions of each candidate's domain), and a pick
made uniformly at random per request. The random pick's correct count
is a sum of independent Bernoulli variables, one per question, with p
the share of candidate pools that answer it correctly; the script gives
its exact mean, standard deviation and the exact probability that it
scores at least as many as the selector. Net losses against the full
model are full-model correct minus served correct.

Usage:
    random_pool_eval.py FEATURES.pt POOL_MATRIX.json OUT.json
        [--n-train 128]
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runtime_identity, write_json  # noqa: E402

KEY = "correct_per_char"


def unit(x):
    return x / x.norm(dim=1, keepdim=True)


def count_pmf(ps):
    """Exact distribution of a sum of independent Bernoulli(p)."""
    pmf = [1.0]
    for p in ps:
        nxt = [0.0] * (len(pmf) + 1)
        for k, v in enumerate(pmf):
            nxt[k] += v * (1 - p)
            nxt[k + 1] += v * p
        pmf = nxt
    return pmf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("matrix")
    ap.add_argument("out")
    ap.add_argument("--n-train", type=int, default=128)
    args = ap.parse_args()

    F = torch.load(args.features, weights_only=False)
    with open(args.matrix) as f:
        M = json.load(f)
    pools = [m for m in M["models"] if m != "full"]
    domains = list(F["domains"])
    ok = {}
    for m, cells in M["models"].items():
        ok[m] = {}
        for d in domains:
            rows = {r["id"]: r[KEY] for r in cells[d]["rows"]}
            ok[m][d] = [bool(rows[i]) for i in F["domains"][d]["test"]["ids"]]

    def emb(d, split):
        x = F["domains"][d][split]["emb"].double()
        return unit(x[: args.n_train] if split == "train" else x)

    cases = [(d, list(pools), "unpooled") for d in domains if d not in pools]
    cases += [(d, [p for p in pools if p != d], "held-out") for d in pools]

    res = {
        "runtime": runtime_identity(),
        "args": vars(args),
        "pools": pools,
        "cases": {},
    }
    hdr = (
        f"{'case':22} {'full':>4} {'sel':>4} {'rand':>6} {'sd':>4} "
        f"{'P(r>=s)':>7} {'best':>12} {'worst':>12}"
    )
    print(hdr)
    for d, cand, kind in cases:
        C = unit(torch.stack([emb(c, "train").mean(0) for c in cand]))
        pick = (emb(d, "test") @ C.T).argmax(1).tolist()
        n = len(pick)
        sel = sum(ok[cand[k]][d][i] for i, k in enumerate(pick))
        ps = [sum(ok[c][d][i] for c in cand) / len(cand) for i in range(n)]
        pmf = count_pmf(ps)
        mean = sum(ps)
        sd = math.sqrt(sum(p * (1 - p) for p in ps))
        per_pool = {c: sum(ok[c][d]) for c in cand}
        best = max(per_pool, key=per_pool.get)
        worst = min(per_pool, key=per_pool.get)
        full = sum(ok["full"][d])
        r = {
            "kind": kind,
            "candidates": cand,
            "n": n,
            "full_correct": full,
            "pool_correct": per_pool,
            "selector_correct": sel,
            "selector_picks": {c: pick.count(k) for k, c in enumerate(cand)},
            "random_mean": mean,
            "random_sd": sd,
            "random_p_at_least_selector": sum(pmf[sel:]),
            "random_p_above_selector": sum(pmf[sel + 1 :]),
            "any_pool_correct": sum(max(ok[c][d][i] for c in cand) for i in range(n)),
            "net_losses_vs_full": {
                "selector": full - sel,
                "random_mean": full - mean,
                "best_pool": full - per_pool[best],
                "worst_pool": full - per_pool[worst],
            },
        }
        res["cases"][f"{d} ({kind})"] = r
        print(
            f"{d + ' (' + kind + ')':22} {full:4d} {sel:4d} {mean:6.1f} "
            f"{sd:4.1f} {r['random_p_at_least_selector']:7.3f} "
            f"{best + ' ' + str(per_pool[best]):>12} "
            f"{worst + ' ' + str(per_pool[worst]):>12}"
        )
    write_json(args.out, res)


if __name__ == "__main__":
    main()
