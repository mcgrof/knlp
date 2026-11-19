#!/usr/bin/env python3
"""Score a selector that can abstain to the full model (CPU).

A server that serves each request on one expert pool must still answer
requests from domains that have no pool. The embedding-centroid
selector (mixed_stream.py) always picks a pool, so such requests run on
a wrong pool and lose answers. This script asks whether the selector
can instead abstain, sending a request far from every pool to the full
model, without sending many pooled requests there too.

Inputs are the prompt features from selector_features.py and the
pool-by-domain matrix from pool_matrix.py. The pick is the cosine-
nearest centroid of the mean input embedding, with centroids from the
first --n-train training questions of each pooled domain. A request is
abstained when a score falls below a threshold; it is then served by
the full model, otherwise by the picked pool.

Scores, all computed from pooled-domain training questions only:

    maxcos      cosine similarity to the nearest centroid
    knn         mean cosine similarity to the --k nearest pooled
                training questions

Each threshold is a quantile q (--quantiles) of the score over the
pooled training questions, computed out of fold (--folds): a question's
score comes from centroids or neighbours fitted without its fold. So q
is the share of pooled training questions that would be abstained.
Nothing is fitted or chosen on test questions, and no unpooled-domain
question is used.

A further rule, known-other, uses unpooled-domain training questions:
it adds a centroid per unpooled domain (the first --n-train training
questions) and abstains when one of those wins. It shows what knowing
the unpooled domain in advance buys, not what an unseen domain gets.

For each rule the script reports, per domain, the share abstained,
correct answers served, and net losses against the full model and,
for pooled domains, against the domain's own pool (the oracle pick).

Usage:
    abstain_eval.py FEATURES.pt POOL_MATRIX.json OUT.json
        [--n-train 128] [--quantiles 0.02 0.05 0.1 0.2] [--k 8]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runtime_identity, write_json  # noqa: E402

KEY = "correct_per_char"


def unit(x):
    return x / x.norm(dim=1, keepdim=True)


def centroids(X, y, n_cls):
    return unit(torch.stack([X[y == c].mean(0) for c in range(n_cls)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("matrix")
    ap.add_argument("out")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument(
        "--quantiles", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2]
    )
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    F = torch.load(args.features, weights_only=False)
    with open(args.matrix) as f:
        M = json.load(f)
    classes = [m for m in M["models"] if m != "full"]
    domains = list(F["domains"])
    other = [d for d in domains if d not in classes]
    ok = {}
    for m, cells in M["models"].items():
        ok[m] = {}
        for d in domains:
            rows = {r["id"]: r[KEY] for r in cells[d]["rows"]}
            ok[m][d] = [rows[i] for i in F["domains"][d]["test"]["ids"]]

    def emb(d, split):
        x = F["domains"][d][split]["emb"].double()
        return unit(x[: args.n_train] if split == "train" else x)

    Xtr = torch.cat([emb(c, "train") for c in classes])
    ytr = torch.cat([torch.full((args.n_train,), i) for i, _ in enumerate(classes)])
    assert len(Xtr) == args.n_train * len(classes)
    C = centroids(Xtr, ytr, len(classes))

    def knn_score(X, ref):
        s = X @ ref.T
        return s.topk(args.k, dim=1).values.mean(1)

    # Out-of-fold scores of the pooled training questions.
    fold = torch.arange(len(Xtr)) % args.folds
    oof = {"maxcos": torch.empty(len(Xtr), dtype=torch.double)}
    oof["knn"] = torch.empty(len(Xtr), dtype=torch.double)
    for f in range(args.folds):
        tr, te = fold != f, fold == f
        Cf = centroids(Xtr[tr], ytr[tr], len(classes))
        oof["maxcos"][te] = (Xtr[te] @ Cf.T).max(1).values
        oof["knn"][te] = knn_score(Xtr[te], Xtr[tr])

    test = {d: emb(d, "test") for d in domains}
    pick = {d: (test[d] @ C.T).argmax(1) for d in domains}
    score = {
        "maxcos": {d: (test[d] @ C.T).max(1).values for d in domains},
        "knn": {d: knn_score(test[d], Xtr) for d in domains},
    }

    def serve(abst):
        """Per-domain report for a boolean abstain mask per domain."""
        rep = {}
        for d in domains:
            got, full = [], ok["full"][d]
            for i, a in enumerate(abst[d].tolist()):
                m = "full" if a else classes[int(pick[d][i])]
                got.append(ok[m][d][i])

            def net(base):
                lost = sum(b and not g for b, g in zip(base, got))
                return lost - sum(g and not b for b, g in zip(base, got))

            r = {
                "n": len(got),
                "abstained": int(abst[d].sum()),
                "correct": sum(got),
                "full_correct": sum(full),
                "net_losses_vs_full": net(full),
            }
            if d in classes:
                r["net_losses_vs_oracle"] = net(ok[d][d])
            rep[d] = r
        pooled = [d for d in domains if d in classes]
        rep["_summary"] = {
            "pooled_abstain_rate": sum(rep[d]["abstained"] for d in pooled)
            / sum(rep[d]["n"] for d in pooled),
            "other_abstain_rate": (
                sum(rep[d]["abstained"] for d in other)
                / sum(rep[d]["n"] for d in other)
                if other
                else None
            ),
            "worst_pooled_vs_oracle": max(
                rep[d]["net_losses_vs_oracle"] for d in pooled
            ),
            "worst_other_vs_full": (
                max(rep[d]["net_losses_vs_full"] for d in other) if other else None
            ),
        }
        return rep

    res = {
        "runtime": runtime_identity(),
        "args": vars(args),
        "classes": classes,
        "other": other,
        "rules": {},
    }
    no = {d: torch.zeros(len(test[d]), dtype=torch.bool) for d in domains}
    res["rules"]["never"] = serve(no)
    every = {d: torch.tensor([d in other] * len(test[d])) for d in domains}
    res["rules"]["oracle-abstain"] = serve(every)
    for s in ("maxcos", "knn"):
        for q in args.quantiles:
            t = float(torch.quantile(oof[s], q))
            rep = serve({d: score[s][d] < t for d in domains})
            rep["_summary"]["threshold"] = t
            res["rules"][f"{s}-q{q}"] = rep
    if other:
        # One centroid per unpooled domain, as for the pooled ones.
        Co = unit(torch.stack([emb(d, "train").mean(0) for d in other]))
        res["rules"]["known-other"] = serve(
            {
                d: (test[d] @ Co.T).max(1).values > (test[d] @ C.T).max(1).values
                for d in domains
            }
        )
    res["score_quantiles_pooled_train"] = {
        s: {
            str(q): float(torch.quantile(oof[s], q))
            for q in (0.02, 0.05, 0.1, 0.2, 0.5)
        }
        for s in oof
    }
    res["score_quantiles_test"] = {
        s: {
            d: {str(q): float(torch.quantile(score[s][d], q)) for q in (0.1, 0.5, 0.9)}
            for d in domains
        }
        for s in score
    }

    print(
        "rule              pooled_abst  other_abst  worst_pooled_vs_oracle  other_vs_full"
    )
    for name, rep in res["rules"].items():
        s = rep["_summary"]
        print(
            f"{name:16}  {s['pooled_abstain_rate']:10.1%}  "
            f"{(s['other_abstain_rate'] or 0):10.1%}  {s['worst_pooled_vs_oracle']:22d}"
            f"  {s['worst_other_vs_full']}"
        )
    write_json(args.out, res)


if __name__ == "__main__":
    main()
