#!/usr/bin/env python3
"""Fit domain selectors and score the quality they serve (CPU).

Inputs are the prompt features from selector_features.py and the
pool-by-domain matrix from pool_matrix.py. Each selector picks one of
the pools (the classes are the domains that have a pool) from a
request's question text, before the main forward pass. A pick is judged
two ways: whether it names the request's own domain, and the quality
actually served, i.e. whether the chosen pool answers the item correctly
in the matrix. The served result is paired against the oracle pick (the
request's own pool) and against the full model. Requests from a domain
with no pool are scored only against the full model.

Selectors, each fitted on the first N training requests of every pooled
domain, for each N in --n-train:

    tokens-nb          multinomial naive Bayes over token ids, add-one
                       smoothing, equal priors
    emb-centroid       nearest class centroid by cosine, mean input
                       embedding
    emb-logreg         multinomial logistic regression, same feature
    routerK-centroid   nearest centroid by cosine over the mean routed
                       softmax of the first K layers' routers
    routerK-logreg     logistic regression, same feature

Logistic regression standardizes features with training statistics and
minimizes mean cross-entropy plus lam times the squared weights, with
L-BFGS. lam is chosen from --lams by four-fold cross-validation on the
training requests only (ties go to the larger lam). Nothing is fitted or
chosen on test requests.

A selector qualifies when, in every pooled domain, its served quality
has at most --max-net-losses net losses against the oracle pick on the
primary (per character) metric.

Usage:
    selector_eval.py FEATURES.pt POOL_MATRIX.json OUT.json
        [--n-train 8 32 128] [--depths 1 2 4 16]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runtime_identity, write_json  # noqa

KEY = "correct_per_char"


class TokenNB:
    def __init__(self, X, y, classes, vocab):
        self.classes = classes
        self.tab = []
        for c in range(len(classes)):
            cnt = Counter(t for doc, yy in zip(X, y) if yy == c for t in doc)
            self.tab.append((cnt, sum(cnt.values()) + vocab))

    def predict(self, doc):
        sc = [
            sum(math.log((cnt.get(t, 0) + 1) / tot) for t in doc)
            for cnt, tot in self.tab
        ]
        return max(range(len(sc)), key=sc.__getitem__)


class Centroid:
    def __init__(self, X, y, classes):
        X = X / X.norm(dim=1, keepdim=True)
        self.C = torch.stack([X[y == c].mean(0) for c in range(len(classes))])
        self.C = self.C / self.C.norm(dim=1, keepdim=True)

    def predict(self, x):
        return int((self.C @ (x / x.norm())).argmax())


def _fit_lr(X, y, nc, lam):
    W = torch.zeros(X.shape[1], nc, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(nc, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [W, b], max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + lam * (W**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


class LogReg:
    def __init__(self, X, y, classes, lams):
        X = X.double()
        self.mu = X.mean(0)
        self.sd = X.std(0).clamp_min(1e-6)
        Z = (X - self.mu) / self.sd
        nc = len(classes)
        # Four stratified folds: the j-th example of each class goes to fold j % 4.
        fold = torch.zeros(len(y), dtype=torch.long)
        for c in range(nc):
            idx = (y == c).nonzero().flatten()
            fold[idx] = torch.arange(len(idx)) % 4
        cv = {}
        for lam in lams:
            ok = 0
            for f in range(4):
                tr, te = fold != f, fold == f
                mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
                W, b = _fit_lr((X[tr] - mu) / sd, y[tr], nc, lam)
                ok += int(((((X[te] - mu) / sd) @ W + b).argmax(1) == y[te]).sum())
            cv[lam] = ok / len(y)
        best = max(cv.values())
        self.lam = max(lam for lam in lams if cv[lam] == best)
        self.cv = {str(k): v for k, v in cv.items()}
        self.W, self.b = _fit_lr(Z, y, nc, self.lam)

    def predict(self, x):
        z = (x.double() - self.mu) / self.sd
        return int((z @ self.W + self.b).argmax())


def paired(base, other):
    lost = sum(1 for a, b in zip(base, other) if a and not b)
    won = sum(1 for a, b in zip(base, other) if b and not a)
    return {"correct": sum(other), "base_correct": sum(base), "net_losses": lost - won}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("matrix")
    ap.add_argument("out")
    ap.add_argument("--n-train", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 16])
    ap.add_argument(
        "--lams", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    )
    ap.add_argument("--max-net-losses", type=int, default=4)
    args = ap.parse_args()

    # Our own features file; it holds the runtime identity, not only tensors.
    F = torch.load(args.features, weights_only=False)
    with open(args.matrix) as f:
        M = json.load(f)
    classes = [m for m in M["models"] if m != "full"]
    domains = list(F["domains"])
    assert all(c in domains for c in classes), (classes, domains)
    unseen = [d for d in domains if d not in classes]

    # Per-item correctness: ok[model][domain] in the features' test order.
    ok = {}
    for m, cells in M["models"].items():
        ok[m] = {}
        for d in domains:
            rows = {r["id"]: r[KEY] for r in cells[d]["rows"]}
            ok[m][d] = [rows[i] for i in F["domains"][d]["test"]["ids"]]

    def feat(kind, split, d, depth=None):
        x = F["domains"][d][split]
        if kind == "tokens":
            return x["tokens"]
        if kind == "emb":
            return x["emb"].double()
        return x["router"][:, :depth, :].reshape(len(x["ids"]), -1).double()

    specs = [("tokens-nb", "tokens", None, "nb"), ("emb-centroid", "emb", None, "c")]
    specs.append(("emb-logreg", "emb", None, "lr"))
    for k in args.depths:
        specs.append((f"router{k}-centroid", "router", k, "c"))
        specs.append((f"router{k}-logreg", "router", k, "lr"))

    def serve(picks):
        """Served-quality summary for one set of picks {domain: [class idx]}."""
        out = {}
        for d in domains:
            served = [ok[classes[p]][d][i] for i, p in enumerate(picks[d])]
            o = {
                "picks": dict(Counter(classes[p] for p in picks[d])),
                "vs_full": paired(ok["full"][d], served),
            }
            if d in classes:
                o["vs_oracle"] = paired(ok[d][d], served)
                o["selected_right"] = sum(classes[p] == d for p in picks[d])
            out[d] = o
        worst = max(out[d]["vs_oracle"]["net_losses"] for d in classes)
        return out, worst

    res = {
        "runtime": runtime_identity(),
        "args": vars(args),
        "classes": classes,
        "unseen_domains": unseen,
        "feature_timing_ms": F["timing"],
        "baselines": {},
        "selectors": {},
    }
    # Baselines: the oracle pick and every fixed pool.
    n_test = {d: len(F["domains"][d]["test"]["ids"]) for d in domains}
    oracle = {d: [classes.index(d) if d in classes else 0] * n_test[d] for d in domains}
    o, w = serve(oracle)
    res["baselines"]["oracle"] = {"serve": o, "worst_net_losses": w}
    for ci, c in enumerate(classes):
        o, w = serve({d: [ci] * n_test[d] for d in domains})
        res["baselines"][f"always-{c}"] = {"serve": o, "worst_net_losses": w}

    for n in args.n_train:
        for name, kind, depth, model in specs:
            Xs, ys = [], []
            for ci, c in enumerate(classes):
                x = feat(kind, "train", c, depth)
                assert len(x) >= n, (c, len(x), n)
                Xs.append(x[:n])
                ys += [ci] * n
            y = torch.tensor(ys)
            t0 = time.perf_counter()
            if model == "nb":
                sel = TokenNB([t for X in Xs for t in X], ys, classes, F["vocab"])
            elif model == "c":
                sel = Centroid(torch.cat(Xs), y, classes)
            else:
                sel = LogReg(torch.cat(Xs), y, classes, args.lams)
            fit_s = time.perf_counter() - t0
            picks, ts = {}, []
            for d in domains:
                X = feat(kind, "test", d, depth)
                picks[d] = []
                for i in range(n_test[d]):
                    t0 = time.perf_counter()
                    picks[d].append(sel.predict(X[i]))
                    ts.append(time.perf_counter() - t0)
            o, w = serve(picks)
            right = sum(o[d]["selected_right"] for d in classes)
            total = sum(n_test[d] for d in classes)
            entry = {
                "n_train_per_domain": n,
                "selected_right": right,
                "n": total,
                "worst_net_losses_vs_oracle": w,
                "qualifies": w <= args.max_net_losses,
                "head_ms_median": sorted(ts)[len(ts) // 2] * 1e3,
                "fit_s": fit_s,
                "serve": o,
            }
            if depth is not None:
                entry["router_layers"] = depth
                entry["feature_ms_median"] = F["timing"][str(depth)]["median_ms"]
            if model == "lr":
                entry["lam"], entry["cv"] = sel.lam, sel.cv
            res["selectors"][f"{name}@{n}"] = entry
            losses = " ".join(
                f"{d}:{o[d]['vs_oracle']['net_losses']:+d}" for d in classes
            )
            print(
                f"N={n:>3} {name:>18}: {right}/{total} right, net losses vs "
                f"oracle {losses}{' QUALIFIES' if entry['qualifies'] else ''}",
                flush=True,
            )
    write_json(args.out, res)


if __name__ == "__main__":
    main()
