#!/usr/bin/env python3
"""Pick an expert pool from the prompt text alone (CPU).

A serving process that keeps one expert pool resident must decide, when a
request arrives and before the model runs, which pool the request needs,
so that a switch can start early. This script fits two cheap selectors
on the calibration examples that built each pool (eight per domain) and
tests them on held-out requests of every domain.

Each selector sees only the request's question: the text after the last
"Question:" marker, without the few-shot demonstrations, which would name
the domain outright, and without the trailing "Answer:".

    tokens     multinomial naive Bayes over the model tokenizer's token
               ids, add-one smoothing, equal class priors
    embedding  the mean of the model's input embedding rows for the
               question's tokens (no transformer layer runs), assigned to
               the domain whose calibration centroid has the highest
               cosine similarity

    router     with --router (GPU), the mean routed-expert softmax of
               layer 0's router over the question's tokens, assigned to
               the nearest calibration centroid by cosine. Only the
               embeddings and layer 0's attention run; the forward pass
               stops inside the router, before any expert weight is read.
               EMO trains its routers to keep a document on a consistent
               set of experts, so this is the model's own domain signal.

The embedding table is read straight from the checkpoint's safetensors
shard. The script reports per-domain accuracy, every misassigned request,
each selector's decision margin and the selection time per request.

Usage:
    domain_selector.py MODEL_DIR OUT.json \\
        --domain arc CAL.json TEST.json --domain gsm8k CAL.json TEST.json \\
        [--router]
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

EMBED = "model.embed_tokens.weight"


def question(context):
    q = context.rsplit("Question:", 1)[-1]
    if q.rstrip().endswith("Answer:"):
        q = q.rstrip()[: -len("Answer:")]
    return q.strip()


def load_embedding(model_dir):
    from safetensors import safe_open

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][EMBED]
    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
        return f.get_tensor(EMBED).float()


class TokenBayes:
    def __init__(self, docs_by_domain, vocab):
        self.vocab = vocab
        self.logp = {}
        for d, docs in docs_by_domain.items():
            c = Counter(t for doc in docs for t in doc)
            total = sum(c.values()) + vocab
            self.logp[d] = (c, total)

    def scores(self, doc):
        out = {}
        for d, (c, total) in self.logp.items():
            out[d] = sum(math.log((c.get(t, 0) + 1) / total) for t in doc)
        return out


class EmbedCentroid:
    def __init__(self, emb, docs_by_domain):
        self.emb = emb
        self.cent = {}
        for d, docs in docs_by_domain.items():
            v = torch.stack([self.vec(doc) for doc in docs]).mean(0)
            self.cent[d] = v / v.norm()

    def vec(self, doc):
        v = self.emb[torch.tensor(doc)].mean(0)
        return v / v.norm()

    def scores(self, doc):
        v = self.vec(doc)
        return {d: float(v @ c) for d, c in self.cent.items()}


class StopAtRouter(Exception):
    pass


class RouterCentroid:
    """Layer-0 router signature; see the module docstring."""

    def __init__(self, model, docs_by_domain):
        self.model = model
        moe = model.model.layers[0].mlp
        self.routed = moe.num_experts - moe.num_shared_experts
        self.cap = []

        def hook(mod, inp, out):
            self.cap.append(out.detach().float())
            raise StopAtRouter()

        moe.gate.register_forward_hook(hook)
        self.cent = {}
        for d, docs in docs_by_domain.items():
            v = torch.stack([self.vec(doc) for doc in docs]).mean(0)
            self.cent[d] = v / v.norm()

    @torch.no_grad()
    def vec(self, doc):
        self.cap.clear()
        try:
            self.model(input_ids=torch.tensor([doc], device="cuda"))
        except StopAtRouter:
            pass
        assert len(self.cap) == 1
        p = torch.softmax(self.cap[0][:, : self.routed], dim=-1).mean(0)
        return (p / p.norm()).cpu()

    def scores(self, doc):
        v = self.vec(doc)
        torch.cuda.synchronize()
        return {d: float(v @ c) for d, c in self.cent.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument(
        "--domain",
        nargs=3,
        action="append",
        metavar=("NAME", "CAL", "TEST"),
        required=True,
    )
    ap.add_argument("--router", action="store_true", help="add the GPU selector")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa
    cal, test = {}, {}
    for name, cal_p, test_p in args.domain:
        with open(cal_p) as f:
            cal[name] = [question(it["context"]) for it in json.load(f)["items"]]
        with open(test_p) as f:
            test[name] = [
                (it["id"], question(it["context"])) for it in json.load(f)["items"]
            ]
    cal_ids = {d: [enc(q) for q in qs] for d, qs in cal.items()}
    for d in cal:
        overlap = set(cal[d]) & {q for _, q in test[d]}
        assert not overlap, f"{d}: calibration question in the test set"

    emb = load_embedding(args.model_dir)
    selectors = {
        "tokens": TokenBayes(cal_ids, len(tok)),
        "embedding": EmbedCentroid(emb, cal_ids),
    }
    if args.router:
        from common import load_model

        model, _ = load_model(args.model_dir)
        selectors["router"] = RouterCentroid(model, cal_ids)
    res = {
        "runtime": runtime_identity(),
        "domains": {d: {"calibration": len(cal[d]), "test": len(test[d])} for d in cal},
        "selectors": {},
    }
    for sname, sel in selectors.items():
        rows, times = [], []
        for d, items in test.items():
            for iid, q in items:
                t0 = time.perf_counter()
                sc = sel.scores(enc(q))
                times.append(time.perf_counter() - t0)
                ranked = sorted(sc, key=sc.get, reverse=True)
                rows.append(
                    {
                        "domain": d,
                        "id": iid,
                        "pred": ranked[0],
                        "margin": sc[ranked[0]] - sc[ranked[1]],
                        "scores": sc,
                    }
                )
        per = {
            d: {
                "correct": sum(r["pred"] == d for r in rows if r["domain"] == d),
                "n": sum(r["domain"] == d for r in rows),
            }
            for d in test
        }
        wrong = [r for r in rows if r["pred"] != r["domain"]]
        right = sorted(r["margin"] for r in rows if r["pred"] == r["domain"])
        res["selectors"][sname] = {
            "per_domain": per,
            "correct": sum(p["correct"] for p in per.values()),
            "n": len(rows),
            "wrong": [
                {k: r[k] for k in ("domain", "id", "pred", "margin")} for r in wrong
            ],
            "smallest_correct_margins": right[:5],
            "select_ms_median": sorted(times)[len(times) // 2] * 1e3,
            "rows": rows,
        }
        print(sname, per, f"{len(wrong)} wrong", flush=True)
    write_json(args.out, res)


if __name__ == "__main__":
    main()
