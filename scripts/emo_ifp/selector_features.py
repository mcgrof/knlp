#!/usr/bin/env python3
"""Compute prompt features for domain selectors (GPU).

A selector must pick a request's expert pool before the main forward
pass. This script computes, for each request's question text (the text
after the last "Question:" marker, without the few-shot demonstrations
and without "Answer:", as in domain_selector.py), the features the
selectors in selector_eval.py use:

    tokens   the question's token ids;
    emb      the mean of the model's input embedding rows over the
             question's tokens (no transformer layer runs);
    router   for each layer, the mean routed-expert softmax of that
             layer's router over the question's tokens, excluding the
             first token. EMO sends the first token of every document to
             one dedicated expert cluster whatever the domain (EMO paper,
             appendix D), so it carries no domain signal.

The router features of layer k depend only on layers 0 to k-1, so one
forward pass of the full model gives the features of every depth. A
selector that reads the routers of the first k layers needs those
layers' full expert sets resident; the cost of that tier is reported by
selector_eval.py. To time such a selector, the script reruns the test
questions with the forward pass stopped inside layer k-1's router, for
each k in --depths, and records the median time with a GPU synchronize.

The full model runs with the fused MoE kernel and the tuned tile table.

On a pool server the routers of layers 1 and later do not see the full
model: layer k's router reads the hidden state that layers 0 to k-1
computed with whichever pool is resident. With --resident NAME=POOL,
the script recomputes the router features once per named pool, with
that pool's experts in every layer, run as the pool server runs them:
each layer's router keeps only the pool's rows (routed experts, then
the shared expert), and the fused kernel reads expert k of the pool
from the full stacked weights through a slot map (fused_moe.py). The
features are still the full router's softmax over all routed experts,
computed from the same hidden state, so they are comparable across
pools and with the full-model features. Layer 0's router has no MoE
before it, so its features must not change with the pool; the script
checks that they are bitwise equal to the full-model ones.

Usage:
    selector_features.py MODEL_DIR OUT.pt --configs TUNED.json \\
        --domain arc TRAIN.json TEST.json --domain health TRAIN.json TEST.json
        [--depths 1 2 4 16] [--resident NAME=POOL.json ...]
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, runtime_identity  # noqa
from domain_selector import question  # noqa


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--domain", nargs=3, action="append", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 16])
    ap.add_argument("--resident", action="append", default=[], metavar="NAME=POOL")
    args = ap.parse_args()

    from fused_moe import ConfigTable, install_fused

    model, tok = load_model(args.model_dir)
    # Read the expert counts before the fused block replaces the MoE.
    moe0 = model.model.layers[0].mlp
    routed = moe0.num_experts - moe0.num_shared_experts
    install_fused(model, ConfigTable(args.configs))
    layers = model.model.layers
    emb_w = model.model.embed_tokens.weight
    cap = {}
    stop_at = [None]
    resident = [False]

    def make_hook(li):
        def hook(mod, inp, out):
            if resident[0]:
                return  # the pre-hook took the full router's logits
            cap[li] = out.detach()
            if stop_at[0] == li:
                raise Stop()

        return hook

    for li, layer in enumerate(layers):
        layer.mlp.gate.register_forward_hook(make_hook(li))

    # Resident-pool emulation: the full router weights, kept for the
    # features, and a pre-hook that computes them from the layer input.
    full_gate = [layer.mlp.gate.weight.data.clone() for layer in layers]

    def make_pre(li):
        def pre(mod, inp):
            if resident[0]:
                x = inp[0].reshape(-1, inp[0].size(-1))
                cap[li] = F.linear(x, full_gate[li])
                if stop_at[0] == li:
                    raise Stop()

        return pre

    for li, layer in enumerate(layers):
        layer.mlp.register_forward_pre_hook(make_pre(li))

    def set_resident(pool):
        for li, layer in enumerate(layers):
            m = layer.mlp
            if pool is None:
                m.gate.weight.data = full_gate[li]
                m.slot_map = None
                continue
            r = pool["layers"][str(li)]["routed"]
            sh = pool["layers"][str(li)]["shared"]
            assert sh == list(range(routed, routed + len(sh))), sh
            m.gate.weight.data = full_gate[li][r + sh].contiguous()
            m.slot_map = torch.tensor(r, dtype=torch.int64, device="cuda")
        resident[0] = pool is not None

    @torch.no_grad()
    def run(ids, stop=None):
        cap.clear()
        stop_at[0] = stop
        try:
            model(input_ids=torch.tensor([ids], device="cuda"))
        except Stop:
            pass
        stop_at[0] = None

    out = {
        "runtime": runtime_identity(),
        "args": vars(args),
        "vocab": len(tok),
        "routed": routed,
        "num_layers": len(layers),
        "domains": {},
    }

    def router_feats(ids):
        run(ids)
        with torch.no_grad():
            return torch.stack(
                [
                    torch.softmax(cap[li][1:, :routed].float(), dim=-1).mean(0)
                    for li in range(len(layers))
                ]
            ).cpu()

    for name, train_p, test_p in args.domain:
        out["domains"][name] = {}
        for split, path in (("train", train_p), ("test", test_p)):
            with open(path) as f:
                items = json.load(f)["items"]
            ids_l, toks, embs, routs = [], [], [], []
            for it in items:
                ids = tok(question(it["context"]), add_special_tokens=False)[
                    "input_ids"
                ]
                assert len(ids) >= 2, it["id"]
                r = router_feats(ids)
                with torch.no_grad():
                    e = emb_w[torch.tensor(ids, device="cuda")].float().mean(0)
                ids_l.append(it["id"])
                toks.append(ids)
                embs.append(e.cpu())
                routs.append(r)
            out["domains"][name][split] = {
                "ids": ids_l,
                "tokens": toks,
                "emb": torch.stack(embs),
                "router": torch.stack(routs),
            }
            print(f"{name} {split}: {len(items)} items", flush=True)

    # Router features with each named pool resident.
    out["resident"] = {}
    for spec in args.resident:
        rname, rpath = spec.split("=", 1)
        with open(rpath) as f:
            pool = json.load(f)
        set_resident(pool)
        res = {"pool": rpath, "domains": {}}
        for name, dom in out["domains"].items():
            res["domains"][name] = {}
            for split, d in dom.items():
                R = torch.stack([router_feats(ids) for ids in d["tokens"]])
                # Layer 0's router has no MoE before it.
                assert torch.equal(R[:, 0], d["router"][:, 0]), (rname, name, split)
                res["domains"][name][split] = {"router": R}
        set_resident(None)
        out["resident"][rname] = res
        print(f"resident {rname}: done", flush=True)

    # Selection time for a selector reading the first k routers.
    test_ids = [t for d in out["domains"].values() for t in d["test"]["tokens"]]
    timing = {}
    for k in args.depths:
        stop = k - 1 if k < len(layers) else None
        for ids in test_ids[:8]:
            run(ids, stop)
        torch.cuda.synchronize()
        ts = []
        for ids in test_ids:
            t0 = time.perf_counter()
            run(ids, stop)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        timing[str(k)] = {
            "median_ms": ts[len(ts) // 2] * 1e3,
            "p90_ms": ts[int(len(ts) * 0.9)] * 1e3,
            "n": len(ts),
        }
        print(f"routers of {k} layers: {timing[str(k)]}", flush=True)
    out["timing"] = timing
    out["question_tokens"] = {
        "median": sorted(len(t) for t in test_ids)[len(test_ids) // 2],
        "max": max(len(t) for t in test_ids),
    }
    torch.save(out, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
