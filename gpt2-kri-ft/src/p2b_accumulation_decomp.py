#!/usr/bin/env python3
"""P2.0b — accumulation decomposition of attention "diffuseness" (lattice/KRI plan).

The P2.0 battery measured single-query block attention pooled over GQA heads and found
GQA head-pooling is only a modest (~0.06-0.08) concentration reducer -- but the mechanism
finding's top1%_mass_share (0.36 on Qwen2.5-0.5B) is the concentration of RECEIVED mass
ACCUMULATED over ALL query positions × heads × layers. That query-accumulation axis (the
one H2O actually consumes) was untested. This reproduces the original token-level metric
and DECOMPOSES the drop across aggregation axes, so we can say WHERE Qwen2.5's diffuseness
comes from:

  single_query   : mean over query positions of a single query's token-attention top1%
                   / norm-entropy concentration (NO accumulation) -- per-head peakiness.
  accum_perhead  : accumulate received mass over ALL causal queries, per head, then the
                   token-level top1% / concentration, averaged over heads/layers
                   (query-accumulation effect = single_query -> accum_perhead).
  accum_pooled_kv: sum received mass over the query heads sharing each KV head, per layer
                   (GQA head-pooling effect = accum_perhead -> accum_pooled_kv).
  global         : sum received over ALL heads AND layers -> one token distribution; this
                   is the ORIGINAL top1%_mass_share (layer+head pooling on top).

Reproduces ~0.36 for Qwen2.5-0.5B if faithful. Then the decomposition attributes the drop.
Non-Qwen large-GQA-group controls (Llama-3.2-3B g3, -1B g4) break the family↔group
confound. WikiText-103 L1024, matching the mechanism finding. Free W7900; no_grad.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from niah_task import load_filler_sentences
from diag_perhead_oracle import capture_q
from niah_evict_perhead import get_keys


def top1_and_conc(mass, dim=-1, frac=0.01, eps=1e-12):
    """(top-frac mass share, 1 - H/log n) of a nonneg mass vector along dim."""
    p = mass / mass.sum(dim, keepdim=True).clamp_min(eps)
    n = p.shape[dim]
    k = max(1, int(math.ceil(frac * n)))
    t1 = p.topk(k, dim=dim).values.sum(dim)
    H = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim)
    conc = 1.0 - H / math.log(n) if n > 1 else torch.ones_like(t1)
    return t1, conc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--eval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    group = Hq // Hkv
    nL = model.config.num_hidden_layers
    Dh = getattr(model.config, "head_dim", model.config.hidden_size // Hq)
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)

    acc = {k: [] for k in [
        "single_query_t1", "single_query_conc",
        "accum_perhead_t1", "accum_perhead_conc",
        "accum_pooledkv_t1", "accum_pooledkv_conc",
        "global_t1", "global_conc",
    ]}
    n_used = 0
    for ei in range(args.eval):
        text, need = "", args.length * 5
        while len(text) < need:
            text += sents[rng.randrange(len(sents))] + " "
        ids = tok(text, add_special_tokens=False)["input_ids"][: args.length]
        if len(ids) < 256:
            continue
        ids_t = torch.tensor(ids)
        T = len(ids) - 1
        past = model(ids_t[:-1].unsqueeze(0).to(device), use_cache=True).past_key_values
        keys = get_keys(past)  # list[layer] of [1, Hkv, T, D]
        # capture post-RoPE Q at ALL prefill positions
        qs, _ = capture_q(model, ids_t[:-1], list(range(T)), device)  # list[layer] [Hq,T,D]
        n_used += 1

        global_received = torch.zeros(T, device=device)  # summed over heads AND layers
        # a few sampled single-query positions for the no-accumulation baseline
        sq_pos = sorted(set(int(x) for x in torch.linspace(T // 4, T - 1, 8).tolist()))

        for li in range(nL):
            q = qs[li].float()  # [Hq, T, D]
            k = keys[li][0].float().repeat_interleave(group, dim=0)  # [Hq, T, D]
            scale = 1.0 / math.sqrt(Dh)
            # full causal attention per head, accumulate received mass over queries
            # logits[h, i, j] = q_i . k_j ; causal j<=i ; softmax over j
            logits = torch.einsum("hid,hjd->hij", q, k) * scale  # [Hq, T, T]
            mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
            logits.masked_fill_(mask.unsqueeze(0), float("-inf"))
            a = logits.softmax(-1)  # [Hq, T, T] attention (row=query)
            # single-query baseline (no accumulation): top1 of each sampled query row
            for p in sq_pos:
                t1, c = top1_and_conc(a[:, p, : p + 1], dim=-1)  # per head at query p
                acc["single_query_t1"].extend(t1.tolist())
                acc["single_query_conc"].extend(c.tolist())
            # accumulate received mass over all queries -> [Hq, T]
            received = a.sum(1)  # column sum = received mass per key token, per head
            del logits, a
            # accum per-head token-level concentration
            t1, c = top1_and_conc(received, dim=-1)
            acc["accum_perhead_t1"].extend(t1.tolist())
            acc["accum_perhead_conc"].extend(c.tolist())
            # pool over the query heads sharing each kv-head
            pooled = received.view(Hkv, group, T).sum(1)  # [Hkv, T]
            t1, c = top1_and_conc(pooled, dim=-1)
            acc["accum_pooledkv_t1"].extend(t1.tolist())
            acc["accum_pooledkv_conc"].extend(c.tolist())
            global_received += received.sum(0)  # add this layer's all-head received
            torch.cuda.empty_cache()

        # global: the ORIGINAL top1%_mass_share (all heads + all layers)
        t1, c = top1_and_conc(global_received, dim=-1)
        acc["global_t1"].append(t1.item())
        acc["global_conc"].append(c.item())
        print(f"  [{n_used}/{args.eval}] T={T} global_top1={t1.item():.3f}", flush=True)

    def summ(xs):
        if not xs:
            return None
        t = torch.tensor(xs, dtype=torch.float)
        return {"n": len(xs), "mean": round(t.mean().item(), 4), "median": round(t.median().item(), 4)}

    out = {
        "model": args.model, "Hq": Hq, "Hkv": Hkv, "group": group, "nL": nL,
        "length": args.length, "n_used": n_used,
        "levels": {k: summ(v) for k, v in acc.items()},
    }
    # decomposition of the top1% concentration drop (higher top1 = peakier)
    sq = summ(acc["single_query_t1"])["mean"]
    ap_ = summ(acc["accum_perhead_t1"])["mean"]
    pk = summ(acc["accum_pooledkv_t1"])["mean"]
    gl = summ(acc["global_t1"])["mean"]
    out["decomp_top1"] = {
        "single_query": sq,
        "accum_perhead": ap_,
        "pooled_kv": pk,
        "global_original": gl,
        "drop_query_accumulation": round(sq - ap_, 4),
        "drop_head_pooling": round(ap_ - pk, 4),
        "drop_layer_pooling": round(pk - gl, 4),
        "note": "which axis produces the diffuseness; global should ~reproduce the "
        "mechanism-finding top1%_mass_share",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["decomp_top1"], indent=2))
    print(f"P2B_DECOMP_DONE {args.out}")


if __name__ == "__main__":
    main()
