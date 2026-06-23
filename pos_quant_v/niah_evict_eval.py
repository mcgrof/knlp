#!/usr/bin/env python3
"""Lattice-KRI gap #3: KV-eviction routers on a REAL long-context retrieval task.

The lattice verdict closed gaps #1 (7B) and #2 (multi-seed) on KL/PPL, but the
last gap is a real long-context TASK where eviction actually bites. This evicts
KV blocks at decode time and measures NIAH retrieval accuracy at matched retained
KV, comparing the lattice residual_rel router against H2O, StreamingLLM
(sink_recent), relevance-only, random, and full cache.

Tractable and fits 7B at L4096 because it uses SDPA prefill + single-token decode
(no full attention matrix, no full-vocab logprobs -- which is what OOM'd the KL
harness). Eviction is GLOBAL token-level (block-level keep/evict shared across
layers/heads) via a 2D attention mask on the decode -- a legitimate KV-eviction
scheme that answers "does residual_rel keep the answer-bearing blocks better."

Block relevance uses q.k_centroid (no full attention matrix); the query probe is
the last prefill position's key, aggregated over layers/kv-heads (the validated
lattice last-position-K-probe). H2O accumulates attention mass over a sample of
query positions.

Usage:
  python3 pos_quant_v/niah_evict_eval.py --model Qwen/Qwen2.5-7B-Instruct \
      --length 4096 --needles 4 --eval 20 --out OUT/niah_evict.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, token_content_mask, load_filler_sentences


def get_keys(cache):
    if hasattr(cache, "key_cache"):
        return cache.key_cache
    return [layer.keys for layer in cache.layers]


@torch.no_grad()
def block_centroids(cache, bs):
    """Global per-block key centroid [NB, D] (mean over layers, kv-heads, tokens
    in block) and the query probe (last-position key, global) [D]."""
    keys = get_keys(cache)  # list [1,Hkv,T,D]
    T = keys[0].shape[2]
    NB = (T + bs - 1) // bs
    D = keys[0].shape[3]
    # global key per token = mean over layers, heads -> [T, D]
    kt = torch.stack([k[0].float().mean(0) for k in keys], 0).mean(0)  # [T, D] fp32
    kt = F.normalize(kt, dim=-1)
    cent = torch.zeros(NB, D, device=kt.device)
    cnt = torch.zeros(NB, device=kt.device)
    idx = torch.arange(T, device=kt.device) // bs
    cent.index_add_(0, idx, kt)
    cnt.index_add_(0, idx, torch.ones(T, device=kt.device))
    cent = cent / cnt.clamp(min=1).unsqueeze(1)
    q = kt[-1]  # last-position key as query probe
    return F.normalize(cent, dim=-1), F.normalize(q, dim=0), kt, idx


def residual_rel_select(cent, q, k_budget):
    """Greedy orthogonal matching pursuit (the lattice residual_rel router)."""
    NB = cent.shape[0]
    chosen = torch.zeros(NB, dtype=torch.bool, device=cent.device)
    q_resid = q.clone()
    picks = []
    for _ in range(min(k_budget, NB)):
        score = (cent * F.normalize(q_resid, dim=0)).sum(-1)
        score = score.masked_fill(chosen, float("-inf"))
        p = int(score.argmax())
        chosen[p] = True
        picks.append(p)
        kc = cent[p]
        q_resid = q_resid - (q_resid @ kc) * kc
    return picks


def rel_only_select(cent, q, k_budget):
    score = (cent * q).sum(-1)
    return torch.topk(score, min(k_budget, cent.shape[0])).indices.tolist()


def h2o_select(kt, idx, NB, k_budget, n_probe=48):
    """Heavy-hitter: accumulate attention mass over a sample of query positions."""
    T = kt.shape[0]
    probe_pos = torch.linspace(0, T - 1, n_probe).long()
    mass = torch.zeros(NB, device=kt.device)
    for p in probe_pos:
        scores = kt[: p + 1] @ kt[p]  # [p+1]
        a = F.softmax(scores, dim=-1)
        mass.index_add_(0, idx[: p + 1], a)
    return torch.topk(mass, min(k_budget, NB)).indices.tolist()


def keep_mask_from_blocks(blocks, NB, T, bs, sink_blocks, recent_blocks):
    keep_b = torch.zeros(NB, dtype=torch.bool)
    for b in blocks:
        keep_b[b] = True
    for b in range(sink_blocks):
        keep_b[b] = True
    for b in range(max(0, NB - recent_blocks), NB):
        keep_b[b] = True
    tok = torch.zeros(T, dtype=torch.bool)
    for b in range(NB):
        if keep_b[b]:
            tok[b * bs : min(T, (b + 1) * bs)] = True
    return tok


def slice_cache(past, kept_idx):
    """Fresh DynamicCache keeping only kept token positions (real eviction). The
    kept keys retain their original RoPE (baked in at prefill), so attention with
    a true-position query stays correct."""
    from transformers import DynamicCache

    new = DynamicCache()
    for i, layer in enumerate(past.layers):
        new.update(
            layer.keys[:, :, kept_idx, :].clone(),
            layer.values[:, :, kept_idx, :].clone(),
            i,
        )
    return new


@torch.no_grad()
def decode_evicted(model, last_id, past, keep_tok, device, T, max_new=20):
    """Evict (drop) non-kept KV, decode the answer with true position_ids so the
    kept keys' original RoPE gives correct relative positions."""
    kept_idx = keep_tok.nonzero(as_tuple=True)[0].to(device)
    cur = slice_cache(past, kept_idx)
    cache_len = int(kept_idx.numel())
    ids = last_id.view(1, 1).to(device)
    gen = []
    for s in range(max_new):
        # position_ids = TRUE position (for RoPE vs the kept keys' original RoPE);
        # cache_position = actual slot in the evicted cache (so HF sizes the mask
        # to the small cache, not the large true position).
        pos = torch.tensor([[T + s]], device=device)
        cpos = torch.tensor([cache_len + s], device=device)
        out = model(
            ids,
            past_key_values=cur,
            position_ids=pos,
            cache_position=cpos,
            use_cache=True,
        )
        nxt = out.logits[0, -1].argmax()
        gen.append(int(nxt))
        cur = out.past_key_values
        ids = nxt.view(1, 1)
    del cur, out
    torch.cuda.empty_cache()
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-blocks", type=int, default=8)
    ap.add_argument("--budgets", default="16,32,64,128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)
    bs = args.block_size
    budgets = [int(x) for x in args.budgets.split(",")]
    routers = ["full", "residual_rel", "rel_only", "h2o", "sink_recent", "random"]

    samples = []
    for _ in range(args.eval):
        text, spans, needles, qa = build_context(
            tok, args.length, args.needles, sents, rng
        )
        ids, _ = token_content_mask(tok, text, spans, 10**9)
        samples.append((torch.tensor(ids), qa[1]))

    def correct(gen, val):
        return val in tok.decode(gen)

    agg = {r: {b: [0, 0] for b in budgets} for r in routers}  # [hits, retained_sum]
    agg["full"] = {0: [0, 0]}
    for ids, val in samples:
        past = model(ids[:-1].unsqueeze(0).to(device), use_cache=True).past_key_values
        T = get_keys(past)[0].shape[2]
        NB = (T + bs - 1) // bs
        cent, q, kt, idx = block_centroids(past, bs)
        # full baseline
        gen = decode_evicted(
            model, ids[-1], past, torch.ones(T, dtype=torch.bool), device, T
        )
        agg["full"][0][0] += correct(gen, val)
        agg["full"][0][1] += T
        for b in budgets:
            for r in routers:
                if r == "full":
                    continue
                if r == "residual_rel":
                    blocks = residual_rel_select(cent, q, b)
                elif r == "rel_only":
                    blocks = rel_only_select(cent, q, b)
                elif r == "h2o":
                    blocks = h2o_select(kt, idx, NB, b)
                elif r == "sink_recent":
                    blocks = []
                elif r == "random":
                    blocks = rng.sample(range(NB), min(b, NB))
                keep = keep_mask_from_blocks(
                    blocks, NB, T, bs, args.sink_blocks, args.recent_blocks
                )
                gen = decode_evicted(model, ids[-1], past, keep, device, T)
                agg[r][b][0] += correct(gen, val)
                agg[r][b][1] += int(keep.sum())
        del past, cent, q, kt, idx
        torch.cuda.empty_cache()

    n = len(samples)
    result = {"model": args.model, "length": args.length, "eval": n, "arms": {}}
    print(f"[{args.model}] L={args.length} eval={n}  (acc @ retained-KV fraction)")
    fb = agg["full"][0]
    print(f"  full           acc={fb[0]/n:.3f}  ret=1.000")
    result["arms"]["full"] = {"acc": fb[0] / n, "ret": 1.0}
    for r in routers:
        if r == "full":
            continue
        for b in budgets:
            hits, ret = agg[r][b]
            acc = hits / n
            retf = ret / (n * agg["full"][0][1] / n) if agg["full"][0][1] else 0
            result["arms"][f"{r}_K{b}"] = {"acc": acc, "ret": retf}
            print(f"  {r:13s} K={b:>3} acc={acc:.3f}  ret={retf:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
