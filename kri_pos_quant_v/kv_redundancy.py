#!/usr/bin/env python3
"""Per-family long-context KV redundancy diagnostic.

Tests the hypothesis for why lattice residual_rel (anti-redundancy block dedup)
beats H2O on SmolLM2/Mistral/Llama but loses on Qwen2.5: residual_rel pays off
only when long-context KV is REDUNDANT (many similar blocks -> dedup keeps
diverse, informative ones). It measures, per model, the redundancy of long-context
key blocks: mean off-diagonal cosine similarity of block key-centroids and the
effective rank of the block-centroid matrix (per layer/kv-head, averaged). Higher
mean-sim / lower effective-rank = more redundant.

Usage:
  python3 kri_pos_quant_v/kv_redundancy.py --models Qwen/Qwen2.5-1.5B,...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def get_keys(cache):
    if hasattr(cache, "key_cache"):
        return cache.key_cache
    return [layer.keys for layer in cache.layers]


@torch.no_grad()
def redundancy(model, ids, device, bs=16):
    """Mean off-diagonal block-centroid cosine sim + effective rank, avg over
    layers/kv-heads."""
    past = model(ids.unsqueeze(0).to(device), use_cache=True).past_key_values
    keys = get_keys(past)
    sims, eranks = [], []
    for k in keys:  # [1,H,T,D]
        H, T, D = k.shape[1], k.shape[2], k.shape[3]
        NB = T // bs
        if NB < 4:
            continue
        kk = k[0, :, : NB * bs, :].float().view(H, NB, bs, D).mean(2)  # [H,NB,D]
        kn = F.normalize(kk, dim=-1)
        for h in range(H):
            S = kn[h] @ kn[h].t()  # [NB,NB]
            off = S - torch.eye(NB, device=device)
            sims.append((off.sum() / (NB * (NB - 1))).item())
            # effective rank from singular values of the centroid matrix
            sv = torch.linalg.svdvals(kk[h].float())
            p = sv / sv.sum()
            erank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
            eranks.append(erank / NB)  # normalized to [0,1]
    del past
    torch.cuda.empty_cache()
    return sum(sims) / len(sims), sum(eranks) / len(eranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="Qwen/Qwen2.5-1.5B,HuggingFaceTB/SmolLM2-1.7B,meta-llama/Llama-3.2-3B,mistralai/Mistral-7B-v0.1",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    device = torch.device(args.device)
    result = {"length": args.length, "models": {}}
    for mp in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16).to(
            device
        )
        model.eval()
        enc = tok(text, return_tensors="pt").input_ids[0]
        rng = random.Random(args.seed)
        msim, merank = [], []
        for _ in range(args.n):
            s = rng.randint(0, enc.numel() - args.length - 1)
            sim, er = redundancy(model, enc[s : s + args.length], device)
            msim.append(sim)
            merank.append(er)
        tag = mp.split("/")[-1]
        ms = sum(msim) / len(msim)
        me = sum(merank) / len(merank)
        result["models"][tag] = {"mean_block_sim": ms, "norm_effrank": me}
        print(
            f"  {tag:18s} mean_block_sim={ms:.4f}  norm_effrank={me:.4f} "
            f"(higher sim / lower effrank = more redundant)",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
