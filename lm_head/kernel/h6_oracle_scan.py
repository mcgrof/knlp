#!/usr/bin/env python3
"""LM-head cluster-routing oracle scan (H6).

Cheap upper-bound scan of the output-vocabulary routing structure: cluster the
vocabulary into C groups, then ask "if a router picked the m best clusters
perfectly, would the true next token still be covered, and at what fetch
fraction?" No router/predictor is trained here -- this measures only the oracle
ceiling, to see whether that ceiling rises with model scale before committing
PEFT compute.

The oracle ranks clusters by the dense model's own aggregated next-token
probability mass and keeps the top-m. Metrics, averaged over eval positions:

  fetch     fraction of LM-head rows touched (size of selected clusters / V)
  truenext  the true next token's cluster is among the selected m
  top10     all of the dense top-10 tokens' clusters are selected
  ppl_nofb  perplexity decoding with ONLY the selected clusters' logits
            (no fallback); per-position nll capped for a finite mean

Cluster methods:
  idblock    contiguous token-id slabs, cluster = min(token_id // slab, C-1).
             centroid-free, single integer divide, addressable.
  kmeans_l2  k-means on the L2-normalized rows of the unembedding matrix.
             stronger but learned and model-specific.

Usage:
  python3 lm_head/h6_oracle_scan.py --model EleutherAI/pythia-410m \
      --num-seqs 200 --seq-len 512 --out OUT/pythia-410m.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def build_idblock(vocab_size: int, C: int) -> torch.Tensor:
    slab = math.ceil(vocab_size / C)
    ids = torch.arange(vocab_size)
    return torch.clamp(ids // slab, max=C - 1).long()


def build_kmeans_l2(W_U: torch.Tensor, C: int, seed: int) -> torch.Tensor:
    import numpy as np

    X = W_U.detach().float().cpu().numpy()
    # L2-normalize rows so k-means groups by output direction, not magnitude.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    # Full Lloyd k-means converges to balanced clusters; MiniBatch collapses
    # into one mega-cluster even with reassignment, on both 50k and 128k-152k
    # vocabularies. Use full Lloyd at every size, with cheaper init/iter counts
    # for very large vocabularies to keep the runtime tractable.
    from sklearn.cluster import KMeans

    if Xn.shape[0] <= 60000:
        n_init, max_iter = 4, 300
    else:
        n_init, max_iter = 2, 100
    km = KMeans(
        n_clusters=C,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        init="k-means++",
    )
    labels = km.fit_predict(Xn)
    return torch.from_numpy(labels).long()


def load_eval_ids(tokenizer, num_seqs: int, seq_len: int):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    need = num_seqs * seq_len
    if enc.numel() < need:
        num_seqs = enc.numel() // seq_len
        need = num_seqs * seq_len
    enc = enc[:need].view(num_seqs, seq_len)
    return enc


@torch.no_grad()
def oracle_scan(
    model, input_ids, cluster_of, C, m_list, device, nll_cap=30.0, pos_chunk=128
):
    """Return per-(method handled by caller) metrics dict keyed by m."""
    V = cluster_of.numel()
    cluster_of = cluster_of.to(device)
    cluster_sizes = torch.bincount(cluster_of, minlength=C).float()  # [C]

    # accumulators per m
    acc = {
        m: {"fetch": 0.0, "truenext": 0.0, "top10": 0.0, "nll": 0.0, "n": 0}
        for m in m_list
    }

    B = input_ids.shape[0]
    for b in range(B):
        ids = input_ids[b : b + 1].to(device)
        out = model(ids)
        logits = out.logits[0, :-1, :].float()  # [T-1, V] predict next
        y = ids[0, 1:].to(device)  # [T-1] true next
        N = logits.shape[0]
        for s in range(0, N, pos_chunk):
            lg = logits[s : s + pos_chunk]  # [n,V]
            yy = y[s : s + pos_chunk]  # [n]
            n = lg.shape[0]
            probs = F.softmax(lg, dim=-1)  # [n,V]
            cof = cluster_of.unsqueeze(0).expand(n, V)  # [n,V]
            cmass = torch.zeros(n, C, device=device).scatter_add_(1, cof, probs)
            top10 = lg.topk(10, dim=-1).indices  # [n,10]
            top10_cl = cluster_of[top10]  # [n,10]
            y_cl = cluster_of[yy]  # [n]
            for m in m_list:
                sel = cmass.topk(m, dim=-1).indices  # [n,m]
                sel_mask = torch.zeros(n, C, dtype=torch.bool, device=device)
                sel_mask.scatter_(1, sel, True)
                # fetch = sum of selected cluster sizes / V (per position)
                fetch = (sel_mask.float() * cluster_sizes).sum(1) / V  # [n]
                tn = sel_mask.gather(1, y_cl.unsqueeze(1)).squeeze(1)  # [n] bool
                t10 = sel_mask.gather(1, top10_cl).all(dim=1)  # [n] bool
                covered_tok = sel_mask.gather(1, cof)  # [n,V] bool
                masked = lg.masked_fill(~covered_tok, -1e9)
                logp = F.log_softmax(masked, dim=-1)
                nll = (-logp.gather(1, yy.unsqueeze(1)).squeeze(1)).clamp(max=nll_cap)
                a = acc[m]
                a["fetch"] += fetch.sum().item()
                a["truenext"] += tn.float().sum().item()
                a["top10"] += t10.float().sum().item()
                a["nll"] += nll.sum().item()
                a["n"] += n
    res = {}
    for m, a in acc.items():
        nn = max(a["n"], 1)
        res[m] = {
            "fetch": a["fetch"] / nn,
            "truenext": a["truenext"] / nn,
            "top10": a["top10"] / nn,
            "ppl_nofb": math.exp(a["nll"] / nn),
        }
    return res


@torch.no_grad()
def dense_ppl(model, input_ids, device):
    tot_nll, tot_n = 0.0, 0
    for b in range(input_ids.shape[0]):
        ids = input_ids[b : b + 1].to(device)
        logits = model(ids).logits[0, :-1, :].float()
        y = ids[0, 1:].to(device)
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, y.unsqueeze(1)).squeeze(1)
        tot_nll += nll.sum().item()
        tot_n += y.numel()
    return math.exp(tot_nll / max(tot_n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-seqs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--m", default="8,16,32,64")
    ap.add_argument("--methods", default="idblock,kmeans_l2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    W_U = model.get_output_embeddings().weight  # [V, d]
    V = W_U.shape[0]
    C = args.clusters
    m_list = [int(x) for x in args.m.split(",")]
    methods = args.methods.split(",")

    ids = load_eval_ids(tok, args.num_seqs, args.seq_len)
    print(
        f"[{args.model}] V={V} d={W_U.shape[1]} eval={tuple(ids.shape)} "
        f"C={C} m={m_list} methods={methods}",
        flush=True,
    )

    dppl = dense_ppl(model, ids, device)
    print(f"  dense_ppl={dppl:.2f}", flush=True)

    result = {
        "model": args.model,
        "vocab_size": int(V),
        "d_model": int(W_U.shape[1]),
        "tied_embeddings": bool(
            model.get_input_embeddings().weight.data_ptr() == W_U.data_ptr()
        ),
        "C": C,
        "m_list": m_list,
        "eval_seqs": int(ids.shape[0]),
        "eval_seq_len": int(ids.shape[1]),
        "dense_ppl": dppl,
        "methods": {},
    }

    for method in methods:
        if method == "idblock":
            cof = build_idblock(V, C)
        elif method == "kmeans_l2":
            cof = build_kmeans_l2(W_U, C, args.seed)
        else:
            raise SystemExit(f"unknown method {method}")
        cl_sizes = torch.bincount(cof, minlength=C)
        res = oracle_scan(model, ids, cof, C, m_list, device)
        result["methods"][method] = {
            "cluster_size_min": int(cl_sizes.min()),
            "cluster_size_max": int(cl_sizes.max()),
            "cluster_size_max_frac": float(cl_sizes.max()) / V,
            "by_m": {str(m): res[m] for m in m_list},
        }
        for m in m_list:
            r = res[m]
            print(
                f"  {method:10s} m={m:3d}  fetch={r['fetch']:.3f}  "
                f"truenext={r['truenext']:.3f}  top10={r['top10']:.3f}  "
                f"ppl_nofb={r['ppl_nofb']:.1f}",
                flush=True,
            )

    result["wall_seconds"] = round(time.time() - t0, 1)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"[wrote] {outp}  ({result['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
