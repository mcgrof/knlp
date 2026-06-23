#!/usr/bin/env python3
"""Certified-decode replay kill-test for LM-head cluster pruning.

Offline replay over real hidden states: for each vocabulary partition, simulate
the certified greedy expansion and measure how many LM-head rows must actually
be fetched to *prove* the full-vocabulary argmax. This is the reality check on
whether progressive certified decoding saves anything, and whether an addressable
idblock partition can do it or whether only direction-coherent (k-means) clusters
can.

The certificate (per position, given hidden state h):
  open clusters in decreasing upper-bound order; track ell_star = max fetched
  logit; stop as soon as ell_star >= the largest upper bound among unopened
  clusters -- then no unfetched token can beat ell_star, so the argmax is
  certified. Lossless by construction; cost adapts per token.

Per-cluster ball bound on any member logit:
  w_v.h = centroid_c.h + (w_v - centroid_c).h <= centroid_c.h + r_c ||h||
  with centroid_c the mean unembedding row of cluster c and r_c its max member
  distance to that centroid. Computed on the raw rows regardless of how the
  clustering was chosen, so it is a valid upper bound for every partition.

Reports, per partition, the distribution of fetched fraction (rows touched / V)
to certify greedy argmax, and the share of tokens certified under 12.5/25/50%.

Usage:
  python3 kri_lm_head/replay_kill_test.py --model EleutherAI/pythia-410m \
      --positions 2000 --partitions idblock,kmeans_raw,kmeans_l2 --out OUT/x.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from h6_oracle_scan import build_idblock, build_kmeans_l2
from predictor_baseline import load_split, capture


def build_kmeans_raw(W_U, C, seed):
    import numpy as np

    X = W_U.detach().float().cpu().numpy()
    # Full Lloyd is too slow at large C; MiniBatch is the right tool for many
    # small clusters (the regime where bound radius gets small).
    if C <= 512 and X.shape[0] <= 60000:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=C, random_state=seed, n_init=4, max_iter=300)
    else:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=C,
            random_state=seed,
            batch_size=8192,
            n_init=3,
            max_iter=200,
            init="k-means++",
        )
    return torch.from_numpy(km.fit_predict(X)).long()


@torch.no_grad()
def cluster_geometry(W_U, cluster_of, C, device):
    """centroid_c [C,d], radius r_c [C], size [C] on device (raw rows)."""
    W = W_U.float().to(device)
    cof = cluster_of.to(device)
    d = W.shape[1]
    sizes = torch.bincount(cof, minlength=C).float()  # [C]
    cent = torch.zeros(C, d, device=device).index_add_(0, cof, W)
    cent = cent / sizes.clamp(min=1).unsqueeze(1)  # [C,d]
    # radius: max member distance to its centroid
    diff = W - cent[cof]  # [V,d]
    dist = diff.norm(dim=1)  # [V]
    r = torch.zeros(C, device=device).scatter_reduce(
        0, cof, dist, reduce="amax", include_self=True
    )
    return cent, r, sizes


@torch.no_grad()
def replay_partition(H, W_U, cluster_of, C, device, pos_chunk=128):
    """Mean/quantile fetched fraction to certify greedy argmax."""
    cof = cluster_of.to(device)
    Wt = W_U.float().to(device).t().contiguous()  # [d, V]
    V = Wt.shape[1]
    cent, r, sizes = cluster_geometry(W_U, cluster_of, C, device)
    sizes_row = sizes  # [C]
    fracs = []
    certified_token_ok = 0
    total = 0
    N = H.shape[0]
    for s in range(0, N, pos_chunk):
        Hc = H[s : s + pos_chunk].float().to(device)  # [nc, d]
        nc = Hc.shape[0]
        logits = Hc @ Wt  # [nc, V]
        # per-cluster max real logit
        ml = torch.full((nc, C), float("-inf"), device=device)
        ml.scatter_reduce_(
            1, cof.unsqueeze(0).expand(nc, V), logits, reduce="amax", include_self=True
        )
        true_argmax = logits.argmax(1)  # [nc]
        true_cl = cof[true_argmax]  # [nc]
        hn = Hc.norm(dim=1, keepdim=True)  # [nc,1]
        U = Hc @ cent.t() + hn * r.unsqueeze(0)  # [nc, C] ball upper bound
        # certified greedy expansion, vectorized
        order = U.argsort(dim=1, descending=True)  # [nc,C]
        U_sorted = U.gather(1, order)
        ml_sorted = ml.gather(1, order)
        size_sorted = sizes_row[order]  # [nc,C]
        ell_star = ml_sorted.cummax(dim=1).values  # [nc,C]
        U_next = torch.cat(
            [U_sorted[:, 1:], torch.full((nc, 1), float("-inf"), device=device)], dim=1
        )
        certified = ell_star >= U_next  # [nc,C]
        stop_k = certified.float().argmax(dim=1)  # first True
        rows = size_sorted.cumsum(dim=1)
        fetched = rows.gather(1, stop_k.unsqueeze(1)).squeeze(1)  # [nc]
        fracs.append((fetched / V).cpu())
        # sanity: true argmax cluster opened by stop_k (rank in order < stop_k+1)
        rank_true = (order == true_cl.unsqueeze(1)).float().argmax(1)
        certified_token_ok += (rank_true <= stop_k).sum().item()
        total += nc
    f = torch.cat(fracs)
    return {
        "mean": f.mean().item(),
        "median": f.median().item(),
        "p90": f.quantile(0.90).item(),
        "p99": f.quantile(0.99).item(),
        "frac_under_0.125": (f <= 0.125).float().mean().item(),
        "frac_under_0.25": (f <= 0.25).float().mean().item(),
        "frac_under_0.50": (f <= 0.50).float().mean().item(),
        "argmax_certified_ok": certified_token_ok / max(total, 1),
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--positions", type=int, default=4000)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--partitions", default="idblock,kmeans_raw,kmeans_l2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()
    head = model.get_output_embeddings()
    W_U = head.weight.detach()
    V, d = W_U.shape
    C = args.clusters

    ids = load_split(tok, "val", args.seqs, args.seq_len)
    H, _, _ = capture(model, head, ids, device, pos_cap=args.positions)
    print(f"[{args.model}] V={V} d={d} positions={H.shape[0]} C={C}", flush=True)

    result = {
        "model": args.model,
        "vocab_size": int(V),
        "d_model": int(d),
        "C": C,
        "positions": int(H.shape[0]),
        "bound": "ball (centroid + radius*||h||)",
        "partitions": {},
    }
    builders = {
        "idblock": lambda: build_idblock(V, C),
        "kmeans_raw": lambda: build_kmeans_raw(W_U, C, args.seed),
        "kmeans_l2": lambda: build_kmeans_l2(W_U, C, args.seed),
    }
    for name in args.partitions.split(","):
        cof = builders[name]()
        cl_sizes = torch.bincount(cof, minlength=C)
        res = replay_partition(H, W_U, cof, C, device)
        res["cluster_maxfrac"] = float(cl_sizes.max()) / V
        result["partitions"][name] = res
        print(
            f"  {name:11s} fetched%: mean={res['mean']*100:.1f} "
            f"median={res['median']*100:.1f} p90={res['p90']*100:.1f}  "
            f"<=12.5%:{res['frac_under_0.125']*100:.0f}% <=25%:{res['frac_under_0.25']*100:.0f}%  "
            f"cert_ok={res['argmax_certified_ok']:.3f}",
            flush=True,
        )

    result["wall_seconds"] = round(time.time() - t0, 1)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"[wrote] {outp}  ({result['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
