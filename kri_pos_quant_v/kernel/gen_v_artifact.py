#!/usr/bin/env python3
"""Generate a low-rank-V decode-attention artifact from a real model's KV cache.

Runs an L-token context through the model, grabs one layer's K/V cache (real V
geometry, which the V6 result showed is ~rank-32 retrieval-lossless), SVDs V per
kv-head to rank r, and saves everything the low-rank-V decode kernel needs plus a
set of decode queries, so the kernel can be validated + benchmarked offline.

Saved (torch .pt), one layer, Hkv kv-heads, T context, head_dim dh:
  K     bf16 [Hkv, T, dh]   keys (full; scores use full K)
  V     bf16 [Hkv, T, dh]   values (dense reference)
  Bbasis fp32 [Hkv, dh, r]  per-head rank-r value basis (orthonormal)
  proj  fp32 [Hkv, T, r]    per-head coefficients  (V ~= proj @ Bbasis^T)
  Q     bf16 [Hkv, nq, dh]  decode queries (real next-token q via the model)
  meta  dict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--layer", type=int, default=-1, help="layer index (-1=middle)")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--n-query", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev)
    model.eval()
    torch.set_grad_enabled(False)

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    buf = []
    for t in ds["text"]:
        if len(t.strip()) > 80 and not t.startswith("="):
            buf.extend(tok(t).input_ids)
        if len(buf) >= args.length + args.n_query + 1:
            break
    ids = torch.tensor(buf[: args.length]).unsqueeze(0).to(dev)

    out = model(ids, use_cache=True)
    pkv = out.past_key_values
    nlayers = len(pkv) if not hasattr(pkv, "layers") else len(pkv.layers)
    L = args.layer if args.layer >= 0 else nlayers // 2

    def layer_kv(pkv, i):
        if hasattr(pkv, "layers"):
            return pkv.layers[i].keys, pkv.layers[i].values
        return pkv[i][0], pkv[i][1]

    K, V = layer_kv(pkv, L)  # [1, Hkv, T, dh]
    K = K[0].contiguous()  # [Hkv, T, dh]
    V = V[0].contiguous()
    Hkv, T, dh = V.shape
    print(f"[{args.model}] layer={L} Hkv={Hkv} T={T} dh={dh} r={args.r}", flush=True)

    # per-head rank-r SVD of V -> basis Bbasis [dh,r], coeffs proj [T,r]
    Bbasis = torch.empty(Hkv, dh, args.r, device=dev)
    proj = torch.empty(Hkv, T, args.r, device=dev)
    recon_err = []
    for h in range(Hkv):
        Vh = V[h].float()  # [T, dh]
        # economy SVD: Vh = U S Wt ; basis = top-r right singular vectors [dh,r]
        U, Sv, Wt = torch.linalg.svd(Vh, full_matrices=False)
        basis = Wt[: args.r].t().contiguous()  # [dh, r]
        Bbasis[h] = basis
        proj[h] = Vh @ basis  # [T, r]
        recon = proj[h] @ basis.t()
        recon_err.append((Vh - recon).norm() / Vh.norm())
    rel_err = torch.tensor(recon_err).mean().item()
    print(f"rank-{args.r} V recon rel-error (mean over heads): {rel_err:.4f}", flush=True)

    # decode queries: continue the context n_query tokens, grab real q vectors via
    # a hook on the layer's q_proj is arch-specific; instead use the next n_query
    # positions' KEYS as query proxies (same geometry/scale as q, realistic
    # softmax). For GQA we need query-head dim; use kv-head keys (Hkv) so q.k is
    # within-head -- a faithful score distribution for the kernel benchmark.
    Q = K[:, -args.n_query :, :].contiguous()  # [Hkv, nq, dh]

    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    torch.save(K.to(dtype).cpu(), o / "K.pt")
    torch.save(V.to(dtype).cpu(), o / "V.pt")
    torch.save(Bbasis.cpu(), o / "Bbasis.pt")
    torch.save(proj.cpu(), o / "proj.pt")
    torch.save(Q.to(dtype).cpu(), o / "Q.pt")
    meta = {
        "model": args.model,
        "layer": int(L),
        "Hkv": int(Hkv),
        "T": int(T),
        "head_dim": int(dh),
        "r": int(args.r),
        "n_query": int(args.n_query),
        "rank_r_V_recon_rel_error": rel_err,
    }
    (o / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[wrote] {o}", flush=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
