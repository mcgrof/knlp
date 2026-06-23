#!/usr/bin/env python3
"""Shadow-bound certified-decode replay for idblock LM-head pruning.

Tests whether a per-token projection-residual upper bound -- decoupled from the
partition geometry -- lets an addressable idblock partition certify the dense
argmax while reading only a small fraction of the LM head. This is the candidate
that could make "idblock + lossless certificate" non-contradictory, by replacing
the cluster ball bound (whose slack r_c*||h|| killed every partition at routing
granularity) with a bound whose slack is per-token and shrinks when the hidden
states are low-rank.

The bound. Pick an orthonormal basis B (d x r), r << d. Offline store, per row,
a_v = B^T w_v and the residual norm delta_v = ||w_v - B B^T w_v||. At runtime
q = B^T h and rho = ||h - B q||. Then for every token

    w_v . h = (B B^T w_v).h + (w_v - B B^T w_v).h_perp
            <= a_v . q + delta_v * rho                  (Cauchy-Schwarz)

a valid upper bound that needs only the compact shadow head (V x r) plus one
scalar delta per row -- streamable in token-id order, no k-means, no permutation.

Certified greedy decode (per position): reduce per-token bounds to idblock slab
maxima U_b; open slabs in descending U_b; track ell* = max real fetched logit;
stop (strict) when ell* > max U_b over unopened slabs -> dense argmax certified.

Cost model. Reading the shadow head costs (r+1)/d of the dense-head bytes every
token; the certified exact fetch adds the fetched fraction. Total byte ratio =
(r+1)/d + fetched_fraction, to be compared against a quantized dense GEMV
(int8 ~0.5, int4 ~0.25), which is the baseline this must beat.

Usage:
  python3 kri_lm_head/shadow_bound_replay.py --model Qwen/Qwen2.5-7B \
      --bases hidden_pca,w_svd,random --rs 16,32,64,128,256 --out OUT/x.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from h6_oracle_scan import build_idblock
from predictor_baseline import load_split, capture


@torch.no_grad()
def topk_importance(H, W_U, device, K=64, hchunk=256, vchunk=16384):
    """Per-token count of being in the dense top-K across captured positions."""
    Hd = H.float().to(device)
    Wt = W_U.float().to(device).t().contiguous()
    weight = torch.zeros(W_U.shape[0], device=device)
    for s in range(0, Hd.shape[0], hchunk):
        lg = Hd[s : s + hchunk] @ Wt
        tk = lg.topk(K, dim=1).indices.reshape(-1)
        weight.scatter_add_(0, tk, torch.ones_like(tk, dtype=weight.dtype))
    return weight


@torch.no_grad()
def build_basis(kind, H, W_U, r, device, lam=0.5, vchunk=16384):
    d = W_U.shape[1]
    if kind == "random":
        g = torch.randn(d, r, device=device)
        q, _ = torch.linalg.qr(g)
        return q[:, :r].contiguous()
    if kind == "hidden_pca":
        Hd = H.float().to(device)
        M = Hd.t() @ Hd  # [d,d] second moment (uncentered: keeps mean direction)
    elif kind == "w_svd":
        M = torch.zeros(d, d, device=device)
        for s in range(0, W_U.shape[0], vchunk):
            Wc = W_U[s : s + vchunk].float().to(device)
            M += Wc.t() @ Wc
    elif kind == "logit_aware":
        # blend hidden-state directions (small rho) with importance-weighted
        # W_U directions (small delta for tokens that actually reach the top).
        Hd = H.float().to(device)
        Mh = Hd.t() @ Hd
        w = topk_importance(H, W_U, device).clamp(min=0).sqrt()  # [V]
        Mw = torch.zeros(d, d, device=device)
        for s in range(0, W_U.shape[0], vchunk):
            Wc = W_U[s : s + vchunk].float().to(device) * w[s : s + vchunk].unsqueeze(1)
            Mw += Wc.t() @ Wc
        M = Mh / Mh.norm() + lam * Mw / Mw.norm()
    else:
        raise ValueError(kind)
    evals, evecs = torch.linalg.eigh(M)  # ascending
    return evecs[:, -r:].contiguous()  # top-r directions [d,r]


@torch.no_grad()
def shadow_precompute(W_U, B, device, vchunk=16384):
    V, d = W_U.shape
    r = B.shape[1]
    Bt = B.t().contiguous()
    a = torch.empty(V, r, device=device)
    delta = torch.empty(V, device=device)
    for s in range(0, V, vchunk):
        Wc = W_U[s : s + vchunk].float().to(device)
        ac = Wc @ B
        a[s : s + vchunk] = ac
        delta[s : s + vchunk] = (Wc - ac @ Bt).norm(dim=1)
    return a, delta


def quantize_cols(a, bits):
    """Per-column symmetric quantization of the shadow coefficients, with the
    L2 norm of the worst-case quantization error so the bound stays valid:
    |(a - ahat).q| <= ||a - ahat|| * ||q|| <= err_norm * ||q||."""
    qmax = 2 ** (bits - 1) - 1
    scale = (a.abs().amax(0) / qmax).clamp(min=1e-12)  # [r]
    ahat = torch.round(a / scale).clamp(-qmax, qmax) * scale
    err_norm = (scale / 2).norm().item()  # sqrt(sum_j (scale_j/2)^2)
    return ahat, err_norm


@torch.no_grad()
def replay(H, W_U, B, a, delta, cof, C, device, aq_err_norm=0.0, pchunk=64):
    V, d = W_U.shape
    sizes = torch.bincount(cof, minlength=C).float()
    Wt = W_U.float().to(device).t().contiguous()
    Bt = B.t().contiguous()
    fracs = []
    thrs = (0.125, 0.25, 0.35)
    cert_at = {t: 0 for t in thrs}
    argmax_ok = 0
    total = 0
    N = H.shape[0]
    for s in range(0, N, pchunk):
        Hc = H[s : s + pchunk].float().to(device)
        nc = Hc.shape[0]
        logits = Hc @ Wt  # [nc,V] real logits (no bias: tied / bias-free heads)
        q = Hc @ B  # [nc,r]
        rho = (Hc - q @ Bt).norm(dim=1)  # [nc]
        qnorm = q.norm(dim=1)  # [nc]
        # outward slack from shadow quantization (0 when fp16): added to every
        # token equally, so it raises the bar the incumbent must beat.
        U = (
            q @ a.t()
            + rho.unsqueeze(1) * delta.unsqueeze(0)
            + (aq_err_norm * qnorm).unsqueeze(1)
        )  # [nc,V] upper bound
        idx = cof.unsqueeze(0).expand(nc, V)
        ml = torch.full((nc, C), float("-inf"), device=device)
        ml.scatter_reduce_(1, idx, logits, reduce="amax", include_self=True)
        Ub = torch.full((nc, C), float("-inf"), device=device)
        Ub.scatter_reduce_(1, idx, U, reduce="amax", include_self=True)
        true_cl = cof[logits.argmax(1)]
        order = Ub.argsort(1, descending=True)
        Ub_s = Ub.gather(1, order)
        ml_s = ml.gather(1, order)
        size_s = sizes[order]
        ell = ml_s.cummax(1).values
        Ub_next = torch.cat(
            [Ub_s[:, 1:], torch.full((nc, 1), float("-inf"), device=device)], 1
        )
        certified = ell > Ub_next  # strict, for tie-safe losslessness
        stop_k = certified.float().argmax(1)
        rows = size_s.cumsum(1)
        fetched = rows.gather(1, stop_k.unsqueeze(1)).squeeze(1)
        frac = fetched / V
        fracs.append(frac.cpu())
        for t in thrs:
            cert_at[t] += (frac <= t).sum().item()
        rank_true = (order == true_cl.unsqueeze(1)).float().argmax(1)
        argmax_ok += (rank_true <= stop_k).sum().item()
        total += nc
    f = torch.cat(fracs)
    return {
        "fetched_mean": f.mean().item(),
        "fetched_median": f.median().item(),
        "fetched_p95": f.quantile(0.95).item(),
        "cert_at_0.125": cert_at[0.125] / total,
        "cert_at_0.25": cert_at[0.25] / total,
        "cert_at_0.35": cert_at[0.35] / total,
        "argmax_match": argmax_ok / total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--positions", type=int, default=1500)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--bases", default="hidden_pca,w_svd,random")
    ap.add_argument("--rs", default="16,32,64,128,256")
    ap.add_argument("--shadow-bits", type=int, default=16)
    ap.add_argument("--laware-lambda", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()
    head = model.get_output_embeddings()
    if getattr(head, "bias", None) is not None:
        print("WARN: head has bias; bound ignores it (results approximate)", flush=True)
    W_U = head.weight.detach()
    V, d = W_U.shape
    C = args.clusters
    cof = build_idblock(V, C).to(device)

    ids = load_split(tok, "val", args.seqs, args.seq_len)
    H, _, _ = capture(model, head, ids, device, pos_cap=args.positions)
    print(f"[{args.model}] V={V} d={d} positions={H.shape[0]} C={C}", flush=True)

    result = {
        "model": args.model,
        "vocab_size": int(V),
        "d_model": int(d),
        "C": C,
        "positions": int(H.shape[0]),
        "partition": "idblock",
        "quantized_dense_baseline_byteratio": {"int8": 0.5, "int4": 0.25},
        "runs": [],
    }
    rs = [int(x) for x in args.rs.split(",")]
    for basis in args.bases.split(","):
        for r in rs:
            B = build_basis(basis, H, W_U, r, device, lam=args.laware_lambda)
            a, delta = shadow_precompute(W_U, B, device)
            aq_err = 0.0
            if args.shadow_bits < 16:
                a, aq_err = quantize_cols(a, args.shadow_bits)
            m = replay(H, W_U, B, a, delta, cof, C, device, aq_err_norm=aq_err)
            # shadow bytes: a is V*r at shadow_bits, delta is V scalars at fp16
            shadow_ratio = (r * (args.shadow_bits / 16.0) + 1) / d
            total_ratio = shadow_ratio + m["fetched_mean"]
            rec = {
                "basis": basis,
                "r": r,
                "shadow_bits": args.shadow_bits,
                "shadow_byteratio": shadow_ratio,
                "total_byteratio": total_ratio,
                **m,
            }
            result["runs"].append(rec)
            print(
                f"  {basis:11s} r={r:>3}  fetched%: mean={m['fetched_mean']*100:.1f} "
                f"p95={m['fetched_p95']*100:.1f}  cert<=12.5%:{m['cert_at_0.125']*100:.0f}% "
                f"cert<=35%:{m['cert_at_0.35']*100:.0f}%  shadow={shadow_ratio*100:.1f}% "
                f"TOTAL={total_ratio*100:.1f}%  amatch={m['argmax_match']:.3f}",
                flush=True,
            )
            del a, delta, B
            torch.cuda.empty_cache()

    result["wall_seconds"] = round(time.time() - t0, 1)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"[wrote] {outp}  ({result['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
