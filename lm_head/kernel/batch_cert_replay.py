#!/usr/bin/env python3
"""Offline batch-certification replay (ChatGPT-Pro's decisive cheap experiment).

For each (batch B, shadow rank r), simulate the certified decode over real batches
and record the inputs to the H100 Pareto kill-condition T/T_dense >= tau_shadow +
u_all_cert/gamma. Hardware-independent (pure simulation, no deployment kernel).

Per r: build the hidden-PCA basis, the int8 shadow bound, the EXACT per-slab max
logits. Per B: sample many batches and record
  - fixed-K first pass: per-token cert rate c, P(all tokens certify), union frac u
  - adaptive ALL-certify (no dense fallback): u_all_cert = distinct slabs read
    until EVERY token in the batch certifies (perfect global slab cache), with
    p50/p95/p99, and the extra slabs per failed token
  - token-slab pair counts + active-tokens-per-slab (for the CSR sparse-pair kernel)
tau_shadow(r) = (r+1)/d * 0.5 (int8 shadow bytes / dense head bytes).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from h6_oracle_scan import build_idblock  # noqa: F401  (kept for parity)
from predictor_baseline import load_split, capture
from shadow_bound_replay import build_basis, shadow_precompute, quantize_cols


@torch.no_grad()
def bounds_and_truth(H, W_U, Bb, a, delta, aq_err, C, S, device, chunk=64):
    """Ub[N,C] (shadow upper bound per slab) and ml[N,C] (EXACT per-slab max
    logit), computed in position-chunks to bound memory."""
    N, d = H.shape
    V = W_U.shape[0]
    Bt = Bb.t().contiguous()
    Wt = W_U.float().t().contiguous()
    Ub = torch.empty(N, C, device=device)
    ml = torch.empty(N, C, device=device)
    for s in range(0, N, chunk):
        Hc = H[s:s + chunk].float().to(Bb.device)
        q = Hc @ Bb                                   # [c,r]
        rho = (Hc - q @ Bt).norm(dim=1)               # [c]
        qn = q.norm(dim=1)
        U = q @ a.t() + rho[:, None] * delta[None, :] + (aq_err * qn)[:, None]
        Ub[s:s + chunk] = U.view(-1, C, S).amax(2)
        ml[s:s + chunk] = (Hc @ Wt).view(-1, C, S).amax(2)
    return Ub, ml


def simulate(Ub, ml, K, n_batches, Bn, gen):
    """Returns dict of cert/union stats for batch size Bn."""
    N, C = Ub.shape
    cert_rate = []          # per-token cert within K
    all_cert = []           # P(all B tokens certify within K)
    union_K = []            # fixed-K first-pass union fraction
    u_allc = []             # adaptive all-certify union fraction
    pairs_req = []          # token-slab pairs requested fixed-K (=Bn*K)
    pairs_union = []        # Bn * union_slabs (the dense-union overcompute)
    extra_per_fail = []     # avg extra slabs a failed token needs past K
    for _ in range(n_batches):
        idx = torch.randint(0, N, (Bn,), generator=gen, device=Ub.device)
        ub = Ub[idx]                                  # [B,C]
        m = ml[idx]                                   # [B,C]
        Ub_s, order = ub.sort(1, descending=True)
        m_s = m.gather(1, order)
        ell = m_s.cummax(1).values                    # incumbent as slabs open
        Ub_next = torch.cat([Ub_s[:, 1:],
                             torch.full((Bn, 1), -1e30, device=ub.device)], 1)
        cert = ell > Ub_next                          # [B,C] certified-by-step-k
        stopk = cert.float().argmax(1)                # first certifying step
        # tokens that never certify within K
        cert_byK = (stopk < K)
        cert_rate.append(cert_byK.float().mean().item())
        all_cert.append(float(bool(cert_byK.all())))
        # fixed-K union
        opened_K = order[:, :K]
        union_K.append(torch.unique(opened_K).numel() / C)
        pairs_req.append(Bn * K)
        pairs_union.append(Bn * torch.unique(opened_K).numel())
        # adaptive all-certify: each token opens up to its own stop_k (+1)
        need = (stopk + 1).clamp(max=C)
        slabs = set()
        ef = []
        for b in range(Bn):
            sb = order[b, :int(need[b])].tolist()
            slabs.update(sb)
            ef.append(max(0, int(need[b]) - K))
        u_allc.append(len(slabs) / C)
        extra_per_fail.append(sum(ef) / max(1, sum(1 for e in ef if e > 0)))
    t = lambda xs: sorted(xs)
    pct = lambda xs, p: t(xs)[min(len(xs) - 1, int(p * len(xs)))]
    return {
        "batch": Bn, "K": K,
        "cert_rate_mean": sum(cert_rate) / len(cert_rate),
        "p_all_certify": sum(all_cert) / len(all_cert),
        "union_K_mean": sum(union_K) / len(union_K),
        "u_allcert_mean": sum(u_allc) / len(u_allc),
        "u_allcert_p50": pct(u_allc, 0.50),
        "u_allcert_p95": pct(u_allc, 0.95),
        "u_allcert_p99": pct(u_allc, 0.99),
        "extra_slabs_per_fail": sum(extra_per_fail) / len(extra_per_fail),
        "active_tokens_per_union_slab": (sum(pairs_req) / sum(pairs_union)) * Bn
        if sum(pairs_union) else 0.0,
        "overcompute_ratio": sum(pairs_union) / max(1, sum(pairs_req)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--rs", default="256,512,768,1024,1280")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--k-frac", type=float, default=0.06)
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                     torch_dtype=torch.bfloat16)
    model.to(dev).eval()
    head = model.get_output_embeddings()
    W_U = head.weight.detach()
    V, d = W_U.shape
    C = args.clusters
    S = V // C
    ids = load_split(tok, "val", args.seqs, 512)
    H, _, _ = capture(model, head, ids, dev, pos_cap=args.positions)
    del model
    torch.cuda.empty_cache()
    print(f"[{args.model}] V={V} d={d} C={C} S={S} N={H.shape[0]}", flush=True)

    rs = [int(x) for x in args.rs.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    result = {"model": args.model, "V": V, "d": d, "C": C, "S": S,
              "k_frac": args.k_frac, "runs": []}
    for r in rs:
        Bb = build_basis("hidden_pca", H, W_U, r, dev)
        a, delta = shadow_precompute(W_U, Bb, dev)
        a, aq_err = quantize_cols(a, 8)
        Ub, ml = bounds_and_truth(H, W_U, Bb, a, delta, aq_err, C, S, dev)
        tau_shadow = (r + 1) / d * 0.5
        K = max(1, int(args.k_frac * C))
        for Bn in batches:
            st = simulate(Ub, ml, K, args.n_batches, Bn, gen)
            st["r"] = r
            st["tau_shadow"] = tau_shadow
            # ideal lower bounds at a few gamma (sparse/dense bw ratio)
            st["lb_fixedK_g1"] = tau_shadow + st["union_K_mean"] + (1 - st["cert_rate_mean"] ** Bn)
            for g in (1.0, 0.7, 0.5, 0.4):
                st[f"lb_adaptive_g{g}"] = tau_shadow + st["u_allcert_mean"] / g
            result["runs"].append(st)
            print(f"r={r:>4} B={Bn:>2} cert={st['cert_rate_mean']:.3f} "
                  f"Pall={st['p_all_certify']:.3f} uK={100*st['union_K_mean']:.1f}% "
                  f"u_allc={100*st['u_allcert_mean']:.1f}% (p95 "
                  f"{100*st['u_allcert_p95']:.1f}%) lb_adapt_g0.7="
                  f"{st['lb_adaptive_g0.7']:.2f} overcompute={st['overcompute_ratio']:.1f}x",
                  flush=True)
        del Ub, ml, Bb, a, delta
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
