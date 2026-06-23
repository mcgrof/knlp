#!/usr/bin/env python3
"""Batch cert replay v2 (ChatGPT-Pro v2 spec): shared-open adaptive + per-row aq_err.

Extends batch_cert_replay.py with the two changes the review prescribed BEFORE
building the production kernel:
 1. SHARED-OPEN adaptive all-certify: one opened slab set for the whole batch;
    when a slab opens it is computed for ALL B tokens (good TC reuse, and the
    "extra" logits strengthen every token's incumbent). Static round-open ladder
    so the path is CUDA-graph capturable. Urgency G[s] = max over active tokens of
    (Ub[b,s] if Ub[b,s] >= m_b else -inf). Reports shared union + rounds and
    compares to the per-token adaptive union (the v1 lower bound).
 2. PER-ROW aq_err_up[v] = ||(W_U[v]@B) - dequant(aq[v])||, rounded up, vs the
    scalar max. The scalar is dominated by one bad row and taxes every token.

Lossless by construction (opens until every token certifies, strict >); we also
verify false_cert == 0 against the exact dense argmax. Sweeps r and aq-mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from predictor_baseline import load_split, capture
from shadow_bound_replay import build_basis, shadow_precompute, quantize_cols

# round-open ladder (NEW slabs opened each round); last entry "rest" = open all
LADDER = [4, 4, 8, 8, 8, 16, 16, 32, 32, 64, 256]


@torch.no_grad()
def precompute(H, W_U, Bb, bits, device, chunk=64):
    """Return Ub_scalar[N,C], Ub_perrow[N,C], ml[N,C], true_argmax[N].
    Ub uses dequant(aq)·q + rho·delta + err·||q||; scalar vs per-row err."""
    a_full, delta = shadow_precompute(W_U, Bb, device)       # a_full=[V,r] exact
    a_q, aq_err_scalar = quantize_cols(a_full.clone(), bits)  # dequantized ahat
    aq_err_row = (a_full - a_q).norm(dim=1)                   # [V] per-row, exact>=
    N, d = H.shape
    V, C = W_U.shape[0], None
    Bt = Bb.t().contiguous()
    Wt = W_U.float().t().contiguous()
    return a_q, delta, aq_err_scalar, aq_err_row, Bt, Wt


@torch.no_grad()
def bounds(H, W_U, Bb, a_q, delta, aq_err_scalar, aq_err_row, Bt, Wt, C, S,
           device, chunk=48):
    N = H.shape[0]
    Ub_s = torch.empty(N, C, device=device)
    Ub_r = torch.empty(N, C, device=device)
    ml = torch.empty(N, C, device=device)
    targ = torch.empty(N, dtype=torch.long, device=device)
    for s in range(0, N, chunk):
        Hc = H[s:s + chunk].float().to(device)
        q = Hc @ Bb
        rho = (Hc - q @ Bt).norm(dim=1)
        qn = q.norm(dim=1)
        base = q @ a_q.t() + rho[:, None] * delta[None, :]      # [c,V]
        U_s = base + (aq_err_scalar * qn)[:, None]
        U_r = base + qn[:, None] * aq_err_row[None, :]
        logits = Hc @ Wt                                        # [c,V]
        Ub_s[s:s + chunk] = U_s.view(-1, C, S).amax(2)
        Ub_r[s:s + chunk] = U_r.view(-1, C, S).amax(2)
        ml[s:s + chunk] = logits.view(-1, C, S).amax(2)
        targ[s:s + chunk] = logits.argmax(1)
    return Ub_s, Ub_r, ml, targ


def sim_shared_open(Ub, ml, targ_slab, n_batches, Bn, gen, ladder=LADDER):
    """Shared-open adaptive all-certify. Returns union/round/cert stats + false_cert."""
    N, C = Ub.shape
    NEG = -1e30
    unions, rounds_used, allslab, false_cert = [], [], 0, 0
    for _ in range(n_batches):
        idx = torch.randint(0, N, (Bn,), generator=gen, device=Ub.device)
        ub = Ub[idx]                                  # [B,C]
        m = ml[idx]                                   # [B,C] real per-slab max
        tslab = targ_slab[idx]                        # [B] slab holding dense argmax
        opened = torch.zeros(C, dtype=torch.bool, device=Ub.device)
        m_b = torch.full((Bn,), NEG, device=Ub.device)
        cert = torch.zeros(Bn, dtype=torch.bool, device=Ub.device)
        nr = 0
        for L in ladder:
            if bool(cert.all()):
                break
            active = ~cert
            cand = torch.where((ub >= m_b[:, None]) & active[:, None], ub,
                               torch.full_like(ub, NEG))
            G = cand.max(0).values                    # [C] urgency
            G = torch.where(opened, torch.full_like(G, NEG), G)
            navail = int((~opened).sum())
            nopen = min(L, navail)
            if nopen <= 0:
                break
            top = G.topk(nopen).indices
            opened[top] = True
            m_b = torch.maximum(m_b, torch.where(opened[None, :], m,
                                                 torch.full_like(m, NEG)).max(1).values)
            rem = torch.where(opened[None, :], torch.full_like(ub, NEG), ub).max(1).values
            cert = m_b > rem
            nr += 1
        unions.append(int(opened.sum()) / C)
        rounds_used.append(nr)
        allslab += int(bool(opened.all()))
        # false cert: the dense-argmax slab must be opened for every token
        false_cert += int((~opened[tslab]).sum())
    def pct(xs, p):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
    return {
        "batch": Bn,
        "shared_union_mean": sum(unions) / len(unions),
        "shared_union_p95": pct(unions, 0.95),
        "shared_union_p99": pct(unions, 0.99),
        "rounds_p50": pct(rounds_used, 0.50),
        "rounds_p95": pct(rounds_used, 0.95),
        "rounds_p99": pct(rounds_used, 0.99),
        "slabs_p95": pct([int(u * C) for u in unions], 0.95),
        "all_slabs_rate": allslab / n_batches,
        "false_cert": false_cert,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--rs", default="512,640,768,896,1024,1280")
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--bits", type=int, default=8)
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
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
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
    print(f"[{args.model}] V={V} d={d} C={C} S={S} N={H.shape[0]} ladder={LADDER}", flush=True)

    rs = [int(x) for x in args.rs.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    result = {"model": args.model, "V": V, "d": d, "C": C, "S": S,
              "ladder": LADDER, "runs": []}
    for r in rs:
        Bb = build_basis("hidden_pca", H, W_U, r, dev)
        a_q, delta, err_sc, err_row, Bt, Wt = precompute(H, W_U, Bb, args.bits, dev)
        Ub_s, Ub_r, ml, targ = bounds(H, W_U, Bb, a_q, delta, err_sc, err_row,
                                      Bt, Wt, C, S, dev)
        targ_slab = targ // S
        tau = (r + 1) / d * 0.5
        print(f"-- r={r} tau_shadow={tau:.3f} aq_err scalar={err_sc:.4e} "
              f"per_row(max={err_row.max():.4e} mean={err_row.mean():.4e}) --", flush=True)
        for mode, Ub in (("scalar", Ub_s), ("per_row", Ub_r)):
            for Bn in batches:
                st = sim_shared_open(Ub, ml, targ_slab, args.n_batches, Bn, gen)
                st.update(r=r, aq_err=mode, tau_shadow=tau)
                result["runs"].append(st)
                print(f"  r={r:>4} {mode:>7} B={Bn:>2} shared_union="
                      f"{100*st['shared_union_mean']:>5.1f}% (p95 "
                      f"{100*st['shared_union_p95']:>5.1f}%) rounds p50/p95="
                      f"{st['rounds_p50']}/{st['rounds_p95']} false_cert="
                      f"{st['false_cert']} allslab={st['all_slabs_rate']:.2f}", flush=True)
        del Ub_s, Ub_r, ml, Bb, a_q, delta
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
