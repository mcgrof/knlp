#!/usr/bin/env python3
"""Quest decode (dynamic per-step page selection) + projected-Quest variant + 2D
bandwidth accounting. Tests whether the probe's page-selection accuracy win
(query-PCA projection > Quest's min/max box) becomes a real BANDWIDTH win.

Three selectors, all DYNAMIC (full KV resident; per decode step the current query
picks top-K pages to read):
  box   -- vanilla Quest: page upper bound from per-page key min/max.
  proj  -- projected-Quest: page score max_k (B^T q)(B^T k), B = query-PCA basis.
  full  -- read everything (anchor).
Static evictors (Recent-Q etc.) are a different axis (resident memory) and are
not the comparison here.

Read-bytes/token is computed ANALYTICALLY per method (resident memory is
unchanged for all dynamic methods; only read traffic differs):
  full : 2 * T * d                         (read all K+V every step)
  box  : 2*NB*d (min/max scan) + 2*(K+s+r)*bs*d   (selected pages' K+V)
  proj : T*ratio_r (B^T k scan) + 2*(K+s+r)*bs*d  (heavier scan, same pages)
in units of d-element reads per token, per (layer, kv-head); ratio_r=r/bs adjusts
the per-token projection metadata. The Pareto is task accuracy vs read-bytes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, load_filler_sentences
from niah_evict_perhead import answer_block_indices, get_keys
from diag_perhead_oracle import capture_q, get_values

NEG = -1e9
_QUEST = {"on": False}


def quest_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    g = module.num_key_value_groups
    ks = key.repeat_interleave(g, dim=1)
    vs = value.repeat_interleave(g, dim=1)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    st = getattr(module, "_quest", None)
    if _QUEST["on"] and st is not None and query.shape[2] == 1:  # decode step
        Bn, Hq, _, _ = aw.shape
        Kc = ks.shape[-2]
        Tp, NB, K = st["Tp"], st["NB"], st["K"]
        Hkv = key.shape[1]
        q = query[:, :, 0, :].view(Bn, Hkv, g, -1).mean(2).float()  # [B,Hkv,D]
        if st["mode"] == "box":
            qk = q.unsqueeze(2)  # [B,Hkv,1,D]
            ub = torch.where(qk > 0, qk * st["kmax"], qk * st["kmin"]).sum(-1)
        else:  # proj
            qp = torch.einsum("bhd,hdr->bhr", q, st["B"])  # [B,Hkv,r]
            ts = torch.einsum("bhr,htr->bht", qp, st["Bk"])  # [B,Hkv,Tp]
            ub = ts.new_full((Bn, Hkv, NB), NEG)
            t2p = st["tok2page"].view(1, 1, Tp).expand(Bn, Hkv, Tp)
            ub.scatter_reduce_(2, t2p, ts, reduce="amax", include_self=True)
        topp = ub.topk(min(K, NB), dim=-1).indices  # [B,Hkv,K]
        pk = torch.zeros(Bn, Hkv, NB, dtype=torch.bool, device=aw.device)
        pk.scatter_(2, topp, True)
        tk = pk.gather(2, st["tok2page"].view(1, 1, Tp).expand(Bn, Hkv, Tp))
        tk[:, :, : st["sink"]] = True
        tk[:, :, st["recent"] :] = True
        keepq = tk.repeat_interleave(g, dim=1)  # [B,Hq,Tp]
        z = aw.new_zeros(())
        nz = aw.new_full((), NEG)
        bias = aw.new_zeros(Bn, Hq, Kc)
        bias[:, :, :Tp] = torch.where(keepq, z, nz)
        aw = aw + bias.view(Bn, Hq, 1, Kc)
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, aw


def install(model):
    model.config._attn_implementation = "eager"
    mods = set()
    for m in model.modules():
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx"):
            m._quest = None
            mods.add(type(m).__module__)
    for mn in mods:
        mod = sys.modules.get(mn)
        if mod is not None and hasattr(mod, "eager_attention_forward"):
            mod.eager_attention_forward = quest_attention


def attn_mods(model):
    return [
        m
        for m in model.modules()
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx")
    ]


def clone_cache(past):
    from transformers import DynamicCache

    new = DynamicCache()
    for i, layer in enumerate(past.layers):
        new.update(layer.keys.clone(), layer.values.clone(), i)
    return new


@torch.no_grad()
def decode(model, last_id, past, device, T, max_new):
    ids = torch.tensor([[int(last_id)]], device=device)
    cur, gen = past, []
    _QUEST["on"] = True
    for s in range(max_new):
        out = model(
            ids,
            past_key_values=cur,
            position_ids=torch.tensor([[T + s]], device=device),
            cache_position=torch.tensor([T + s], device=device),
            use_cache=True,
        )
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        cur = out.past_key_values
        ids = torch.tensor([[nxt]], device=device)
    _QUEST["on"] = False
    return gen


def fit_basis(model, tok, sents, rng, length, needles, n, bs, device, r):
    """query-PCA basis B per (layer,kv-head) from n calibration last-pos queries."""
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    g = Hq // Hkv
    acc = None
    for _ in range(n):
        text, spans, _, (qk, qv) = build_context(tok, length, needles, sents, rng)
        ids, ans = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)[:-1]
        T = ids_t.shape[0]
        qs, past = capture_q(model, ids_t, [T - 1], device)
        if acc is None:
            acc = [[[] for _ in range(Hkv)] for _ in range(len(qs))]
        for li in range(len(qs)):
            q = qs[li][:, 0, :].view(Hkv, g, -1).mean(1)  # [Hkv,D]
            for hh in range(Hkv):
                acc[li][hh].append(q[hh].cpu())
        del past
        torch.cuda.empty_cache()
    nL = len(acc)
    B = [[None] * Hkv for _ in range(nL)]
    for li in range(nL):
        for hh in range(Hkv):
            Q = torch.stack(acc[li][hh], 0).to(device)
            Q = Q - Q.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(Q, full_matrices=False)
            B[li][hh] = Vh[: min(r, Vh.shape[0])].T.contiguous()
    return B, nL, Hkv


def read_bytes_per_tok(mode, K, NB, bs, d, r, sink, recent):
    """d-element reads per generated token, per (layer, kv-head). Resident memory
    is unchanged for all (full cache kept); only read traffic differs."""
    pages = min(K + sink + recent, NB)
    sel_kv = 2 * pages * bs * d  # K+V of selected pages
    if mode == "full":
        return 2 * NB * bs * d
    if mode == "box":
        return 2 * NB * d + sel_kv  # per-page min/max scan + selected K+V
    if mode == "proj":
        return NB * bs * r + sel_kv  # B^T k scan (r per token) + selected K+V
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--eval", type=int, default=80)
    ap.add_argument("--rank", type=int, default=0, help="0 = head_dim/2")
    ap.add_argument("--budgets", default="8,16,32")
    ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--recent", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    install(model)
    bs = args.block_size
    d = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    r = args.rank or d // 2
    budgets = [int(x) for x in args.budgets.split(",")]
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)

    print(f"fitting query-PCA basis (r={r}, calib={args.calib}) ...", flush=True)
    Bbasis, nL, Hkv = fit_basis(
        model, tok, sents, rng, args.length, args.needles, args.calib, bs, device, r
    )
    mods = {m.layer_idx: m for m in attn_mods(model)}

    methods = ["full", "box", "proj"]
    acc = {m: {K: [0, 0] for K in budgets} for m in methods}
    n_used = 0
    for ei in range(args.eval):
        text, spans, needles_l, (qk, qv) = build_context(
            tok, args.length, args.needles, sents, rng
        )
        ids, ans = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)[:-1]
        T = ids_t.shape[0]
        NB = (T + bs - 1) // bs
        if not ans or max(ans) >= NB:
            continue
        last_id = ids[-1]
        past = model(ids_t.unsqueeze(0).to(device), use_cache=True).past_key_values
        keys, vals = get_keys(past), get_values(past)
        tok2page = torch.arange(T, device=device) // bs
        # precompute per-layer page bounds + projected keys
        layer_st = []
        for li in range(nL):
            kl = keys[li][0].float()  # [Hkv,T,D]
            kmin = kl.new_full((Hkv, NB, d), 1e30)
            kmax = kl.new_full((Hkv, NB, d), -1e30)
            t2 = tok2page.view(1, T, 1).expand(Hkv, T, d)
            kmin.scatter_reduce_(1, t2, kl, reduce="amin", include_self=True)
            kmax.scatter_reduce_(1, t2, kl, reduce="amax", include_self=True)
            Bm = torch.stack([Bbasis[li][hh] for hh in range(Hkv)], 0)  # [Hkv,D,r]
            Bk = torch.einsum("htd,hdr->htr", kl, Bm)  # [Hkv,T,r]
            layer_st.append((kmin, kmax, Bm, Bk))
        n_used += 1
        for K in budgets:
            for mode in methods:
                if mode == "full":
                    for m in mods.values():
                        m._quest = None
                else:
                    for li, m in mods.items():
                        kmin, kmax, Bm, Bk = layer_st[li]
                        m._quest = dict(
                            mode=mode,
                            kmin=kmin,
                            kmax=kmax,
                            B=Bm,
                            Bk=Bk,
                            tok2page=tok2page,
                            NB=NB,
                            K=K,
                            Tp=T,
                            sink=args.sink * bs,
                            recent=max(0, T - args.recent * bs),
                        )
                gen = decode(model, last_id, clone_cache(past), device, T, args.max_new)
                ok = qv in tok.decode(gen)
                acc[mode][K][0] += int(ok)
                acc[mode][K][1] += 1
        for m in mods.values():
            m._quest = None
        del past, layer_st
        torch.cuda.empty_cache()
        if (ei + 1) % 10 == 0:
            print(f"  {n_used}/{args.eval}", flush=True)

    # Pareto: accuracy + analytical read-bytes/token (relative to full=1.0)
    full_rb = read_bytes_per_tok("full", 0, NB, bs, d, r, 0, 0)
    res = {
        "model": args.model,
        "rank": r,
        "head_dim": d,
        "eval_used": n_used,
        "budgets": budgets,
        "pareto": {},
    }
    print(f"\n[{args.model}] r={r}/{d}  Quest 2D Pareto (acc @ read-frac of full)")
    print(f"{'method':6s} " + " ".join(f"K{K}".rjust(16) for K in budgets))
    for mode in methods:
        cells, pareto = [], {}
        for K in budgets:
            a = acc[mode][K][0] / max(1, acc[mode][K][1])
            rb = read_bytes_per_tok(mode, K, NB, bs, d, r, args.sink, args.recent)
            frac = rb / full_rb
            pareto[f"K{K}"] = {"acc": a, "read_frac": frac}
            cells.append(f"{a:.2f}@{frac*100:4.1f}%")
        res["pareto"][mode] = pareto
        print(f"{mode:6s} " + " ".join(c.rjust(16) for c in cells))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
