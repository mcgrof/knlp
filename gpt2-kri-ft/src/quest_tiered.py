#!/usr/bin/env python3
"""Tiered / offloaded KV prefetch-filter experiment.

The cost model that in-GPU Quest does NOT have: KV is split into a HOT tier
(resident in HBM, attended for free) and a COLD tier (on CXL/SSD). At decode, a
small RESIDENT shadow scores cold pages and we prefetch the top-Kc -- *reading a
cold page is the expensive event*, and the resident shadow is the scarce HBM. So
the axes are: task accuracy, COLD-FETCH traffic (Kc pages/step from the slow
tier), and RESIDENT footprint (hot pages + shadow bytes). This is the one regime
where a low-rank projected shadow (small footprint) could matter even though it
selects worse pages than the full min/max box.

Hot tier = SnapKV top-M pages (static, from the observation window) + sink +
recent. Cold tier = the rest. Each decode step the live query scores cold pages
with a filter and prefetches top-Kc; attention sees hot + prefetched-cold +
sink + recent + generated.

Cold-page shadow per page (resident HBM, d-elements):
  box     : 2d   (full min/max)
  sparq   : 2d   (full min/max; reads top-k1 coords at score time)
  projbox : 2r   (rotated query-PCA min/max) <- the small-footprint candidate

Filters compared on: task accuracy vs cold-fetches Kc, at a fixed hot budget M,
plus the resident shadow footprint. Run on long context where SnapKV cannot keep
everything (so cold pages genuinely carry answers).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, load_filler_sentences
from niah_evict_perhead import answer_block_indices, get_keys
from diag_perhead_oracle import capture_q, get_values

NEG = -1e9
_T = {"on": False}


def tiered_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    g = module.num_key_value_groups
    ks = key.repeat_interleave(g, dim=1)
    vs = value.repeat_interleave(g, dim=1)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    st = getattr(module, "_t", None)
    if _T["on"] and st is not None and query.shape[2] == 1:
        Bn, Hq, _, _ = aw.shape
        Kc = ks.shape[-2]
        Tp, NB, Kf = st["Tp"], st["NB"], st["Kc"]
        Hkv = key.shape[1]
        q = query[:, :, 0, :].view(Bn, Hkv, g, -1).mean(2).float()  # [B,Hkv,D]
        if st["kind"] == "box":
            qk = q.unsqueeze(2)
            ub = torch.where(qk > 0, qk * st["kmax"], qk * st["kmin"]).sum(-1)
        elif st["kind"] == "sparq":
            k1 = st["k1"]
            sel = q.abs().topk(min(k1, q.shape[-1]), dim=-1).indices
            qs = torch.gather(q, 2, sel)
            kmx = torch.gather(
                st["kmax"].unsqueeze(0).expand(Bn, -1, -1, -1),
                3,
                sel.unsqueeze(2).expand(-1, -1, NB, -1),
            )
            kmn = torch.gather(
                st["kmin"].unsqueeze(0).expand(Bn, -1, -1, -1),
                3,
                sel.unsqueeze(2).expand(-1, -1, NB, -1),
            )
            qe = qs.unsqueeze(2)
            ub = torch.where(qe > 0, qe * kmx, qe * kmn).sum(-1)
        else:  # projbox
            qp = torch.einsum("bhd,hdr->bhr", q, st["B"])
            qe = qp.unsqueeze(2)
            ub = torch.where(qe > 0, qe * st["zmax"], qe * st["zmin"]).sum(-1)
        hot = st["hot"].unsqueeze(0).expand(Bn, -1, -1)  # [B,Hkv,NB]
        ub = ub.masked_fill(hot, NEG)  # don't spend cold budget on hot pages
        topp = ub.topk(min(Kf, NB), dim=-1).indices
        pk = hot.clone()
        pk.scatter_(2, topp, True)  # hot + prefetched cold
        tk = pk.gather(2, st["tok2page"].view(1, 1, Tp).expand(Bn, Hkv, Tp))
        tk[:, :, : st["sink"]] = True
        tk[:, :, st["recent"] :] = True
        keepq = tk.repeat_interleave(g, dim=1)
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
            m._t = None
            mods.add(type(m).__module__)
    for mn in mods:
        mod = sys.modules.get(mn)
        if mod is not None and hasattr(mod, "eager_attention_forward"):
            mod.eager_attention_forward = tiered_attention


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
    _T["on"] = True
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
    _T["on"] = False
    return gen


def fit_basis(model, tok, sents, rng, length, needles, n, bs, device, r):
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
            q = qs[li][:, 0, :].view(Hkv, g, -1).mean(1)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--eval", type=int, default=80)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--k1", type=int, default=32)
    ap.add_argument("--hot-frac", type=float, default=0.10, help="SnapKV hot pages")
    ap.add_argument("--cold", default="2,4,8,16", help="cold prefetch budgets Kc")
    ap.add_argument("--recent-window", type=int, default=16)
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
    colds = [int(x) for x in args.cold.split(",")]
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)
    W = args.recent_window

    print(f"basis r={r} calib={args.calib} ...", flush=True)
    Bb, nL, Hkv = fit_basis(
        model, tok, sents, rng, args.length, args.needles, args.calib, bs, device, r
    )
    mods = {m.layer_idx: m for m in attn_mods(model)}
    Hq = model.config.num_attention_heads
    g = Hq // Hkv

    # Named offloading systems:
    #   quest    = dynamic box selection, NO permanent hot tier (all pages
    #              compete; resident box metadata 2d/page over all NB pages)
    #   rocketkv = SnapKV permanent hot tier + SparQ box-on-top|q| cold filter
    #   projbox  = SnapKV hot tier + low-rank query-PCA shadow (2r/page) cold
    methods = ["full", "quest", "rocketkv", "projbox"]
    KIND = {"quest": "box", "rocketkv": "sparq", "projbox": "projbox"}
    USE_HOT = {"quest": False, "rocketkv": True, "projbox": True}
    acc = {m: {Kc: [0, 0] for Kc in colds} for m in methods}
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
        recent = list(range(max(0, T - W), T))
        last_id = ids[-1]
        qs, past = capture_q(model, ids_t, sorted(set(recent + [T - 1])), device)
        keys, vals = get_keys(past), get_values(past)
        tok2page = torch.arange(T, device=device) // bs
        M = max(1, int(args.hot_frac * NB))
        ridx = torch.tensor(list(range(len(recent))), device=device)
        layer_st = []
        for li in range(nL):
            kl = keys[li][0].float()
            kmin = kl.new_full((Hkv, NB, d), 1e30)
            kmax = kl.new_full((Hkv, NB, d), -1e30)
            t2 = tok2page.view(1, T, 1).expand(Hkv, T, d)
            kmin.scatter_reduce_(1, t2, kl, reduce="amin", include_self=True)
            kmax.scatter_reduce_(1, t2, kl, reduce="amax", include_self=True)
            Bm = torch.stack([Bb[li][hh] for hh in range(Hkv)], 0)
            Bk = torch.einsum("htd,hdr->htr", kl, Bm)
            rr = Bk.shape[-1]
            zmin = Bk.new_full((Hkv, NB, rr), 1e30)
            zmax = Bk.new_full((Hkv, NB, rr), -1e30)
            t2r = tok2page.view(1, T, 1).expand(Hkv, T, rr)
            zmin.scatter_reduce_(1, t2r, Bk, reduce="amin", include_self=True)
            zmax.scatter_reduce_(1, t2r, Bk, reduce="amax", include_self=True)
            # SnapKV hot set: obs-window attention -> per-page mass -> top-M
            qo = qs[li][:, ridx, :].view(Hkv, g, len(recent), -1)  # [Hkv,g,W,D]
            sc = torch.einsum("hgwd,htd->hgwt", qo, kl) / math.sqrt(d)
            imp = sc.softmax(dim=-1).sum(dim=(1, 2))  # [Hkv,T]
            pimp = imp.new_full((Hkv, NB), 0.0)
            pimp.scatter_reduce_(
                1,
                tok2page.view(1, T).expand(Hkv, T),
                imp,
                reduce="sum",
                include_self=True,
            )
            hotp = pimp.topk(min(M, NB), dim=-1).indices
            hot = torch.zeros(Hkv, NB, dtype=torch.bool, device=device)
            hot.scatter_(1, hotp, True)
            layer_st.append((kmin, kmax, Bm, zmin, zmax, hot))
        n_used += 1
        # full is Kc-independent -> decode once, replicate across budgets
        for m in mods.values():
            m._t = None
        gen = decode(model, last_id, clone_cache(past), device, T, args.max_new)
        ok_full = int(qv in tok.decode(gen))
        for Kc in colds:
            acc["full"][Kc][0] += ok_full
            acc["full"][Kc][1] += 1
        for Kc in colds:
            for mode in methods:
                if mode == "full":
                    continue
                kind, use_hot = KIND[mode], USE_HOT[mode]
                for li, m in mods.items():
                    kmin, kmax, Bm, zmin, zmax, hot = layer_st[li]
                    hh = hot if use_hot else torch.zeros_like(hot)
                    m._t = dict(
                        kind=kind,
                        kmin=kmin,
                        kmax=kmax,
                        B=Bm,
                        zmin=zmin,
                        zmax=zmax,
                        hot=hh,
                        k1=args.k1,
                        tok2page=tok2page,
                        NB=NB,
                        Kc=Kc,
                        Tp=T,
                        sink=args.sink * bs,
                        recent=max(0, T - args.recent * bs),
                    )
                gen = decode(model, last_id, clone_cache(past), device, T, args.max_new)
                ok = qv in tok.decode(gen)
                acc[mode][Kc][0] += int(ok)
                acc[mode][Kc][1] += 1
        for m in mods.values():
            m._t = None
        del past, layer_st
        torch.cuda.empty_cache()
        if (ei + 1) % 10 == 0:
            print(f"  {n_used}/{args.eval}", flush=True)

    M = max(1, int(args.hot_frac * NB))
    # Resident HBM model, per (layer, kv-head), in d-elements:
    #   metadata/page  : quest 2d (all NB), rocketkv 2d (cold NB-M), projbox 2r (cold)
    #   hot KV         : quest 0, rocketkv/projbox  M*bs*d*2 (K+V) full-precision
    meta_pp = {"quest": 2 * d, "rocketkv": 2 * d, "projbox": 2 * r}
    hot_pages = {"quest": 0, "rocketkv": M, "projbox": M}
    resident = {}
    for mth in ["quest", "rocketkv", "projbox"]:
        npages_meta = NB if mth == "quest" else (NB - M)
        meta_de = meta_pp[mth] * npages_meta
        hotkv_de = hot_pages[mth] * bs * d * 2
        resident[mth] = {
            "metadata_d_per_page": meta_pp[mth],
            "metadata_pages": npages_meta,
            "hot_pages": hot_pages[mth],
            "resident_d_elements": meta_de + hotkv_de,
        }
    res = {
        "model": args.model,
        "rank": r,
        "head_dim": d,
        "block_size": bs,
        "eval_used": n_used,
        "hot_pages": M,
        "NB": NB,
        "colds": colds,
        "resident": resident,
        "tiered": {},
    }
    print(f"\n[{args.model}] TIERED  hot={M}/{NB} pages, r={r}/{d}")
    print(
        f"  resident d-elem/head: "
        + "  ".join(f"{k} {v['resident_d_elements']}" for k, v in resident.items())
    )
    print(f"{'method':8s} " + " ".join(f"Kc{Kc}(acc)".rjust(12) for Kc in colds))
    for mode in methods:
        cells, row = [], {}
        for Kc in colds:
            a = acc[mode][Kc][0] / max(1, acc[mode][Kc][1])
            row[f"Kc{Kc}"] = a
            cells.append(f"{a:.2f}")
        res["tiered"][mode] = row
        print(f"{mode:8s} " + " ".join(c.rjust(12) for c in cells))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
