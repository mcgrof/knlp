#!/usr/bin/env python3
"""P2.2 — faithful routing-quality test: attention-output reconstruction e_S.

P2.1's routing-negative used recall vs biased true-mass, which structurally favors the
biased (full) centroids. This replaces the proxy with the FAITHFUL metric the doc names:
per (layer, kv-head, query), evict all but the router-selected top-B blocks (+ sink +
recent), recompute the attention output over the KEPT keys only, and measure relative
error to the full-cache output:

    e_S = || o_full - o_kept || / || o_full ||.

o_full uses the model's TRUE (biased) attention, so this is a fair test: it asks whether
the blocks a router KEEPS actually carry the attention output — the real KV-routing job —
regardless of whether the bias is "content" or not.

Cells: routers {recent-Q (α=0), residual_rel (α=1), h2o} × centroids {full, content-only
(key − R_j b_K)}. Decisive: on Qwen2.5, does content-only centroid give LOWER e_S for
residual_rel (routing rescue) — the KL-faithful version of P2.1? And does residual_rel beat
recent-Q on e_S on plain text (the general-LM regime where residual_rel wins on KL)? No-bias
models are the no-op control (full ≡ content). Free W7900, no_grad.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import load_filler_sentences
from diag_perhead_oracle import capture_q, get_values
from niah_evict_perhead import get_keys
from kv_selectors import leaky_residual, top_k_mass


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_centroids(k, bs):
    Hkv, T, D = k.shape
    NB = (T + bs - 1) // bs
    idx = (torch.arange(T, device=k.device) // bs).clamp_max(NB - 1)
    kn = F.normalize(k.float(), dim=-1)
    cent = torch.zeros(Hkv, NB, D, device=k.device)
    cent.index_add_(1, idx, kn)
    cnt = torch.zeros(Hkv, NB, 1, device=k.device)
    cnt.index_add_(1, idx, torch.ones(Hkv, T, 1, device=k.device))
    return F.normalize(cent / cnt.clamp_min(1), dim=-1), idx


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--eval", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--n-probe", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    group = Hq // Hkv
    nL = model.config.num_hidden_layers
    Dh = getattr(model.config, "head_dim", model.config.hidden_size // Hq)
    bs = args.block_size
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)

    bK = []
    has_bias = False
    for li in range(nL):
        b = getattr(model.model.layers[li].self_attn.k_proj, "bias", None)
        bK.append(None if b is None else b.detach().float().view(Hkv, Dh).to(device))
        has_bias = has_bias or b is not None

    acc = {}

    def push(k, v):
        acc.setdefault(k, []).append(float(v))

    n_used = 0
    for ei in range(args.eval):
        text, need = "", args.length * 5
        while len(text) < need:
            text += sents[rng.randrange(len(sents))] + " "
        ids = tok(text, add_special_tokens=False)["input_ids"][: args.length]
        if len(ids) < 256:
            continue
        ids_t = torch.tensor(ids)
        T = len(ids) - 1
        NB = (T + bs - 1) // bs
        if NB < args.budget + args.sink + 4:
            continue
        past = model(ids_t[:-1].unsqueeze(0).to(device), use_cache=True).past_key_values
        keys, vals = get_keys(past), get_values(past)
        recent = list(range(max(0, T - args.recent_window), T))
        probes = sorted(set(int(x) for x in torch.linspace(T // 4, T - 1, args.n_probe).tolist()))
        sel_pos = sorted(set(recent + probes + [T - 1]))
        pos_of = {p: i for i, p in enumerate(sel_pos)}
        qs, _ = capture_q(model, ids_t[:-1], sel_pos, device)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        dummy = torch.zeros(1, T, Dh, device=device, dtype=next(model.parameters()).dtype)
        cos, sin = model.model.rotary_emb(dummy, pos_ids)
        cos, sin = cos[0].float(), sin[0].float()
        n_used += 1
        sink_recent = set(range(args.sink))

        for li in range(nL):
            kfull = keys[li][0].float()  # [Hkv, T, D]
            v = vals[li][0].float()  # [Hkv, T, D]
            cent_full, idx = build_centroids(kfull, bs)
            if bK[li] is not None:
                rb = bK[li].unsqueeze(1) * cos.unsqueeze(0) + rotate_half(bK[li].unsqueeze(1)) * sin.unsqueeze(0)
                cent_cont, _ = build_centroids(kfull - rb, bs)
            else:
                cent_cont = cent_full
            ql = qs[li].float()
            # recent-Q probe
            q_rec = ql[:, torch.tensor([pos_of[p] for p in recent], device=device), :]
            q_rec = q_rec.view(Hkv, group, len(recent), Dh)
            p_rec = F.normalize(F.normalize(q_rec, dim=-1).mean((1, 2)), dim=-1)

            for pp in probes:
                ppos = pp
                cur_b = ppos // bs
                nb = cur_b + 1
                rec_blocks = set(range(max(0, cur_b - args.recent_window // bs), cur_b + 1))
                keepbase = sink_recent | rec_blocks
                qh = ql[:, pos_of[ppos], :]  # [Hq, D]
                ke = kfull.repeat_interleave(group, dim=0)[:, : ppos + 1, :]
                ve = v.repeat_interleave(group, dim=0)[:, : ppos + 1, :]
                logits = torch.einsum("hd,htd->ht", qh, ke) / math.sqrt(Dh)  # [Hq, ppos+1]
                a_full = logits.softmax(-1)
                o_full = torch.einsum("ht,htd->hd", a_full, ve)  # [Hq, D]
                tok_block = idx[: ppos + 1]  # [ppos+1]

                # mass-based h2o (true attention mass per block, pooled to kv)
                blmass = torch.zeros(Hq, nb, device=device)
                blmass.index_add_(1, tok_block, a_full)
                mass_kv = blmass.view(Hkv, group, nb).sum(1)

                for cname, cent in (("full", cent_full), ("content", cent_cont)):
                    cc = cent[:, :nb, :]
                    routers = {
                        "rel": leaky_residual(p_rec, cc, args.budget, alpha=0.0),
                        "resid": leaky_residual(p_rec, cc, args.budget, alpha=1.0),
                    }
                    if cname == "full":
                        routers["h2o"] = top_k_mass(mass_kv, args.budget)
                    for rname, picks in routers.items():
                        # kept token mask per query head (expand kv picks to q heads)
                        for hh in range(Hkv):
                            kb = set(picks[hh].tolist()) | keepbase
                            keep_tok = torch.tensor(
                                [t for t in range(ppos + 1) if int(tok_block[t]) in kb],
                                device=device,
                            )
                            for gq in range(group):
                                qi = hh * group + gq
                                lg = logits[qi, keep_tok]
                                aS = lg.softmax(-1)
                                oS = (aS.unsqueeze(-1) * ve[qi, keep_tok, :]).sum(0)
                                eS = (o_full[qi] - oS).norm() / o_full[qi].norm().clamp_min(1e-6)
                                push(f"eS_{rname}_{cname}", eS.item())
            torch.cuda.empty_cache()
        print(f"  [{n_used}/{args.eval}] T={T}", flush=True)

    def summ(xs):
        if not xs:
            return None
        t = torch.tensor(xs, dtype=torch.float)
        return {"n": len(xs), "mean": round(t.mean().item(), 4), "median": round(t.median().item(), 4)}

    m = {k: (summ(v)["mean"] if summ(v) else None) for k, v in acc.items()}
    out = {
        "model": args.model, "Hkv": Hkv, "group": group, "nL": nL, "has_k_bias": has_bias,
        "length": args.length, "budget": args.budget, "n_used": n_used,
        "metrics": {k: summ(v) for k, v in sorted(acc.items())},
        "routing_read": {
            "eS_rel_full": m.get("eS_rel_full"),
            "eS_resid_full": m.get("eS_resid_full"),
            "eS_h2o_full": m.get("eS_h2o_full"),
            "eS_rel_content": m.get("eS_rel_content"),
            "eS_resid_content": m.get("eS_resid_content"),
            "resid_debias_rescue": (
                round(m["eS_resid_full"] - m["eS_resid_content"], 4)
                if m.get("eS_resid_full") and m.get("eS_resid_content") else None
            ),
            "resid_vs_rel_full": (
                round(m["eS_resid_full"] - m["eS_rel_full"], 4)
                if m.get("eS_resid_full") and m.get("eS_rel_full") else None
            ),
            "note": "lower eS = better reconstruction. resid_debias_rescue>0 means "
            "content-only centroids improve residual_rel routing (KL-faithful P2.1). "
            "resid_vs_rel_full<0 means residual_rel beats recent-Q on general-LM output.",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["routing_read"], indent=2))
    print(f"P2_2_ROUTING_DONE {args.out}")


if __name__ == "__main__":
    main()
