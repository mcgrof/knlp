#!/usr/bin/env python3
"""P2.1 — router-only K-bias debias: the centroid-contamination causal test.

The K projection bias b_K is added PRE-RoPE, so a cached post-RoPE key is
  k_postRoPE_j = R_j (W_K x_j + b_K) = R_j c_j + R_j b_K.
A uniform b_K would cancel under softmax, but after RoPE the rotated bias R_j b_K is
position-dependent and survives. It can pull every block CENTROID toward a common
low-dimensional direction (coherent centroids), which poisons cosine relevance and makes
`residual_rel`'s first pick strip a direction the other blocks share.

This test changes ONLY the keys used to build router centroids (the model's true
attention is untouched):
  k_content_j = k_postRoPE_j - R_j b_K.
It compares block centroids built from FULL vs CONTENT-only keys, on Qwen2.5-0.5B/1.5B
(large b_K) vs Qwen3-0.6B / SmolLM2-1.7B (NO b_K -> debias is a no-op control). Metrics:
centroid coherence (mean pairwise |cos|), PC1 variance share, rho_centroid
(||rotated-bias centroid|| / ||content centroid||), and — the decisive one — router
calibration: top-B recall of the blocks selected by residual_rel (α=1) and recent-Q (α=0)
against the TRUE block attention mass, FULL vs CONTENT centroids.

Decisive read: on Qwen2.5, content-only centroids are LESS coherent AND rescue
residual_rel's block selection (recall_content >> recall_full), while on the no-bias
control full == content. That confirms the K-center poisons router geometry directly —
stronger and cleaner than "K-bias makes attention diffuse". Free W7900, no_grad.
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
from kv_selectors import leaky_residual, top_k_relevance


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_centroids(keys_layer, bs):
    """keys_layer [Hkv, T, D] -> L2-normalized block centroids [Hkv, NB, D] + idx[T]."""
    Hkv, T, D = keys_layer.shape
    NB = (T + bs - 1) // bs
    idx = (torch.arange(T, device=keys_layer.device) // bs).clamp_max(NB - 1)
    kn = F.normalize(keys_layer.float(), dim=-1)
    cent = torch.zeros(Hkv, NB, D, device=keys_layer.device)
    cent.index_add_(1, idx, kn)  # add each token's unit key to its block, per kv-head
    cnt = torch.zeros(Hkv, NB, 1, device=keys_layer.device)
    cnt.index_add_(1, idx, torch.ones(Hkv, T, 1, device=keys_layer.device))
    cent = cent / cnt.clamp_min(1)
    cent = F.normalize(cent, dim=-1)
    return cent, idx


def coherence(cent, keep):
    """mean pairwise |cosine| among kept block centroids, per kv-head -> scalar mean."""
    c = cent[:, keep, :]  # [Hkv, K, D] (already unit norm)
    G = torch.einsum("hkd,hjd->hkj", c, c).abs()  # [Hkv,K,K]
    K = c.shape[1]
    eye = torch.eye(K, device=c.device, dtype=torch.bool)
    off = G[:, ~eye].view(c.shape[0], -1)
    return off.mean().item()


def pc1_share(cent, keep):
    c = cent[:, keep, :].float()  # [Hkv,K,D]
    c = c - c.mean(1, keepdim=True)
    vals = []
    for h in range(c.shape[0]):
        s = torch.linalg.svdvals(c[h])
        vals.append((s[0] ** 2 / (s**2).sum().clamp_min(1e-12)).item())
    return sum(vals) / len(vals)


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).to(device)
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

    # per-layer k_proj bias b_K reshaped [Hkv, Dh]; None where absent
    bK = []
    has_bias = False
    for li in range(nL):
        b = getattr(model.model.layers[li].self_attn.k_proj, "bias", None)
        if b is None:
            bK.append(None)
        else:
            bK.append(b.detach().float().view(Hkv, Dh).to(device))
            has_bias = True

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
        last = T - 1
        recent = list(range(max(0, T - args.recent_window), T))
        sel_pos = sorted(set(recent + [last]))
        pos_of = {p: i for i, p in enumerate(sel_pos)}
        qs, _ = capture_q(model, ids_t[:-1], sel_pos, device)

        # RoPE cos/sin for ALL positions (to rotate b_K); reuse model rotary_emb
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        dummy = torch.zeros(1, T, Dh, device=device, dtype=next(model.parameters()).dtype)
        cos, sin = model.model.rotary_emb(dummy, pos_ids)
        cos = cos[0].float()  # [T, Dh]
        sin = sin[0].float()
        n_used += 1

        keep = torch.tensor(
            [b for b in range(NB) if b >= args.sink and b < NB - 1], device=device
        )[: max(8, args.budget * 3)]

        for li in range(nL):
            kfull = keys[li][0].float()  # [Hkv, T, D]
            cent_full, idx = build_centroids(kfull, bs)
            # content-only keys: subtract rotated bias R_j b_K
            if bK[li] is not None:
                rb = bK[li].unsqueeze(1) * cos.unsqueeze(0) + rotate_half(
                    bK[li].unsqueeze(1)
                ) * sin.unsqueeze(0)  # [Hkv, T, D]
                kcont = kfull - rb
            else:
                kcont = kfull
            cent_cont, _ = build_centroids(kcont, bs)

            push("coherence_full", coherence(cent_full, keep))
            push("coherence_content", coherence(cent_cont, keep))
            push("pc1_full", pc1_share(cent_full, keep))
            push("pc1_content", pc1_share(cent_cont, keep))
            if bK[li] is not None:
                # rho_centroid = ||rotated-bias block centroid|| / ||content centroid||
                rb_cent = torch.zeros(Hkv, NB, Dh, device=device)
                rb_cent.index_add_(1, idx, rb)
                cnt = torch.zeros(Hkv, NB, 1, device=device)
                cnt.index_add_(1, idx, torch.ones(Hkv, T, 1, device=device))
                rb_cent = rb_cent / cnt.clamp_min(1)
                cc = torch.zeros(Hkv, NB, Dh, device=device)
                cc.index_add_(1, idx, kcont)
                cc = cc / cnt.clamp_min(1)
                rho = (rb_cent[:, keep].norm(dim=-1) / cc[:, keep].norm(dim=-1).clamp_min(1e-6))
                push("rho_centroid", rho.mean().item())

            # ---- router calibration at the decode (last) query ----
            qL = qs[li][:, pos_of[last], :].float()  # [Hq, D]
            kexp = kfull.repeat_interleave(group, dim=0)
            aL = (torch.einsum("hd,htd->ht", qL, kexp[:, :T, :]) / math.sqrt(Dh)).softmax(-1)
            true_mass = torch.zeros(Hq, NB, device=device)
            true_mass.index_add_(1, idx, aL)
            true_mass = true_mass.view(Hkv, group, NB).sum(1)  # [Hkv, NB]
            B = args.budget
            tb = [set(true_mass[h].topk(B).indices.tolist()) for h in range(Hkv)]
            # recent-Q probe
            q_rec = qs[li][:, torch.tensor([pos_of[p] for p in recent], device=device), :]
            q_rec = q_rec.float().view(Hkv, group, len(recent), Dh)
            p_rec = F.normalize(F.normalize(q_rec, dim=-1).mean((1, 2)), dim=-1)

            for cname, cent in (("full", cent_full), ("content", cent_cont)):
                # residual_rel (α=1) picks, and recent-Q (α=0) picks
                for sel_name, a in (("resid", 1.0), ("rel", 0.0)):
                    picks = leaky_residual(p_rec, cent, B, alpha=a)  # [Hkv, B]
                    rec = sum(
                        len(set(picks[h].tolist()) & tb[h]) for h in range(Hkv)
                    ) / (B * Hkv)
                    push(f"{sel_name}_recall_{cname}", rec)
        print(f"  [{n_used}/{args.eval}] T={T} NB={NB}", flush=True)

    def summ(xs):
        if not xs:
            return None
        t = torch.tensor(xs, dtype=torch.float)
        return {"n": len(xs), "mean": round(t.mean().item(), 4), "median": round(t.median().item(), 4)}

    out = {
        "model": args.model, "Hq": Hq, "Hkv": Hkv, "group": group, "nL": nL,
        "has_k_bias": has_bias, "length": args.length, "budget": args.budget,
        "n_used": n_used, "metrics": {k: summ(v) for k, v in sorted(acc.items())},
    }
    m = {k: (summ(v)["mean"] if summ(v) else None) for k, v in acc.items()}
    out["debias_read"] = {
        "coherence_full": m.get("coherence_full"),
        "coherence_content": m.get("coherence_content"),
        "coherence_drop": (round(m["coherence_full"] - m["coherence_content"], 4)
                           if m.get("coherence_full") and m.get("coherence_content") else None),
        "rho_centroid": m.get("rho_centroid"),
        "resid_recall_full": m.get("resid_recall_full"),
        "resid_recall_content": m.get("resid_recall_content"),
        "resid_recall_rescue": (round(m["resid_recall_content"] - m["resid_recall_full"], 4)
                                if m.get("resid_recall_content") and m.get("resid_recall_full") else None),
        "rel_recall_full": m.get("rel_recall_full"),
        "rel_recall_content": m.get("rel_recall_content"),
        "note": "on a K-biased model, content-only should DROP coherence and RESCUE "
        "resid_recall (content>full); on a no-bias control both are ~equal (no-op)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["debias_read"], indent=2))
    print(f"P2_1_DEBIAS_DONE {args.out}")


if __name__ == "__main__":
    main()
