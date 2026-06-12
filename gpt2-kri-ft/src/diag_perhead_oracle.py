#!/usr/bin/env python3
"""Phase A-minus: per-head oracle + probe diagnostics for the portfolio router.

Before any portfolio grid runs, this answers (per ChatGPT-Pro review, see
knlp-key-results/lattice-kri/portfolio-router-20260612/PLAN.md):

  1. Probe bakeoff -- rank of the NIAH answer block under relevance scored by
     the legacy last-position-K probe vs TRUE post-RoPE Q (group-mean over the
     GQA q-heads sharing each kv-head) vs mean/max recent-Q probes.
  2. List-union oracle -- does ANY ingredient list (relevance R, mass H,
     diversity D, value V) rank the answer block within budget K at all. If
     the union ceiling cannot retain the needle, no portfolio composition can.
  3. Value-arm diagnostic -- block ranking by value-norm (V), a key-free axis
     the current router ignores entirely.
  4. Head-level mechanism test -- per-head accumulated true-Q attention-mass
     concentration, correlated with the per-head H2O-vs-residual advantage
     (mechanism vs model-family astrology).

True Q is captured with q_proj forward hooks at selected positions, then the
layer's q_norm (Qwen3) and rotary embedding are applied post-hoc so probe and
key live in the same post-RoPE space (cached keys are post-RoPE). GQA is
handled by aggregating across the q-heads that share each kv head.

Usage:
  python3 diag_perhead_oracle.py --model HuggingFaceTB/SmolLM2-360M \
      --length 2048 --eval 20 --seed 0 --out OUT/diag.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, load_filler_sentences
from niah_evict_perhead import answer_block_indices, get_keys

BUDGETS = (8, 16, 32, 64)
LISTS = (
    "R_lastk",
    "R_trueq",
    "R_recentq_mean",
    "R_recentq_max",
    "D_resid",
    "H_kproxy",
    "H_trueq",
    "V_vnorm",
)
UNIONS = {
    "U_RD": ("R_trueq", "D_resid"),
    "U_RH": ("R_trueq", "H_trueq"),
    "U_RHD": ("R_trueq", "H_trueq", "D_resid"),
    "U_RHDV": ("R_trueq", "H_trueq", "D_resid", "V_vnorm"),
    "U_all": LISTS,
}


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def get_values(cache):
    if hasattr(cache, "value_cache"):
        return cache.value_cache
    return [layer.values for layer in cache.layers]


def capture_q(model, ids_t, sel_pos, device):
    """Prefill forward with q_proj hooks; returns per-layer post-RoPE per-head
    Q at sel_pos: list of [Hq, P, D] float32, plus the past_key_values."""
    layers = model.model.layers
    raw = [None] * len(layers)
    hooks = []
    sel_t = torch.tensor(sel_pos, device=device)

    def mk(i):
        def hook(_m, _inp, out):
            raw[i] = out[0, sel_t, :].detach().float()  # [P, Hq*D]

        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(mk(i)))
    try:
        past = model(ids_t.unsqueeze(0).to(device), use_cache=True).past_key_values
    finally:
        for h in hooks:
            h.remove()

    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    rotary = model.model.rotary_emb
    pos_ids = sel_t.unsqueeze(0)
    dummy = torch.zeros(1, len(sel_pos), 1, device=device, dtype=torch.float32)
    cos, sin = rotary(dummy, pos_ids)
    cos, sin = cos[0].float(), sin[0].float()  # [P, D]

    qs = []
    for i, layer in enumerate(layers):
        P = raw[i].shape[0]
        q = raw[i].view(P, -1, head_dim).permute(1, 0, 2)  # [Hq, P, D]
        qn = getattr(layer.self_attn, "q_norm", None)
        if qn is not None:  # Qwen3: per-head RMSNorm before RoPE
            q = qn(q.to(next(qn.parameters()).dtype)).float()
        qs.append(q * cos.unsqueeze(0) + rotate_half(q) * sin.unsqueeze(0))
    return qs, past


def block_stats(kl, vl, bs):
    """Per-layer [Hkv,T,D] keys/values -> (centroids [Hkv,NB,D] normalised,
    vnorm [Hkv,NB], idx [T])."""
    Hkv, T, D = kl.shape
    NB = (T + bs - 1) // bs
    idx = torch.arange(T, device=kl.device) // bs
    khn = F.normalize(kl.float(), dim=-1)
    cent = torch.zeros(Hkv, NB, D, device=kl.device)
    cnt = torch.zeros(NB, device=kl.device)
    cent.index_add_(1, idx, khn)
    cnt.index_add_(0, idx, torch.ones(T, device=kl.device))
    cent = F.normalize(cent / cnt.clamp(min=1).view(1, NB, 1), dim=-1)
    vn = torch.zeros(Hkv, NB, device=kl.device)
    vn.index_add_(1, idx, vl.float().norm(dim=-1))
    vn = vn / cnt.clamp(min=1).view(1, NB)
    return cent, vn, idx, khn


def resid_ranks(cent, probe, ans, max_steps=64):
    """Vectorised greedy residual_rel over heads; returns per-head pick-step of
    the answer (censored at max_steps)."""
    Hkv, NB, D = cent.shape
    q_resid = probe.clone()
    chosen = torch.zeros(Hkv, NB, dtype=torch.bool, device=cent.device)
    rank = torch.full((Hkv,), max_steps, dtype=torch.long, device=cent.device)
    ans_t = torch.tensor(sorted(ans), device=cent.device)
    for step in range(min(max_steps, NB)):
        score = torch.einsum("hd,hnd->hn", F.normalize(q_resid, dim=-1), cent)
        score = score.masked_fill(chosen, float("-inf"))
        pick = score.argmax(dim=-1)  # [Hkv]
        hit = (pick.unsqueeze(1) == ans_t.unsqueeze(0)).any(dim=1)
        rank = torch.where(hit & (rank == max_steps), step, rank)
        chosen.scatter_(1, pick.unsqueeze(1), True)
        kc = torch.gather(cent, 1, pick.view(Hkv, 1, 1).expand(Hkv, 1, D)).squeeze(1)
        q_resid = q_resid - (q_resid * kc).sum(-1, keepdim=True) * kc
    return rank


def order_rank(score, ans):
    """Rank (0-based) of the best answer block under descending score [H,NB]."""
    order = score.argsort(dim=-1, descending=True)  # [H, NB]
    ans_t = torch.tensor(sorted(ans), device=score.device)
    isans = (order.unsqueeze(-1) == ans_t.view(1, 1, -1)).any(-1)  # [H, NB]
    return isans.float().argmax(dim=-1)  # first True position


def mass_trueq(qlayer, probe_sel, khn_raw, kl_raw, idx, NB, group):
    """Accumulated true-Q attention mass per (kv-head, block). qlayer [Hq,P,D]
    post-RoPE; probe_sel = indices into P of the mass probes; kl_raw [Hkv,T,D]."""
    Hq, _, D = qlayer.shape
    Hkv = kl_raw.shape[0]
    T = kl_raw.shape[1]
    mass = torch.zeros(Hkv, NB, device=kl_raw.device)
    kf = kl_raw.float()
    scale = 1.0 / math.sqrt(D)
    for pi, ppos in probe_sel:
        qp = qlayer[:, pi, :]  # [Hq, D]
        kv = kf[:, : ppos + 1, :]  # [Hkv, p+1, D]
        kx = kv.repeat_interleave(group, dim=0)  # [Hq, p+1, D]
        a = F.softmax(torch.einsum("hd,htd->ht", qp, kx) * scale, dim=-1)
        a = a.view(Hkv, group, -1).sum(1)  # sum q-heads sharing the kv head
        mass.index_add_(1, idx[: ppos + 1], a)
    return mass


def spearman(x, y):
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx = (rx - rx.mean()) / (rx.std() + 1e-9)
    ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-blocks", type=int, default=8)
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--mass-probes", type=int, default=48)
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
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)
    bs = args.block_size

    ranks = {name: [] for name in LISTS}  # per (head,sample) answer rank
    head_key = []  # (layer, kvhead) per entry, aligned with ranks lists
    conc_top1, conc_ent = [], []  # per entry
    auto_keep = [0, 0]
    n_used = 0

    for ei in range(args.eval):
        text, spans, needles, (qk, qv) = build_context(
            tok, args.length, args.needles, sents, rng
        )
        ids, ans = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)[:-1]
        T = ids_t.shape[0]
        NB = (T + bs - 1) // bs
        if not ans or max(ans) >= NB:
            continue
        n_used += 1
        in_auto = any(b < args.sink_blocks or b >= NB - args.recent_blocks for b in ans)
        auto_keep[0] += int(in_auto)
        auto_keep[1] += 1

        # selected positions: last, recent window, mass probes
        recent = list(range(max(0, T - args.recent_window), T))
        probes = sorted(
            set(int(x) for x in torch.linspace(0, T - 2, args.mass_probes).tolist())
        )
        sel_pos = sorted(set(recent + probes + [T - 1]))
        pos_of = {p: i for i, p in enumerate(sel_pos)}
        probe_sel = [(pos_of[p], p) for p in probes]

        qs, past = capture_q(model, ids_t, sel_pos, device)
        keys, vals = get_keys(past), get_values(past)

        for li in range(len(keys)):
            kl = keys[li][0]  # [Hkv, T, D]
            vl = vals[li][0]
            cent, vn, idx, khn = block_stats(kl, vl, bs)
            ql = qs[li]  # [Hq, P, D] post-RoPE float32

            # probes per kv-head
            p_lastk = F.normalize(khn[:, -1, :], dim=-1)  # [Hkv, D]
            q_last = ql[:, pos_of[T - 1], :].view(Hkv, group, -1)
            p_trueq = F.normalize(F.normalize(q_last, dim=-1).mean(1), dim=-1)
            ridx = torch.tensor([pos_of[p] for p in recent], device=device)
            q_rec = ql[:, ridx, :].view(Hkv, group, len(recent), -1)
            q_recn = F.normalize(q_rec, dim=-1)
            p_recmean = F.normalize(q_recn.mean(dim=(1, 2)), dim=-1)

            # relevance lists
            ranks["R_lastk"].append(
                order_rank(torch.einsum("hd,hnd->hn", p_lastk, cent), ans)
            )
            ranks["R_trueq"].append(
                order_rank(torch.einsum("hd,hnd->hn", p_trueq, cent), ans)
            )
            ranks["R_recentq_mean"].append(
                order_rank(torch.einsum("hd,hnd->hn", p_recmean, cent), ans)
            )
            sc_max = torch.einsum("hgwd,hnd->hgwn", q_recn, cent).amax(dim=(1, 2))
            ranks["R_recentq_max"].append(order_rank(sc_max, ans))

            # diversity (the current residual_rel, legacy probe)
            ranks["D_resid"].append(resid_ranks(cent, p_lastk, ans))

            # mass lists
            m_kproxy = torch.zeros(Hkv, NB, device=device)
            for _, ppos in probe_sel:
                a = F.softmax(
                    torch.einsum("hd,htd->ht", khn[:, ppos, :], khn[:, : ppos + 1, :]),
                    dim=-1,
                )
                m_kproxy.index_add_(1, idx[: ppos + 1], a)
            ranks["H_kproxy"].append(order_rank(m_kproxy, ans))
            m_true = mass_trueq(ql, probe_sel, khn, kl, idx, NB, group)
            ranks["H_trueq"].append(order_rank(m_true, ans))

            # value arm
            ranks["V_vnorm"].append(order_rank(vn, ans))

            # per-head mass concentration (true-Q mass)
            p = m_true / m_true.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            conc_top1.append(p.amax(dim=-1))
            ent = -(p * (p + 1e-12).log()).sum(-1) / math.log(NB)
            conc_ent.append(ent)
            head_key.extend([(li, h) for h in range(Hkv)])

        del past, qs
        torch.cuda.empty_cache()
        print(f"  sample {ei + 1}/{args.eval} done (T={T} NB={NB})", flush=True)

    # ---- aggregate ----
    out = {
        "model": args.model,
        "length": args.length,
        "eval_used": n_used,
        "block_size": bs,
        "seed": args.seed,
        "Hq": Hq,
        "Hkv": Hkv,
        "auto_keep_frac": auto_keep[0] / max(1, auto_keep[1]),
        "lists": {},
        "unions": {},
        "mechanism": {},
    }
    cat = {name: torch.cat(ranks[name]) for name in LISTS}
    for name in LISTS:
        r = cat[name].float()
        out["lists"][name] = {
            "retention": {f"K{K}": float((r < K).float().mean()) for K in BUDGETS},
            "p50_rank": float(r.median()),
            "p90_rank": float(r.quantile(0.9)),
        }
    for uname, members in UNIONS.items():
        r = torch.stack([cat[m] for m in members]).min(dim=0).values.float()
        out["unions"][uname] = {
            "retention": {f"K{K}": float((r < K).float().mean()) for K in BUDGETS},
            "p50_rank": float(r.median()),
        }
    # mechanism: per-head H2O-vs-resid advantage vs concentration
    t1 = torch.cat(conc_top1)
    en = torch.cat(conc_ent)
    adv16 = (cat["H_trueq"] < 16).float() - (cat["D_resid"] < 16).float()
    out["mechanism"] = {
        "mean_top1_mass_share": float(t1.mean()),
        "mean_mass_entropy": float(en.mean()),
        "spearman_adv16_vs_top1share": spearman(adv16, t1),
        "spearman_adv16_vs_entropy": spearman(adv16, en),
    }

    print(
        f"\n[{args.model}] L={args.length} samples={n_used} "
        f"auto_keep={out['auto_keep_frac']:.3f}"
    )
    print(f"{'list':16s} " + " ".join(f"K{K:>3}" for K in BUDGETS) + "  p50")
    for name in LISTS:
        d = out["lists"][name]
        row = " ".join(f"{d['retention'][f'K{K}']:.2f}" for K in BUDGETS)
        print(f"{name:16s} {row}  {d['p50_rank']:.0f}")
    for uname in UNIONS:
        d = out["unions"][uname]
        row = " ".join(f"{d['retention'][f'K{K}']:.2f}" for K in BUDGETS)
        print(f"{uname:16s} {row}  {d['p50_rank']:.0f}")
    m = out["mechanism"]
    print(
        f"mech: top1share={m['mean_top1_mass_share']:.3f} "
        f"ent={m['mean_mass_entropy']:.3f} "
        f"rho(adv,top1)={m['spearman_adv16_vs_top1share']:.3f} "
        f"rho(adv,ent)={m['spearman_adv16_vs_entropy']:.3f}"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
