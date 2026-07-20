#!/usr/bin/env python3
"""P2.0 — granularity-correct attention battery (lattice/KRI plan, regroup-20260720).

Central question: is Qwen2.5's apparent "diffuse attention" (low top1%_mass_share in the
mechanism finding) GENUINE per-head flatness, or an AGGREGATION ARTIFACT — peaky heads
that DISAGREE within a shared GQA group, or peaky attention whose preferred blocks DRIFT
over positions, averaged into something that only looks diffuse?

We measure attention concentration at the correct granularity, from captured post-RoPE Q
and cached post-RoPE K (no full T×T map materialized), on plain packed WikiText text:

  D_local      : per-(layer, query-head, query-position) block-attention concentration
                 (mean over positions/heads) — peakiness BEFORE any cross-head pooling.
  D_aggregate  : concentration of the block-attention distribution POOLED over query
                 heads (within a KV group) and positions — peakiness AFTER pooling.
  head_disagree: within each shared KV group, mean pairwise Jensen-Shannon divergence of
                 the query heads' block-attention distributions (high = peaky-but-
                 disagreeing, a GQA confound the old aggregate hides).
  top1_perhead / top1_pooled : the mechanism-finding statistic recomputed per-head then
                 averaged, vs on the pooled distribution — the artifact, side by side.
  persistence  : Jaccard overlap of the top-B important blocks between the last query
                 position and earlier probe positions (temporal drift).
  calib_recentq / calib_lastk : router calibration — Spearman + top-B recall of the
                 block ranking by cos(probe, block-centroid) against the TRUE block
                 attention mass, for the recent-Q-mean and last-K probes.

Decisive read: D_local high + D_aggregate low (and high head_disagree) ⇒ diffuseness is a
pooling artifact, and the "Qwen2.5 flat attention" language must be dropped. All at
per-query-head AND per-KV-group granularity, sink/recent tokens included and excluded.
Free W7900; torch.no_grad; no training.
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
from diag_perhead_oracle import capture_q, block_stats, get_values
from niah_evict_perhead import get_keys


def norm_entropy_conc(p, dim=-1, eps=1e-12):
    """Concentration = 1 - H(p)/log(n): 1 = one-hot, 0 = uniform."""
    n = p.shape[dim]
    if n <= 1:
        return torch.ones(p.shape[:-1], device=p.device)
    H = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim)
    return 1.0 - H / math.log(n)


def top_frac_mass(p, frac=0.01, dim=-1):
    """Mass on the top-ceil(frac*n) entries (the mechanism-finding statistic)."""
    n = p.shape[dim]
    k = max(1, int(math.ceil(frac * n)))
    return p.topk(k, dim=dim).values.sum(dim)


def js_div(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a.clamp_min(eps) * (a.clamp_min(eps).log() - b.clamp_min(eps).log())).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


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
    ap.add_argument("--sink", type=int, default=4, help="sink blocks to optionally exclude")
    ap.add_argument("--n-probe", type=int, default=16, help="query positions sampled for stats")
    ap.add_argument("--budget", type=int, default=16, help="B for top-budget/persistence/recall")
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

    acc = {}  # metric -> list of scalars

    def push(k, v):
        acc.setdefault(k, [])
        if isinstance(v, torch.Tensor):
            acc[k].extend(v.flatten().tolist())
        else:
            acc[k].append(float(v))

    n_used = 0
    for ei in range(args.eval):
        # pack plain wikitext to ~length tokens
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

        # query positions to analyze: last position + recent window + spread probes
        last = T - 1
        recent = list(range(max(0, T - args.recent_window), T))
        probes = sorted(
            set(int(x) for x in torch.linspace(bs * 2, T - 2, args.n_probe).tolist())
        )
        sel_pos = sorted(set(recent + probes + [last]))
        pos_of = {p: i for i, p in enumerate(sel_pos)}
        qs, _ = capture_q(model, ids_t[:-1], sel_pos, device)
        n_used += 1

        for li in range(nL):
            cent, _vn, idx, khn = block_stats(keys[li][0], vals[li][0], bs)  # cent[Hkv,NB,D], idx[T]
            kl_raw = keys[li][0].float()  # [Hkv, T, D]
            ql = qs[li].float()  # [Hq, P, D]
            idx = idx.to(device)

            # ---- attention at probe positions, per query-head, block-level ----
            # For a query at absolute pos ppos, causal over key positions 0..ppos.
            # Compute block-attention per (query-head, probe): softmax over keys, sum to blocks.
            probe_list = [(pos_of[p], p) for p in probes if p >= bs * 2]
            # per-KV-group container: block dists per query head, for disagreement
            for (pi, ppos) in probe_list:
                nb_causal = (ppos // bs) + 1  # blocks up to and including current
                if nb_causal < args.budget + args.sink + 2:
                    continue
                q = ql[:, pi, :]  # [Hq, D]
                kexp = kl_raw.repeat_interleave(group, dim=0)  # [Hq, T, D]
                logits = torch.einsum("hd,htd->ht", q, kexp[:, : ppos + 1, :]) / math.sqrt(Dh)
                a = logits.softmax(-1)  # [Hq, ppos+1] token attention
                # sum tokens into blocks
                bl = torch.zeros(Hq, nb_causal, device=device)
                bl.index_add_(1, idx[: ppos + 1], a)
                bl_norm = bl / bl.sum(-1, keepdim=True).clamp_min(1e-12)  # [Hq, nb]
                # exclude sink+recent variant
                keepmask = torch.ones(nb_causal, dtype=torch.bool, device=device)
                keepmask[: args.sink] = False
                cur_b = ppos // bs
                keepmask[max(0, cur_b - 1) : cur_b + 1] = False  # local/current
                bl_ex = bl.clone()
                bl_ex[:, ~keepmask] = 0
                bl_ex = bl_ex / bl_ex.sum(-1, keepdim=True).clamp_min(1e-12)

                # D_local: per-query-head block concentration (incl & excl)
                push("D_local_inc", norm_entropy_conc(bl_norm))
                push("D_local_exc", norm_entropy_conc(bl_ex))
                push("top1_perhead_inc", top_frac_mass(bl_norm))
                push("top1_perhead_exc", top_frac_mass(bl_ex))

                # per-KV-group pooling -> D_aggregate + head disagreement
                blg = bl_norm.view(Hkv, group, nb_causal)
                pooled = blg.mean(1)  # [Hkv, nb] pooled over query heads in the group
                pooled = pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-12)
                push("D_aggregate", norm_entropy_conc(pooled))
                push("top1_pooled", top_frac_mass(pooled))
                # head disagreement: mean pairwise JS within group
                if group >= 2:
                    for gi in range(group):
                        for gj in range(gi + 1, group):
                            push("head_disagree", js_div(blg[:, gi], blg[:, gj]))

            # ---- router calibration at the LAST position (the decode query) ----
            pi_last = pos_of[last]
            q_last = ql[:, pi_last, :].view(Hkv, group, Dh)
            # true block mass at last position (per kv-head, sum over group)
            qL = ql[:, pi_last, :]  # [Hq, D]
            kexp = kl_raw.repeat_interleave(group, dim=0)
            logitsL = torch.einsum("hd,htd->ht", qL, kexp[:, :T, :]) / math.sqrt(Dh)
            aL = logitsL.softmax(-1)  # [Hq, T]
            blL = torch.zeros(Hq, NB, device=device)
            blL.index_add_(1, idx[:T], aL)
            true_mass = blL.view(Hkv, group, NB).sum(1)  # [Hkv, NB]
            # probes: recent-Q mean & last-K
            q_rec = ql[:, torch.tensor([pos_of[p] for p in recent], device=device), :]
            q_rec = q_rec.view(Hkv, group, len(recent), Dh)
            p_rec = F.normalize(F.normalize(q_rec, dim=-1).mean((1, 2)), dim=-1)  # [Hkv,D]
            p_lastk = F.normalize(khn[:, -1, :].float(), dim=-1)  # [Hkv, D]
            for name, probe in (("recentq", p_rec), ("lastk", p_lastk)):
                sc = torch.einsum("hd,hnd->hn", probe.float(), cent.float())  # [Hkv, NB]
                # Spearman per kv-head between sc and true_mass
                for hh in range(Hkv):
                    a_r = sc[hh].argsort().argsort().float()
                    b_r = true_mass[hh].argsort().argsort().float()
                    a_c = a_r - a_r.mean()
                    b_c = b_r - b_r.mean()
                    rho = (a_c * b_c).sum() / (a_c.norm() * b_c.norm()).clamp_min(1e-12)
                    push(f"calib_{name}_spearman", rho)
                # top-B recall: fraction of true top-B blocks in probe top-B
                B = args.budget
                tb = true_mass.topk(B, dim=-1).indices  # [Hkv, B]
                pb = sc.topk(B, dim=-1).indices
                for hh in range(Hkv):
                    push(
                        f"calib_{name}_topB_recall",
                        len(set(tb[hh].tolist()) & set(pb[hh].tolist())) / B,
                    )

            # ---- temporal persistence: top-B blocks last vs earliest probe ----
            if probe_list:
                pi0, pp0 = probe_list[0]
                q0 = ql[:, pi0, :]
                kexp0 = kl_raw.repeat_interleave(group, dim=0)
                lg0 = torch.einsum("hd,htd->ht", q0, kexp0[:, : pp0 + 1, :]) / math.sqrt(Dh)
                a0 = lg0.softmax(-1)
                nb0 = (pp0 // bs) + 1
                bl0 = torch.zeros(Hq, nb0, device=device)
                bl0.index_add_(1, idx[: pp0 + 1], a0)
                bl0g = bl0.view(Hkv, group, nb0).sum(1)
                B0 = min(args.budget, nb0)
                topL = true_mass[:, :nb0].topk(B0, dim=-1).indices
                top0 = bl0g.topk(B0, dim=-1).indices
                for hh in range(Hkv):
                    inter = len(set(topL[hh].tolist()) & set(top0[hh].tolist()))
                    push("persistence_jaccard", inter / (2 * B0 - inter))
        print(f"  [{n_used}/{args.eval}] T={T} NB={NB}", flush=True)

    def summ(xs):
        if not xs:
            return None
        t = torch.tensor(xs, dtype=torch.float)
        return {
            "n": len(xs),
            "mean": round(t.mean().item(), 4),
            "median": round(t.median().item(), 4),
        }

    out = {
        "model": args.model,
        "Hq": Hq,
        "Hkv": Hkv,
        "group": group,
        "nL": nL,
        "length": args.length,
        "block_size": bs,
        "budget": args.budget,
        "n_used": n_used,
        "metrics": {k: summ(v) for k, v in sorted(acc.items())},
    }
    # decisive artifact flag: is "diffuse" an aggregation artifact?
    dl = summ(acc.get("D_local_inc", []))
    da = summ(acc.get("D_aggregate", []))
    hd = summ(acc.get("head_disagree", []))
    out["diffuseness_read"] = {
        "D_local_mean": dl["mean"] if dl else None,
        "D_aggregate_mean": da["mean"] if da else None,
        "head_disagree_mean": hd["mean"] if hd else None,
        "top1_perhead_mean": (summ(acc.get("top1_perhead_inc", [])) or {}).get("mean"),
        "top1_pooled_mean": (summ(acc.get("top1_pooled", [])) or {}).get("mean"),
        "artifact_hint": (
            "if D_local >> D_aggregate (and head_disagree high), the 'diffuse' label is a "
            "GQA/pooling artifact, not genuine per-head flatness"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["diffuseness_read"], indent=2))
    print(f"P2_BATTERY_DONE {args.out}")


if __name__ == "__main__":
    main()
