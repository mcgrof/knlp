#!/usr/bin/env python3
"""Phase-B: offline head-role classification (Razor/Duo-shaped).

Classifies every (layer, kv-head) into one of three roles on CALIBRATION data
(NIAH samples at a calibration seed, disjoint from the eval seeds), so the
per-head router is fixed before any gate evaluation:

  R  retrieval-capable: the recent-Q relevance list ranks the planted answer
     block within K=16 on >= thr_r of calibration samples. These heads get the
     full multi-probe relevance budget (the Phase-A lesson: the reserve IS the
     retrieval method).
  H  diffuse: not retrieval-capable AND mean top-1 attention-mass share
     <= thr_diffuse. These heads get a mass quota (the Gate-Q lesson: exact
     mass fixes Qwen2.5 KL at matched memory).
  D  everything else: legacy unrestricted residual_rel (the Gate-G lesson:
     SmolLM2 small-K wants no quotas and no pool restriction).

Thresholds are family-agnostic constants chosen once on the dev trio and
frozen for the held-out families (training-free != tuning-free discipline).

Usage:
  python3 head_roles.py --model HuggingFaceTB/SmolLM2-360M \
      --calib-seed 99 --eval 10 --out roles/SmolLM2-360M.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from diag_perhead_oracle import block_stats, capture_q, get_values, mass_trueq
from niah_evict_perhead import answer_block_indices, get_keys
from niah_task import build_context, load_filler_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--mass-probes", type=int, default=48)
    ap.add_argument("--calib-seed", type=int, default=99)
    ap.add_argument("--retr-k", type=int, default=16)
    ap.add_argument("--thr-r", type=float, default=0.5)
    ap.add_argument("--thr-diffuse", type=float, default=0.45)
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
    sents = load_filler_sentences(args.calib_seed)
    rng = random.Random(args.calib_seed)
    bs = args.block_size

    retr = torch.zeros(nL, Hkv)
    top1 = torch.zeros(nL, Hkv)
    n_used = 0
    for ei in range(args.eval):
        text, spans, _, (qk, qv) = build_context(
            tok, args.length, args.needles, sents, rng
        )
        ids, ans = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)[:-1]
        T = ids_t.shape[0]
        NB = (T + bs - 1) // bs
        if not ans or max(ans) >= NB:
            continue
        n_used += 1
        ans_t = torch.tensor(sorted(ans))
        recent = list(range(max(0, T - args.recent_window), T))
        probes = sorted(
            set(int(x) for x in torch.linspace(0, T - 2, args.mass_probes).tolist())
        )
        sel_pos = sorted(set(recent + probes + [T - 1]))
        pos_of = {p: i for i, p in enumerate(sel_pos)}
        probe_sel = [(pos_of[p], p) for p in probes]
        qs, past = capture_q(model, ids_t, sel_pos, device)
        keys, vals = get_keys(past), get_values(past)
        for li in range(nL):
            kl, vl = keys[li][0], vals[li][0]
            cent, _vn, idx, khn = block_stats(kl, vl, bs)
            ql = qs[li]
            ridx = torch.tensor([pos_of[p] for p in recent], device=device)
            q_rec = ql[:, ridx, :].view(Hkv, group, len(recent), -1)
            p_rec = F.normalize(F.normalize(q_rec, dim=-1).mean(dim=(1, 2)), dim=-1)
            order = torch.einsum("hd,hnd->hn", p_rec, cent).argsort(
                dim=-1, descending=True
            )[:, : args.retr_k]
            hit = (
                (order.unsqueeze(-1) == ans_t.to(device).view(1, 1, -1)).any(-1).any(-1)
            )
            retr[li] += hit.float().cpu()
            m = mass_trueq(ql, probe_sel, khn, kl, idx, NB, group)
            p = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            top1[li] += p.amax(dim=-1).cpu()
        del past, qs
        torch.cuda.empty_cache()
        print(f"  calib {ei + 1}/{args.eval}", flush=True)

    retr /= max(1, n_used)
    top1 /= max(1, n_used)
    roles = []
    counts = {"R": 0, "H": 0, "D": 0}
    for li in range(nL):
        row = []
        for hh in range(Hkv):
            if retr[li, hh] >= args.thr_r:
                r = "R"
            elif top1[li, hh] <= args.thr_diffuse:
                r = "H"
            else:
                r = "D"
            row.append(r)
            counts[r] += 1
        roles.append(row)
    out = {
        "model": args.model,
        "calib_seed": args.calib_seed,
        "eval_used": n_used,
        "thr_r": args.thr_r,
        "thr_diffuse": args.thr_diffuse,
        "retr_k": args.retr_k,
        "counts": counts,
        "roles": roles,
        "retr": retr.tolist(),
        "top1": top1.tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    tot = sum(counts.values())
    print(
        f"[{args.model}] roles: R={counts['R']}/{tot} H={counts['H']}/{tot} "
        f"D={counts['D']}/{tot} -> {args.out}"
    )


if __name__ == "__main__":
    main()
