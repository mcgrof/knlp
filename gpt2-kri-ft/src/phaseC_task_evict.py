#!/usr/bin/env python3
"""Phase-C Tier-1: downstream-task accuracy under per-head KV eviction.

Generates real answers (greedy) under each eviction router and scores task
accuracy vs retained-KV fraction -- the honest memory axis. Faithfulness of the
per-head masked decode to physical eviction is established in
phaseC_faithcheck.py (bit-identical in fp32). All routers run through the SAME
masked-decode path so they share numerics; only the kept-block set differs.

Tasks:
  niah     -- RULER-class needle retrieval (single/multi via --needles), score =
              the queried value appears in the generated continuation.
  gsm8k    -- 8-shot CoT reasoning, score = exact final-integer match. Short
              context: a regression panel, NOT a headline (report retained frac
              so "no harm" vs "nothing evicted" is never conflated).

Routers per (layer, kv-head) at budget K blocks (+ sink + recent always):
  full, headrole (needs --roles_json), h2o_true (exact mass), relq_recent
  (recent-Q relevance), residual_rel (legacy unrestricted OMP), sink_recent,
  random.

Usage:
  python3 phaseC_task_evict.py --task niah --model HuggingFaceTB/SmolLM2-1.7B \
     --length 4096 --eval 200 --budgets 8,16,32,64 \
     --roles_json roles/SmolLM2-1.7B.json --out out/niah_smol17.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, load_filler_sentences
from niah_evict_perhead import answer_block_indices, get_keys
from diag_perhead_oracle import block_stats, capture_q, get_values, mass_trueq
from portfolio import compose, qualified_pool
from phaseC_faithcheck import install_patch, attn_modules, masked_decode, _EVICT


def batch_cache(past, B):
    """Replicate a prefill DynamicCache across a batch dim of size B."""
    from transformers import DynamicCache

    new = DynamicCache()
    for i, layer in enumerate(past.layers):
        new.update(
            layer.keys.expand(B, -1, -1, -1).contiguous(),
            layer.values.expand(B, -1, -1, -1).contiguous(),
            i,
        )
    return new


@torch.no_grad()
def batched_masked_decode(
    model, last_id, past, keep_per_layer_batched, device, T, max_new
):
    """Decode B combos at once. keep_per_layer_batched: list over layers of
    [B,Hkv,T] bool. Returns list of B token-id lists."""
    B = keep_per_layer_batched[0].shape[0]
    mods = {m.layer_idx: m for m in attn_modules(model)}
    for li, m in mods.items():
        m._evict_keep = keep_per_layer_batched[li].to(device)
    _EVICT["on"] = True
    ids = last_id.view(1, 1).expand(B, 1).contiguous().to(device)
    cur = past
    gens = [[] for _ in range(B)]
    for s in range(max_new):
        pos = torch.full((B, 1), T + s, device=device, dtype=torch.long)
        cpos = torch.tensor([T + s], device=device)
        out = model(
            ids,
            past_key_values=cur,
            position_ids=pos,
            cache_position=cpos,
            use_cache=True,
        )
        nxt = out.logits[:, -1].argmax(dim=-1)  # [B]
        for b in range(B):
            gens[b].append(int(nxt[b]))
        cur = out.past_key_values
        ids = nxt.view(B, 1)
    _EVICT["on"] = False
    for m in mods.values():
        m._evict_keep = None
    return gens


ROUTERS = (
    "full",
    "headrole",
    "h2o_true",
    "relq_recent",
    "residual_rel",
    "sink_recent",
    "random",
)


def resid_sequence(cent, probe, pool_mask, steps):
    Hkv, NB, D = cent.shape
    q_resid = probe.clone()
    chosen = torch.zeros(Hkv, NB, dtype=torch.bool, device=cent.device)
    seq = []
    for _ in range(min(steps, NB)):
        sc = torch.einsum("hd,hnd->hn", F.normalize(q_resid, dim=-1), cent)
        sc = sc.masked_fill(~pool_mask | chosen, float("-inf"))
        pick = sc.argmax(dim=-1)
        seq.append(pick)
        chosen.scatter_(1, pick.unsqueeze(1), True)
        kc = torch.gather(cent, 1, pick.view(Hkv, 1, 1).expand(Hkv, 1, D)).squeeze(1)
        q_resid = q_resid - (q_resid * kc).sum(-1, keepdim=True) * kc
    return torch.stack(seq, dim=-1)


@torch.no_grad()
def per_head_orders(model, ids_t, device, bs, recent_window, mass_probes):
    """Prefill once; return (past, T, NB, Hkv, lists) where lists has the
    per-(layer,kv-head) orderings each router needs."""
    T = ids_t.shape[0]
    NB = (T + bs - 1) // bs
    recent = list(range(max(0, T - recent_window), T))
    probes = sorted(set(int(x) for x in torch.linspace(0, T - 2, mass_probes).tolist()))
    sel_pos = sorted(set(recent + probes + [T - 1]))
    pos_of = {p: i for i, p in enumerate(sel_pos)}
    probe_sel = [(pos_of[p], p) for p in probes]
    qs, past = capture_q(model, ids_t, sel_pos, device)
    keys, vals = get_keys(past), get_values(past)
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    group = Hq // Hkv
    ridx = torch.tensor([pos_of[p] for p in recent], device=device)
    lists = []  # per layer: dict of orderings
    Kmax = 64
    for li in range(len(keys)):
        kl, vl = keys[li][0], vals[li][0]
        cent, vn, idx, khn = block_stats(kl, vl, bs)
        ql = qs[li]
        p_lastk = F.normalize(khn[:, -1, :], dim=-1)
        q_rec = ql[:, ridx, :].view(Hkv, group, len(recent), -1)
        p_rec = F.normalize(F.normalize(q_rec, dim=-1).mean(dim=(1, 2)), dim=-1)
        r1o = torch.einsum("hd,hnd->hn", p_rec, cent).argsort(dim=-1, descending=True)
        r2o = torch.einsum("hd,hnd->hn", p_lastk, cent).argsort(dim=-1, descending=True)
        m_true = mass_trueq(ql, probe_sel, khn, kl, idx, NB, group)
        ho = m_true.argsort(dim=-1, descending=True)
        pool_mask = torch.zeros(Hkv, NB, dtype=torch.bool, device=device)
        pool_mask.scatter_(1, r1o[:, : 2 * Kmax], True)
        pool_mask.scatter_(1, r2o[:, : 2 * Kmax], True)
        d_seq = resid_sequence(cent, p_lastk, pool_mask, Kmax)
        allmask = torch.ones(Hkv, NB, dtype=torch.bool, device=device)
        d_un = resid_sequence(cent, p_lastk, allmask, Kmax)
        lists.append(
            dict(
                r1=r1o.tolist(),
                r2=r2o.tolist(),
                h=ho.tolist(),
                d=d_seq.tolist(),
                d_un=d_un.tolist(),
            )
        )
    return past, T, NB, Hkv, lists


def keep_mask_for(
    router, K, lists, roles, NB, Hkv, nL, bs, T, sink, recent_b, rng, device
):
    """Build keep_per_layer: list over layers of [Hkv,T] bool (token keep)."""
    sink_blocks = set(range(sink))
    recent_blocks = set(range(max(0, NB - recent_b), NB))
    out = []
    for li in range(nL):
        L = lists[li]
        keep_blocks = [None] * Hkv
        for hh in range(Hkv):
            if router == "full":
                sel = list(range(NB))
            elif router == "sink_recent":
                sel = []
            elif router == "relq_recent":
                sel = L["r1"][hh][:K]
            elif router == "residual_rel":
                sel = L["d_un"][hh][:K]
            elif router == "h2o_true":
                sel = L["h"][hh][:K]
            elif router == "random":
                sel = rng.sample(range(NB), min(K, NB))
            elif router == "headrole":
                role = roles[li][hh]
                if role == "R":
                    sel = compose(
                        L["r1"][hh], L["r2"][hh], L["h"][hh], L["d"][hh], K, K, 0
                    )
                elif role == "H":
                    hn = max(1, -(-K // 4))
                    sel = compose(
                        L["r1"][hh],
                        L["r2"][hh],
                        L["h"][hh],
                        L["d"][hh],
                        K,
                        K - hn,
                        hn,
                        pool=qualified_pool(L["r1"][hh], L["r2"][hh], K),
                    )
                else:
                    sel = L["d_un"][hh][:K]
            else:
                raise ValueError(router)
            kb = set(sel) | sink_blocks | recent_blocks
            keep_blocks[hh] = kb
        km = torch.zeros(Hkv, T, dtype=torch.bool)
        for hh in range(Hkv):
            for b_ in keep_blocks[hh]:
                km[hh, b_ * bs : min(T, (b_ + 1) * bs)] = True
        out.append(km.to(device))
    return out


def retained_frac(keep_per_layer, T):
    tot = sum(km.sum().item() for km in keep_per_layer)
    cells = sum(km.numel() for km in keep_per_layer)
    return tot / cells


# ---------------- GSM8K ----------------
def gsm_extract(s):
    m = re.findall(r"-?\d[\d,]*", s.replace(",", ""))
    return m[-1] if m else None


def build_gsm8k(tok, n_shot, seed):
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    shots = rng.sample(range(len(train)), n_shot)
    pre = ""
    for i in shots:
        ex = train[i]
        pre += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    return pre, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("niah", "gsm8k"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-blocks", type=int, default=8)
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--mass-probes", type=int, default=48)
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--n-shot", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roles_json", default=None)
    ap.add_argument("--routers", default=",".join(ROUTERS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    install_patch(model)
    nL = model.config.num_hidden_layers
    bs = args.block_size
    budgets = [int(x) for x in args.budgets.split(",")]
    routers = args.routers.split(",")
    roles = json.load(open(args.roles_json))["roles"] if args.roles_json else None
    rng = random.Random(args.seed)

    if args.task == "gsm8k":
        pre, test = build_gsm8k(tok, args.n_shot, args.seed)
        max_new = max(args.max_new, 256)
    else:
        sents = load_filler_sentences(args.seed)
        max_new = args.max_new

    # acc[router][K] = [hits, n]; retained[router][K] = [sum_frac, n]
    acc = {r: {K: [0, 0] for K in budgets} for r in routers}
    acc_full = [0, 0]
    ret = {r: {K: [0.0, 0] for K in budgets} for r in routers}
    n_used = 0
    idxs = list(range(len(test))) if args.task == "gsm8k" else list(range(args.eval))
    rng.shuffle(idxs)

    ei = 0
    for sample_i in idxs:
        if n_used >= args.eval:
            break
        ans_blocks = None
        if args.task == "niah":
            text, spans, needles, (qk, qv) = build_context(
                tok, args.length, args.needles, sents, rng
            )
            ids, ans_blocks = answer_block_indices(tok, text, qv, spans, bs, 10**9)
            gold = qv
        else:
            ex = test[sample_i]
            prompt = pre + f"Question: {ex['question']}\nAnswer:"
            ids = tok(prompt, add_special_tokens=True)["input_ids"]
            gold = gsm_extract(ex["answer"])
        # both tasks: prefill all-but-last, decode from the last token
        last_id = torch.tensor(ids[-1])
        ids_t = torch.tensor(ids[:-1])
        T = ids_t.shape[0]
        NB = (T + bs - 1) // bs
        if T < bs * (args.sink_blocks + args.recent_blocks + 4):
            continue

        past0, T, NB, Hkv, lists = per_head_orders(
            model, ids_t, device, bs, args.recent_window, args.mass_probes
        )
        if args.task == "niah" and (not ans_blocks or max(ans_blocks) >= NB):
            del past0
            torch.cuda.empty_cache()
            continue
        n_used += 1
        ei += 1

        # build all (router,K) combos; full/sink_recent are K-independent (one each)
        combos = []
        for router in routers:
            ks = [budgets[0]] if router in ("full", "sink_recent") else budgets
            for K in ks:
                combos.append((router, K))
        keeps = [
            keep_mask_for(
                router,
                K,
                lists,
                roles,
                NB,
                Hkv,
                nL,
                bs,
                T,
                args.sink_blocks,
                args.recent_blocks,
                rng,
                device,
            )
            for (router, K) in combos
        ]
        keep_batched = [
            torch.stack([keeps[c][li] for c in range(len(combos))], 0)
            for li in range(nL)
        ]
        gens = batched_masked_decode(
            model,
            last_id,
            batch_cache(past0, len(combos)),
            keep_batched,
            device,
            T,
            max_new,
        )
        for ci, (router, K) in enumerate(combos):
            txt = tok.decode(gens[ci])
            if args.task == "niah":
                ok = gold in txt
            else:
                pred = gsm_extract(txt.split("Question:")[0])
                ok = pred is not None and gold is not None and pred == gold
            rf = retained_frac(keeps[ci], T)
            ks = budgets if router not in ("full", "sink_recent") else budgets
            for Kw in (budgets if router in ("full", "sink_recent") else [K]):
                acc[router][Kw][0] += int(ok)
                acc[router][Kw][1] += 1
                ret[router][Kw][0] += rf
                ret[router][Kw][1] += 1
        del past0
        torch.cuda.empty_cache()
        if ei % 10 == 0 or ei < 3:
            hr = acc.get("headrole", {}).get(
                budgets[1] if len(budgets) > 1 else budgets[0]
            )
            fu = acc.get("full", {}).get(budgets[0])
            print(
                (
                    f"  [{n_used}/{args.eval}] full={fu[0]}/{fu[1]} "
                    f"headrole@K={budgets[1] if len(budgets)>1 else budgets[0]}="
                    f"{hr[0]}/{hr[1]}"
                    if hr and fu
                    else f"  [{n_used}/{args.eval}]"
                ),
                flush=True,
            )

    res = {
        "task": args.task,
        "model": args.model,
        "length": args.length if args.task == "niah" else None,
        "eval_used": n_used,
        "seed": args.seed,
        "budgets": budgets,
        "block_size": bs,
        "sink_blocks": args.sink_blocks,
        "recent_blocks": args.recent_blocks,
        "accuracy": {},
        "retained_frac": {},
    }
    print(f"\n[{args.model}] {args.task}  (n={n_used})")
    print(f"{'router':14s} " + " ".join(f"K{K:>3d}" for K in budgets))
    for r in routers:
        accs, rets = {}, {}
        cells = []
        for K in budgets:
            h, nn = acc[r][K]
            a = h / nn if nn else None
            rf = ret[r][K][0] / ret[r][K][1] if ret[r][K][1] else None
            accs[f"K{K}"] = a
            rets[f"K{K}"] = rf
            cells.append(f"{a:.2f}" if a is not None else "  - ")
        res["accuracy"][r] = accs
        res["retained_frac"][r] = rets
        print(f"{r:14s} " + " ".join(cells))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
