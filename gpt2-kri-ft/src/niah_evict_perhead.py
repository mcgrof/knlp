#!/usr/bin/env python3
"""Lattice gap #3, PER-HEAD: KV-eviction routers on a real NIAH task, scored by
whether each (layer, kv-head) keeps the answer-bearing block.

The global harness (niah_evict_eval.py) collapses keys over layers AND kv-heads
into one shared keep-mask. That handicaps residual_rel: its anti-redundancy
orthogonal-matching-pursuit selection earns its keep PER HEAD, where each head's
query geometry differs. Real deployed KV-eviction (H2O et al.) is per-head too.

This evaluates the routers per (layer, kv-head): for each head we build the
block-key centroids and the last-position query probe for THAT head, run the
router at budget K, and ask whether the answer block (the KV block holding the
queried needle's value tokens) survives in that head's kept set (router blocks +
sink + recent). The headline metric is answer-block retention averaged over
(layer, kv-head, sample): the fraction of heads that can still see the answer.
If no head keeps the answer block the value is unrecoverable, so this retention
is the per-head quantity that gates task success and is exactly what the global
aggregation washed out.

We also report the model's actual greedy-decode accuracy under the matched
GLOBAL eviction (the existing tractable slice_cache path) as the task anchor.

Usage:
  python3 niah_evict_perhead.py --model HuggingFaceTB/SmolLM2-1.7B \
      --length 4096 --needles 4 --eval 20 --out OUT/perhead.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, token_content_mask, load_filler_sentences


def get_keys(cache):
    if hasattr(cache, "key_cache"):
        return cache.key_cache
    return [layer.keys for layer in cache.layers]


def answer_block_indices(tok, text, qv, content_spans, bs, max_tokens):
    """Token positions (block indices) of the queried value `qv` in the context.

    Locates the LAST occurrence of qv's character span (the queried needle's
    value), maps it through the offset mapping to token positions, and returns
    the set of block indices those tokens fall in. The query is appended after
    the context, so the value's own block is what eviction must preserve.
    """
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"][:max_tokens]
    offs = enc["offset_mapping"][:max_tokens]
    # char span of qv: last occurrence inside the body (not the question tail)
    cidx = text.rfind(" is " + qv + ".")
    if cidx < 0:
        cidx = text.rfind(qv)
        vs, ve = cidx, cidx + len(qv)
    else:
        vs = cidx + len(" is ")
        ve = vs + len(qv)
    blocks = set()
    for t, (a, b) in enumerate(offs):
        if a < ve and b > vs:
            blocks.add(t // bs)
    return ids, blocks


# ---- per-head routers (operate on one head's [NB,D] centroids + [D] probe) ----


def residual_rel_select(cent, q, k_budget):
    """Greedy orthogonal matching pursuit (the lattice residual_rel router)."""
    NB = cent.shape[0]
    chosen = torch.zeros(NB, dtype=torch.bool, device=cent.device)
    q_resid = q.clone()
    picks = []
    for _ in range(min(k_budget, NB)):
        score = (cent * F.normalize(q_resid, dim=0)).sum(-1)
        score = score.masked_fill(chosen, float("-inf"))
        p = int(score.argmax())
        chosen[p] = True
        picks.append(p)
        kc = cent[p]
        q_resid = q_resid - (q_resid @ kc) * kc
    return picks


def rel_only_select(cent, q, k_budget):
    score = (cent * q).sum(-1)
    return torch.topk(score, min(k_budget, cent.shape[0])).indices.tolist()


def h2o_select_head(kh, idx, NB, k_budget, n_probe=48):
    """Per-head heavy-hitter: accumulate attention mass over sampled queries.
    kh is one head's per-token key [T,D] (normalised)."""
    T = kh.shape[0]
    probe_pos = torch.linspace(0, T - 1, n_probe).long()
    mass = torch.zeros(NB, device=kh.device)
    for p in probe_pos:
        scores = kh[: p + 1] @ kh[p]
        a = F.softmax(scores, dim=-1)
        mass.index_add_(0, idx[: p + 1], a)
    return torch.topk(mass, min(k_budget, NB)).indices.tolist()


def head_centroids(kh, bs):
    """One head's per-token key [T,D] -> (block centroids [NB,D] normalised,
    last-position probe [D] normalised, token idx->block)."""
    T, D = kh.shape
    NB = (T + bs - 1) // bs
    khn = F.normalize(kh.float(), dim=-1)
    cent = torch.zeros(NB, D, device=kh.device)
    cnt = torch.zeros(NB, device=kh.device)
    idx = torch.arange(T, device=kh.device) // bs
    cent.index_add_(0, idx, khn)
    cnt.index_add_(0, idx, torch.ones(T, device=kh.device))
    cent = cent / cnt.clamp(min=1).unsqueeze(1)
    return F.normalize(cent, dim=-1), F.normalize(khn[-1], dim=0), idx


def keeps_answer(blocks, NB, sink_blocks, recent_blocks, ans_blocks):
    keep = set(blocks)
    keep.update(range(sink_blocks))
    keep.update(range(max(0, NB - recent_blocks), NB))
    return any(b in keep for b in ans_blocks)


# ---- global decode anchor (reuses the tractable slice_cache path) ----


def slice_cache(past, kept_idx):
    from transformers import DynamicCache

    new = DynamicCache()
    for i, layer in enumerate(past.layers):
        new.update(
            layer.keys[:, :, kept_idx, :].clone(),
            layer.values[:, :, kept_idx, :].clone(),
            i,
        )
    return new


@torch.no_grad()
def decode_evicted(model, last_id, past, keep_tok, device, T, max_new=20):
    kept_idx = keep_tok.nonzero(as_tuple=True)[0].to(device)
    cur = slice_cache(past, kept_idx)
    cache_len = int(kept_idx.numel())
    ids = last_id.view(1, 1).to(device)
    gen = []
    for s in range(max_new):
        pos = torch.tensor([[T + s]], device=device)
        cpos = torch.tensor([cache_len + s], device=device)
        out = model(
            ids,
            past_key_values=cur,
            position_ids=pos,
            cache_position=cpos,
            use_cache=True,
        )
        nxt = out.logits[0, -1].argmax()
        gen.append(int(nxt))
        cur = out.past_key_values
        ids = nxt.view(1, 1)
    del cur, out
    torch.cuda.empty_cache()
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-blocks", type=int, default=8)
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode-anchor", action="store_true", help="also run the "
                    "global slice_cache decode accuracy anchor (slower)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    # Inference-only eval: disable autograd globally so the prefill forward does
    # NOT retain a backward graph (that graph OOMs at >=1.7B on the 48GB W7900;
    # forward VALUES are identical, so this only frees memory). decode_evicted is
    # already no_grad; this covers the prefill + per-head selection too.
    torch.set_grad_enabled(False)
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)
    bs = args.block_size
    budgets = [int(x) for x in args.budgets.split(",")]
    routers = ["residual_rel", "rel_only", "h2o", "sink_recent", "random"]

    # per-head answer-block retention: hits[router][K] = (kept_heads, total_heads)
    ret = {r: {b: [0, 0] for b in budgets} for r in routers}
    # global decode anchor: acc[router][K] = (hits, n); plus full
    dec = {r: {b: [0, 0] for b in budgets} for r in routers}
    dec["full"] = {0: [0, 0]}

    for ei in range(args.eval):
        text, spans, needles, qa = build_context(
            tok, args.length, args.needles, sents, rng
        )
        qk, qv = qa
        ids, ans_blocks = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)
        past = model(
            ids_t[:-1].unsqueeze(0).to(device), use_cache=True
        ).past_key_values
        keys = get_keys(past)  # list per layer [1, Hkv, T, D]
        T = keys[0].shape[2]
        NB = (T + bs - 1) // bs
        if not ans_blocks or max(ans_blocks) >= NB:
            del past
            torch.cuda.empty_cache()
            continue

        # ----- per-head retention -----
        for li, kl in enumerate(keys):
            Hkv = kl.shape[1]
            for h in range(Hkv):
                kh = kl[0, h]  # [T, D]
                cent, q, idx = head_centroids(kh, bs)
                khn = F.normalize(kh.float(), dim=-1)
                for b in budgets:
                    for r in routers:
                        if r == "residual_rel":
                            blocks = residual_rel_select(cent, q, b)
                        elif r == "rel_only":
                            blocks = rel_only_select(cent, q, b)
                        elif r == "h2o":
                            blocks = h2o_select_head(khn, idx, NB, b)
                        elif r == "sink_recent":
                            blocks = []
                        elif r == "random":
                            blocks = rng.sample(range(NB), min(b, NB))
                        ok = keeps_answer(
                            blocks, NB, args.sink_blocks, args.recent_blocks,
                            ans_blocks,
                        )
                        ret[r][b][0] += int(ok)
                        ret[r][b][1] += 1

        # ----- optional global decode anchor -----
        if args.decode_anchor:
            # global per-token key (mean over layers/heads) for selection
            kt = torch.stack(
                [k[0].float().mean(0) for k in keys], 0
            ).mean(0)
            kt = F.normalize(kt, dim=-1)
            gcent = torch.zeros(NB, kt.shape[1], device=kt.device)
            gcnt = torch.zeros(NB, device=kt.device)
            gidx = torch.arange(T, device=kt.device) // bs
            gcent.index_add_(0, gidx, kt)
            gcnt.index_add_(0, gidx, torch.ones(T, device=kt.device))
            gcent = F.normalize(gcent / gcnt.clamp(min=1).unsqueeze(1), dim=-1)
            gq = F.normalize(kt[-1], dim=0)

            def keep_tok_from(blocks):
                kb = torch.zeros(NB, dtype=torch.bool)
                for b_ in blocks:
                    kb[b_] = True
                for b_ in range(args.sink_blocks):
                    kb[b_] = True
                for b_ in range(max(0, NB - args.recent_blocks), NB):
                    kb[b_] = True
                tokm = torch.zeros(T, dtype=torch.bool)
                for b_ in range(NB):
                    if kb[b_]:
                        tokm[b_ * bs : min(T, (b_ + 1) * bs)] = True
                return tokm

            gen = decode_evicted(
                model, ids_t[-1], past, torch.ones(T, dtype=torch.bool), device, T
            )
            dec["full"][0][0] += int(qv in tok.decode(gen))
            dec["full"][0][1] += 1
            for b in budgets:
                for r in routers:
                    if r == "residual_rel":
                        blocks = residual_rel_select(gcent, gq, b)
                    elif r == "rel_only":
                        blocks = rel_only_select(gcent, gq, b)
                    elif r == "h2o":
                        blocks = h2o_select_head(kt, gidx, NB, b)
                    elif r == "sink_recent":
                        blocks = []
                    elif r == "random":
                        blocks = rng.sample(range(NB), min(b, NB))
                    keep = keep_tok_from(blocks)
                    gen = decode_evicted(model, ids_t[-1], past, keep, device, T)
                    dec[r][b][0] += int(qv in tok.decode(gen))
                    dec[r][b][1] += 1

        del past
        torch.cuda.empty_cache()
        print(f"  sample {ei + 1}/{args.eval} done", flush=True)

    result = {
        "model": args.model,
        "length": args.length,
        "eval": args.eval,
        "block_size": bs,
        "per_head_answer_retention": {},
        "global_decode_acc": {},
    }
    print(f"\n[{args.model}] L={args.length}  PER-HEAD answer-block retention")
    print("(fraction of (layer,kv-head) pairs whose kept set sees the answer)")
    for r in routers:
        result["per_head_answer_retention"][r] = {}
        for b in budgets:
            hit, n = ret[r][b]
            frac = hit / max(1, n)
            result["per_head_answer_retention"][r][f"K{b}"] = frac
            print(f"  {r:13s} K={b:>3} retention={frac:.4f}")
    if args.decode_anchor:
        print("\nGLOBAL decode accuracy anchor")
        fb = dec["full"][0]
        print(f"  full           acc={fb[0]/max(1,fb[1]):.3f}")
        result["global_decode_acc"]["full"] = fb[0] / max(1, fb[1])
        for r in routers:
            result["global_decode_acc"][r] = {}
            for b in budgets:
                hit, n = dec[r][b]
                acc = hit / max(1, n)
                result["global_decode_acc"][r][f"K{b}"] = acc
                print(f"  {r:13s} K={b:>3} acc={acc:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
