#!/usr/bin/env python3
"""Phase-C Tier-0 faithfulness check (GATES the whole downstream-value sweep).

Codex's highest-value objection: a per-head attention MASK over a full KV cache
is only a valid proxy for real KV EVICTION if masking and physically deleting
the evicted K/V produce the same decode. We prove it here.

Mechanism: register a custom attention function `lattice_evict` in
ALL_ATTENTION_FUNCTIONS and point every layer at it via
config._attn_implementation. Prefill runs with eviction OFF (full attention =
real hidden states, exactly as a normal prefill). Then per-(layer, kv-head)
keep sets are stashed on each attention module and eviction is switched ON for
incremental decode: each new query attends to the FULL cache but with evicted
prefill key columns biased to -inf per kv-head; all generated positions stay
visible. That is real KV-eviction semantics (prefill computed once under full
attention; decode denied access to evicted entries) without a ragged cache.

Faithfulness test: drive a GLOBAL keep policy (same kept positions for every
head) through (A) the existing PHYSICAL slice decode (niah_evict_perhead.
decode_evicted, which really deletes KV columns and renumbers the cache) and
(B) this masked decode. For a global policy the two MUST produce identical
greedy tokens and near-identical logits. If they match, the per-head masked
decode (which a dense physical cache cannot express) is trustworthy and the
sweep proceeds. If not, the sweep stops.

Usage:
  python3 phaseC_faithcheck.py --model HuggingFaceTB/SmolLM2-360M \
      --length 2048 --eval 30 --max-new 16 --out out/faith_smol360.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, load_filler_sentences
from niah_evict_perhead import (
    answer_block_indices,
    decode_evicted,
    get_keys,
    head_centroids,
    rel_only_select,
)

NEG = -1e9
_EVICT = {"on": False}


def clone_cache(past):
    """Deep-copy a DynamicCache so independent decoders don't alias/mutate the
    one prefill cache (model(...) appends in place, which corrupts reuse)."""
    from transformers import DynamicCache

    new = DynamicCache()
    for i, layer in enumerate(past.layers):
        new.update(layer.keys.clone(), layer.values.clone(), i)
    return new


def repeat_kv_cols(keep_kv, groups):
    """[Hkv, T] bool -> [Hq, T] bool by repeating each kv-head `groups` times."""
    return keep_kv.repeat_interleave(groups, dim=0)


def lattice_evict_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    g = module.num_key_value_groups
    key_states = key.repeat_interleave(g, dim=1)
    value_states = value.repeat_interleave(g, dim=1)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    keep = getattr(module, "_evict_keep", None)  # [Hkv,T] or [B,Hkv,T] bool
    if _EVICT["on"] and keep is not None:
        B = attn_weights.shape[0]
        Hq = attn_weights.shape[1]
        K = key_states.shape[-2]
        z = attn_weights.new_zeros(())
        nz = attn_weights.new_full((), NEG)
        if keep.dim() == 2:  # [Hkv,T] -> broadcast over batch
            keepq = keep.repeat_interleave(g, dim=0)  # [Hq,Tp]
            Tp = keepq.shape[1]
            bias = attn_weights.new_zeros(Hq, K)
            bias[:, :Tp] = torch.where(keepq, z, nz)
            attn_weights = attn_weights + bias.view(1, Hq, 1, K)
        else:  # [B,Hkv,T] per-batch-per-head
            keepq = keep.repeat_interleave(g, dim=1)  # [B,Hq,Tp]
            Tp = keepq.shape[2]
            bias = attn_weights.new_zeros(B, Hq, K)
            bias[:, :, :Tp] = torch.where(keepq, z, nz)
            attn_weights = attn_weights + bias.view(B, Hq, 1, K)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def install_patch(model):
    """Patch the per-family module-global `eager_attention_forward` that each
    attention layer resolves at call time, and KEEP _attn_implementation="eager"
    so HF still builds the additive CAUSAL mask (a custom impl name makes HF skip
    causal-mask construction -> bidirectional attention -> garbage). Our wrapper
    receives that causal mask in `attention_mask` and adds the eviction bias on
    top."""
    import sys

    model.config._attn_implementation = "eager"
    mods = set()
    for m in model.modules():
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx"):
            m._evict_keep = None
            mods.add(type(m).__module__)
    for modname in mods:
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, "eager_attention_forward"):
            mod.eager_attention_forward = lattice_evict_attention


def attn_modules(model):
    return [
        m
        for m in model.modules()
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx")
    ]


@torch.no_grad()
def masked_decode(model, last_id, past, keep_tok_blocks_per_layer, device, T, max_new):
    """Decode with eviction ON. keep_tok_blocks_per_layer: list over layers of
    [Hkv, T] bool keep masks (token granularity). Generated positions always
    kept (bias only spans the first T prefill columns)."""
    mods = {m.layer_idx: m for m in attn_modules(model)}
    for li, m in mods.items():
        m._evict_keep = keep_tok_blocks_per_layer[li].to(device)
    _EVICT["on"] = True
    ids = last_id.view(1, 1).to(device)
    cur = past
    cache_len = T
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
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        cur = out.past_key_values
        ids = torch.tensor([[nxt]], device=device)
    _EVICT["on"] = False
    for m in mods.values():
        m._evict_keep = None
    return gen


@torch.no_grad()
def masked_decode_logits(model, last_id, past, keep_per_layer, device, T):
    """One-step masked logits (for logit-level comparison vs physical slice)."""
    mods = {m.layer_idx: m for m in attn_modules(model)}
    for li, m in mods.items():
        m._evict_keep = keep_per_layer[li].to(device)
    _EVICT["on"] = True
    out = model(
        last_id.view(1, 1).to(device),
        past_key_values=past,
        position_ids=torch.tensor([[T]], device=device),
        cache_position=torch.tensor([T], device=device),
        use_cache=True,
    )
    _EVICT["on"] = False
    for m in mods.values():
        m._evict_keep = None
    return out.logits[0, -1].float()


@torch.no_grad()
def sliced_decode_logits(model, last_id, past, keep_tok, device, T):
    from niah_evict_perhead import slice_cache

    kept_idx = keep_tok.nonzero(as_tuple=True)[0].to(device)
    cur = slice_cache(past, kept_idx)
    out = model(
        last_id.view(1, 1).to(device),
        past_key_values=cur,
        position_ids=torch.tensor([[T]], device=device),
        cache_position=torch.tensor([int(kept_idx.numel())], device=device),
        use_cache=True,
    )
    return out.logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=30)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-blocks", type=int, default=8)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
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

    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)
    bs = args.block_size
    nL = model.config.num_hidden_layers
    Hkv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    n_tok_match = n = 0
    max_logit_diffs = []
    tok_seq_match = 0
    for ei in range(args.eval):
        text, spans, needles, (qk, qv) = build_context(
            tok, args.length, args.needles, sents, rng
        )
        ids, ans_blocks = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)
        T = ids_t.shape[0] - 1
        past = model(ids_t[:-1].unsqueeze(0).to(device), use_cache=True).past_key_values
        keys = get_keys(past)
        NB = (T + bs - 1) // bs
        if not ans_blocks or max(ans_blocks) >= NB:
            del past
            torch.cuda.empty_cache()
            continue
        n += 1

        # GLOBAL keep policy: rel_only on the layer-mean key centroids (one block
        # set shared by all heads/layers) + sink + recent.
        kt = torch.stack([k[0].float().mean(0) for k in keys], 0).mean(0)  # [T,D]
        cent, q, idx = head_centroids(kt, bs)
        sel = rel_only_select(cent, q, args.budget)
        keep_blocks = (
            set(sel)
            | set(range(args.sink_blocks))
            | set(range(max(0, NB - args.recent_blocks), NB))
        )
        keep_tok = torch.zeros(T, dtype=torch.bool)
        for b_ in keep_blocks:
            keep_tok[b_ * bs : min(T, (b_ + 1) * bs)] = True

        # per-layer per-head keep (global: same for all heads) at TOKEN granularity
        keep_kv = keep_tok.view(1, T).expand(Hkv, T).contiguous()
        keep_per_layer = [keep_kv.clone() for _ in range(nL)]

        # logit-level: physical slice vs masked, one step
        last_id = ids_t[-1]
        lg_slice = sliced_decode_logits(
            model, last_id, clone_cache(past), keep_tok, device, T
        )
        lg_mask = masked_decode_logits(
            model, last_id, clone_cache(past), keep_per_layer, device, T
        )
        max_logit_diffs.append(float((lg_slice - lg_mask).abs().max()))

        # token-level: full greedy decode both ways (each on its own cache copy)
        gen_slice = decode_evicted(
            model, last_id, clone_cache(past), keep_tok, device, T, args.max_new
        )
        gen_mask = masked_decode(
            model, last_id, clone_cache(past), keep_per_layer, device, T, args.max_new
        )
        match = sum(int(a == b) for a, b in zip(gen_slice, gen_mask))
        n_tok_match += match
        if gen_slice == gen_mask:
            tok_seq_match += 1

        del past
        torch.cuda.empty_cache()
        print(
            f"  {ei + 1}/{args.eval} logit_maxdiff={max_logit_diffs[-1]:.4f} "
            f"tok_match={match}/{args.max_new} seq_eq={gen_slice == gen_mask}",
            flush=True,
        )

    res = {
        "model": args.model,
        "length": args.length,
        "eval_used": n,
        "budget": args.budget,
        "max_new": args.max_new,
        "mean_max_logit_diff": sum(max_logit_diffs) / max(1, len(max_logit_diffs)),
        "worst_max_logit_diff": max(max_logit_diffs) if max_logit_diffs else None,
        "token_match_frac": n_tok_match / max(1, n * args.max_new),
        "seq_exact_match_frac": tok_seq_match / max(1, n),
    }
    print(json.dumps(res, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")
    ok = res["token_match_frac"] > 0.98 and res["mean_max_logit_diff"] < 0.05
    print("FAITHFUL" if ok else "DIVERGENT — sweep must not proceed on the mask")


if __name__ == "__main__":
    main()
