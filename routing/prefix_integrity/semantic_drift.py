# SPDX-License-Identifier: GPL-2.0
"""MVP 3: semantic drift of a candidate against the full-cartridge baseline.

The metric/invariant path grades the cache contract without a model. This module
adds the orthogonal axis: does the algorithm change what the model would say?
It loads a real model and the cartridge KV, then for each query compares the
next-token distribution under the full cartridge against the distribution under
the candidate -- a block selector applied as an attention mask over prefix
positions, or a KV codec applied to the cached tensors.

Block selection reuses the same adapters as the CPU harness, so the GPU drift
and the CPU manifest agree on what each algorithm keeps. The query's position
ids are pinned to the original prefix offset for both baseline and candidate, so
only the attended key set (or the codec transform) differs -- RoPE phase of the
query is held fixed and the drift is attributable to the algorithm, not to
re-indexing.

This runs on a GPU pod (transformers + torch). The CPU harness folds its output
in via --semantic-json. Designed to be scp'd to a pod standalone or imported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def block_keep_mask(num_blocks, block_size, selected_blocks, prefix_len):
    """Token-level keep mask over the prefix (1 keep, 0 drop), length prefix_len."""
    import torch

    keep = torch.zeros(prefix_len, dtype=torch.long)
    sel = set(int(b) for b in selected_blocks)
    for b in sel:
        lo = b * block_size
        hi = min(prefix_len, (b + 1) * block_size)
        if lo < prefix_len:
            keep[lo:hi] = 1
    return keep


def _build_cache(cartridge_kv, device, dtype):
    """Build a DynamicCache from list[(k, v)] per layer."""
    import torch
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(cartridge_kv):
        cache.update(
            k.to(device=device, dtype=dtype),
            v.to(device=device, dtype=dtype),
            layer_idx,
        )
    return cache


# Each codec is a (key-transform, value-transform) pair. Symmetric codecs apply
# the same precision to K and V; asymmetric ones apply different precisions,
# which is the whole point of the decode-paper asymmetry (protect the fragile
# keys at 16-bit, compress the tolerant values).
#
# Scale granularity matters as much as bit-width. A single per-tensor scale on
# post-RoPE keys manufactures key fragility -- the known pathology is per-channel
# outliers in the RoPE'd key space, so a coarse global scale clips them. To make
# a k16v8-vs-k8v16 key-vs-value isolation FAIR, keys are quantized per-channel
# (a scale per head-dim channel, KIVI-style) and values per-token (a scale per
# position). The per-tensor codecs are kept only as labeled "naive stress" rows,
# and the k8v16 pair (fair vs naive) shows whether fair scaling rescues the keys.
# Value tensors carry a leading (batch,) head, seq, dim layout, so the last two
# dims are (seq, head_dim): per-channel = amax over seq (dim -2), per-token =
# amax over head_dim (dim -1).
def _identity(t):
    return t.clone()


def _fp8(t):
    import torch

    return t.to(torch.float8_e4m3fn).to(t.dtype)


def _intN_perchannel(t, bits):
    """Symmetric int-N with one scale per channel (shared across positions)."""
    import torch

    qmax = (1 << (bits - 1)) - 1
    f = t.float()
    scale = f.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8) / qmax
    q = (f / scale).round().clamp(-qmax, qmax)
    return (q * scale).to(t.dtype)


def _intN_pertoken(t, bits):
    """Symmetric int-N with one scale per token (shared across channels)."""
    import torch

    qmax = (1 << (bits - 1)) - 1
    f = t.float()
    scale = f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (f / scale).round().clamp(-qmax, qmax)
    return (q * scale).to(t.dtype)


def _intN_roundtrip(t, bits):
    """Symmetric int-N with a single per-tensor scale (naive; stress only)."""
    import torch

    qmax = (1 << (bits - 1)) - 1
    f = t.float()
    scale = f.abs().amax().clamp(min=1e-8) / qmax
    q = (f / scale).round().clamp(-qmax, qmax)
    return (q * scale).to(t.dtype)


_CODECS = {
    "none": (_identity, _identity),
    # naive per-tensor stress rows (single global scale)
    "int8": (lambda t: _intN_roundtrip(t, 8), lambda t: _intN_roundtrip(t, 8)),
    "fp8": (_fp8, _fp8),
    "k8v16_pt": (lambda t: _intN_roundtrip(t, 8), _identity),
    # fair-granularity asymmetric: keys per-channel, values per-token
    "k16v8": (_identity, lambda t: _intN_pertoken(t, 8)),
    "k16v4": (_identity, lambda t: _intN_pertoken(t, 4)),
    "k8v16": (lambda t: _intN_perchannel(t, 8), _identity),
    "k8v8": (lambda t: _intN_perchannel(t, 8), lambda t: _intN_pertoken(t, 8)),
}


def _codec_transform(cartridge_kv, codec):
    """Return a transformed copy of the cartridge KV for codec-mode drift.

    none      : identity (sanity check -> ~0 drift; also the bf16 reload control)
    int8/fp8  : per-tensor symmetric round-trip on both K and V (naive stress)
    k8v16_pt  : keys int8 per-TENSOR, values bf16 (naive-K stress; vs fair k8v16)
    k16v8     : keys bf16, values int8 per-token (fair; drift is V-only)
    k16v4     : keys bf16, values int4 per-token (fair; pushes V)
    k8v16     : keys int8 per-channel, values bf16 (fair; drift is K-only)
    k8v8      : keys int8 per-channel, values int8 per-token (fair symmetric)
    """
    if codec not in _CODECS:
        raise ValueError(f"unknown codec {codec}")
    kfn, vfn = _CODECS[codec]
    return [(kfn(k), vfn(v)) for k, v in cartridge_kv]


def _next_token_probs(model, input_ids, cache, prefix_keep, device):
    """Forward one query over the cache; return softmax of the last-step logits.

    prefix_keep is a 0/1 mask over the prefix positions (None = keep all). The
    query's position ids are pinned after the full prefix so RoPE is fixed.
    """
    import torch

    prefix_len = cache.get_seq_length()
    q_len = input_ids.shape[1]
    if prefix_keep is None:
        prefix_mask = torch.ones(prefix_len, dtype=torch.long)
    else:
        prefix_mask = prefix_keep
    attn = (
        torch.cat([prefix_mask, torch.ones(q_len, dtype=torch.long)])
        .unsqueeze(0)
        .to(device)
    )
    pos = torch.arange(prefix_len, prefix_len + q_len, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(device),
            past_key_values=cache,
            attention_mask=attn,
            position_ids=pos,
            use_cache=True,
        )
    logits = out.logits[0, -1, :].float()
    return torch.softmax(logits, dim=-1)


def _kl(p, q, eps=1e-8):
    import torch

    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * (p / q).log()).sum())


def run_drift(
    model,
    tokenizer,
    cartridge_kv,
    queries,
    adapter,
    budget_k,
    block_size,
    device,
    dtype,
    mode="selector",
    codec="none",
    topk=5,
):
    """Compare candidate vs full-cartridge next-token distribution per query.

    Returns {kl_mean, kl_max, top1_agreement, topk_agreement, n, per_query}.
    """
    import torch

    prefix_len = cartridge_kv[0][0].shape[2]
    num_blocks = math.ceil(prefix_len / block_size)

    # Candidate cartridge tensors (codec mode transforms them once).
    cand_kv = _codec_transform(cartridge_kv, codec) if mode == "codec" else cartridge_kv

    kls, top1, topk_agree, per_query = [], 0, [], []
    for qi, q in enumerate(queries):
        text = q.get("query") or q.get("question") or q.get("text") or q["id"]
        ids = tokenizer(text, return_tensors="pt").input_ids

        # Baseline: full cartridge, no masking, original tensors.
        base_cache = _build_cache(cartridge_kv, device, dtype)
        p_base = _next_token_probs(model, ids, base_cache, None, device)

        if mode == "selector":
            sel = adapter.select_blocks(q, num_blocks, budget_k)
            keep = block_keep_mask(num_blocks, block_size, sel, prefix_len)
            cand_cache = _build_cache(cartridge_kv, device, dtype)
            p_cand = _next_token_probs(model, ids, cand_cache, keep, device)
        else:  # codec
            cand_cache = _build_cache(cand_kv, device, dtype)
            p_cand = _next_token_probs(model, ids, cand_cache, None, device)

        kl = _kl(p_base, p_cand)
        a1 = int(p_base.argmax().item() == p_cand.argmax().item())
        tb = set(p_base.topk(topk).indices.tolist())
        tc = set(p_cand.topk(topk).indices.tolist())
        ov = len(tb & tc) / topk

        kls.append(kl)
        top1 += a1
        topk_agree.append(ov)
        per_query.append(
            {"id": q.get("id", f"q{qi}"), "kl": kl, "top1": a1, "topk": ov}
        )

    n = max(1, len(queries))
    return {
        "kl_mean": sum(kls) / n,
        "kl_max": max(kls) if kls else 0.0,
        "top1_agreement": top1 / n,
        "topk_agreement": sum(topk_agree) / n,
        "n": len(queries),
        "mode": mode,
        "codec": codec,
        "per_query": per_query,
    }


def _load_queries(path, limit):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"query": line}
            obj.setdefault("id", f"q{i}")
            out.append(obj)
            if limit and len(out) >= limit:
                break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="PIA semantic drift (GPU)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cartridge", required=True, help="cartridge dir or .pt")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--algorithm", default="query_aware")
    ap.add_argument("--algorithm-config", default=None)
    ap.add_argument("--mode", default="selector", choices=["selector", "codec"])
    ap.add_argument(
        "--codec",
        default="none",
        choices=sorted(_CODECS),
        help="KV round-trip codec; k*/v* names are asymmetric (see _CODECS)",
    )
    ap.add_argument("--budget-k", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--pins", default="A1R2")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--out", required=True, help="semantic.json output path")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
    from routing.prefix_integrity.adapters import load_adapter, parse_pins
    from routing.prefix_integrity.cartridge_view import load_tensors

    pin = parse_pins(args.pins) or (1, 2, 0)
    config = {}
    if args.algorithm_config:
        with open(args.algorithm_config) as f:
            config = json.load(f)
    config.setdefault("anchor", pin[0])
    config.setdefault("recent", pin[1])
    adapter = load_adapter(args.algorithm, config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"loading {args.model} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()

    cartridge_kv = load_tensors(args.cartridge)
    queries = _load_queries(args.queries, args.limit)
    print(
        f"cartridge: {len(cartridge_kv)} layers, prefix {cartridge_kv[0][0].shape[2]} "
        f"tokens; {len(queries)} queries; mode={args.mode} codec={args.codec}",
        flush=True,
    )

    res = run_drift(
        model,
        tok,
        cartridge_kv,
        queries,
        adapter,
        args.budget_k,
        args.block_size,
        device,
        dtype,
        mode=args.mode,
        codec=args.codec,
    )
    # Repairable is a placeholder until xa25 overlay is wired; default False.
    payload = {
        "kl": res["kl_mean"],
        "top1": res["top1_agreement"],
        "repairable": False,
        "detail": res,
        "algorithm": args.algorithm,
        "model": args.model,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"drift: kl_mean={res['kl_mean']:.4f} kl_max={res['kl_max']:.4f} "
        f"top1={res['top1_agreement']:.3f} topk={res['topk_agreement']:.3f} "
        f"-> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
