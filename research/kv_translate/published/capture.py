#!/usr/bin/env python3
"""Collect the calibration traces both published methods are fitted on.

One model at a time. The two models never need to share a card during
calibration, because each is run over the same token sequences independently
and its keys and values written to disk; the fit then reads the files. That
halves the memory the calibration stage needs and means a lost instance costs
one model's pass rather than both.

Keys are stored as content: the cache holds them after rotary embedding, and
the rotation is removed here using each token's own position, so what lands on
disk is what the regression is defined over.

For the receiving model this also computes the attention sensitivities the
weighted method needs. Those come from the receiver's own attention at fixed
causal boundaries and are the only part of calibration that looks at queries.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from .torch_ops import apply_rope, head_dim_of, rope_theta_of, strip_rope


def cache_layers(cache):
    """(keys, values) per layer, across the cache layouts transformers has used."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(kv[0], kv[1]) for kv in cache]


def make_cache(pairs):
    """A cache object the model will accept, from per-layer (keys, values)."""
    from transformers import DynamicCache

    pairs = [(k, v) for k, v in pairs]
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(pairs))
    return DynamicCache(ddp_cache_data=pairs)


def attention_modules(model):
    return [layer.self_attn for layer in model.model.layers]


def geometry(model):
    cfg = model.config
    return {
        "layers": int(cfg.num_hidden_layers),
        "attention_heads": int(cfg.num_attention_heads),
        "kv_heads": int(cfg.num_key_value_heads),
        "head_dim": head_dim_of(cfg),
        "rope_theta": rope_theta_of(cfg),
    }


class QueryTap:
    """Queries as they enter attention, before rotation.

    Hooked on the query normalisation where the architecture has one, because
    that is the last thing applied before the rotary embedding; on models
    without it the projection output is the same quantity.
    """

    def __init__(self, model, geom):
        self.geom = geom
        self.q = {}
        self.handles = []
        for i, attn in enumerate(attention_modules(model)):
            mod = getattr(attn, "q_norm", None) or attn.q_proj
            self.handles.append(mod.register_forward_hook(self._hook(i)))

    def _hook(self, i):
        def fn(_m, _inp, out):
            b, t = out.shape[0], out.shape[1]
            self.q[i] = out.reshape(
                b, t, self.geom["attention_heads"], self.geom["head_dim"]
            ).detach()

        return fn

    def close(self):
        for h in self.handles:
            h.remove()


def sensitivities_for_layer(q, k_rot, v, boundaries, sample_pos, geom):
    """Attention sensitivities for one layer of one sequence.

    q is [T, heads, d] content queries, k_rot is [kv_heads, T, d] keys as
    cached (rotated), v is [kv_heads, T, d]. Returns key and value
    sensitivities of shape [kv_heads, n_samples], already normalised to unit
    mass per boundary and summed over boundaries with equal weight.

    Eligible positions at boundary c are the sampled positions strictly before
    c: the boundary token is the query, and the prefix it attends over is what
    a transferred cache would supply.
    """
    dev = q.device
    H, Hkv, d = geom["attention_heads"], geom["kv_heads"], geom["head_dim"]
    rep = H // Hkv
    T = k_rot.shape[1]
    bpos = torch.as_tensor(boundaries, device=dev)
    qb = q[bpos].to(torch.float32)  # [B, H, d]
    qb = apply_rope(qb.transpose(0, 1), bpos, geom["rope_theta"]).transpose(0, 1)
    kh = k_rot.to(torch.float32).repeat_interleave(rep, dim=0)  # [H, T, d]
    vh = v.to(torch.float32).repeat_interleave(rep, dim=0)
    s = torch.einsum("bhd,htd->bht", qb, kh) / math.sqrt(d)
    pos = torch.arange(T, device=dev)
    causal = pos[None, None, :] < bpos[:, None, None]
    s = s.masked_fill(~causal, float("-inf"))
    a = torch.softmax(s, dim=-1)  # [B, H, T]
    a2 = a * a
    o = torch.einsum("bht,htd->bhd", a, vh)  # [B, H, d]
    v2 = (vh * vh).sum(-1)  # [H, T]
    o2 = (o * o).sum(-1)  # [B, H]
    vo = torch.einsum("htd,bhd->bht", vh, o)
    dist2 = (v2[None] - 2.0 * vo + o2[:, :, None]).clamp_min(0.0)
    qn2 = (qb * qb).sum(-1)  # [B, H]
    rk = a2 * dist2 * qn2[:, :, None] / d
    rv = a2
    # sum the query heads that share each key-value head
    B = rk.shape[0]
    rk = rk.reshape(B, Hkv, rep, T).sum(2)
    rv = rv.reshape(B, Hkv, rep, T).sum(2)
    sp = torch.as_tensor(sample_pos, device=dev)
    rk, rv = rk[:, :, sp], rv[:, :, sp]  # [B, Hkv, S]
    eligible = (sp[None, :] < bpos[:, None])[:, None, :]  # [B, 1, S]
    out = []
    for r in (rk, rv):
        r = r * eligible
        mass = r.sum(-1, keepdim=True)
        r = torch.where(mass > 0, r / mass.clamp_min(1e-30), torch.zeros_like(r))
        out.append(r.sum(0))  # equal mass per boundary
    return out[0], out[1]


@torch.no_grad()
def capture(
    model,
    token_ids,
    out_dir,
    *,
    stride=4,
    batch_size=8,
    boundaries=None,
    progress=None,
):
    """Run the model over token_ids [N, T] and write content K and V to disk.

    Files are one float16 array per layer and component, shaped
    [rows, kv_heads, head_dim] with rows = N * (T // stride). When boundaries
    are given the attention sensitivities are written alongside, shaped
    [layers, 2, kv_heads, rows] with component 0 the keys.
    """
    os.makedirs(out_dir, exist_ok=True)
    geom = geometry(model)
    N, T = token_ids.shape
    sample_pos = list(range(0, T, stride))
    S = len(sample_pos)
    rows = N * S
    L, Hkv, d = geom["layers"], geom["kv_heads"], geom["head_dim"]
    dev = next(model.parameters()).device

    mm = {}
    for layer in range(L):
        for comp in ("k", "v"):
            mm[(layer, comp)] = np.lib.format.open_memmap(
                os.path.join(out_dir, f"L{layer:03d}.{comp}.npy"),
                mode="w+",
                dtype=np.float16,
                shape=(rows, Hkv, d),
            )
    sens = None
    tap = None
    if boundaries is not None:
        sens = np.zeros((L, 2, Hkv, rows), dtype=np.float32)
        tap = QueryTap(model, geom)

    sp = torch.as_tensor(sample_pos, device=dev)
    positions = torch.arange(T, device=dev)
    done = 0
    for start in range(0, N, batch_size):
        ids = token_ids[start : start + batch_size].to(dev)
        out = model(input_ids=ids, use_cache=True)
        layers = cache_layers(out.past_key_values)
        bsz = ids.shape[0]
        for layer, (k, v) in enumerate(layers):
            # k, v: [bsz, kv_heads, T, d]; keys are rotated in the cache
            k_content = strip_rope(k.to(torch.float32), positions, geom["rope_theta"])
            ks = k_content[:, :, sp, :].permute(0, 2, 1, 3).reshape(bsz * S, Hkv, d)
            vs = v[:, :, sp, :].permute(0, 2, 1, 3).reshape(bsz * S, Hkv, d)
            lo, hi = done * S, (done + bsz) * S
            mm[(layer, "k")][lo:hi] = ks.to(torch.float16).cpu().numpy()
            mm[(layer, "v")][lo:hi] = vs.to(torch.float16).cpu().numpy()
            if sens is not None:
                q = tap.q[layer]  # [bsz, T, H, d]
                for b in range(bsz):
                    rk, rv = sensitivities_for_layer(
                        q[b], k[b], v[b], boundaries, sample_pos, geom
                    )
                    r0 = (done + b) * S
                    sens[layer, 0, :, r0 : r0 + S] = rk.cpu().numpy()
                    sens[layer, 1, :, r0 : r0 + S] = rv.cpu().numpy()
        done += bsz
        if progress:
            progress(done, N)

    for m in mm.values():
        m.flush()
    if tap is not None:
        tap.close()
        np.save(os.path.join(out_dir, "sensitivities.npy"), sens)
    meta = {
        "geometry": geom,
        "sequences": int(N),
        "sequence_length": int(T),
        "stride": int(stride),
        "sample_positions": "range(0, T, stride)",
        "rows": int(rows),
        "dtype": "float16",
        "keys_are": "content (rotary removed at each token's own position)",
        "has_sensitivities": boundaries is not None,
        "boundaries": list(boundaries) if boundaries is not None else None,
        "eligibility": "sampled positions strictly before the boundary",
    }
    with open(os.path.join(out_dir, "CAPTURE.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta
