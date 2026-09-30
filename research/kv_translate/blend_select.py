# SPDX-License-Identifier: GPL-2.0
"""A bounded Qwen3 adapter for key-deviation replay selection.

LMCache's blender at commit
``e73629e286258f5477f83f6d99de4977ad7b3bf7`` computes receiver keys at a
configured check layer, ranks tokens by squared key deviation from the cached
keys, and carries the selected token states through the remaining layers.  A
check at layer zero needs no earlier dense transformer layer: native layer-zero
keys are a function only of the token embeddings, input normalisation, key
projection, Qwen3 key normalisation, and RoPE.  The ordinary sparse replay can
then carry exactly those tokens through every layer against the mixed cache.

This module implements that small Hugging Face adapter.  It is deliberately
not described as an end-to-end LMCache reproduction: it fixes the check layer
at zero, returns the scores used to choose tokens, and leaves cache scatter and
causal replay to :mod:`research.kv_translate.replay`.
"""

from __future__ import annotations

import torch


def rank_tokens_by_key_deviation(fresh, cached, count):
    """Rank the token axis by squared K deviation, with stable ties.

    Both inputs use the receiver cache layout ``[1, kv_heads, tokens,
    head_dim]``.  Head and channel deviations are summed; tokens are not
    pooled into blocks.  Returned indices are sorted into original sequence
    order for replay, while ``scores`` retains one value per token.
    """
    if fresh.shape != cached.shape:
        raise ValueError(
            f"fresh key shape {tuple(fresh.shape)} != cached {tuple(cached.shape)}"
        )
    if fresh.dim() != 4 or fresh.shape[0] != 1:
        raise ValueError("keys must have cache layout [1, kv_heads, tokens, head_dim]")
    tokens = int(fresh.shape[2])
    count = int(count)
    if not 0 <= count <= tokens:
        raise ValueError(f"selected token count {count} is outside 0..{tokens}")
    scores = (fresh.float() - cached.float()).square().sum(dim=(0, 1, 3))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=fresh.device), scores

    # Transfer one short vector rather than synchronising once per token.
    # Python's stable ordering makes the lower token index win exact ties,
    # independent of the device sort implementation.
    host_scores = scores.detach().cpu().tolist()
    ranked = sorted(range(tokens), key=lambda i: (-host_scores[i], i))[:count]
    selected = torch.tensor(sorted(ranked), dtype=torch.long, device=fresh.device)
    return selected, scores


@torch.no_grad()
def qwen3_layer0_keys(model, input_ids, position_ids=None):
    """Compute Qwen3's cached layer-zero keys without running attention.

    The operations and their order match ``Qwen3Attention.forward`` in
    transformers 4.55: input RMSNorm, projection, per-head ``k_norm``, then
    the model's rotary embedding exactly once.  No receiver cache, values,
    attention output, MLP output, or later layer is computed.
    """
    trunk = getattr(model, "model", None)
    if trunk is None or not getattr(trunk, "layers", None):
        raise TypeError("model does not expose Qwen3 decoder layers")
    layer = trunk.layers[0]
    attn = layer.self_attn
    if not hasattr(attn, "k_norm") or not hasattr(attn, "head_dim"):
        raise TypeError("layer zero is not a supported Qwen3 attention layer")
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, tokens]")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if position_ids is None:
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    else:
        position_ids = position_ids.to(device)
    if position_ids.shape != input_ids.shape:
        raise ValueError(
            f"position_ids shape {tuple(position_ids.shape)} != input_ids "
            f"{tuple(input_ids.shape)}"
        )

    hidden = trunk.embed_tokens(input_ids)
    normalised = layer.input_layernorm(hidden)
    shape = (*normalised.shape[:-1], -1, attn.head_dim)
    keys = attn.k_norm(attn.k_proj(normalised).view(shape)).transpose(1, 2)
    cos, sin = trunk.rotary_emb(hidden, position_ids)

    # Import the model's own rotation rather than reconstructing a rope_theta
    # approximation.  Passing K in both slots avoids allocating the much wider
    # query tensor; only the returned key is used.
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
    return keys


@torch.no_grad()
def select_qwen3_layer0_key_deviation(model, input_ids, mapped_key, count):
    """Select tokens by native-vs-mapped layer-zero K deviation."""
    fresh = qwen3_layer0_keys(model, input_ids)
    selected, scores = rank_tokens_by_key_deviation(fresh, mapped_key, count)
    return selected, scores, fresh
