# SPDX-License-Identifier: GPL-2.0
"""Cheap query->prefix attention scorer for PIA, without output_attentions.

The PIA selectors (reuse-replay, kill-test, codec sweep) scored prefix tokens by
running the model with output_attentions=True, which materializes a
[layers, heads, T, T] tensor and does not fit past ~4-5K tokens. This computes
the same query-attention signal without that tensor: a forward hook on each
attention module captures the layer input and RoPE cos/sin, recomputes the query
rows' q and the prefix's k for that layer, and reduces q_obs . k_prefix to a
per-prefix-token score inside the hook -- so only one layer's scores are ever
resident. Memory is O(heads * obs * prefix), not O(layers * heads * T^2), and the
context can be 16-32K.

Targets rotate-half RoPE (Qwen2.5, Llama-3, Phi-4). The score matches the
attention-mass reduction the output_attentions path used: softmax over the prefix
of the last `obs` query rows, mean over those rows, max over heads, max across
layers -- the SnapKV-shaped observation-window signal.
"""

from __future__ import annotations

import torch


def _rotate_half(x):
    h1, h2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-h2, h1), dim=-1)


def _apply_rope(t, cos, sin):
    # t: [n_head, S, D]; cos/sin: [S, D]
    return t * cos.unsqueeze(0) + _rotate_half(t) * sin.unsqueeze(0)


@torch.no_grad()
def _row_prefix_attention(model, full_ids, prefix_len, row_lo, reduce):
    """Run one forward and reduce each layer's attention onto the prefix.

    A pre-hook on every attention module recomputes that layer's q for rows
    [row_lo, T) and k for all T positions, applies RoPE, and takes the softmax
    over the full causal support. That matches the model's post-softmax
    attention[:, rows, :] exactly, but only materializes [Hq, rows, T], never
    [Hq, T, T]. The hook then slices the prefix, takes the mean over the rows
    and the max over heads, and passes the [prefix_len] vector to `reduce`.
    Returns the elementwise max of the reduced vectors across layers.
    """
    layers = model.model.layers
    cfg = model.config
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", Hq)
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    scale = D**-0.5
    T = full_ids.shape[1]
    contribs = {}

    def layer_hook(idx):
        attn = layers[idx].self_attn

        def hook(mod, args, kwargs):
            x = (args[0] if args else kwargs["hidden_states"])[0].float()  # [T, d]
            pe = kwargs["position_embeddings"]
            cos, sin = pe[0][0].float(), pe[1][0].float()  # [T, D]
            q = x @ attn.q_proj.weight.float().t()
            if attn.q_proj.bias is not None:
                q = q + attn.q_proj.bias.float()
            k = x @ attn.k_proj.weight.float().t()
            if attn.k_proj.bias is not None:
                k = k + attn.k_proj.bias.float()
            q = _apply_rope(q.view(T, Hq, D).transpose(0, 1), cos, sin)
            k = _apply_rope(k.view(T, Hkv, D).transpose(0, 1), cos, sin)
            q_rows = q[:, row_lo:, :]  # [Hq, rows, D]
            k_rep = k.repeat_interleave(Hq // Hkv, dim=0)  # [Hq, T, D]
            logits = torch.einsum("hod,htd->hot", q_rows, k_rep) * scale
            row_pos = torch.arange(row_lo, T, device=logits.device)
            key_idx = torch.arange(T, device=logits.device)
            future = key_idx.unsqueeze(0) > row_pos.unsqueeze(1)  # [rows, T]
            logits = logits.masked_fill(future.unsqueeze(0), float("-inf"))
            w = torch.softmax(logits, dim=-1)  # over the full causal support
            col = w[:, :, :prefix_len].mean(dim=1).max(dim=0).values  # [prefix]
            contribs[idx] = reduce(col)

        return hook

    handles = [
        layers[i].self_attn.register_forward_pre_hook(layer_hook(i), with_kwargs=True)
        for i in range(len(layers))
    ]
    try:
        model(full_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    score = None
    for idx in sorted(contribs):
        c = contribs[idx]
        score = c if score is None else torch.maximum(score, c)
    return score


@torch.no_grad()
def qk_prefix_scores(model, full_ids, prefix_len, obs, device):
    """Per-prefix-token attention score from the last `obs` query rows.

    full_ids is [1, T] = prefix tokens then query tokens. Returns a
    [prefix_len] float tensor: for each prefix position, the max over layers of
    (max over heads of the mean over the observation-window rows of the
    softmax-normalized q.k attention onto that position).
    """
    T = full_ids.shape[1]
    return _row_prefix_attention(model, full_ids, prefix_len, T - obs, lambda c: c)


@torch.no_grad()
def qk_block_mass(model, full_ids, prefix_len, row_lo, blocks, device):
    """Block-level twin of the kill-test's attention_block_mass(), without
    output_attentions. Scores attention FROM rows [row_lo, T) onto each prefix
    block: per layer, softmax over the causal support, mean over the scoring
    rows, max over heads, sum over the block's tokens; then max across layers.
    Returns a CPU [len(blocks)] tensor. Only [Hq, rows, T] is ever resident,
    which is what lets the kill-test run on a 40GB card."""
    lo_idx = torch.tensor([lo for lo, _ in blocks], device=device)
    hi_idx = torch.tensor([hi for _, hi in blocks], device=device)

    def block_sums(col):
        cs = torch.cat([torch.zeros(1, device=col.device), col.cumsum(0)])
        return (cs[hi_idx] - cs[lo_idx]).cpu()  # [len(blocks)]

    return _row_prefix_attention(model, full_ids, prefix_len, row_lo, block_sums)


@torch.no_grad()
def qk_top_tokens(model, full_ids, prefix_len, keep_tokens, obs, device):
    """Top-`keep_tokens` prefix positions by qk_prefix_scores (a set)."""
    s = qk_prefix_scores(model, full_ids, prefix_len, obs, device)
    keep = torch.topk(s, min(keep_tokens, prefix_len)).indices
    return set(keep.tolist())
