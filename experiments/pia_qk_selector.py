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
def qk_prefix_scores(model, full_ids, prefix_len, obs, device):
    """Per-prefix-token attention score from the last `obs` query rows.

    full_ids is [1, T] = prefix tokens then query tokens. Returns a
    [prefix_len] float tensor: for each prefix position, the max over layers of
    (max over heads of the mean over the observation-window rows of the
    softmax-normalized q.k attention onto that position).
    """
    layers = model.model.layers
    cfg = model.config
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", Hq)
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    scale = D**-0.5
    T = full_ids.shape[1]
    q_lo = T - obs  # observation-window rows (last obs positions, all in query)
    score = torch.zeros(prefix_len, device=device)

    # one forward, a pre-hook on every layer, each computing its own contribution
    # into `contribs` and keeping only a [prefix] vector resident.
    contribs = {}

    def layer_hook(idx):
        attn = model.model.layers[idx].self_attn

        def hook(mod, args, kwargs):
            x = (args[0] if args else kwargs["hidden_states"])[0].float()  # [T, d]
            pe = kwargs["position_embeddings"]
            cos, sin = pe[0][0].float(), pe[1][0].float()  # [T, D]
            Wq, Wk = attn.q_proj.weight.float(), attn.k_proj.weight.float()
            bq = attn.q_proj.bias
            bk = attn.k_proj.bias
            q = x @ Wq.t()
            if bq is not None:
                q = q + bq.float()
            k = x @ Wk.t()
            if bk is not None:
                k = k + bk.float()
            q = q.view(T, Hq, D).transpose(0, 1)  # [Hq, T, D]
            k = k.view(T, Hkv, D).transpose(0, 1)  # [Hkv, T, D]
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            q_obs = q[:, q_lo:, :]  # [Hq, obs, D]  (only the obs rows are scored)
            rep = Hq // Hkv
            k_rep = k.repeat_interleave(rep, dim=0)  # [Hq, T, D]
            # attention FROM the obs rows over ALL keys, then softmax over the
            # full causal support and slice the prefix -- matches the model's
            # post-softmax attention[:, obs_rows, :prefix_len] exactly, but only
            # materializes [Hq, obs, T] (cheap at 16-32K), never [Hq, T, T].
            logits = torch.einsum("hod,htd->hot", q_obs, k_rep) * scale
            obs_pos = torch.arange(q_lo, T, device=logits.device)  # abs positions
            key_idx = torch.arange(T, device=logits.device)
            future = key_idx.unsqueeze(0) > obs_pos.unsqueeze(1)  # [obs, T]
            logits = logits.masked_fill(future.unsqueeze(0), float("-inf"))
            w = torch.softmax(logits, dim=-1)  # over the full causal support
            col = w[:, :, :prefix_len].mean(dim=1)  # [Hq, prefix] mean over obs
            contribs[idx] = col.max(dim=0).values  # [prefix] max over heads

        return hook

    handles = [
        model.model.layers[i].self_attn.register_forward_pre_hook(
            layer_hook(i), with_kwargs=True
        )
        for i in range(len(layers))
    ]
    try:
        model(full_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    for idx in contribs:
        score = torch.maximum(score, contribs[idx])
    return score


@torch.no_grad()
def qk_top_tokens(model, full_ids, prefix_len, keep_tokens, obs, device):
    """Top-`keep_tokens` prefix positions by qk_prefix_scores (a set)."""
    s = qk_prefix_scores(model, full_ids, prefix_len, obs, device)
    keep = torch.topk(s, min(keep_tokens, prefix_len)).indices
    return set(keep.tolist())
