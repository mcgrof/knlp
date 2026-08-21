# SPDX-License-Identifier: GPL-2.0
"""Attention-level damage metrics for lossy KV block codecs.

Raw K/V reconstruction error is a poor ranking signal; what matters
is how much the attention output moves. These helpers replace one
sealed block of a captured cache with its codec reconstruction,
leave the rest of the context at full precision, and measure the
resulting attention damage for every causal query position after
the block. All math runs in fp32.
"""

from __future__ import annotations

import math

import torch


def causal_attention(q, k, v, q_positions):
    """Full attention for query rows at absolute positions
    q_positions (ascending), causally masked over k/v of length T.

    Returns (probs [Tq, T], out [Tq, d], logits [Tq, T]).
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    d = q32.shape[-1]
    logits = q32 @ k32.transpose(-1, -2) / math.sqrt(d)
    t = k32.shape[0]
    pos = torch.as_tensor(q_positions, device=logits.device).unsqueeze(1)
    mask = torch.arange(t, device=logits.device).unsqueeze(0) > pos
    logits = logits.masked_fill(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return probs, probs @ v32, logits


def reference_pass(q_rows, q_positions, k, v, block_end):
    """Filter probes to those after the block and run the reference
    attention once. The result is reusable across every candidate
    reconstruction of the same block."""
    keep = [i for i, p in enumerate(q_positions) if p >= block_end]
    if not keep:
        return None
    q_rows = q_rows[keep]
    q_positions = [q_positions[i] for i in keep]
    probs, out, logits = causal_attention(q_rows, k, v, q_positions)
    return q_rows, q_positions, probs, out, logits


def block_replacement_metrics(
    q_rows, q_positions, k, v, block_start, block_end, k_hat, v_hat, ref=None
):
    """Attention damage from replacing one block of the cache.

    q_rows: [Tq, d] query probes at absolute cache positions
    q_positions (may be strided); k, v: [T, d] full-precision cache;
    k_hat, v_hat: [B, d] reconstruction of positions
    [block_start, block_end). Only probes strictly after the block
    are evaluated (they are the ones that can attend into it). Pass
    ref from reference_pass() to amortize the clean-cache side.
    """
    if ref is None:
        ref = reference_pass(q_rows, q_positions, k, v, block_end)
    if ref is None:
        return None
    q_rows, q_positions, probs_ref, out_ref, logits_ref = ref
    t = k.shape[0]
    k_mod, v_mod = k.clone(), v.clone()
    k_mod[block_start:block_end] = k_hat.to(k.dtype)
    v_mod[block_start:block_end] = v_hat.to(v.dtype)
    probs_hat, out_hat, logits_hat = causal_attention(q_rows, k_mod, v_mod, q_positions)

    blk = slice(block_start, block_end)
    lref, lhat = logits_ref[:, blk], logits_hat[:, blk]
    logit_rel = float((lhat - lref).norm() / (lref.norm() + 1e-12))
    logit_max = float((lhat - lref).abs().max())

    p = probs_ref.clamp_min(1e-12)
    ph = probs_hat.clamp_min(1e-12)
    kl = float((p * (p / ph).log()).sum(-1).mean())

    top1 = float((probs_ref.argmax(-1) == probs_hat.argmax(-1)).float().mean())
    k_top = min(4, t)
    ref_top = probs_ref.topk(k_top, dim=-1).indices
    hat_top = probs_hat.topk(k_top, dim=-1).indices
    overlap = (ref_top.unsqueeze(-1) == hat_top.unsqueeze(-2)).any(-1).float()
    top4 = float(overlap.mean())

    out_rel = float((out_hat - out_ref).norm() / (out_ref.norm() + 1e-12))

    return {
        "qk_logit_rel_err": logit_rel,
        "qk_logit_max_err": logit_max,
        "attn_kl": kl,
        "top1_agreement": top1,
        "top4_overlap": top4,
        "attn_out_rel_err": out_rel,
        "k_rel_err": float(
            (k_hat.float() - k[blk].float()).norm() / (k[blk].float().norm() + 1e-12)
        ),
        "v_rel_err": float(
            (v_hat.float() - v[blk].float()).norm() / (v[blk].float().norm() + 1e-12)
        ),
        "num_queries": len(q_positions),
    }
