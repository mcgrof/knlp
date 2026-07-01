# SPDX-License-Identifier: GPL-2.0
"""Per-block scoring, including KRI-D-sum.

KRI-D-sum is the production block selector in the knlp KRI family. Its per-block
score is the norm of the summed keys plus the norm of the summed values over the
block's tokens, aggregated over layers and heads -- the same quantity computed in
gpt2-kri-ft/src/canonical_kri.py::kri_energy_mask (score = ||sum_t K_t|| +
||sum_t V_t||). It is query-independent: computed once from the prefix KV, so a
block's tier and score do not depend on the decode query.

For the emulation we also expose the oracle score (measured attention mass per
block) so KRI-D-sum's choices can be compared against the blocks that actually
receive attention.
"""

from __future__ import annotations

import math


def kri_d_sum_scores(kv, block_size: int) -> list:
    """KRI-D-sum per block. kv is a list of (K, V) per layer, each shaped
    (batch, n_kv_heads, seq_len, head_dim). Returns one score per block,
    aggregated over layers and heads. Requires torch.
    """
    import torch

    n_layers = len(kv)
    seq_len = kv[0][0].shape[2]
    num_blocks = math.ceil(seq_len / block_size)
    acc = torch.zeros(num_blocks, dtype=torch.float32)
    for k, v in kv:
        k = k.float()
        v = v.float()
        b, h, t, d = k.shape
        for bi in range(num_blocks):
            lo, hi = bi * block_size, min(t, (bi + 1) * block_size)
            if lo >= hi:
                continue
            # ||sum_t K_t|| + ||sum_t V_t||, summed over heads, mean over batch
            ks = k[:, :, lo:hi, :].sum(dim=2)  # [b, h, d]
            vs = v[:, :, lo:hi, :].sum(dim=2)
            s = ks.norm(dim=-1).sum(dim=-1) + vs.norm(dim=-1).sum(dim=-1)  # [b]
            acc[bi] += float(s.mean().item())
    return (acc / max(1, n_layers)).tolist()


def recency_scores(num_blocks: int) -> list:
    """Higher for more recent blocks (monotone in block_id)."""
    return [float(b) for b in range(num_blocks)]


def fifo_scores(num_blocks: int) -> list:
    """Lower (older) first out: score = block_id, evict the smallest. Same shape
    as recency but named for intent -- FIFO evicts oldest, recency keeps newest.
    """
    return [float(b) for b in range(num_blocks)]


def attention_mass_scores(attn_weights, block_size: int, num_blocks: int) -> list:
    """Oracle: fraction of attention mass each block receives. attn_weights is a
    per-layer list of tensors (batch, heads, q_len, k_len); we take the query's
    last-row attention, sum per block, aggregate over layers and heads. Requires
    torch. This is the ground truth KRI-D-sum is trying to approximate.
    """
    import torch

    acc = torch.zeros(num_blocks, dtype=torch.float32)
    layers = 0
    for aw in attn_weights:
        if aw is None:
            continue
        layers += 1
        last = aw.float()[:, :, -1, :]  # [b, h, k_len]
        klen = last.shape[-1]
        for bi in range(num_blocks):
            lo, hi = bi * block_size, min(klen, (bi + 1) * block_size)
            if lo < hi:
                acc[bi] += float(last[:, :, lo:hi].sum().item())
    total = float(acc.sum().item())
    if total > 0:
        acc = acc / total
    return acc.tolist()
