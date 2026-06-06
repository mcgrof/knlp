"""Trellis-as-a-block-selector — the bounded-memory "surprise" router.

This exists to settle whether Trellis-KRI's earlier loss to KRI-Q+N was a
real result or an eval artifact, by scoring KV blocks with the Trellis /
DeltaNet write-surprise signal and dropping it into the same per-position
screen as Lattice-KRI and the KRI family.

Unlike KRI-Q (query-conditioned), this is *query-agnostic*: it runs a
parameter-free gated-delta bounded memory over the L2-normalised keys and
measures, per token, how much the memory still mispredicts the value —
the residual ||v_t - phi(S_{t-1} k_t)||, i.e. how much new information that
token writes into the bounded state. Blocks with the most cumulative
write-surprise are kept. `nonlinear=True` applies the SiLU writer that
distinguishes Trellis from plain (linear) DeltaNet.

Because the signal is query-agnostic, the single-probe-vs-per-position
distinction barely touches it — so a clean per-position comparison directly
tests whether query-agnostic bounded-memory selection simply *is* worse than
query-conditioned routing, rather than having been sunk by a weak surface.

Caveat: write-surprise conflates "informative" with "unpredictable/noisy" —
the known weakness of write-energy importance. This is a faithful
representative of "Trellis as a selector", not a tuned router.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .kri_mask import block_index_of, num_blocks


@torch.no_grad()
def score_trellis_blocks(k, v, block_size, nonlinear=True):
    """Query-agnostic per-block write-surprise, shape [B,H,NB]. O(T) once.

    Runs S_t = S_{t-1} + (v_t - phi(S_{t-1} k_t)) k_t^T over the sequence
    (delta rule, beta=1, keys L2-normalised) and aggregates the per-token
    surprise ||v_t - phi(S_{t-1} k_t)|| into block means. Computed in fp32
    for numerical stability of the unbounded fast-weight state.
    """
    B, H, T, D = k.shape
    NB = num_blocks(T, block_size)
    kk = F.normalize(k.float(), dim=-1)
    vv = v.float()
    S = torch.zeros(B, H, D, D, device=k.device, dtype=torch.float32)
    surprise = torch.zeros(B, H, T, device=k.device, dtype=torch.float32)
    for t in range(T):
        kt = kk[:, :, t, :]
        pred = torch.einsum("bhij,bhj->bhi", S, kt)
        if nonlinear:
            pred = F.silu(pred)
        err = vv[:, :, t, :] - pred
        surprise[:, :, t] = err.norm(dim=-1)
        S = S + torch.einsum("bhi,bhj->bhij", err, kt)
    pad = NB * block_size - T
    if pad > 0:
        surprise = torch.cat([surprise, surprise.new_zeros(B, H, pad)], dim=2)
        counts = surprise.new_full((NB,), float(block_size))
        counts[-1] = float(block_size - pad)
    else:
        counts = surprise.new_full((NB,), float(block_size))
    return surprise.view(B, H, NB, block_size).sum(-1) / counts.view(1, 1, NB)


def select_trellis_blocks(q_vec, k, v, cfg, t_query):
    """select_fn-compatible selector: top-k eligible prefix blocks by
    write-surprise (q_vec ignored — query-agnostic). Same return contract as
    select_kri_blocks. NOTE: recomputes the O(T) recurrence on every call, so
    inside a per-position loop the eval uses a cached path instead."""
    B, H, T, D = k.shape
    bs = cfg.block_size
    NB = num_blocks(T, bs)
    local_first = block_index_of(max(0, t_query - cfg.local_window_tokens), bs)
    eligible = torch.zeros(NB, dtype=torch.bool, device=k.device)
    eligible[:local_first] = True
    for p in cfg.protected_blocks:
        if 0 <= p < NB:
            eligible[p] = False
    k_eff = min(cfg.global_topk_blocks, int(eligible.sum()))
    if k_eff <= 0:
        return q_vec.new_zeros(B, H, 0, dtype=torch.long)
    sb = score_trellis_blocks(k, v, bs).to(k.dtype)  # [B,H,NB]
    neg = torch.finfo(sb.dtype).min
    sb = sb.masked_fill(~eligible.view(1, 1, NB), neg)
    return sb.topk(k_eff, dim=-1).indices  # [B,H,K]
