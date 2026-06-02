"""Canonical KRI-family router implementations for the cross-router
generalisation test (Phase 2.6.2 / task #31).

These implement the routers documented in the broader KRI family literature so we
can ask: does KRI-FT (a model fine-tuned under our internal
KRI-Q+N curriculum) also tolerate the *canonical* KRI variants?

Variants in this module:

  KRI-Q          query-conditioned cosine of layer-0 W_q projected
                 query against block-mean Keys
                 (the original training-free KRI-Q)
  KRI-Q-window   KRI-Q where the selected K must form a contiguous
                 window — anchor at the most-relevant block and take
                 ceil(K/2) before + floor(K/2) after
  KRI-G          geometric, query-agnostic: K-means cluster centers
                 in block-mean Key space; pick block whose centroid
                 is closest to each cluster center
  KRI-D          decoded / kv-sum: per-block sum-of-K and sum-of-V
                 norms; pick blocks with highest score (proxy for
                 attention-mass receivers)

All routers consume the K/V tensors that `GPT2KRI.collect_kv`
already returns. They produce a `[B, H, T, T]` boolean attention
mask suitable for `GPT2KRI.forward(attn_mask=...)` with the same
sink + local window structure used elsewhere in this repo.

Caveat about scope. The canonical KRI work was developed and
validated on Qwen2.5-7B at 16K context with cartridge-style
priors baked offline once per document. We are porting the
*shape* of those routers to a per-batch eval-time computation on
GPT-2 small at 1024 context. Quantitative numbers from these
implementations should NOT be cross-compared with the prior
Qwen2.5-7B legal-cartridge numbers; the point is the
*generalisation pattern*: does KRI-FT survive across these
shapes the way it survives across our internal KRI-Q+N?
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def num_blocks(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


def block_key_centroids(k_layer: torch.Tensor, block_size: int) -> torch.Tensor:
    """[B, H, NB, D] L2-normalised mean Key per block."""
    B, H, T, D = k_layer.shape
    NB = num_blocks(T, block_size)
    pad = NB * block_size - T
    if pad > 0:
        k_layer = torch.cat([k_layer, k_layer.new_zeros(B, H, pad, D)], dim=2)
    k_block = k_layer.view(B, H, NB, block_size, D)
    centroids = k_block.mean(dim=3)
    return F.normalize(centroids, dim=-1)


def block_value_centroids(v_layer: torch.Tensor, block_size: int) -> torch.Tensor:
    """[B, H, NB, D] mean Value per block (no normalisation; used for
    energy / sum-norm scoring)."""
    B, H, T, D = v_layer.shape
    NB = num_blocks(T, block_size)
    pad = NB * block_size - T
    if pad > 0:
        v_layer = torch.cat([v_layer, v_layer.new_zeros(B, H, pad, D)], dim=2)
    return v_layer.view(B, H, NB, block_size, D).mean(dim=3)


def _causal_base_mask(seq_len: int, block_size: int, local_window_tokens: int,
                     sink_blocks: int, B: int, H: int, device: torch.device
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the sink + recent + same-block base mask shared across
    routers. Returns (base_mask, causal) — the second is the strict
    lower-triangle.
    """
    T = seq_len
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    idx = torch.arange(T, device=device)
    diff = idx.view(T, 1) - idx.view(1, T)
    local = (diff >= 0) & (diff < local_window_tokens)
    sink = torch.zeros(T, dtype=torch.bool, device=device)
    sink[: sink_blocks * block_size] = True
    sink_keep = sink.view(1, T) & causal
    same_block = (idx.view(T, 1) // block_size == idx.view(1, T) // block_size) & causal
    base_2d = (local & causal) | sink_keep | same_block
    return base_2d.view(1, 1, T, T).expand(B, H, T, T).clone(), causal


def _select_blocks_to_mask(visible_blocks: torch.Tensor, T: int,
                          block_size: int, B: int, H: int,
                          base_mask: torch.Tensor, causal: torch.Tensor
                          ) -> torch.Tensor:
    """Given visible_blocks as a [B, H, K] int tensor (or [B, K]
    broadcastable to H), add those blocks' columns to base_mask and
    AND with causal.
    """
    if visible_blocks.dim() == 2:
        visible_blocks = visible_blocks.unsqueeze(1).expand(-1, H, -1)
    out = base_mask.clone()
    for b in range(B):
        for h in range(H):
            for blk in visible_blocks[b, h].tolist():
                lo = int(blk) * block_size
                hi = min(T, (int(blk) + 1) * block_size)
                out[b, h, :, lo:hi] |= True
    return out & causal.view(1, 1, T, T)


# ---------------------------------------------------------------------------
# KRI-Q (canonical training-free query-conditioned scoring)
# ---------------------------------------------------------------------------

def kri_q_mask(k_per_layer: List[torch.Tensor], v_per_layer: List[torch.Tensor],
              q_per_layer: List[torch.Tensor],
              seq_len: int, block_size: int, local_window_tokens: int,
              sink_blocks: int, topk_blocks: int,
              score_layer_index: int = 0,
              device: Optional[torch.device] = None) -> torch.Tensor:
    """Canonical KRI-Q: cosine(query, block-mean Key) at the chosen
    layer (default layer 0, matching the prior KRI-Q work which used
    the model's layer-0 W_q projection as the "geometric probe"
    into the cache's Key space).

    The "query" here is a single representative per (batch, head).
    For the prior cartridge work this is the query embedding fed to
    the cartridge; for our streaming-text eval we use the K vector
    at the final position as a stand-in (well-defined and in the
    same space).
    """
    if device is None:
        device = k_per_layer[0].device
    B, H, T, D = k_per_layer[score_layer_index].shape
    base, causal = _causal_base_mask(seq_len, block_size, local_window_tokens,
                                       sink_blocks, B, H, device)
    NB = num_blocks(T, block_size)
    L = min(score_layer_index, len(k_per_layer) - 1)
    centroids = block_key_centroids(k_per_layer[L], block_size)  # [B,H,NB,D]
    # Use last-position K as the "query probe" (geometric probe into Key
    # space, matching KRI-Q's framing). Normalise.
    q_probe = F.normalize(q_per_layer[L][:, :, -1, :], dim=-1)    # [B,H,D]
    score = torch.einsum("bhnd,bhd->bhn", centroids, q_probe)
    # Mask out the sink + local window + same-block blocks; they're
    # already in `base` and we don't double-count.
    eligible = torch.ones(NB, dtype=torch.bool, device=device)
    for b in range(sink_blocks):
        eligible[b] = False
    last_local_block = max(0, (seq_len - local_window_tokens)) // block_size
    for b in range(last_local_block, NB):
        eligible[b] = False
    neg = torch.finfo(score.dtype).min
    score = score.masked_fill(~eligible.view(1, 1, NB), neg)
    k_eff = min(topk_blocks, int(eligible.sum().item()))
    if k_eff <= 0:
        return base
    sel = score.topk(k_eff, dim=-1).indices  # [B,H,K]
    return _select_blocks_to_mask(sel, T, block_size, B, H, base, causal)


# ---------------------------------------------------------------------------
# KRI-Q-window (KRI-Q with contiguous-window structure around the anchor)
# ---------------------------------------------------------------------------

def kri_q_window_mask(k_per_layer, v_per_layer, q_per_layer,
                     seq_len: int, block_size: int, local_window_tokens: int,
                     sink_blocks: int, topk_blocks: int,
                     score_layer_index: int = 0,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """KRI-Q-window: same KRI-Q scoring, but the chosen K blocks must
    form a contiguous window centered on the anchor (the block with
    the highest cosine score). This is the variant that achieved
    1.00 word overlap at K=1 / K=8 on the prior legal-cartridge
    benchmark.
    """
    if device is None:
        device = k_per_layer[0].device
    B, H, T, D = k_per_layer[score_layer_index].shape
    base, causal = _causal_base_mask(seq_len, block_size, local_window_tokens,
                                       sink_blocks, B, H, device)
    NB = num_blocks(T, block_size)
    L = min(score_layer_index, len(k_per_layer) - 1)
    centroids = block_key_centroids(k_per_layer[L], block_size)
    q_probe = F.normalize(q_per_layer[L][:, :, -1, :], dim=-1)
    score = torch.einsum("bhnd,bhd->bhn", centroids, q_probe)
    # Mask ineligible (sink + local) so the anchor can't land there
    eligible = torch.ones(NB, dtype=torch.bool, device=device)
    for b in range(sink_blocks):
        eligible[b] = False
    last_local_block = max(0, (seq_len - local_window_tokens)) // block_size
    for b in range(last_local_block, NB):
        eligible[b] = False
    neg = torch.finfo(score.dtype).min
    score = score.masked_fill(~eligible.view(1, 1, NB), neg)
    # Per (b, h): find anchor, take a window
    anchors = score.argmax(dim=-1)  # [B,H]
    half_before = topk_blocks // 2
    half_after = topk_blocks - half_before - 1
    out = base.clone()
    for b in range(B):
        for h in range(H):
            anc = int(anchors[b, h].item())
            lo_block = max(0, anc - half_before)
            hi_block = min(NB - 1, anc + half_after)
            # Clip the window so it stays in the eligible region
            window = list(range(lo_block, hi_block + 1))
            for blk in window:
                if not eligible[blk]:
                    continue
                lo = blk * block_size
                hi = min(T, (blk + 1) * block_size)
                out[b, h, :, lo:hi] |= True
    return out & causal.view(1, 1, T, T)


# ---------------------------------------------------------------------------
# KRI-G (geometric, query-agnostic K-means on block-mean Keys)
# ---------------------------------------------------------------------------

def kri_g_mask(k_per_layer, v_per_layer, q_per_layer,
              seq_len: int, block_size: int, local_window_tokens: int,
              sink_blocks: int, topk_blocks: int,
              score_layer_index: int = 0, n_kmeans_iters: int = 8,
              device: Optional[torch.device] = None) -> torch.Tensor:
    """KRI-G: per (batch, head), run K-means with K=topk_blocks on the
    block-mean Keys; pick the block in each cluster whose centroid
    is closest to the cluster center. This is the "geometric" /
    query-agnostic variant.

    The original cartridge KRI-G is precomputed once per document.
    Our version is a per-batch streaming variant — same algorithm,
    different staging. Still query-agnostic.
    """
    if device is None:
        device = k_per_layer[0].device
    B, H, T, D = k_per_layer[score_layer_index].shape
    base, causal = _causal_base_mask(seq_len, block_size, local_window_tokens,
                                       sink_blocks, B, H, device)
    NB = num_blocks(T, block_size)
    L = min(score_layer_index, len(k_per_layer) - 1)
    centroids = block_key_centroids(k_per_layer[L], block_size)  # [B,H,NB,D]

    eligible_idx = torch.tensor(
        [i for i in range(NB)
         if i >= sink_blocks and i < (max(0, (seq_len - local_window_tokens)) // block_size)],
        device=device, dtype=torch.long,
    )
    if eligible_idx.numel() == 0:
        return base
    elig_centroids = centroids[:, :, eligible_idx, :]  # [B,H,NE,D]
    NE = elig_centroids.shape[2]
    k_eff = min(topk_blocks, NE)
    if k_eff <= 0:
        return base

    # K-means: init cluster centers from k_eff evenly-spaced eligible
    # blocks; run k iterations.
    init_idx = torch.linspace(0, NE - 1, k_eff, device=device).long()
    centers = elig_centroids[:, :, init_idx, :].clone()  # [B,H,K,D]
    for _ in range(n_kmeans_iters):
        # Assign each eligible block to nearest center
        d = torch.einsum("bhnd,bhkd->bhnk", elig_centroids, centers)
        assign = d.argmax(dim=-1)  # [B,H,NE]
        # Recompute centers as the mean of assigned points
        new_centers = torch.zeros_like(centers)
        for cls in range(k_eff):
            mask = (assign == cls).unsqueeze(-1).float()
            n = mask.sum(dim=2).clamp(min=1.0)
            new_centers[:, :, cls, :] = (elig_centroids * mask).sum(dim=2) / n
        centers = F.normalize(new_centers, dim=-1)
    # Final: for each cluster, pick the eligible block whose centroid
    # is closest to that cluster center.
    d = torch.einsum("bhnd,bhkd->bhnk", elig_centroids, centers)
    chosen_within = d.argmax(dim=2)  # [B,H,K] indices into elig_centroids
    chosen_global = eligible_idx[chosen_within]  # [B,H,K] indices into NB
    return _select_blocks_to_mask(chosen_global, T, block_size, B, H, base, causal)


# ---------------------------------------------------------------------------
# KRI-Energy (block-energy prior; NOT the canonical KRI-D)
# ---------------------------------------------------------------------------

def kri_energy_mask(k_per_layer, v_per_layer, q_per_layer,
              seq_len: int, block_size: int, local_window_tokens: int,
              sink_blocks: int, topk_blocks: int,
              score_layer_index: int = 0,
              device: Optional[torch.device] = None) -> torch.Tensor:
    """KRI-Energy: per block, score = ||sum_t K_t|| + ||sum_t V_t||.

    Cheap query-agnostic block-energy proxy. Picks blocks whose
    summed K and V have the largest norm — "this block contains
    content that receives a lot of attention mass."

    Naming note (see CARTRIDGE_LINEAGE.md): this is NOT the
    canonical KRI-D from the production cartridge stack.

    - Production `kri_d_kv_sum` = leave-one-block-out hidden-state
      L2 perturbation (offline; the production leader).
    - Reference `priors.py:kri_d` = leave-one-block-out KL on
      last-token distribution.
    - This function = a cheap content-energy block selector
      that doesn't require model perturbation.

    Earlier versions of this file called this `kri_d_mask`, which
    was misleading. Renamed to `kri_energy_mask` so the literature
    term stays attached to the algorithm that earned it.
    """
    if device is None:
        device = k_per_layer[0].device
    B, H, T, D = k_per_layer[score_layer_index].shape
    base, causal = _causal_base_mask(seq_len, block_size, local_window_tokens,
                                       sink_blocks, B, H, device)
    NB = num_blocks(T, block_size)
    L = min(score_layer_index, len(k_per_layer) - 1)
    k_layer = k_per_layer[L]
    v_layer = v_per_layer[L]
    pad = NB * block_size - T
    if pad > 0:
        k_layer = torch.cat([k_layer, k_layer.new_zeros(B, H, pad, D)], dim=2)
        v_layer = torch.cat([v_layer, v_layer.new_zeros(B, H, pad, D)], dim=2)
    k_block = k_layer.view(B, H, NB, block_size, D).sum(dim=3)  # [B,H,NB,D]
    v_block = v_layer.view(B, H, NB, block_size, D).sum(dim=3)
    score = k_block.norm(dim=-1) + v_block.norm(dim=-1)  # [B,H,NB]

    eligible = torch.ones(NB, dtype=torch.bool, device=device)
    for b in range(sink_blocks):
        eligible[b] = False
    last_local_block = max(0, (seq_len - local_window_tokens)) // block_size
    for b in range(last_local_block, NB):
        eligible[b] = False
    neg = torch.finfo(score.dtype).min
    score = score.masked_fill(~eligible.view(1, 1, NB), neg)
    k_eff = min(topk_blocks, int(eligible.sum().item()))
    if k_eff <= 0:
        return base
    sel = score.topk(k_eff, dim=-1).indices
    return _select_blocks_to_mask(sel, T, block_size, B, H, base, causal)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CANONICAL_ROUTERS = {
    "kri_q":        kri_q_mask,
    "kri_q_window": kri_q_window_mask,
    "kri_g":        kri_g_mask,
    # NOTE: "kri_d" key kept for backward compatibility with the
    # Phase 2.6.2 JSONL outputs. New code should call "kri_energy"
    # to make the algorithmic distinction explicit (see
    # CARTRIDGE_LINEAGE.md).
    "kri_d":        kri_energy_mask,
    "kri_energy":   kri_energy_mask,
}


def canonical_router_mask(name: str, **kwargs) -> torch.Tensor:
    if name not in CANONICAL_ROUTERS:
        raise ValueError(f"Unknown canonical KRI router: {name}; "
                         f"available: {list(CANONICAL_ROUTERS)}")
    return CANONICAL_ROUTERS[name](**kwargs)
