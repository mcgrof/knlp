"""Non-learned KRI router and attention-mask builder.

The router operates on KV **blocks**, not individual tokens. For each
query token past a sampled `prefill_split`, only three regions are
visible:

  1. a local recent-token window `[t - W, t]`,
  2. a protected sink block (block 0, optionally plus user-specified
     protected blocks),
  3. the top-k KRI-selected prefix blocks scored against query q_t.

The score is purely a function of K, V tensors and the query — no
learned parameters. We score per (layer, head) but emit a single
attention mask shared across layers; the per-(layer, head) signal is
collapsed by taking the union of selected blocks across heads of one
representative layer, which is cheap and good enough as a regulariser
during training.

For the strict per-(layer, head) variant, set `per_head=True` and pass
the resulting `[B, H, Tq, Tk]` mask through to attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass
class KRIConfig:
    block_size: int = 16
    local_window_tokens: int = 128
    global_topk_blocks: int = 8
    prefill_split: Optional[int] = None  # if None, pick at random per call
    prefill_split_choices: Tuple[int, ...] = (256, 384, 512, 768)
    local_window_choices: Tuple[int, ...] = (64, 128, 256)
    topk_block_choices: Tuple[int, ...] = (2, 4, 8, 16)
    protected_blocks: Tuple[int, ...] = (0,)
    w_cos: float = 1.00
    w_value_energy: float = 0.20
    w_recency: float = 0.15
    w_novelty: float = 0.50
    w_protected: float = 10.0
    use_novelty: bool = True
    per_head: bool = False
    # Which layer's K/V to use to score blocks when emitting one global
    # mask (per_head=False). Mid-layer is a reasonable default.
    score_layer_index: int = 6

    def sample(self, rng: torch.Generator) -> "KRIConfig":
        """Return a copy with prefill_split / local_window / topk
        sampled fresh from the *_choices fields."""

        def pick(choices):
            i = int(torch.randint(0, len(choices), (1,), generator=rng).item())
            return choices[i]

        c = KRIConfig(**self.__dict__)
        c.prefill_split = pick(self.prefill_split_choices)
        c.local_window_tokens = pick(self.local_window_choices)
        c.global_topk_blocks = pick(self.topk_block_choices)
        return c


def num_blocks(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


def block_index_of(t: int, block_size: int) -> int:
    return t // block_size


def _block_summaries(
    k: torch.Tensor, v: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-block key centroid and value-energy.

    Args:
        k, v: [B, H, T, D] tensors. K is L2-normalized along D before
            averaging, so the centroid is itself a unit-ish vector once
            we renormalize.
        block_size: tokens per block.

    Returns:
        key_centroid: [B, H, NB, D] L2-normalized
        value_energy: [B, H, NB]   RMS energy per block, normalized by
                                   per-sequence mean
    """
    B, H, T, D = k.shape
    NB = num_blocks(T, block_size)
    pad = NB * block_size - T
    if pad > 0:
        k = torch.cat([k, k.new_zeros(B, H, pad, D)], dim=2)
        v = torch.cat([v, v.new_zeros(B, H, pad, D)], dim=2)

    k_block = k.view(B, H, NB, block_size, D)
    v_block = v.view(B, H, NB, block_size, D)

    # Mask out the padded slots when averaging.
    if pad > 0:
        mask = k.new_ones(NB * block_size)
        mask[T:] = 0.0
        mask = mask.view(NB, block_size)  # [NB, block_size]
        counts = mask.sum(-1).clamp(min=1.0)  # [NB]
        k_norm = torch.nn.functional.normalize(k_block, dim=-1)
        k_norm = k_norm * mask.view(1, 1, NB, block_size, 1)
        kc = k_norm.sum(dim=3) / counts.view(1, 1, NB, 1)  # [B,H,NB,D]
        v_sq = (v_block * v_block).sum(-1) * mask.view(1, 1, NB, block_size)
        ve = torch.sqrt(v_sq.sum(-1) / counts.view(1, 1, NB))
    else:
        k_norm = torch.nn.functional.normalize(k_block, dim=-1)
        kc = k_norm.mean(dim=3)  # [B,H,NB,D]
        ve = torch.sqrt((v_block * v_block).sum(-1).mean(-1))  # [B,H,NB]

    # Renormalize centroid so cosine with q is well behaved.
    kc = torch.nn.functional.normalize(kc, dim=-1)

    # Normalize energy per sequence so the weight is comparable across
    # batches; the per-batch mean energy maps to 1.0.
    ve = ve / (ve.mean(dim=-1, keepdim=True) + 1e-6)
    return kc, ve


def _recency_bonus(num_blocks_total: int, t_block: int, device, dtype) -> torch.Tensor:
    """Monotone in block index, scaled to [0, 1]."""
    idx = torch.arange(num_blocks_total, device=device, dtype=dtype)
    if t_block <= 0:
        return torch.zeros_like(idx)
    return torch.clamp(idx / max(1, t_block), min=0.0, max=1.0)


def _novelty_bonus(
    kc_prefix: torch.Tensor, protected_idx: Sequence[int], local_idx: Sequence[int]
) -> torch.Tensor:
    """Approximate 1 - max cosine to already-selected blocks.

    Args:
        kc_prefix: [B, H, NB, D] normalized prefix centroids.
        protected_idx, local_idx: block indices considered already
            protected or already local (and so not novel).
    Returns:
        [B, H, NB] novelty bonus in [0, 1].
    """
    B, H, NB, D = kc_prefix.shape
    seed_idx = sorted(set(list(protected_idx) + list(local_idx)))
    if not seed_idx:
        return kc_prefix.new_ones(B, H, NB) * 0.5  # uninformative
    seed = kc_prefix[:, :, seed_idx, :]  # [B,H,S,D]
    # cosine since both are L2-normalized
    sim = torch.einsum("bhnd,bhsd->bhns", kc_prefix, seed)  # [B,H,NB,S]
    max_sim = sim.max(dim=-1).values  # [B,H,NB]
    return torch.clamp(1.0 - max_sim, min=0.0, max=1.0)


def select_kri_blocks(
    q_vec: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: KRIConfig,
    t_query: int,
) -> torch.Tensor:
    """Select top-k prefix blocks for query position `t_query`.

    Args:
        q_vec: [B, H, D] per-(batch, head) query vector at position
               t_query (taken from the same layer as k/v).
        k, v:  [B, H, T, D] of the layer whose blocks we're selecting.
        cfg:   KRIConfig.
        t_query: current query position (int).

    Returns:
        sel: [B, H, K] long tensor of selected prefix block indices, or
             empty tensor if no prefix is available.

    Notes:
        - Block 0 is always protected and NOT in `sel` (the mask builder
          adds the protected blocks separately).
        - Blocks within the local window (those overlapping
          `[t_query - W, t_query]`) are also excluded from `sel`.
        - sel is clipped to fewer than `global_topk_blocks` if not
          enough prefix blocks exist.
    """
    B, H, T, D = k.shape
    NB = num_blocks(T, cfg.block_size)
    bs = cfg.block_size

    # Block containing the query. Blocks strictly before the local
    # window are eligible prefix blocks.
    t_block = block_index_of(t_query, bs)
    local_start = max(0, t_query - cfg.local_window_tokens)
    local_first_block = block_index_of(local_start, bs)

    kc, ve = _block_summaries(k, v, bs)
    # cosine(q, kc) — q_vec is not pre-normalized; normalize for cosine
    q_n = torch.nn.functional.normalize(q_vec, dim=-1)  # [B,H,D]
    cos = torch.einsum("bhd,bhnd->bhn", q_n, kc)  # [B,H,NB]

    rec = _recency_bonus(NB, t_block, device=k.device, dtype=k.dtype)  # [NB]
    rec_b = rec.view(1, 1, -1).expand(B, H, NB)

    # Eligible mask: prefix blocks only (strictly before local window
    # and not in the protected set; protected blocks are added by the
    # mask builder).
    eligible = torch.zeros(NB, device=k.device, dtype=torch.bool)
    eligible[:local_first_block] = True
    for p in cfg.protected_blocks:
        if 0 <= p < NB:
            eligible[p] = False

    score = cfg.w_cos * cos + cfg.w_value_energy * ve + cfg.w_recency * rec_b
    if cfg.use_novelty:
        nov = _novelty_bonus(
            kc, cfg.protected_blocks, list(range(local_first_block, t_block + 1))
        )
        score = score + cfg.w_novelty * nov
    # Protected bonus is a positive nudge but we keep protected blocks
    # out of `sel` and add them separately in the mask builder.

    # Mask out ineligible blocks with -inf.
    neg_inf = torch.finfo(score.dtype).min
    score = score.masked_fill(~eligible.view(1, 1, NB), neg_inf)

    k_eff = min(cfg.global_topk_blocks, int(eligible.sum().item()))
    if k_eff <= 0:
        return q_vec.new_zeros(B, H, 0, dtype=torch.long)
    sel = score.topk(k_eff, dim=-1).indices  # [B,H,K]
    return sel


def build_kri_mask(
    cfg: KRIConfig,
    seq_len: int,
    batch_size: int,
    n_head: int,
    k_per_layer: Optional[List[torch.Tensor]] = None,
    v_per_layer: Optional[List[torch.Tensor]] = None,
    q_per_layer: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    select_fn=None,
) -> torch.Tensor:
    """Build a [B, H, T, T] boolean mask for KRI-pruned attention.

    The protocol:
      * For query positions t <= cfg.prefill_split: dense causal
        attention (mask all-True along the causal triangle).
      * For t > cfg.prefill_split: only the local window, protected
        sink blocks, the block containing t, and the top-k KRI blocks
        (scored at the configured layer) are visible.

    When `k_per_layer`, `v_per_layer`, `q_per_layer` are given, KRI
    scoring uses the named score layer's tensors. When they are None,
    we fall back to a recency-only selection (newer prefix blocks
    preferred), which is what we use in early training before the
    student has stabilized.

    Args:
        cfg: KRIConfig with prefill_split / local_window / topk set.
        seq_len: query and key length (square mask).
        batch_size, n_head: required to broadcast q-scoring.
        k_per_layer, v_per_layer, q_per_layer: optional [B,H,T,D] lists
        device: device for the mask.

    Returns:
        mask: [B, H, T, T] bool. True = attend, False = masked.
    """
    if device is None:
        device = (
            k_per_layer[0].device if k_per_layer is not None else torch.device("cpu")
        )
    B, H = batch_size, n_head
    T = seq_len
    bs = cfg.block_size
    NB = num_blocks(T, bs)
    prefill_split = cfg.prefill_split if cfg.prefill_split is not None else T // 2
    prefill_split = min(prefill_split, T - 1)
    W = max(1, cfg.local_window_tokens)

    # Mask base: full False. We'll set True where we want to attend.
    mask = torch.zeros(B, H, T, T, dtype=torch.bool, device=device)

    # 1) For prefill region (t <= prefill_split): full causal attention.
    #    Set mask[:, :, t, k] = True for k <= t when t <= prefill_split.
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    # The whole row for t <= prefill_split — copy the causal row.
    rows_prefill = torch.zeros(T, dtype=torch.bool, device=device)
    rows_prefill[: prefill_split + 1] = True
    mask = mask | (causal & rows_prefill.view(T, 1)).view(1, 1, T, T)

    # 2) For t > prefill_split, allow local window [max(0, t-W+1), t].
    #    Note: include t itself; never include the future.
    idx_t = torch.arange(T, device=device)
    idx_k = torch.arange(T, device=device)
    diff = idx_t.view(T, 1) - idx_k.view(1, T)  # [T, T] = t - k
    local_keep = (diff >= 0) & (diff < W)  # within window AND not future
    rows_decode = ~rows_prefill
    mask = mask | (local_keep & rows_decode.view(T, 1)).view(1, 1, T, T)

    # 3) Protected blocks visible to all decode-region queries.
    protected_keep = torch.zeros(T, dtype=torch.bool, device=device)
    for pb in cfg.protected_blocks:
        if 0 <= pb < NB:
            lo = pb * bs
            hi = min(T, (pb + 1) * bs)
            protected_keep[lo:hi] = True
    # AND with causal: even sink blocks must satisfy causal mask.
    pk = protected_keep.view(1, T) & causal  # [T, T]
    mask = mask | (pk & rows_decode.view(T, 1)).view(1, 1, T, T)

    # 4) Add the block that contains t itself (current block).
    #    Helps avoid degenerate attention when t is at the start of a
    #    new block and the local window doesn't cover the whole block.
    t_block_of = idx_t // bs  # [T]
    k_block_of = idx_k // bs  # [T]
    same_block = (t_block_of.view(T, 1) == k_block_of.view(1, T)) & causal
    mask = mask | (same_block & rows_decode.view(T, 1)).view(1, 1, T, T)

    # 5) Top-k KRI blocks per query. If no K/V provided, fall back to
    #    recency-only selection (most recent prefix blocks before the
    #    local window).
    if k_per_layer is None or q_per_layer is None:
        for t in range(prefill_split + 1, T):
            local_first_block = max(0, (t - W + 1)) // bs
            # Eligible prefix blocks: those strictly before local
            # window and not already protected.
            eligible = list(range(0, local_first_block))
            for pb in cfg.protected_blocks:
                if pb in eligible:
                    eligible.remove(pb)
            if not eligible:
                continue
            # newest-first recency
            sel = eligible[-cfg.global_topk_blocks :]
            for b in sel:
                lo = b * bs
                hi = min(T, (b + 1) * bs)
                mask[:, :, t, lo:hi] = True
    else:
        # Use the configured score layer's K, V, q.
        i_layer = min(cfg.score_layer_index, len(k_per_layer) - 1)
        k = k_per_layer[i_layer]  # [B,H,T,D]
        v = v_per_layer[i_layer]  # [B,H,T,D]
        q = q_per_layer[i_layer]  # [B,H,T,D]

        # Score and pick once per query position in decode region. A custom
        # select_fn (same signature as select_kri_blocks) plugs in an
        # alternative router — e.g. Lattice-KRI — reusing all of this
        # sink/recent/current/causal assembly unchanged.
        sel_fn = select_fn if select_fn is not None else select_kri_blocks
        for t in range(prefill_split + 1, T):
            q_t = q[:, :, t, :]
            sel = sel_fn(q_t, k, v, cfg, t_query=t)  # [B,H,K]
            if cfg.per_head:
                # Per-head selection: each head gets its own visible set.
                for b in range(B):
                    for h in range(H):
                        for bk in sel[b, h].tolist():
                            lo = bk * bs
                            hi = min(T, (bk + 1) * bs)
                            mask[b, h, t, lo:hi] = True
            else:
                # Union across heads — one mask shared across heads.
                if sel.numel() == 0:
                    continue
                uniq = torch.unique(sel)
                for bk in uniq.tolist():
                    lo = bk * bs
                    hi = min(T, (bk + 1) * bs)
                    mask[:, :, t, lo:hi] = True

    # Final causal AND — belt and braces against any indexing bug above.
    mask = mask & causal.view(1, 1, T, T)
    return mask


def fixed_policy_mask(
    policy: str,
    seq_len: int,
    batch_size: int,
    n_head: int,
    block_size: int = 16,
    local_window_tokens: int = 128,
    sink_blocks: int = 1,
    topk_blocks: int = 8,
    device: Optional[torch.device] = None,
    k_per_layer: Optional[List[torch.Tensor]] = None,
    v_per_layer: Optional[List[torch.Tensor]] = None,
    q_per_layer: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Build an eval-time attention mask for one of the named policies.

    Policies:
        - full         : no pruning (returns None for the mask in caller)
        - recent       : only the most recent local_window_tokens visible
        - sink_recent  : recent + sink_blocks first blocks
        - random_global: sink + recent + topk uniformly-random prefix
                         blocks (control against `kri`)
        - kri          : KRI selection on top of sink + recent

    Returns:
        (mask, info) where mask is [B, H, T, T] bool and info is a
        small dict reporting the retained-token statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = seq_len
    B, H = batch_size, n_head
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    if policy == "full":
        mask = causal.view(1, 1, T, T).expand(B, H, T, T).clone()
        retained = mask.sum(-1).float().mean().item()
        return mask, {"retained_per_query_avg": retained, "policy": policy}

    # local window (strict): keep [t - W + 1, t]
    idx = torch.arange(T, device=device)
    diff = idx.view(T, 1) - idx.view(1, T)
    local_keep = (diff >= 0) & (diff < local_window_tokens)
    mask2d = local_keep & causal

    # sink blocks
    if policy in ("sink_recent", "kri", "random_global"):
        sink = torch.zeros(T, dtype=torch.bool, device=device)
        sink[: sink_blocks * block_size] = True
        sink_keep = sink.view(1, T) & causal
        mask2d = mask2d | sink_keep

    mask = mask2d.view(1, 1, T, T).expand(B, H, T, T).clone()

    if policy == "recent" or policy == "sink_recent":
        retained = mask.sum(-1).float().mean().item()
        return mask, {
            "retained_per_query_avg": retained,
            "policy": policy,
            "local_window_tokens": local_window_tokens,
            "sink_blocks": sink_blocks if policy == "sink_recent" else 0,
        }

    # random_global: at each query t in the decode region, pick
    # `topk_blocks` blocks uniformly at random from the eligible prefix
    # set (strictly before the local window, not in protected/sink, not
    # the block of t itself). This is the control-baseline for KRI:
    # if random_global matches KRI at iso-budget, then the KRI scoring
    # rule is not contributing anything.
    if policy == "random_global":
        bs = block_size
        NB = num_blocks(T, bs)
        protected = set(range(sink_blocks))
        # Per-query random selection
        for t in range(T):
            local_start_block = max(0, t - local_window_tokens + 1) // bs
            cur_block = t // bs
            eligible = [
                b
                for b in range(local_start_block)
                if b not in protected and b != cur_block
            ]
            if not eligible:
                continue
            k_eff = min(topk_blocks, len(eligible))
            sel = torch.randperm(len(eligible))[:k_eff].tolist()
            for i in sel:
                b = eligible[i]
                lo = b * bs
                hi = min(T, (b + 1) * bs)
                mask[:, :, t, lo:hi] = True
        mask = mask & causal.view(1, 1, T, T)
        retained = mask.sum(-1).float().mean().item()
        return mask, {
            "retained_per_query_avg": retained,
            "policy": policy,
            "local_window_tokens": local_window_tokens,
            "sink_blocks": sink_blocks,
            "topk_blocks": topk_blocks,
            "block_size": block_size,
        }

    # KRI: layer on top
    assert policy == "kri"
    if k_per_layer is None:
        # recency-fallback selection
        cfg = KRIConfig(
            block_size=block_size,
            local_window_tokens=local_window_tokens,
            global_topk_blocks=topk_blocks,
            prefill_split=0,  # entire sequence is "decode" so KRI applies everywhere
            protected_blocks=tuple(range(sink_blocks)),
        )
        kri = build_kri_mask(cfg, T, B, H, device=device)
        mask = mask | kri
    else:
        cfg = KRIConfig(
            block_size=block_size,
            local_window_tokens=local_window_tokens,
            global_topk_blocks=topk_blocks,
            prefill_split=0,
            protected_blocks=tuple(range(sink_blocks)),
        )
        kri = build_kri_mask(
            cfg,
            T,
            B,
            H,
            k_per_layer=k_per_layer,
            v_per_layer=v_per_layer,
            q_per_layer=q_per_layer,
            device=device,
        )
        mask = mask | kri

    mask = mask & causal.view(1, 1, T, T)
    retained = mask.sum(-1).float().mean().item()
    return mask, {
        "retained_per_query_avg": retained,
        "policy": policy,
        "local_window_tokens": local_window_tokens,
        "sink_blocks": sink_blocks,
        "topk_blocks": topk_blocks,
        "block_size": block_size,
    }
