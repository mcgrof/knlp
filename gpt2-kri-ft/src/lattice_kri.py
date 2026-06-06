"""Lattice-inspired KRI router — orthogonal-novelty block selection.

This is NOT a Lattice architecture reproduction. It is a *training-free*
KRI router that borrows one idea from Lattice — selecting support whose
content is orthogonal (adds information) to what is already selected — and
expresses it as a block-selection rule compatible with the existing KRI-FT
mask machinery. There are no learned parameters, no memory slots, and no
model surgery here.

A router in this repo is just a block-selection function. `select_kri_blocks`
(kri_mask.py) scores every eligible prefix block once and takes the top-k.
Lattice-KRI replaces that one-shot score with a *greedy* selection that, at
each step, prefers blocks which are both relevant to the current query and
non-redundant with the blocks already chosen — measured as the norm of the
block summary's residual against an orthonormal basis spanned by the
already-selected summaries.

`select_lattice_blocks` has the exact same signature as `select_kri_blocks`,
so it drops into `build_kri_mask(..., select_fn=select_lattice_blocks)` and
reuses the entire sink/recent/current-block/causal assembly and the causal
correctness tests. `LatticeConfig` subclasses `KRIConfig`, so it is accepted
anywhere a `KRIConfig` is.

Variants (cfg.variant):
  rel_only      score = rel                         (≈ KRI-Q, cos-only)
  orth_only     step 0 = rel, then score = novelty  (novelty after a seed)
  rel_orth      score = rel + lambda_orth*novelty   (the default)
  mmr           score = rel - lambda_redun*redund.  (maximal marginal relevance)
  residual_rel  score = cos(q_resid, k_centroid);   (explain-away query dirs)
                after each pick, q_resid -= proj onto the chosen key centroid

Block summary used for the basis/novelty/redundancy is configurable via
cfg.summary_mode ("k" | "v" | "kv_cat" | "kv_sum"); query relevance `rel`
is always cosine(q_t, k_centroid) regardless of summary_mode, per the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .kri_mask import KRIConfig, block_index_of, num_blocks, _block_summaries
from .canonical_kri import _causal_base_mask, _select_blocks_to_mask

_VARIANTS = ("rel_only", "orth_only", "rel_orth", "mmr", "residual_rel")


@dataclass
class LatticeConfig(KRIConfig):
    """KRIConfig plus the Lattice-KRI selection knobs.

    Inherits block_size / local_window_tokens / global_topk_blocks /
    protected_blocks / per_head / score_layer_index from KRIConfig, so the
    existing mask builder and accounting treat it as a KRIConfig.
    """

    variant: str = "rel_orth"
    lambda_orth: float = 0.25
    lambda_redun: float = 0.25  # only used by the mmr variant
    lambda_age: float = 0.0
    summary_mode: str = "k"  # "k" | "v" | "kv_cat" | "kv_sum"
    include_sink_recent_in_basis: bool = False
    basis_update_eps: float = 1e-6
    selection_mode: str = "greedy"  # "greedy" | "batch_topk_approx"


def _block_summaries_kv(
    k: torch.Tensor, v: torch.Tensor, block_size: int, summary_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x, kc) where kc is the L2-normalised key centroid [B,H,NB,D]
    (used for query relevance) and x is the L2-normalised block summary
    [B,H,NB,Dx] selected by summary_mode (used for novelty/redundancy).

    The centroid recipe reuses kri_mask._block_summaries (per-token L2
    normalise, mean, renormalise) so the key-centroid space is byte-identical
    to select_kri_blocks — that is what makes the rel_only variant collapse
    exactly onto cosine-only KRI-Q. The value centroid is the symmetric
    application of the same recipe to V."""
    kc = _block_summaries(k, k, block_size)[0]  # [B,H,NB,D] == KRI-Q centroid
    if summary_mode == "k":
        return kc, kc
    vc = _block_summaries(v, v, block_size)[0]  # symmetric value centroid
    if summary_mode == "v":
        return vc, kc
    if summary_mode == "kv_cat":
        return F.normalize(torch.cat([kc, vc], dim=-1), dim=-1), kc
    if summary_mode == "kv_sum":
        if kc.shape[-1] != vc.shape[-1]:
            raise ValueError("kv_sum needs matching key/value head dims")
        return F.normalize(kc + vc, dim=-1), kc
    raise ValueError(f"unknown summary_mode: {summary_mode}")


def _minmax_per_query(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalise the last dim to [0,1] (per query, over eligible
    blocks). Flat inputs map to all-zeros."""
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


def select_lattice_blocks(
    q_vec: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: LatticeConfig,
    t_query: int,
) -> torch.Tensor:
    """Greedy orthogonal-novelty selection of prefix blocks for query
    position `t_query`. Same contract as `select_kri_blocks`:

    Args:
        q_vec: [B,H,D] query at position t_query (same layer as k/v).
        k, v:  [B,H,T,D].
        cfg:   LatticeConfig.
        t_query: current query position.

    Returns:
        sel: [B,H,K] long block indices (K <= global_topk_blocks), excluding
             protected/sink blocks and the local window (added by the mask
             builder), or an empty [B,H,0] tensor if no prefix is eligible.
    """
    B, H, T, D = k.shape
    bs = cfg.block_size
    NB = num_blocks(T, bs)
    eps = cfg.basis_update_eps

    # Eligible prefix blocks: strictly before the local window, not protected.
    local_start = max(0, t_query - cfg.local_window_tokens)
    local_first_block = block_index_of(local_start, bs)
    eligible = torch.zeros(NB, dtype=torch.bool, device=k.device)
    eligible[:local_first_block] = True
    for p in cfg.protected_blocks:
        if 0 <= p < NB:
            eligible[p] = False
    elig_idx = eligible.nonzero(as_tuple=False).flatten()  # [NE]
    NE = int(elig_idx.numel())
    k_eff = min(cfg.global_topk_blocks, NE)
    if k_eff <= 0:
        return q_vec.new_zeros(B, H, 0, dtype=torch.long)

    x_all, kc_all = _block_summaries_kv(k, v, bs, cfg.summary_mode)
    xe = x_all[:, :, elig_idx, :]  # [B,H,NE,Dx]
    kce = kc_all[:, :, elig_idx, :]  # [B,H,NE,D]
    q_n = F.normalize(q_vec, dim=-1)  # [B,H,D]
    rel = _minmax_per_query(torch.einsum("bhd,bhnd->bhn", q_n, kce))  # [B,H,NE]

    # Optional age bonus: newer eligible block -> higher (normalised index).
    if cfg.lambda_age != 0.0 and NE > 1:
        age = torch.linspace(0.0, 1.0, NE, device=k.device, dtype=xe.dtype)
        age = age.view(1, 1, NE).expand(B, H, NE)
    else:
        age = None

    variant = cfg.variant
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant}; choices {_VARIANTS}")

    chosen = torch.zeros(B, H, NE, dtype=torch.bool, device=k.device)
    picks: List[torch.Tensor] = []
    basis: List[torch.Tensor] = []  # orthonormal vectors [B,H,Dx]
    sel_sum: List[torch.Tensor] = []  # selected summaries [B,H,Dx]
    q_resid = q_n.clone()  # for residual_rel
    neg_inf = torch.finfo(xe.dtype).min

    # Seed the basis with sink/recent block summaries if requested, so that
    # novelty is measured relative to the compulsory support too.
    if cfg.include_sink_recent_in_basis:
        seed_blocks = [p for p in cfg.protected_blocks if 0 <= p < NB]
        seed_blocks += list(
            range(local_first_block, min(NB, block_index_of(t_query, bs) + 1))
        )
        for b in sorted(set(seed_blocks)):
            e = _orthonormalize(x_all[:, :, b, :], basis, eps)
            if e is not None:
                basis.append(e)

    for step in range(k_eff):
        if variant == "residual_rel":
            score = torch.einsum("bhd,bhnd->bhn", F.normalize(q_resid, dim=-1), kce)
        else:
            novelty = _novelty(xe, basis)  # [B,H,NE] in [0,1]
            if variant == "rel_only":
                score = rel
            elif variant == "orth_only":
                score = rel if step == 0 else novelty
            elif variant == "rel_orth":
                score = rel + cfg.lambda_orth * novelty
            elif variant == "mmr":
                score = rel - cfg.lambda_redun * _redundancy(xe, sel_sum)
            if age is not None:
                score = score + cfg.lambda_age * age

        score = score.masked_fill(chosen, neg_inf)
        pick = score.argmax(dim=-1)  # [B,H]
        chosen.scatter_(-1, pick.unsqueeze(-1), True)
        picks.append(pick)

        x_pick = torch.gather(
            xe, 2, pick.view(B, H, 1, 1).expand(B, H, 1, xe.shape[-1])
        ).squeeze(
            2
        )  # [B,H,Dx]
        sel_sum.append(x_pick)
        e = _orthonormalize(x_pick, basis, eps)
        if e is not None:
            basis.append(e)
        if variant == "residual_rel":
            kc_pick = torch.gather(
                kce, 2, pick.view(B, H, 1, 1).expand(B, H, 1, D)
            ).squeeze(
                2
            )  # [B,H,D]
            coeff = (q_resid * kc_pick).sum(-1, keepdim=True)
            q_resid = q_resid - coeff * kc_pick

    sel_ne = torch.stack(picks, dim=-1)  # [B,H,k_eff] into NE
    return elig_idx[sel_ne]  # -> block indices


def _novelty(xe: torch.Tensor, basis: List[torch.Tensor]) -> torch.Tensor:
    """orth_norm = ||x - proj_basis(x)|| / (||x|| + eps), per eligible block.
    Empty basis -> fully novel (1.0)."""
    if not basis:
        return xe.new_ones(xe.shape[:-1])
    Bmat = torch.stack(basis, dim=2)  # [B,H,S,Dx]
    coeff = torch.einsum("bhnd,bhsd->bhns", xe, Bmat)
    proj = torch.einsum("bhns,bhsd->bhnd", coeff, Bmat)
    resid = xe - proj
    return resid.norm(dim=-1) / (xe.norm(dim=-1) + 1e-6)


def _redundancy(xe: torch.Tensor, sel_sum: List[torch.Tensor]) -> torch.Tensor:
    """max cosine(x_b, selected summaries), clamped >=0; 0 if none selected."""
    if not sel_sum:
        return xe.new_zeros(xe.shape[:-1])
    S = torch.stack(sel_sum, dim=2)  # [B,H,Ns,Dx]
    sim = torch.einsum("bhnd,bhsd->bhns", xe, S)
    return sim.max(dim=-1).values.clamp(min=0.0)


def _orthonormalize(
    x: torch.Tensor, basis: List[torch.Tensor], eps: float
) -> Optional[torch.Tensor]:
    """Gram-Schmidt: residual of x [B,H,Dx] against the orthonormal basis,
    normalised. Returns None where the residual is degenerate (< eps)."""
    r = x
    for e in basis:
        coeff = (x * e).sum(-1, keepdim=True)
        r = r - coeff * e
    nrm = r.norm(dim=-1, keepdim=True)
    if float(nrm.max()) < eps:
        return None
    return r / (nrm + eps)


# ---------------------------------------------------------------------------
# Canonical-registry adapters (eval_canonical_kri-style, last-position probe).
# The primary apples-to-apples comparison uses the per-position path via
# build_kri_mask(select_fn=select_lattice_blocks); these adapters expose the
# same router on the cheaper canonical surface for cross-checking vs kri_q.
# ---------------------------------------------------------------------------


def _make_lattice_canonical(variant: str):
    def _mask(
        k_per_layer,
        v_per_layer,
        q_per_layer,
        seq_len,
        block_size,
        local_window_tokens,
        sink_blocks,
        topk_blocks,
        score_layer_index=0,
        device=None,
        lattice_cfg: Optional[LatticeConfig] = None,
    ):
        if device is None:
            device = k_per_layer[0].device
        L = min(score_layer_index, len(k_per_layer) - 1)
        k, v, q = k_per_layer[L], v_per_layer[L], q_per_layer[L]
        B, H, T, D = k.shape
        base, causal = _causal_base_mask(
            seq_len, block_size, local_window_tokens, sink_blocks, B, H, device
        )
        cfg = LatticeConfig(**(lattice_cfg.__dict__ if lattice_cfg else {}))
        cfg.variant = variant
        cfg.block_size = block_size
        cfg.local_window_tokens = local_window_tokens
        cfg.global_topk_blocks = topk_blocks
        cfg.protected_blocks = tuple(range(sink_blocks))
        sel = select_lattice_blocks(q[:, :, -1, :], k, v, cfg, t_query=seq_len - 1)
        if sel.numel() == 0:
            return base
        return _select_blocks_to_mask(sel, T, block_size, B, H, base, causal)

    return _mask


LATTICE_ROUTERS = {f"lattice_kri_{v}": _make_lattice_canonical(v) for v in _VARIANTS}
# Canonical alias requested by the task: --router lattice_kri == the default.
LATTICE_ROUTERS["lattice_kri"] = _make_lattice_canonical("rel_orth")
