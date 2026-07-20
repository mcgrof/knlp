"""P0: one selector interface for the lattice/KRI line.

Decomposes every router we have into  query-probe x selection-operator, so the
probe (last-K / recent-Q / pre-RoPE) and the operator (top-K relevance, leaky
orthogonal residual, mass) are independent axes instead of a pile of named
cocktails. All operators consume the SAME per-head inputs:

    q0   [H, D]        one query probe per (kv-)head (already normalized upstream
                       or normalized here), whatever probe the caller chose
    kc   [H, NB, D]    L2-normalized block key-centroids
    mass [H, NB]       accumulated attention mass per block (for the H2O operator)

and return, per head, an ORDERED list of selected block indices (best-first),
length min(K, NB). Ordered selections are logged, not just final metrics, so the
equivalence tests can assert block-for-block identity.

The load-bearing operator is `leaky_residual(alpha)`:
    alpha = 0  -> plain top-K relevance == recent-Q top-K   (exact)
    alpha = 1  -> exact-span orthogonal residual            (the intended
                  "residual_rel", using a modified-Gram-Schmidt orthonormal
                  basis of selected centroids, not repeated in-place projection)
    0<alpha<1  -> relevant directions decay gradually instead of vanishing after
                  one pick.

This is the ChatGPT-order-of-actions P0 core; equivalence proofs live in
tests/test_selectors_equiv.py.
"""

from __future__ import annotations

from typing import List, Optional

import torch

_NEG_INF = float("-inf")


def _normrows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def leaky_residual(
    q0: torch.Tensor,  # [H, D]
    kc: torch.Tensor,  # [H, NB, D], expected L2-normalized rows
    K: int,
    alpha: float,
    eps: float = 1e-8,
    log: Optional[list] = None,
    trace: Optional[list] = None,
    ans: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Greedy leaky-orthogonal selection, vectorized over heads.

    Returns [H, K'] long tensor of selected block indices (K' = min(K, NB)),
    best-first. Reduces to plain recent-Q top-K at alpha=0 and to exact-span
    orthogonal residual at alpha=1.

    `log` (list): append per-step picks `{"step", "pick"}`.
    `trace` (list) + `ans` (1-D LongTensor of answer/needle block indices, shared
      across heads): the P1 retrieval-failure diagnostic. Per step, before the pick
      is made, append per-head: `ans_rank` (rank of the best still-unchosen answer
      block under the CURRENT residual query, 0=top), `q_cos` (cos of the residual
      query to q0), `resid_ratio` (‖q0 − α·U Uᵀq0‖ / ‖q0‖), and the pick. This
      records how the answer's rank moves as orthogonal explain-away deflates the
      query — the mechanism test for whether hard projection kills retrieval.
    """
    H, NB, D = kc.shape
    Kk = min(K, NB)
    q0 = _normrows(q0.float())
    kcf = kc.float()
    device = kc.device
    if ans is not None:
        ans = torch.as_tensor(ans, device=device, dtype=torch.long).view(-1)

    chosen = torch.zeros(H, NB, dtype=torch.bool, device=device)
    basis: List[torch.Tensor] = []  # each [H, D], orthonormal per head
    picks: List[torch.Tensor] = []

    for _ in range(Kk):
        if basis and alpha != 0.0:
            # explained component of q0 in the span of selected centroids
            explained = torch.zeros_like(q0)
            for e in basis:
                explained = explained + (q0 * e).sum(-1, keepdim=True) * e
            resid = q0 - alpha * explained
            q_score = _normrows(resid)
            # where residual is numerically exhausted, fall back to q0
            exhausted = resid.norm(dim=-1) <= eps
            if exhausted.any():
                q_score[exhausted] = q0[exhausted]
        else:
            resid = q0
            q_score = q0

        scores = torch.einsum("hd,hnd->hn", q_score, kcf)  # [H, NB]
        scores = scores.masked_fill(chosen, _NEG_INF)
        if trace is not None and ans is not None:
            # per-needle rank [H, A]: for each answer block a, how many blocks
            # score strictly higher under the current residual query (0 = top).
            ans_sc = scores[:, ans]  # [H, A]
            ans_rank = (scores.unsqueeze(1) > ans_sc.unsqueeze(2)).sum(dim=2)  # [H, A]
            chosen_ans = chosen[:, ans]  # [H, A] which needle blocks already picked
            q_cos = (q_score * q0).sum(-1)  # [H]
            resid_ratio = resid.norm(dim=-1)  # [H] (q0 is unit-norm)
            trace.append(
                {
                    "step": len(picks),
                    "ans_rank": ans_rank.tolist(),  # [H][A]
                    "chosen_ans": chosen_ans.tolist(),  # [H][A]
                    "q_cos": [round(x, 4) for x in q_cos.tolist()],
                    "resid_ratio": [round(x, 4) for x in resid_ratio.tolist()],
                }
            )
        pick = scores.argmax(dim=-1)  # [H]
        picks.append(pick)
        chosen.scatter_(-1, pick.unsqueeze(-1), True)
        if trace is not None and ans is not None:
            trace[-1]["pick"] = pick.tolist()
        if log is not None:
            log.append({"step": len(picks) - 1, "pick": pick.tolist()})

        if alpha != 0.0:
            # add the picked centroid to each head's orthonormal basis (MGS)
            u = torch.gather(kcf, 1, pick.view(H, 1, 1).expand(H, 1, D)).squeeze(1)
            for e in basis:
                u = u - (u * e).sum(-1, keepdim=True) * e
            un = u.norm(dim=-1, keepdim=True)
            u = torch.where(un > eps, u / (un + eps), torch.zeros_like(u))
            basis.append(u)

    return torch.stack(picks, dim=-1)  # [H, Kk]


def top_k_relevance(q0: torch.Tensor, kc: torch.Tensor, K: int) -> torch.Tensor:
    """Plain top-K cosine relevance (== leaky_residual at alpha=0, == rel_only,
    == cosine KRI-Q under the same probe)."""
    scores = torch.einsum("hd,hnd->hn", _normrows(q0.float()), kc.float())
    Kk = min(K, kc.shape[1])
    return scores.topk(Kk, dim=-1).indices  # [H, Kk], best-first


def top_k_mass(mass: torch.Tensor, K: int) -> torch.Tensor:
    """H2O: top-K blocks by accumulated attention mass."""
    Kk = min(K, mass.shape[1])
    return mass.float().topk(Kk, dim=-1).indices


def position_matched_random(
    H: int, NB: int, K: int, gen: torch.Generator
) -> torch.Tensor:
    """Position-matched random control: K distinct random blocks per head, drawn
    from the SAME block index range every operator sees (so a NIAH win cannot be
    attributed to relevance when the selector merely kept favorable positions)."""
    Kk = min(K, NB)
    out = torch.empty(H, Kk, dtype=torch.long)
    for h in range(H):
        out[h] = torch.randperm(NB, generator=gen)[:Kk]
    return out


# Operator registry: name -> callable(q0, kc, mass, K, **params) -> [H, K'] picks.
def select(
    operator: str,
    K: int,
    *,
    q0=None,
    kc=None,
    mass=None,
    alpha: float = 1.0,
    gen: Optional[torch.Generator] = None,
    log: Optional[list] = None,
) -> torch.Tensor:
    if operator in ("rel_only", "kri_q", "relq_recent", "top_k_relevance"):
        return top_k_relevance(q0, kc, K)
    if operator in ("residual_rel", "leaky_residual"):
        return leaky_residual(q0, kc, K, alpha=alpha, log=log)
    if operator in ("h2o", "h2o_true", "mass"):
        return top_k_mass(mass, K)
    if operator in ("random", "position_matched_random"):
        H, NB = kc.shape[0], kc.shape[1]
        return position_matched_random(H, NB, K, gen)
    raise ValueError(f"unknown operator {operator!r}")
