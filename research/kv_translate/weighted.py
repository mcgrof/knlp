# SPDX-License-Identifier: GPL-2.0
"""Receiver-weighted calibration for a fixed mapper.

A2 changes one thing about A1 and nothing else: which calibration tokens the
ridge fit tries hardest to get right. The support, the source-layer selection,
the regularisation grid, the mapper's byte count and the code that applies it
online all stay as A1 left them, so any difference is attributable to the
weighting rather than to a bigger or better-tuned map.

The weight is the attention mass a cached position actually receives in the
target model's own prefill. A position every later token looks at is one whose
translation error propagates; a position nothing attends to can be wrong for
free. This is available at calibration time without a backward pass, which is
what keeps it deployable.

The exponent matters and is not arbitrary. An attention output is ``sum_t a_t
v_t``, so for errors that are uncorrelated across positions the expected squared
output error is ``sum_t a_t^2 ||e_t||^2`` -- the square is the principled choice
for values. Keys act through the logit rather than the sum, so the right
exponent there is less obvious and both are scored, with the choice made on a
development split and disclosed rather than quietly tuned.

Weights are normalised to mean one within each target block. Without that the
effective regularisation would move with the weighting and A1's ridge grid
would no longer mean the same thing, which would break the very comparison this
file exists to make.

The memory trick that makes this affordable: A1 has already chosen which source
columns feed each target block, so the weighted Gram matrices are built on the
selected columns only. At the selected width that is megabytes per block rather
than the gigabytes a weighted version of the full feature space would need.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from research.kv_translate.fit import AffineMap, SourceLayout, solve_ridge


def attention_mass(
    attn_layer: torch.Tensor, n_kv_heads: int, n_positions: int
) -> torch.Tensor:
    """Per-(kv head, position) mass received during the prompt's own prefill.

    ``attn_layer`` is ``[B, n_q, T, T]``. Summing over the query axis gives how
    much total attention each cached position receives; query heads are folded
    into their key/value group because that is the granularity a cache block
    has.
    """
    a = attn_layer.float()[0].sum(dim=1)  # [n_q, T_kv]
    if a.shape[0] != n_kv_heads:
        grp = a.shape[0] // n_kv_heads
        a = a.reshape(n_kv_heads, grp, a.shape[-1]).mean(dim=1)
    return a[:, :n_positions]


def normalise(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean one, so the ridge grid keeps meaning what it meant unweighted."""
    return w / w.mean(dim=-1, keepdim=True).clamp(min=eps)


class WeightedAccumulator:
    """Per-target weighted sufficient statistics on a fixed column selection.

    One accumulator per target block, each holding only the columns A1 selected
    for it. Nothing here re-decides support; that would make the comparison
    against A1 a comparison of two different mappers.
    """

    def __init__(self, cols: torch.Tensor, target_dim: int, device="cpu"):
        d = int(cols.numel())
        self.cols = cols
        self.xtx = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.xty = torch.zeros(d, target_dim, dtype=torch.float64, device=device)
        self.xsum = torch.zeros(d, dtype=torch.float64, device=device)
        self.ysum = torch.zeros(target_dim, dtype=torch.float64, device=device)
        self.wsum = 0.0

    @torch.no_grad()
    def add(self, X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor) -> None:
        """``X`` is ``[T, d_full]``, ``Y`` is ``[T, target_dim]``, ``w`` is ``[T]``."""
        Xs = X[:, self.cols.to(X.device)].to(self.xtx.dtype).to(self.xtx.device)
        Yd = Y.to(self.xtx.dtype).to(self.xtx.device)
        wd = w.to(self.xtx.dtype).to(self.xtx.device).unsqueeze(1)
        Xw = Xs * wd
        self.xtx += Xs.transpose(0, 1) @ Xw
        self.xty += Xw.transpose(0, 1) @ Yd
        self.xsum += Xw.sum(0)
        self.ysum += (Yd * wd).sum(0)
        self.wsum += float(wd.sum())

    def centred(self):
        """Weighted-centred ``(G, C, xbar, ybar)``.

        Centring uses the weighted means, which is what makes the intercept the
        optimum of the weighted problem rather than of the unweighted one.
        """
        n = max(self.wsum, 1e-30)
        xb = self.xsum / n
        yb = self.ysum / n
        G = self.xtx - n * torch.outer(xb, xb)
        C = self.xty - n * torch.outer(xb, yb)
        return G, C, xb, yb


def fit_weighted(
    acc: WeightedAccumulator,
    *,
    target_layer: int,
    head: int,
    kind: str,
    layers: Sequence[int],
    ridge: float,
    head_local: bool,
) -> AffineMap:
    G, C, xb, yb = acc.centred()
    M = solve_ridge(G, C, ridge)
    b = yb - M.transpose(0, 1) @ xb
    return AffineMap(
        M=M,
        b=b,
        layers=tuple(int(x) for x in layers),
        head=head,
        kind=kind,
        target_layer=target_layer,
        ridge=float(ridge),
        head_local=bool(head_local),
        n_calib_tokens=int(acc.wsum),
    )
