# SPDX-License-Identifier: GPL-2.0
"""Fitting a source-to-target cache map from streamed sufficient statistics.

The mapper is per target (layer, head, tensor kind): a ridge regression from a
selected set of source layers onto that target block, with the key maps fitted
in the de-rotated content frame and re-rotated at the target position when the
cache is written.

Two things make this cheap enough to do exhaustively.

Sufficient statistics rather than an observation matrix. One pass over the
calibration prompts accumulates the source Gram matrix and the source-target
cross-products; nothing keeps the tokens. Memory is set by the number of source
features, not by the number of calibration tokens, so the calibration set can
grow without the fit getting harder to hold.

And the Gram matrix does not depend on the target. Source-layer selection is
therefore a sub-solve of statistics already accumulated: choosing which source
layers feed a target block costs a slice and a small solve, not a refit. That
turns "compare full fan-in against head-local support over a selected layer
set" from a sweep into an enumeration.

Keys and values get separate maps and separate verdicts throughout. A value map
that works cannot excuse a key map that does not, and the two fail differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch


@dataclass
class SourceLayout:
    """The full source feature space: every source layer and every source head.

    Accumulation always covers the whole space, and the head arm is expressed
    as a *column selection* rather than as a different accumulation. That way
    head-local and full fan-in are two reads of one pass over the calibration
    data instead of two passes, and they are guaranteed to have been fitted on
    byte-identical statistics.
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int

    def feature_slice(self, layer: int, head: int) -> slice:
        per_layer = self.n_kv_heads * self.head_dim
        off = layer * per_layer + head * self.head_dim
        return slice(off, off + self.head_dim)

    def columns_for(
        self, layers: Sequence[int], head: int, head_local: bool
    ) -> torch.Tensor:
        """Columns feeding one target head.

        ``head_local`` takes only the source head with the same index, which is
        well defined here because both models carry the same number of key/value
        heads. ``full fan-in`` takes every source head.
        """
        if head_local and head >= self.n_kv_heads:
            raise ValueError(
                f"head-local support asked for source head {head}, but the "
                f"source has only {self.n_kv_heads} key/value heads. Head "
                "correspondence is only defined when both models carry the "
                "same number; use full fan-in for a pair that does not."
            )
        idx = []
        heads = [head] if head_local else list(range(self.n_kv_heads))
        for li in layers:
            for h in heads:
                sl = self.feature_slice(li, h)
                idx.extend(range(sl.start, sl.stop))
        return torch.tensor(idx, dtype=torch.long)

    @property
    def n_features(self) -> int:
        return self.n_layers * self.n_kv_heads * self.head_dim


class Accumulator:
    """Streams ``X'X``, ``X'Y`` and the means for one tensor kind.

    ``X`` is the source features for a token, ``Y`` the target block for that
    same token. The intercept is handled by centring rather than by an appended
    column, so the Gram matrix stays the same size whichever way the fit is
    later regularised.
    """

    def __init__(
        self, layout: SourceLayout, n_targets: int, target_dim: int, device="cpu"
    ):
        self.layout = layout
        d = layout.n_features
        self.d = d
        self.xtx = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.xty = torch.zeros(
            n_targets, d, target_dim, dtype=torch.float64, device=device
        )
        self.xsum = torch.zeros(d, dtype=torch.float64, device=device)
        self.ysum = torch.zeros(
            n_targets, target_dim, dtype=torch.float64, device=device
        )
        self.n = 0

    @torch.no_grad()
    def add(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """``X`` is ``[T, d]``; ``Y`` is ``[n_targets, T, target_dim]``."""
        Xd = X.to(self.xtx.dtype).to(self.xtx.device)
        Yd = Y.to(self.xtx.dtype).to(self.xtx.device)
        self.xtx += Xd.transpose(0, 1) @ Xd
        self.xty += torch.einsum("td,nte->nde", Xd, Yd)
        self.xsum += Xd.sum(0)
        self.ysum += Yd.sum(1)
        self.n += Xd.shape[0]

    def centred(self, cols: torch.Tensor, target: int):
        """Centred ``(G, C, xbar, ybar)`` restricted to ``cols`` for one target."""
        n = float(self.n)
        xb = self.xsum[cols] / n
        yb = self.ysum[target] / n
        G = self.xtx[cols][:, cols] - n * torch.outer(xb, xb)
        C = self.xty[target][cols] - n * torch.outer(xb, yb)
        return G, C, xb, yb


@dataclass
class AffineMap:
    """One fitted block map, with the provenance a byte count needs."""

    M: torch.Tensor  # [d_sel, target_dim]
    b: torch.Tensor  # [target_dim]
    layers: tuple
    head: int
    kind: str
    target_layer: int
    ridge: float
    head_local: bool
    n_calib_tokens: int
    info: dict = field(default_factory=dict)

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the map at the *input's* precision, not the solver's.

        The fit is solved in float64 because a Gram matrix is ill-conditioned
        and the solve happens once. Applying the result is an ordinary matrix
        multiply that happens on every prefix, and doing it in float64 costs
        roughly thirty times a float32 one on hardware that deprioritises
        double precision. Measured on an A6000 with a 28-layer 4-head target,
        a float64 apply took 3.15 times a native prefill -- turning a method
        that beats re-prefilling into one that loses to it, for no numerical
        benefit at all.
        """
        w = self.M.to(X.dtype)
        return X @ w + self.b.to(X.dtype)

    def to(self, dtype: torch.dtype) -> "AffineMap":
        """Cast the stored map, so the precision it is kept at is explicit."""
        self.M = self.M.to(dtype)
        self.b = self.b.to(dtype)
        return self

    @property
    def n_params(self) -> int:
        return self.M.numel() + self.b.numel()


def solve_ridge(G: torch.Tensor, C: torch.Tensor, ridge: float) -> torch.Tensor:
    """``(G + ridge*I)^-1 C``, by Cholesky with an escape to a pseudo-inverse."""
    d = G.shape[0]
    A = G + ridge * torch.eye(d, dtype=G.dtype, device=G.device)
    try:
        return torch.cholesky_solve(C, torch.linalg.cholesky(A))
    except Exception:  # noqa: BLE001
        return torch.linalg.lstsq(A, C).solution


def fit_block(
    acc: Accumulator,
    *,
    target: int,
    target_layer: int,
    head: int,
    kind: str,
    layers: Sequence[int],
    ridge: float,
    head_local: bool,
) -> AffineMap:
    cols = acc.layout.columns_for(layers, head, head_local)
    G, C, xb, yb = acc.centred(cols, target)
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
        n_calib_tokens=acc.n,
    )


def source_layer_scores(
    acc: Accumulator, target: int, head: int, ridge: float, head_local: bool
) -> list:
    """Residual left by each single source layer, for top-k selection.

    Cheap because the statistics are already accumulated: each candidate is one
    small solve on a slice of the Gram matrix rather than another pass over the
    calibration data.
    """
    out = []
    for li in range(acc.layout.n_layers):
        cols = acc.layout.columns_for([li], head, head_local)
        G, C, _, _ = acc.centred(cols, target)
        M = solve_ridge(G, C, ridge)
        # residual sum of squares up to the target-variance constant
        r = -2.0 * float((M * C).sum()) + float(
            (M.transpose(0, 1) @ G @ M).diagonal().sum()
        )
        out.append((li, r))
    out.sort(key=lambda x: x[1])
    return out


def select_layers(
    acc: Accumulator, target: int, head: int, ridge: float, k: int, head_local: bool
) -> tuple:
    """Top-``k`` source layers for one target block, by single-layer residual.

    Returned in depth order rather than score order, so a selection reads as a
    span of the source stack and two selections can be compared by eye.
    """
    scored = source_layer_scores(acc, target, head, ridge, head_local)
    return tuple(sorted(li for li, _ in scored[:k]))
