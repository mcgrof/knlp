# SPDX-License-Identifier: GPL-2.0
"""Streaming head-axis second-moment accumulator for SymKV.

For each token we have X in R^(H x D) (H KV heads stacked, each D-dim). Treat the
D feature columns of X as D samples of a length-H vector; the head-axis uncentered
second moment is C = E[x x^T] in R^(H x H), estimated by summing X @ X^T over
tokens and dividing by the total column count (tokens * D). Uncentered (raw second
moment, not mean-subtracted) because the consensus mode u0 lives in the mean and
SymKV wants to model it explicitly, not remove it.

Accumulated in float64 for numerical stability regardless of the model dtype.
"""

from __future__ import annotations

import torch


class HeadCovariance:
    def __init__(self, n_heads: int):
        self.n_heads = n_heads
        self.C = torch.zeros(n_heads, n_heads, dtype=torch.float64)
        self.n = 0  # number of length-H column samples seen

    @torch.no_grad()
    def update(self, X: torch.Tensor):
        """X: (..., H, D) -- any leading batch/token dims. Adds sum over all
        leading dims of X X^T and counts D columns per (leading) matrix."""
        assert X.shape[-2] == self.n_heads, f"expected H={self.n_heads}, got {X.shape[-2]}"
        Xf = X.reshape(-1, X.shape[-2], X.shape[-1]).to(torch.float64)  # (N, H, D)
        # sum_n X_n @ X_n^T  == einsum over the batch and D axes
        self.C += torch.einsum("nhd,ngd->hg", Xf, Xf)
        self.n += Xf.shape[0] * Xf.shape[-1]

    def covariance(self) -> torch.Tensor:
        assert self.n > 0, "no samples accumulated"
        return self.C / self.n

    def merge(self, other: "HeadCovariance"):
        assert other.n_heads == self.n_heads
        self.C += other.C
        self.n += other.n
        return self
