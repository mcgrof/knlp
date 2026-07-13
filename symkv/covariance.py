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
        self.C = None      # HxH accumulator, lazily placed on the input's device
        self.n = 0         # number of length-H column samples seen

    @torch.no_grad()
    def update(self, X: torch.Tensor):
        """X: (..., H, D) -- any leading batch/token dims. Adds sum over all
        leading dims of X X^T and counts D columns per (leading) matrix.

        The contraction output is only HxH, so accumulate on the input's device
        in float32 (fast on ROCm; the HxH einsum keeps enough precision for the
        eigenvectors over ~1e5-1e6 samples). covariance() promotes to float64 on
        CPU for the eigendecomposition."""
        assert X.shape[-2] == self.n_heads, f"expected H={self.n_heads}, got {X.shape[-2]}"
        Xf = X.reshape(-1, X.shape[-2], X.shape[-1]).float()  # (N, H, D) on X.device
        c = torch.einsum("nhd,ngd->hg", Xf, Xf)
        if self.C is None:
            self.C = torch.zeros(self.n_heads, self.n_heads, dtype=torch.float32, device=Xf.device)
        self.C += c
        self.n += Xf.shape[0] * Xf.shape[-1]

    def covariance(self) -> torch.Tensor:
        assert self.n > 0, "no samples accumulated"
        return (self.C.double().cpu()) / self.n

    def merge(self, other: "HeadCovariance"):
        assert other.n_heads == self.n_heads
        if other.C is not None:
            oc = other.C.to(self.C.device) if self.C is not None else other.C.clone()
            self.C = oc if self.C is None else self.C + oc
        self.n += other.n
        return self
