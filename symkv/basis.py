# SPDX-License-Identifier: GPL-2.0
"""SymKV head-axis bases.

All bases are H x m column-orthonormal matrices B (B^T B = I_m) used to project the
head axis: Z = B^T X, X_hat = B Z. The methods differ only in how the m columns are
chosen:

  full          -- B = I_H (m must equal H); exact, the identity control.
  mean_only     -- B = [u0]; m=1, keep only the consensus (mean-of-heads) mode.
  pca_head      -- top-m eigenvectors of the head second moment C (generic PCA).
                   By Eckart-Young this MINIMIZES raw reconstruction MSE among all
                   rank-m bases -- the ceiling SymKV must be measured against.
  symkv_raw     -- [u0, top-(m-1) eigenvectors of P_perp C P_perp], P_perp = I - u0 u0^T.
                   Forces the consensus mode as mode 0, then the best complement
                   directions orthogonal to it. Provably >= pca_head raw MSE; the
                   hypothesis is that it can still win on PREDICTIVE quality.
  random_sym    -- [u0, m-1 random directions in u0's complement], orthonormalized.
                   The honesty control: any "symmetry helps" claim must beat this,
                   not just PCA, so the win is attributable to structure not to
                   merely reserving a mode for the mean.
  grouped_mean  -- block-consensus: partition H heads into m groups, each column the
                   normalized indicator of a group (group-average pooling). A cheap
                   structured baseline with no covariance fit at all.

u0 = 1/sqrt(H) * ones(H): the direction along which all heads contribute equally
(consensus). Nothing physical is implied by the name.
"""

from __future__ import annotations

import torch


def consensus_mode(n_heads: int, dtype=torch.float64) -> torch.Tensor:
    """u0 = ones(H)/sqrt(H), unit-norm consensus/mean-of-heads direction."""
    u0 = torch.ones(n_heads, dtype=dtype)
    return u0 / torch.linalg.norm(u0)


def perp_projector(u0: torch.Tensor) -> torch.Tensor:
    """P_perp = I - u0 u0^T, orthogonal projector off the consensus mode."""
    H = u0.shape[0]
    return torch.eye(H, dtype=u0.dtype) - torch.outer(u0, u0)


def _sym_eig_desc(A: torch.Tensor):
    """Eigenpairs of a symmetric matrix, eigenvalues descending."""
    A = 0.5 * (A + A.transpose(-1, -2))  # symmetrize numerical drift
    w, V = torch.linalg.eigh(A)
    idx = torch.argsort(w, descending=True)
    return w[idx], V[:, idx]


def _orthonormalize(B: torch.Tensor) -> torch.Tensor:
    """QR-based column orthonormalization preserving column 0's direction sign."""
    Q, R = torch.linalg.qr(B)
    # fix signs so Q[:,i] keeps the orientation of B[:,i]
    signs = torch.sign(torch.diagonal(R))
    signs[signs == 0] = 1.0
    return Q * signs


def build_basis(method: str, n_heads: int, n_modes: int,
                C: torch.Tensor | None = None, seed: int = 0,
                dtype=torch.float64) -> torch.Tensor:
    """Return an H x m column-orthonormal basis for the given method."""
    H, m = n_heads, n_modes
    if method == "full":
        assert m == H, "full basis requires m == H"
        return torch.eye(H, dtype=dtype)

    u0 = consensus_mode(H, dtype)

    if method == "mean_only":
        assert m == 1, "mean_only is m == 1"
        return u0.reshape(H, 1)

    if method == "pca_head":
        assert C is not None, "pca_head needs the covariance C"
        _, V = _sym_eig_desc(C.to(dtype))
        return V[:, :m].contiguous()

    if method == "symkv_raw":
        assert C is not None, "symkv_raw needs the covariance C"
        if m == 1:
            return u0.reshape(H, 1)
        Pp = perp_projector(u0)
        Cp = Pp @ C.to(dtype) @ Pp
        _, V = _sym_eig_desc(Cp)
        # top (m-1) complement eigenvectors; drop any that collapsed onto u0
        comp = V[:, : m - 1]
        B = torch.cat([u0.reshape(H, 1), comp], dim=1)
        return _orthonormalize(B)

    if method == "random_sym":
        g = torch.Generator().manual_seed(seed)
        if m == 1:
            return u0.reshape(H, 1)
        Pp = perp_projector(u0)
        R = torch.randn(H, m - 1, generator=g, dtype=dtype)
        R = Pp @ R  # push into u0's complement
        B = torch.cat([u0.reshape(H, 1), R], dim=1)
        return _orthonormalize(B)

    if method == "grouped_mean":
        # partition heads into m contiguous groups; column = normalized indicator
        B = torch.zeros(H, m, dtype=dtype)
        edges = torch.linspace(0, H, m + 1).round().to(torch.long)
        for j in range(m):
            a, b = int(edges[j]), int(edges[j + 1])
            if b <= a:
                b = a + 1
            B[a:b, j] = 1.0
            B[:, j] /= torch.linalg.norm(B[:, j])
        return B

    raise ValueError(f"unknown method {method}")
