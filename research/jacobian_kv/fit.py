# SPDX-License-Identifier: GPL-2.0
"""Affine mappers for receiver-weighted KV fitting.

This module is the linear algebra the Jacobian-KV screen rests on: fit an
affine map from a source cache block to a target cache block under an optional
receiver-side output metric, with optional ridge regularization, a rank
constraint, or a sample-dependent metric.

The load-bearing fact, and the reason ``solver="kron"`` exists at all: a
*fixed*, *invertible* output metric applied to every sample cannot change an
unconstrained affine least-squares solution.  The general solver here does not
assume that collapse, so ``tests/jacobian_kv/test_linearization.py`` can
demonstrate it on real numerics instead of restating it.  A receiver metric can
only change the answer through a sample-dependent weighting, a rank or bit
constraint, a regularizer it does not commute with, or some other restricted
mapper class.  An implementation that reports a win from a fixed full-rank
metric on an unconstrained fit is reporting a bug.

Convention throughout: observations are rows.  ``C`` is ``[N, ds]``, ``T`` is
``[N, dt]``, the map is ``T_hat = C @ M + b`` with ``M`` of shape ``[ds, dt]``
and ``b`` of shape ``[dt]``.  The per-sample error is the column vector
``e_x = M.T @ c_x + b - t_x`` and the objective is
``sum_x w_x * e_x.T @ G_x @ e_x + ridge * ||M||_F^2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

# --------------------------------------------------------------------------
# column-major vec helpers
#
# The Kronecker identity vec(A X B) = (B.T kron A) vec(X) holds for
# column-major vec.  torch reshapes row-major, so go through a transpose.
# --------------------------------------------------------------------------


def _vec(M: torch.Tensor) -> torch.Tensor:
    """Column-major vectorisation of ``[ds, dt]`` into ``[ds*dt]``."""
    return M.transpose(-2, -1).reshape(-1)


def _unvec(v: torch.Tensor, ds: int, dt: int) -> torch.Tensor:
    """Inverse of :func:`_vec`."""
    return v.reshape(dt, ds).transpose(0, 1)


def _symmetrize(G: torch.Tensor) -> torch.Tensor:
    return 0.5 * (G + G.transpose(-2, -1))


def psd_sqrt(G: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Symmetric PSD square root ``S`` with ``S @ S == G``.

    Eigenvalues below ``eps`` are clamped to zero, so a rank-deficient metric
    (the categorical Fisher ``diag(p) - p p^T`` always is, it annihilates the
    all-ones direction) produces a rank-deficient root rather than NaNs.
    """
    G = _symmetrize(G.to(torch.float64))
    evals, evecs = torch.linalg.eigh(G)
    evals = torch.clamp(evals, min=eps)
    return (evecs * evals.clamp(min=0.0).sqrt()) @ evecs.transpose(-2, -1)


def psd_inv_sqrt(G: torch.Tensor, rcond: float = 1e-12) -> torch.Tensor:
    """Pseudo-inverse of :func:`psd_sqrt`, for undoing a whitening."""
    G = _symmetrize(G.to(torch.float64))
    evals, evecs = torch.linalg.eigh(G)
    cutoff = rcond * float(evals.abs().max().clamp(min=1e-300))
    inv_root = torch.where(
        evals > cutoff, evals.clamp(min=cutoff).rsqrt(), torch.zeros_like(evals)
    )
    return (evecs * inv_root) @ evecs.transpose(-2, -1)


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------


def _min_norm_lstsq(
    A: torch.Tensor, b: torch.Tensor, rcond: float = 1e-10
) -> torch.Tensor:
    """Minimum-norm least-squares solution of ``A x = b``, via SVD.

    ``b`` may be a vector or a matrix of right-hand sides.

    ``torch.linalg.lstsq`` defaults to a pivoted-QR driver whose rank cutoff is
    not tight enough here, and the systems this module builds are *expected* to
    be rank deficient: the categorical Fisher ``diag(p) - p p^T`` annihilates
    the all-ones direction, so any metric built from it is singular by
    construction.  Solving by SVD with an explicit cutoff makes the returned
    solution the well-defined minimum-norm one on every device and driver,
    which is what lets the null-space behaviour be asserted rather than
    tolerated.
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    cutoff = rcond * S.max().clamp(min=1e-300)
    S_inv = torch.where(S > cutoff, S.reciprocal(), torch.zeros_like(S))
    proj = U.transpose(-2, -1) @ b
    scale = S_inv if b.ndim == 1 else S_inv.unsqueeze(-1)
    return Vh.transpose(-2, -1) @ (scale * proj)


@dataclass
class AffineMap:
    """A fitted affine map plus enough provenance to audit the fit."""

    M: torch.Tensor
    b: torch.Tensor
    solver: str
    rank: Optional[int] = None
    ridge: float = 0.0
    fit_intercept: bool = True
    weighted: bool = False
    per_sample_metric: bool = False
    n_samples: int = 0
    info: dict = field(default_factory=dict)

    def apply(self, C: torch.Tensor) -> torch.Tensor:
        """Map source rows ``[N, ds]`` to predicted target rows ``[N, dt]``."""
        return C.to(self.M.dtype) @ self.M + self.b

    @property
    def param_count(self) -> int:
        return self.M.numel() + self.b.numel()


# --------------------------------------------------------------------------
# objective, for tests and for reporting
# --------------------------------------------------------------------------


def weighted_sq_error(
    C: torch.Tensor,
    T: torch.Tensor,
    amap: AffineMap,
    *,
    G: Optional[torch.Tensor] = None,
    G_per_sample: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``sum_x w_x * e_x.T @ G_x @ e_x`` for a fitted map.  Scalar tensor."""
    C64 = C.to(torch.float64)
    T64 = T.to(torch.float64)
    E = C64 @ amap.M.to(torch.float64) + amap.b.to(torch.float64) - T64  # [N, dt]
    if G_per_sample is not None:
        Gp = G_per_sample.to(torch.float64)
        quad = torch.einsum("nd,nde,ne->n", E, Gp, E)
    elif G is not None:
        G64 = _symmetrize(G.to(torch.float64))
        quad = torch.einsum("nd,de,ne->n", E, G64, E)
    else:
        quad = (E * E).sum(dim=1)
    if sample_weights is not None:
        quad = quad * sample_weights.to(torch.float64)
    return quad.sum()


# --------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------


def _augment(C: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(C.shape[0], 1, dtype=C.dtype, device=C.device)
    return torch.cat([C, ones], dim=1)


def _solve_kron(
    C: torch.Tensor,
    T: torch.Tensor,
    G: Optional[torch.Tensor],
    G_per_sample: Optional[torch.Tensor],
    w: Optional[torch.Tensor],
    ridge_diag: torch.Tensor,
) -> torch.Tensor:
    """Solve the unconstrained weighted normal equations without shortcuts.

    Stationarity of ``sum_x w_x e_x' G_x e_x + M' diag(ridge) M`` is

        sum_x w_x c_x c_x' M G_x + diag(ridge) M = sum_x w_x c_x t_x' G_x

    which in column-major vec form is a dense ``(ds*dt)`` linear system.  No
    assumption is made about ``G`` being invertible or shared across samples,
    which is exactly what makes the invariance a result rather than a
    tautology.  Cost is ``O((ds*dt)^3)``; this path is for screens and tests,
    not for production-sized blocks.
    """
    N, ds = C.shape
    dt = T.shape[1]
    dev, dtype = C.device, C.dtype

    if w is None:
        w = torch.ones(N, dtype=dtype, device=dev)

    A = torch.zeros(ds * dt, ds * dt, dtype=dtype, device=dev)
    rhs = torch.zeros(ds, dt, dtype=dtype, device=dev)

    if G_per_sample is None:
        Gm = (
            torch.eye(dt, dtype=dtype, device=dev)
            if G is None
            else _symmetrize(G.to(dtype))
        )
        CtWC = C.transpose(0, 1) @ (C * w.unsqueeze(1))
        A = torch.kron(Gm.transpose(0, 1).contiguous(), CtWC.contiguous())
        rhs = (C.transpose(0, 1) @ (T * w.unsqueeze(1))) @ Gm
    else:
        Gp = _symmetrize(G_per_sample.to(dtype))
        # sum_x w_x (G_x' kron c_x c_x')  and  sum_x w_x c_x t_x' G_x
        for x in range(N):
            cx = C[x].unsqueeze(1)  # [ds,1]
            ccT = cx @ cx.transpose(0, 1)
            A = A + w[x] * torch.kron(Gp[x].transpose(0, 1).contiguous(), ccT)
            rhs = rhs + w[x] * (cx @ (T[x].unsqueeze(0) @ Gp[x]))

    A = A + torch.diag(ridge_diag.repeat(dt))
    sol = _min_norm_lstsq(A, _vec(rhs))
    return _unvec(sol, ds, dt)


def _reduced_rank(
    C: torch.Tensor,
    T: torch.Tensor,
    G: Optional[torch.Tensor],
    w: Optional[torch.Tensor],
    ridge: float,
    rank: int,
) -> torch.Tensor:
    """Rank-constrained fit under a fixed target metric, by whitening.

    With ``G = S @ S`` and ``S`` invertible, ``||(C M - T) S||_F`` is an
    ordinary Frobenius objective in ``M_tilde = M @ S``, and whitening preserves
    rank.  The reduced-rank solution projects the *fitted values* onto their top
    ``r`` right singular subspace (Reinsel-Velu); with ``ridge > 0`` the same
    projection is applied to the ridge fit (reduced-rank ridge).

    This is the path where a fixed metric legitimately changes the answer: the
    projection is taken in the whitened space, so ``S`` selects which directions
    survive the rank cut.
    """
    dt = T.shape[1]
    dev, dtype = C.device, C.dtype
    if w is not None:
        rw = w.clamp(min=0).sqrt().unsqueeze(1)
        C = C * rw
        T = T * rw
    if G is None:
        S = torch.eye(dt, dtype=dtype, device=dev)
        S_inv = S
    else:
        S = psd_sqrt(G).to(dtype)
        S_inv = psd_inv_sqrt(G).to(dtype)

    Tw = T @ S
    gram = C.transpose(0, 1) @ C
    if ridge > 0:
        gram = gram + ridge * torch.eye(C.shape[1], dtype=dtype, device=dev)
    M_ols = _min_norm_lstsq(gram, C.transpose(0, 1) @ Tw)  # [ds, dt]

    fitted = C @ M_ols  # [N, dt] in whitened target space
    _, _, Vh = torch.linalg.svd(fitted, full_matrices=False)
    r = int(min(rank, Vh.shape[0]))
    Vr = Vh[:r].transpose(0, 1)  # [dt, r]
    M_tilde = M_ols @ (Vr @ Vr.transpose(0, 1))
    return M_tilde @ S_inv


def _alternating_low_rank(
    C: torch.Tensor,
    T: torch.Tensor,
    G_per_sample: torch.Tensor,
    w: Optional[torch.Tensor],
    ridge: float,
    rank: int,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    """Rank-``r`` fit under a *sample-dependent* metric, by alternating solves.

    ``M = A @ B`` with ``A`` of shape ``[ds, r]`` and ``B`` of shape ``[r, dt]``.
    Holding one factor fixed leaves an ordinary weighted linear problem in the
    other, solved exactly by :func:`_solve_kron`.  There is no closed form for
    the joint problem, so this is block coordinate descent: it decreases the
    objective monotonically and is reported with its iteration count rather
    than claimed to be globally optimal.
    """
    dt = T.shape[1]
    dev, dtype = C.device, C.dtype
    zero_r = torch.zeros(rank, dtype=dtype, device=dev)

    # init from the unweighted reduced-rank solution
    M = _reduced_rank(C, T, None, w, ridge, rank)
    U, Sv, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, Sv.shape[0])
    A = U[:, :r] * Sv[:r].sqrt()
    B = (Vh[:r].transpose(0, 1) * Sv[:r].sqrt()).transpose(0, 1)

    prev = None
    iters = 0
    for iters in range(1, max_iter + 1):
        # solve B with A fixed: features are C @ A
        B = _solve_kron(C @ A, T, None, G_per_sample, w, zero_r + ridge)
        # solve A with B fixed: e_x = B' A' c_x - t_x, which is linear in A.
        A = _solve_a_given_b(C, T, B, G_per_sample, w, ridge)
        M = A @ B
        amap = AffineMap(M=M, b=torch.zeros(dt, dtype=dtype, device=dev), solver="als")
        obj = float(
            weighted_sq_error(C, T, amap, G_per_sample=G_per_sample, sample_weights=w)
        )
        if prev is not None and abs(prev - obj) <= tol * max(1.0, abs(prev)):
            prev = obj
            break
        prev = obj
    return A @ B, iters


def _solve_a_given_b(
    C: torch.Tensor,
    T: torch.Tensor,
    B: torch.Tensor,
    G_per_sample: torch.Tensor,
    w: Optional[torch.Tensor],
    ridge: float,
) -> torch.Tensor:
    """Exact solve for ``A`` in ``M = A @ B`` under a per-sample metric.

    ``e_x = B' A' c_x - t_x``, so with ``H_x = B G_x B'`` and
    ``q_x = B G_x t_x`` the stationarity condition in ``A`` is
    ``sum_x w_x c_x c_x' A H_x + ridge*A = sum_x w_x c_x q_x'``.
    """
    N, ds = C.shape
    r = B.shape[0]
    dev, dtype = C.device, C.dtype
    Gp = _symmetrize(G_per_sample.to(dtype))
    H = torch.einsum("rd,nde,se->nrs", B, Gp, B)  # [N, r, r]
    Q = torch.einsum("rd,nde,ne->nr", B, Gp, T)  # [N, r]
    if w is None:
        w = torch.ones(N, dtype=dtype, device=dev)
    A_sys = torch.zeros(ds * r, ds * r, dtype=dtype, device=dev)
    rhs = torch.zeros(ds, r, dtype=dtype, device=dev)
    for x in range(N):
        cx = C[x].unsqueeze(1)
        ccT = cx @ cx.transpose(0, 1)
        A_sys = A_sys + w[x] * torch.kron(H[x].transpose(0, 1).contiguous(), ccT)
        rhs = rhs + w[x] * (cx @ Q[x].unsqueeze(0))
    A_sys = A_sys + ridge * torch.eye(ds * r, dtype=dtype, device=dev)
    sol = _min_norm_lstsq(A_sys, _vec(rhs))
    return _unvec(sol, ds, r)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def fit_affine(
    C: torch.Tensor,
    T: torch.Tensor,
    *,
    G: Optional[torch.Tensor] = None,
    G_per_sample: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    ridge: float = 0.0,
    rank: Optional[int] = None,
    fit_intercept: bool = True,
    solver: str = "auto",
    als_max_iter: int = 25,
    als_tol: float = 1e-10,
) -> AffineMap:
    """Fit ``T ~ C @ M + b`` under an optional receiver-side metric.

    Args:
        C: source rows, ``[N, ds]``.
        T: target rows, ``[N, dt]``.
        G: fixed target-space metric, ``[dt, dt]``, PSD.  ``None`` means
            identity.  On an unconstrained, unregularised fit with invertible
            ``G`` this argument provably does nothing -- that is the point.
        G_per_sample: sample-dependent metric, ``[N, dt, dt]``.  Mutually
            exclusive with ``G``.  This *does* change the solution.
        sample_weights: scalar per-sample weights, ``[N]``.
        ridge: Frobenius penalty on ``M``.  Never applied to the intercept.
        rank: if set, constrain ``rank(M) <= rank``.
        fit_intercept: fit ``b``; otherwise ``b`` is zero.
        solver: ``"auto"``, ``"kron"`` (general, exact, no rank constraint),
            ``"whiten"`` (fixed metric, supports rank), or ``"als"``
            (per-sample metric with a rank constraint).
        als_max_iter, als_tol: block coordinate descent controls for ``"als"``.

    Returns:
        :class:`AffineMap`.
    """
    if G is not None and G_per_sample is not None:
        raise ValueError("pass either G or G_per_sample, not both")
    if C.ndim != 2 or T.ndim != 2:
        raise ValueError(f"C and T must be 2-D, got {tuple(C.shape)} {tuple(T.shape)}")
    if C.shape[0] != T.shape[0]:
        raise ValueError(f"row mismatch: C has {C.shape[0]}, T has {T.shape[0]}")

    out_dtype = C.dtype
    dev = C.device
    C64 = C.to(torch.float64)
    T64 = T.to(torch.float64)
    N, ds = C64.shape
    dt = T64.shape[1]

    G64 = None if G is None else _symmetrize(G.to(torch.float64))
    Gp64 = None if G_per_sample is None else _symmetrize(G_per_sample.to(torch.float64))
    if Gp64 is not None and Gp64.shape[0] != N:
        raise ValueError(f"G_per_sample must have {N} entries, got {Gp64.shape[0]}")
    w64 = None if sample_weights is None else sample_weights.to(torch.float64)

    if solver == "auto":
        if rank is None:
            solver = "kron"
        elif Gp64 is None:
            solver = "whiten"
        else:
            solver = "als"

    info: dict = {}

    if rank is not None and solver == "kron":
        raise ValueError("solver='kron' cannot honour a rank constraint")
    if rank is None and solver == "als":
        raise ValueError("solver='als' is only for rank-constrained per-sample fits")

    if solver == "whiten" and Gp64 is not None:
        raise ValueError("solver='whiten' requires a fixed metric; use 'als'")

    # --- intercept handling -------------------------------------------------
    # With an unconstrained intercept the optimal b is the weighted centring
    # offset whenever the metric is shared and invertible.  That argument does
    # not survive a per-sample metric, so for the unconstrained per-sample case
    # the intercept is solved jointly as an extra source feature instead.
    joint_intercept = fit_intercept and Gp64 is not None and rank is None

    if joint_intercept:
        Cw = _augment(C64)
        ridge_diag = torch.full(
            (ds + 1,), float(ridge), dtype=torch.float64, device=dev
        )
        ridge_diag[-1] = 0.0
        M_aug = _solve_kron(Cw, T64, None, Gp64, w64, ridge_diag)
        M64, b64 = M_aug[:ds], M_aug[ds]
    else:
        if fit_intercept:
            if w64 is None:
                cbar = C64.mean(dim=0)
                tbar = T64.mean(dim=0)
            else:
                sw = w64.sum().clamp(min=1e-300)
                cbar = (C64 * w64.unsqueeze(1)).sum(dim=0) / sw
                tbar = (T64 * w64.unsqueeze(1)).sum(dim=0) / sw
            Cc, Tc = C64 - cbar, T64 - tbar
        else:
            cbar = torch.zeros(ds, dtype=torch.float64, device=dev)
            tbar = torch.zeros(dt, dtype=torch.float64, device=dev)
            Cc, Tc = C64, T64

        if solver == "kron":
            ridge_diag = torch.full(
                (ds,), float(ridge), dtype=torch.float64, device=dev
            )
            M64 = _solve_kron(Cc, Tc, G64, Gp64, w64, ridge_diag)
        elif solver == "whiten":
            M64 = _reduced_rank(Cc, Tc, G64, w64, float(ridge), int(rank))
        elif solver == "als":
            M64, iters = _alternating_low_rank(
                Cc, Tc, Gp64, w64, float(ridge), int(rank), als_max_iter, als_tol
            )
            info["als_iters"] = iters
        else:
            raise ValueError(f"unknown solver {solver!r}")

        b64 = (
            tbar - M64.transpose(0, 1) @ cbar
            if fit_intercept
            else torch.zeros(dt, dtype=torch.float64, device=dev)
        )

    return AffineMap(
        M=M64.to(out_dtype),
        b=b64.to(out_dtype),
        solver=solver,
        rank=rank,
        ridge=float(ridge),
        fit_intercept=fit_intercept,
        weighted=(G64 is not None or Gp64 is not None),
        per_sample_metric=Gp64 is not None,
        n_samples=int(N),
        info=info,
    )
