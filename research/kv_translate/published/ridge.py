#!/usr/bin/env python3
"""The centered weighted ridge both published methods solve, in closed form.

Three conventions here are easy to get wrong and each changes the answer by a
lot rather than a little.

The ridge term is added to the UNNORMALISED Gram. Both papers write
``(X^T X + lambda I)``, and one independent implementation of the baseline
divides the Gram by n first. At the published calibration size of 128,000 rows
that is a hundred-and-twenty-eight-thousand-fold difference in effective
regularisation, which is not a detail.

The weighted means carry an explicit ``1/n``, not ``1/sum(w)``. The paper
writes ``xbar_w = n^-1 sum_i w_i x_i``. The two agree only when the weights
already average to one, which is why the weights are rescaled to mean one
first and why that rescaling is part of the method rather than housekeeping.

The solve is done in float64 on centered data. Accumulating a 8192-wide Gram
over 128,000 rows in float32 loses precision where it matters most, in the
small eigenvalues that the ridge term is there to control.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LAMBDA = 0.01


def rescale_to_mean_one(w):
    """Weights enter the solve averaging one, as the paper specifies.

    This is not cosmetic. The ridge term is absolute, so scaling all weights up
    scales the Gram up and quietly weakens the regularisation; and the 1/n
    convention in the weighted mean is only the weighted mean when the weights
    average one.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    m = w.mean()
    if m <= 0:
        raise ValueError("weights sum to zero, so no row carries any mass")
    return w / m


def weighted_means(X, Y, w):
    """xbar = (1/n) sum w_i x_i, per the paper, NOT sum(w x)/sum(w)."""
    n = X.shape[0]
    return (w @ X) / n, (w @ Y) / n


def sufficient_statistics(X, Y, w, xbar, ybar):
    """A = sum_i w_i xc xc^T and B = sum_i w_i xc yc^T, unnormalised.

    These are exactly the quantities Fused-Fit accumulates in panels. Forming
    them here in one shot is the same mathematics at a slower speed, which is
    the documented relationship between the reference path and the optimised
    one.
    """
    Xc = X - xbar
    Yc = Y - ybar
    Xw = Xc * w[:, None]
    return Xc.T @ Xw, Xw.T @ Yc


def solve(X, Y, w=None, lam=DEFAULT_LAMBDA):
    """Fit one affine map, returning (W, b) and the statistics behind them.

    X is [n, p] source features, Y is [n, d] target values, w is [n] or None
    for the unweighted baseline.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}")
    n, p = X.shape
    if n == 0:
        raise ValueError("no calibration rows")

    w = np.ones(n, dtype=np.float64) if w is None else rescale_to_mean_one(w)
    if w.shape[0] != n:
        raise ValueError(f"weights are length {w.shape[0]} for {n} rows")

    xbar, ybar = weighted_means(X, Y, w)
    A, B = sufficient_statistics(X, Y, w, xbar, ybar)
    # lambda on the UNNORMALISED Gram, as published
    A_reg = A + lam * np.eye(p, dtype=np.float64)
    W = np.linalg.solve(A_reg, B)
    b = ybar - xbar @ W
    return (
        W,
        b,
        {
            "xbar": xbar,
            "ybar": ybar,
            "A": A,
            "B": B,
            "lambda": lam,
            "n": n,
            "p": p,
            "effective_sample_size": kish_ess(w),
        },
    )


def apply_map(X, W, b):
    return np.asarray(X, dtype=np.float64) @ W + b


def kish_ess(w):
    """(sum w)^2 / sum w^2 -- how many equally-weighted rows this is worth."""
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return float(s * s / (w * w).sum()) if s > 0 else 0.0


def held_in_r2(X, Y, W, b, w=None):
    """Weighted R-squared of a fitted map on its own fitting data.

    The selector ranks source layers by exactly this, so it has to use the same
    weighting the fit will use; ranking on an unweighted probe and then fitting
    weighted would select for a different objective than the one optimised.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    w = np.ones(n, dtype=np.float64) if w is None else rescale_to_mean_one(w)
    pred = apply_map(X, W, b)
    ybar = (w @ Y) / n
    ss_res = float((w[:, None] * (Y - pred) ** 2).sum())
    ss_tot = float((w[:, None] * (Y - ybar) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
