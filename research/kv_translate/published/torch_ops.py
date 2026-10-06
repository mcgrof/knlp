#!/usr/bin/env python3
"""The accelerated path: rotary handling and the batched ridge solve.

The numpy modules beside this one are the reference. Everything here has to
agree with them on small inputs, and the tests hold it to that, because a fast
path that drifts from its reference produces a mapper nobody can audit.

Rotary convention is the one Hugging Face uses for the Qwen and Llama families:
the head dimension is split in half and element ``d`` is paired with element
``d + D/2``, rotated by ``position * theta^(-2d/D)``. A cache stores keys after
that rotation, so the content a mapper should be fitted on is the stored key
turned back by its own position's angle.
"""

from __future__ import annotations

import torch


def rope_angles(positions, head_dim, theta, device=None, dtype=torch.float64):
    """cos and sin tables of shape [n_positions, head_dim], halves duplicated."""
    half = head_dim // 2
    inv = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=dtype) * 2.0 / head_dim)
    )
    ang = positions.to(device=device, dtype=dtype)[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def _rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, positions, theta):
    """Rotate content vectors forward. x is [..., n_positions, head_dim]."""
    cos, sin = rope_angles(positions, x.shape[-1], theta, device=x.device)
    xd = x.to(torch.float64)
    return (xd * cos + _rotate_half(xd) * sin).to(x.dtype)


def strip_rope(x, positions, theta):
    """Undo the rotation, recovering content from a stored key."""
    cos, sin = rope_angles(positions, x.shape[-1], theta, device=x.device)
    xd = x.to(torch.float64)
    return (xd * cos - _rotate_half(xd) * sin).to(x.dtype)


def head_dim_of(config):
    """The head width, read rather than derived.

    On Qwen3-32B the hidden size divided by the number of attention heads is
    80 while the real head width is 128, because its attention is not square.
    Deriving the width by division silently corrupts that model's cache.
    """
    hd = getattr(config, "head_dim", None)
    if hd is None:
        raise ValueError(
            "this config carries no head_dim; refusing to derive it by "
            "division, which is wrong for models with non-square attention"
        )
    return int(hd)


def rope_theta_of(config):
    """The rotary base, across the config layouts transformers has used."""
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    if getattr(config, "rope_theta", None) is not None:
        return float(config.rope_theta)
    raise ValueError("no rotary base found in this config")


class Accumulator:
    """Weighted sufficient statistics for regressions that share a design matrix.

    Rows arrive in batches. The Gram and cross terms are accumulated about zero
    together with the weighted sums, and centred once at the end using
    ``A = S_xx - n xbar xbar^T`` under mean-one weights. Accumulating in double
    precision costs memory and buys back the small eigenvalues the ridge term
    is there to control.

    The output may be wider than one head. When every target head reads the
    same features, as in the baseline, the Gram is formed once and the heads'
    targets are stacked as columns, which is the difference between minutes
    and hours at a feature width of eight thousand.
    """

    def __init__(self, p, d, device, dtype=torch.float64):
        self.p, self.d = p, d
        self.sxx = torch.zeros(p, p, device=device, dtype=dtype)
        self.sxy = torch.zeros(p, d, device=device, dtype=dtype)
        self.sx = torch.zeros(p, device=device, dtype=dtype)
        self.sy = torch.zeros(d, device=device, dtype=dtype)
        self.syy = torch.zeros(d, device=device, dtype=dtype)
        self.sw = torch.zeros((), device=device, dtype=dtype)
        self.n = 0

    def add(self, X, Y, w=None):
        X = X.to(self.sxx.dtype)
        Y = Y.to(self.sxx.dtype)
        if w is None:
            Xw, Yw = X, Y
            self.sw += X.shape[0]
        else:
            w = w.to(self.sxx.dtype)
            Xw, Yw = X * w[:, None], Y * w[:, None]
            self.sw += w.sum()
        self.syy += (Yw * Y).sum(0)
        self.sy += Yw.sum(0)
        self.sx += Xw.sum(0)
        self.sxx += Xw.T @ X
        self.sxy += Xw.T @ Y
        self.n += X.shape[0]

    def solve(self, lam):
        """Closed-form centred ridge, regulariser on the unnormalised Gram.

        Weights are rescaled to average one here rather than by the caller, so
        a caller passing raw sensitivities cannot change the effective
        regularisation by accident. Returns the coefficients, the intercept and
        the residual and total sums of squares per output column, from which a
        caller can form R-squared over whatever group of columns is one head.
        """
        n = float(self.n)
        scale = n / self.sw  # brings the weights to mean one
        sxx, sxy = self.sxx * scale, self.sxy * scale
        sx, sy, syy = self.sx * scale, self.sy * scale, self.syy * scale
        xbar, ybar = sx / n, sy / n
        A = sxx - n * torch.outer(xbar, xbar)
        B = sxy - n * torch.outer(xbar, ybar)
        A_reg = A + lam * torch.eye(self.p, device=A.device, dtype=A.dtype)
        L = torch.linalg.cholesky(A_reg)
        W = torch.cholesky_solve(B, L)
        b = ybar - xbar @ W
        ss_tot = syy - n * ybar * ybar
        # residual sum of squares from moments, without revisiting the rows
        ss_res = ss_tot - 2.0 * (W * B).sum(0) + (W * (A @ W)).sum(0)
        return W, b, {"ss_res": ss_res, "ss_tot": ss_tot}


def r_squared(stats, columns=None):
    """R-squared over a group of output columns, or over all of them."""
    res, tot = stats["ss_res"], stats["ss_tot"]
    if columns is not None:
        res, tot = res[columns], tot[columns]
    t = float(tot.sum())
    return 1.0 - float(res.sum()) / t if t > 0 else 0.0
