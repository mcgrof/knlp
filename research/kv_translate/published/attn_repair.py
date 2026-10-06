#!/usr/bin/env python3
"""Attn-Repair: the row weighting that distinguishes CacheBridge from its base.

The idea is that a calibration row matters in proportion to how much the
receiver's attention would actually be disturbed if that position's key or
value were slightly wrong. Rows the receiver barely attends to can be fitted
loosely; rows it leans on cannot.

The sensitivities are first-order surrogates, computed from the target model's
own attention at fixed causal boundaries:

    r_value(i) = sum over query heads u sharing this KV head of a(u,i)^2
    r_key(i)   = sum over those u of a(u,i)^2 * ||v_i - o_u||^2 * ||q_u||^2 / d

with a(u,.) the softmax attention of query u and o_u its output. Keys get the
extra factors because perturbing a key moves the attention distribution, whose
effect scales with how far that value sits from the current output and with
the query's magnitude; perturbing a value only moves the output directly.

The part that keeps this from backfiring is the shrinkage. Raw sensitivities
are extremely uneven, and fitting a 1024-wide regression whose effective
sample size has collapsed to a few hundred rows is worse than fitting it
unweighted. So the weights are pulled toward uniform by the largest factor that
still leaves the Kish effective sample size above twice the feature width.
That floor is the whole safety argument, and it is why this is a repair rather
than a reweighting.
"""

from __future__ import annotations

import numpy as np


def sensitivities(attn, q, v, out, head_dim):
    """Raw per-position key and value sensitivities for one KV head.

    attn is [n_query_heads, n_positions] softmax rows, q is [n_query_heads,
    head_dim], v is [n_positions, head_dim] and out is [n_query_heads,
    head_dim]. All of these come from the TARGET model, which is the point:
    the weighting encodes what the receiver cares about, not what the source
    happened to emphasise.
    """
    attn = np.asarray(attn, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    if attn.ndim != 2:
        raise ValueError(f"attn must be [heads, positions], got {attn.shape}")
    a2 = attn**2
    r_v = a2.sum(axis=0)
    # ||v_i - o_u||^2 for every (u, i) without materialising the difference
    d2 = ((v[None, :, :] - out[:, None, :]) ** 2).sum(axis=-1)
    qn2 = (q**2).sum(axis=-1)
    r_k = (a2 * d2 * qn2[:, None] / head_dim).sum(axis=0)
    return r_k, r_v


def normalise_to_unit_mass(r):
    """Unit mass per (layer, head, component, boundary) before boundaries are summed.

    Without this a boundary with more eligible positions would dominate purely
    by having more of them, so the 32 boundaries would not carry equal mass
    the way the paper specifies.
    """
    r = np.asarray(r, dtype=np.float64)
    s = r.sum()
    return r / s if s > 0 else np.full_like(r, 1.0 / max(len(r), 1))


def combine_boundaries(per_boundary, n_positions):
    """Sum unit-mass boundary contributions with equal weight, then mean-normalise.

    Each boundary only sees the positions causally available to it, so
    contributions are scattered into a full-length vector before summing.
    """
    total = np.zeros(n_positions, dtype=np.float64)
    for idx, r in per_boundary:
        total[np.asarray(idx, dtype=int)] += normalise_to_unit_mass(r)
    m = total.mean()
    return total / m if m > 0 else np.ones(n_positions, dtype=np.float64)


def cv_squared(r):
    """Dispersion of the mean-normalised sensitivities, n^-1 sum (r_i - 1)^2."""
    r = np.asarray(r, dtype=np.float64)
    return float(((r - 1.0) ** 2).mean())


def shrinkage_alpha(r, n, tau):
    """Largest alpha in [0, 1] whose weights keep the Kish ESS at or above tau.

    Kish for w = 1 + alpha (r - 1) with mean-one r is n / (1 + alpha^2 CV^2),
    so the constraint inverts in closed form. When the floor is already
    unreachable at alpha = 0 the method cannot weight at all and returns zero,
    which is the honest outcome rather than an error: the arm degenerates to
    the unweighted solve and must be reported as having done so.
    """
    c2 = cv_squared(r)
    if c2 <= 0:
        return 1.0
    if tau >= n:
        return 0.0
    limit = (n / tau - 1.0) / c2
    if limit <= 0:
        return 0.0
    return float(min(1.0, np.sqrt(limit)))


def default_tau(feature_width, n):
    """The published floor: twice the feature width, capped by the row count."""
    return min(n, 2 * feature_width)


def weights(r, feature_width, tau=None):
    """Mean-normalised sensitivities turned into safe calibration weights."""
    r = np.asarray(r, dtype=np.float64)
    n = r.shape[0]
    m = r.mean()
    r = r / m if m > 0 else np.ones_like(r)
    tau = default_tau(feature_width, n) if tau is None else tau
    alpha = shrinkage_alpha(r, n, tau)
    w = 1.0 + alpha * (r - 1.0)
    # the surrogate is non-negative and mean-one, so this cannot go negative,
    # but a numerical floor keeps a pathological input from inverting a row
    w = np.clip(w, 0.0, None)
    return w, {
        "alpha": alpha,
        "cv_squared": cv_squared(r),
        "tau": tau,
        "n": n,
        "feature_width": feature_width,
        "ess_before": float(n),
        "ess_after": _ess(w),
        "degenerated_to_unweighted": alpha == 0.0,
    }


def _ess(w):
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return float(s * s / (w * w).sum()) if s > 0 else 0.0


def log_spaced_boundaries(first=12, last=1023, count=32):
    """The paper's 32 logarithmically spaced causal prefix boundaries.

    Stated as a function rather than a literal list so the convention is
    visible: inclusive at both ends, deduplicated after rounding, and
    ascending. Rounding can collide at the short end, and silently returning
    fewer than 32 boundaries would change the mass each one carries.
    """
    if count < 2:
        raise ValueError("need at least two boundaries")
    lo, hi = np.log(first), np.log(last)
    raw = np.exp(np.linspace(lo, hi, count))
    out = sorted({int(round(x)) for x in raw})
    return out, {"requested": count, "distinct": len(out), "collided": count - len(out)}
