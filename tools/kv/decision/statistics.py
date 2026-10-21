# SPDX-License-Identifier: MIT
"""Paired question and document inference, without token pseudoreplication.

The matched-pair difference interval inverts Tango's efficient score test:
Tango (1998), Statistics in Medicine 17:891-908, PMID 9595618. It is a
score interval with approximate coverage, not an exact binomial interval.
Gross-loss bounds use exact Clopper-Pearson limits instead.
"""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, norm


def _counts(gains, losses, n):
    if any(type(x) is not int for x in (gains, losses, n)):
        raise ValueError("Counts must be integers")
    if n <= 0 or min(gains, losses) < 0 or gains + losses > n:
        raise ValueError("Invalid paired counts")


def score(gains, losses, n, delta):
    """Signed efficient score at a candidate-minus-baseline risk difference.

    At fixed delta, maximize the trinomial likelihood over total discordance
    probability t, with probabilities (t+delta)/2, (t-delta)/2, 1-t. Solving
    its quadratic gives the constrained nuisance MLE used in the variance.
    """
    _counts(gains, losses, n)
    if not -1 <= delta <= 1:
        raise ValueError("Risk difference outside [-1, 1]")
    d = gains - losses
    residual = d - n * delta
    if residual == 0:
        return 0.0
    if abs(delta) == 1:
        return math.copysign(math.inf, residual)
    m = gains + losses
    a = m + d * delta
    discriminant = a * a - 4 * n * (d * delta - (n - m) * delta * delta)
    t = (a + math.sqrt(max(0.0, discriminant))) / (2 * n)
    t = min(1.0, max(abs(delta), t))
    variance = n * (t - delta * delta)
    return residual / math.sqrt(variance)


def paired_interval(gains, losses, n, alpha=0.05, two_sided=True):
    """Return bounds for candidate minus baseline (proportions, not pp).

    For two_sided=False, each returned endpoint is a one-sided 1-alpha bound;
    the two endpoints together are not a 1-alpha simultaneous interval.
    """
    _counts(gains, losses, n)
    if not 0 < alpha < 0.5:
        raise ValueError("Alpha must lie in (0, .5)")
    z = float(norm.isf(alpha / 2 if two_sided else alpha))
    point = (gains - losses) / n
    lower = (
        -1.0
        if losses == n
        else brentq(lambda x: score(gains, losses, n, x) - z, -1, point, xtol=1e-13)
    )
    upper = (
        1.0
        if gains == n
        else brentq(lambda x: score(gains, losses, n, x) + z, point, 1, xtol=1e-13)
    )
    return [float(lower), float(upper)]


def loss_bounds(losses, n, alpha=0.05):
    _counts(0, losses, n)
    if not 0 < alpha < 0.5:
        raise ValueError("Alpha must lie in (0, .5)")
    return [
        0.0 if losses == 0 else float(beta.ppf(alpha, losses, n - losses + 1)),
        1.0 if losses == n else float(beta.isf(alpha, losses + 1, n - losses)),
    ]


def paired_binary(native, candidate, alpha=0.05):
    if set(native) != set(candidate) or not native:
        raise ValueError("Paired item IDs must agree and be nonempty")
    if any(type(v) is not bool for side in (native, candidate) for v in side.values()):
        raise ValueError("Outcomes must be scored booleans")
    n = len(native)
    gains = sum(not native[i] and candidate[i] for i in native)
    losses = sum(native[i] and not candidate[i] for i in native)
    return dict(
        n=n,
        native_correct=sum(native.values()),
        candidate_correct=sum(candidate.values()),
        losses=losses,
        gains=gains,
        net_delta=(gains - losses) / n,
        gross_loss_one_sided_bounds=loss_bounds(losses, n, alpha),
        net_one_sided_bounds=paired_interval(gains, losses, n, alpha, False),
        net_interval_95=paired_interval(gains, losses, n),
        certificate_alpha=alpha,
        paired_method="Tango 1998 efficient-score inversion",
    )


def paired_nll(native, candidate, alpha=0.025 / 12, seed=20261020, resamples=100000):
    """Paired document bootstrap for mean NLL and its relative change."""
    left, right = np.asarray(native, dtype=np.float64), np.asarray(
        candidate, dtype=np.float64
    )
    if left.ndim != 1 or left.shape != right.shape or len(left) < 2:
        raise ValueError("Need aligned independent document means")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or min(left) <= 0:
        raise ValueError("Nonfinite NLL or nonpositive native document mean")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(left), size=(resamples, len(left)))
    baseline = left[ix].mean(1)
    absolute = (right - left)[ix].mean(1)
    relative = absolute / baseline
    return dict(
        n_documents=len(left),
        native_mean=float(left.mean()),
        candidate_mean=float(right.mean()),
        absolute_delta=float((right - left).mean()),
        relative_delta=float((right - left).mean() / left.mean()),
        absolute_one_sided_bounds=np.quantile(
            absolute, [alpha, 1 - alpha], method="linear"
        ).tolist(),
        relative_one_sided_bounds=np.quantile(
            relative, [alpha, 1 - alpha], method="linear"
        ).tolist(),
        alpha=alpha,
        resamples=resamples,
        seed=seed,
        quantile="linear",
    )
