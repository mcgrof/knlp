# SPDX-License-Identifier: GPL-2.0
"""Exact cache damage, and the statistics the Stage 0 gate is decided on.

Two halves.  The first measures what a cache perturbation actually costs the
receiver: teacher-forced KL between the clean and perturbed continuation
distributions, the change in negative log-likelihood of the real continuation,
and the argmax flip rate.  These are the ground truth every candidate metric is
ranked against; nothing here is a surrogate.

The second half is the gate arithmetic.  The promotion rule is a *rank*
correlation improvement with a paired bootstrap interval that excludes zero, so
the bootstrap has to resample perturbations and recompute Spearman on each
resample -- a bootstrap of a statistic that is not a mean, which the existing
``tools/kv/fp8_failure/statistics.py`` helpers do not cover.  Its ``_quantile``
and ``benjamini_hochberg`` are reused as-is; the resampling is added here.

Determinism: every function takes an explicit seed and uses the same integer
LCG as the atlas statistics, so a gate verdict is reproducible from the seed
alone with no global random state involved.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.kv.fp8_failure.statistics import (  # noqa: E402
    _quantile,
    benjamini_hochberg,
)

__all__ = [
    "teacher_forced_kl",
    "delta_nll",
    "argmax_flip_rate",
    "attention_output_cosine",
    "spearman",
    "kendall_distance",
    "paired_bootstrap_statistic_delta",
    "paired_bootstrap_spearman_delta",
    "bootstrap_p_two_sided",
    "GateResult",
    "evaluate_gate",
    "benjamini_hochberg",
]


# ---------------------------------------------------------------------------
# exact damage
# ---------------------------------------------------------------------------


def teacher_forced_kl(
    clean_logits: torch.Tensor, pert_logits: torch.Tensor
) -> torch.Tensor:
    """``KL(clean || perturbed)`` per continuation position, in float32.

    Teacher forced: both runs see the same real continuation tokens, so this is
    a distribution distance at matched inputs and not a dynamic-decode drift.
    Keeping that distinction is why the two are never mixed in one column.
    """
    lp = torch.log_softmax(clean_logits.float(), dim=-1)
    lq = torch.log_softmax(pert_logits.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(dim=-1)


def delta_nll(
    clean_logits: torch.Tensor, pert_logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Per-position increase in NLL of the *real* continuation tokens."""
    lp = torch.log_softmax(clean_logits.float(), dim=-1)
    lq = torch.log_softmax(pert_logits.float(), dim=-1)
    idx = target_ids.unsqueeze(-1)
    return (-lq.gather(-1, idx) + lp.gather(-1, idx)).squeeze(-1)


def argmax_flip_rate(clean_logits: torch.Tensor, pert_logits: torch.Tensor) -> float:
    """Fraction of continuation positions whose greedy token changes."""
    a = clean_logits.argmax(dim=-1)
    b = pert_logits.argmax(dim=-1)
    return float((a != b).float().mean())


def attention_output_cosine(clean: torch.Tensor, pert: torch.Tensor) -> float:
    """Cosine between clean and perturbed attention outputs, flattened."""
    a = clean.float().reshape(-1)
    b = pert.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


# ---------------------------------------------------------------------------
# rank statistics
# ---------------------------------------------------------------------------


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- the Spearman convention.

    Fully vectorised on purpose.  The gate bootstraps a rank correlation, so
    this runs once per resample per comparison: on a Stage 0 cell that is of
    order a hundred thousand calls over a few thousand items each, and a
    per-element Python loop turns a half-minute of statistics into hours.
    """
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    # start of each run of equal values
    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    np.not_equal(sx[1:], sx[:-1], out=starts_mask[1:])
    group_of_sorted = np.cumsum(starts_mask) - 1
    bounds = np.append(np.flatnonzero(starts_mask), n)
    # average of the 1-based ranks spanned by each tie group
    avg = (bounds[:-1] + bounds[1:] - 1) * 0.5 + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[group_of_sorted]
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with tie-corrected average ranks."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2:
        return float("nan")
    rx, ry = _rankdata(xa), _rankdata(ya)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if den == 0.0:
        return float("nan")
    return float((rx * ry).sum() / den)


def kendall_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Fraction of pairs the two orderings disagree on, ties excluded.

    This is what "the method ranking changes by less than 5 percent between
    four and eight probes" is measured with: 0 means identical orderings, 0.5
    means unrelated.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    n = xa.size
    if n < 2:
        return float("nan")
    dx = np.sign(xa[:, None] - xa[None, :])
    dy = np.sign(ya[:, None] - ya[None, :])
    iu = np.triu_indices(n, k=1)
    a, b = dx[iu], dy[iu]
    valid = (a != 0) & (b != 0)
    if not valid.any():
        return float("nan")
    return float((a[valid] != b[valid]).mean())


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _resampler(seed: int) -> np.random.Generator:
    """A sound, seeded resampler.  Deterministic, and not the in-repo LCG.

    ``tools/kv/fp8_failure/statistics.py`` resamples with
    ``state = (1103515245*state + 12345) & 0x7FFFFFFF; idx = state % n``.
    Taking ``n`` modulo a power of two reads only the LCG's low bits, whose
    period is exactly ``2**k``, so each block of ``n`` draws is a *permutation*
    of ``0..n-1`` and every resample reproduces the original sample.  Measured
    on this tree: CI width is exactly 0.0 at n = 8, 16, 32, 64 and 128.  A
    zero-width interval "excludes zero" for any nonzero delta, so a gate built
    on it passes on pure noise.

    PCG64 seeded explicitly gives reproducibility without that failure mode.
    ``tests/jacobian_kv/test_weighted_fit.py`` pins the non-degeneracy at the
    exact sizes where the LCG collapses.
    """
    return np.random.default_rng(seed)


def paired_bootstrap_statistic_delta(
    a: Sequence[float],
    b: Sequence[float],
    truth: Sequence[float],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    return_boots: bool = False,
):
    """CI for ``statistic(a, truth) - statistic(b, truth)`` under a paired resample.

    The resample is over *items* (here, perturbations), and both metrics are
    rescored on the identical resample, which is what makes the interval a
    paired one.  Returns ``(delta, lo, hi)``.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    tt = np.asarray(truth, dtype=np.float64)
    if not (aa.size == bb.size == tt.size):
        raise ValueError("paired bootstrap needs aligned same-length samples")
    n = aa.size
    point = statistic(aa, tt) - statistic(bb, tt)
    rng = _resampler(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = statistic(aa[idx], tt[idx]) - statistic(bb[idx], tt[idx])
        if not math.isnan(d):
            boots.append(d)
    if not boots:
        return point, float("nan"), float("nan")
    boots.sort()
    lo = _quantile(boots, alpha / 2)
    hi = _quantile(boots, 1 - alpha / 2)
    if hi <= lo:
        raise RuntimeError(
            f"degenerate bootstrap interval at n={n} (width {hi - lo:.3e}); "
            "the resampler is not resampling"
        )
    if return_boots:
        return point, lo, hi, boots
    return point, lo, hi


def bootstrap_p_two_sided(boots: Sequence[float]) -> float:
    """Two-sided bootstrap p for ``delta != 0``, floored at ``1/n_boot``.

    Nothing in the repo produced p-values, so ``benjamini_hochberg`` had no
    inputs.  These are what the BH family across {K, V} x {model} x {seed} is
    built from.
    """
    b = list(boots)
    if not b:
        return float("nan")
    below = sum(1 for x in b if x <= 0.0)
    p = 2.0 * min(below, len(b) - below) / len(b)
    return min(1.0, max(p, 1.0 / len(b)))


def paired_bootstrap_spearman_delta(
    a: Sequence[float],
    b: Sequence[float],
    truth: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    return_boots: bool = False,
):
    """The Stage 0 gate statistic: paired CI on the Spearman improvement."""
    return paired_bootstrap_statistic_delta(
        a,
        b,
        truth,
        lambda u, v: spearman(u, v),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        return_boots=return_boots,
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """A pre-registered Stage 0 verdict for one tensor kind on one model."""

    model: str
    kind: str
    n_items: int
    truth_name: str
    rho: dict
    best_baseline: str
    best_baseline_rho: float
    candidate: str
    candidate_rho: float
    delta: float
    ci_lo: float
    ci_hi: float
    p_value: float
    probe_stability: dict
    passed: bool
    reasons: list

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def evaluate_gate(
    scores: dict,
    truth: Sequence[float],
    *,
    model: str,
    kind: str,
    candidate: str,
    baselines: Sequence[str],
    truth_name: str = "teacher_forced_kl",
    min_delta: float = 0.10,
    probe_variants: dict | None = None,
    max_rank_change: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> GateResult:
    """Apply the plan's Stage 0 promotion rule, and say why it failed if it did.

    The rule, pre-registered and not negotiable after seeing numbers: the
    candidate must improve held-out Spearman against exact damage by at least
    ``min_delta`` over the *best* cheap baseline, the paired bootstrap 95%
    interval on that improvement must exclude zero, and the ordering the
    candidate induces must be stable between four and eight probes.
    """
    rho = {name: spearman(vals, truth) for name, vals in scores.items()}
    usable = [b for b in baselines if b in rho and not math.isnan(rho[b])]
    if not usable:
        raise ValueError("no usable baseline scores")
    best = max(usable, key=lambda b: rho[b])

    delta, lo, hi, boots = paired_bootstrap_spearman_delta(
        scores[candidate],
        scores[best],
        truth,
        n_boot=n_boot,
        seed=seed,
        return_boots=True,
    )
    p_value = bootstrap_p_two_sided(boots)

    stability: dict = {}
    if (
        probe_variants
        and probe_variants.get(4) is not None
        and probe_variants.get(8) is not None
    ):
        stability = {
            "kendall_distance_4_to_8": kendall_distance(
                probe_variants[4], probe_variants[8]
            ),
            "spearman_4_to_8": spearman(probe_variants[4], probe_variants[8]),
        }

    reasons = []
    if delta < min_delta:
        reasons.append(
            f"delta rho {delta:.4f} below the pre-registered {min_delta:.2f}"
        )
    if lo <= 0.0:
        where = "includes zero" if hi >= 0.0 else "lies entirely below zero"
        reasons.append(f"paired bootstrap CI [{lo:.4f}, {hi:.4f}] {where}")
    kd = stability.get("kendall_distance_4_to_8")
    if kd is not None and not math.isnan(kd) and kd > max_rank_change:
        reasons.append(
            f"ordering moves {kd:.3f} between 4 and 8 probes, above {max_rank_change:.2f}"
        )
    if probe_variants and not stability:
        reasons.append("probe stability not measured (missing 4- or 8-probe scores)")

    return GateResult(
        model=model,
        kind=kind,
        n_items=len(truth),
        truth_name=truth_name,
        rho=rho,
        best_baseline=best,
        best_baseline_rho=rho[best],
        candidate=candidate,
        candidate_rho=rho[candidate],
        delta=delta,
        ci_lo=lo,
        ci_hi=hi,
        p_value=p_value,
        probe_stability=stability,
        passed=not reasons,
        reasons=reasons,
    )
