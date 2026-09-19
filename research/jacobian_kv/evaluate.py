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
    "bootstrap_p_one_sided_positive",
    "blocked_spearman",
    "cluster_bootstrap_blocked_delta",
    "method_ranking_distance",
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
    """Per-position increase in NLL of the *next* continuation token.

    The logits at position ``i`` predict token ``i+1``, so the targets are
    shifted.  Scoring ``target_ids[i]`` against ``logits[:, i]`` would measure
    the model's probability of re-emitting the token already at that input
    position, which is a different quantity and a much easier one.  The last
    position has no next token inside the continuation and is dropped.
    """
    if target_ids.shape[-1] < 2:
        return torch.zeros(
            target_ids.shape[0], 0, device=clean_logits.device, dtype=torch.float32
        )
    lp = torch.log_softmax(clean_logits[:, :-1].float(), dim=-1)
    lq = torch.log_softmax(pert_logits[:, :-1].float(), dim=-1)
    idx = target_ids[:, 1:].unsqueeze(-1)
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
        nan = float("nan")
        return (point, nan, nan, []) if return_boots else (point, nan, nan)
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


def bootstrap_p_one_sided_positive(boots: Sequence[float]) -> float:
    """p for "the improvement is greater than zero", floored at ``1/n_boot``.

    The gate is a one-sided claim -- the candidate must be *better* -- so a
    two-sided p rewards a candidate that is significantly worse with the
    smallest possible value, and a family-wise correction built on it counts
    those cells as discoveries.  This reports the mass at or below zero.
    """
    b = list(boots)
    if not b:
        return float("nan")
    below = sum(1 for x in b if x <= 0.0)
    return min(1.0, max(below / len(b), 1.0 / len(b)))


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
# the blocked statistic the gate is decided on
# ---------------------------------------------------------------------------


def group_rhos(
    scores: Sequence[float],
    truth: Sequence[float],
    groups: Sequence,
    min_size: int = 5,
) -> dict:
    """Spearman within each matched-norm group, keyed by group label.

    Correlating across groups would let a metric be credited for tracking how
    damage scales with layer, with error magnitude and with context, none of
    which is the ranking skill the screen is testing.  Inside a group every
    perturbation has the same norm at the same layer in the same context, so
    what is left is direction alone.
    """
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    idx: dict = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    out = {}
    for g, rows in idx.items():
        if len(rows) < min_size:
            continue
        r = spearman(x[rows], y[rows])
        if not math.isnan(r):
            out[g] = r
    return out


def _fisher_mean(rhos: Sequence[float]) -> float:
    """Average correlations on the Fisher z scale, returned as a correlation.

    Averaging raw correlations compresses each arm by an amount that grows
    with its spread across groups, so an arm with more variable groups is
    penalised for that alone.  The z transform removes the asymmetry.
    """
    if not rhos:
        return float("nan")
    z = np.arctanh(np.clip(np.asarray(rhos, dtype=np.float64), -0.999999, 0.999999))
    return float(np.tanh(z.mean()))


def blocked_spearman(
    scores: Sequence[float],
    truth: Sequence[float],
    groups: Sequence,
    min_size: int = 5,
) -> float:
    """The headline statistic: Fisher-averaged within-group rank correlation."""
    return _fisher_mean(list(group_rhos(scores, truth, groups, min_size).values()))


def cluster_bootstrap_blocked_delta(
    a: Sequence[float],
    b: Sequence[float],
    truth: Sequence[float],
    groups: Sequence,
    clusters: Sequence,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    min_size: int = 5,
    return_boots: bool = False,
):
    """Paired interval on the blocked improvement, resampling whole contexts.

    Every perturbation from one held-out context shares that context's prefill,
    its clean reference and its attention, so the rows are clustered and not
    2500 independent observations.  Resampling rows would narrow the interval
    the gate depends on.  The unit resampled here is the context.

    Because each group's correlation depends only on that group's rows, the
    per-group values are computed once and the bootstrap resamples which
    contexts contribute, which is both exact and cheap.
    """
    ra = group_rhos(a, truth, groups, min_size)
    rb = group_rhos(b, truth, groups, min_size)
    shared = [g for g in ra if g in rb]
    if not shared:
        raise ValueError("no matched-norm group is usable for both arms")

    by_cluster: dict = {}
    g_of_row = {}
    for g, c in zip(groups, clusters):
        g_of_row.setdefault(g, c)
    for g in shared:
        by_cluster.setdefault(g_of_row[g], []).append(g)
    keys = sorted(by_cluster, key=str)

    def stat(cluster_list):
        za, zb = [], []
        for c in cluster_list:
            for g in by_cluster[c]:
                za.append(ra[g])
                zb.append(rb[g])
        return _fisher_mean(za) - _fisher_mean(zb)

    point = stat(keys)
    rng = _resampler(seed)
    n = len(keys)
    boots = []
    for _ in range(n_boot):
        draw = [keys[i] for i in rng.integers(0, n, size=n)]
        d = stat(draw)
        if not math.isnan(d):
            boots.append(d)
    if not boots:
        nan = float("nan")
        return (point, nan, nan, []) if return_boots else (point, nan, nan)
    boots.sort()
    lo = _quantile(boots, alpha / 2)
    hi = _quantile(boots, 1 - alpha / 2)
    if hi <= lo:
        # A zero-width interval is a bug when the resampler is not resampling,
        # and a fact when the two arms score identically -- which happens for
        # real, for instance when a rank truncation removes nothing because the
        # metric was already within that rank.  Distinguish the two instead of
        # failing the run on the second.
        differs = any(abs(ra[g] - rb[g]) > 1e-12 for g in shared)
        if differs and n >= 8:
            raise RuntimeError(
                f"degenerate cluster bootstrap at n_clusters={n} although the "
                "arms differ; the resampler is not resampling"
            )
    return (point, lo, hi, boots) if return_boots else (point, lo, hi)


def method_ranking_distance(
    scores_by_name: dict,
    truth: Sequence[float],
    groups: Sequence,
    family_a: dict,
    family_b: dict,
    min_size: int = 5,
) -> float:
    """How far the ordering of the competing *methods* moves between two budgets.

    The plan's stability condition is written about the method ranking, so this
    scores every method under each probe budget and compares the two orderings.
    The per-candidate reordering of individual perturbations is a stricter
    reading and is reported alongside it rather than substituted for it.
    """
    names = sorted(set(family_a) | set(family_b) | set(scores_by_name))
    va, vb = [], []
    for n in names:
        sa = family_a.get(n, scores_by_name.get(n))
        sb = family_b.get(n, scores_by_name.get(n))
        if sa is None or sb is None:
            continue
        va.append(blocked_spearman(sa, truth, groups, min_size))
        vb.append(blocked_spearman(sb, truth, groups, min_size))
    return kendall_distance(va, vb)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """A pre-registered Stage 0 verdict for one tensor kind on one model."""

    model: str
    kind: str
    n_items: int
    n_groups: int
    n_clusters: int
    truth_name: str
    statistic: str
    rho: dict
    baseline_tier: str
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
        return dict(self.__dict__)


def evaluate_gate(
    scores: dict,
    truth: Sequence[float],
    groups: Sequence,
    clusters: Sequence,
    *,
    model: str,
    kind: str,
    candidate: str,
    baselines: Sequence[str],
    baseline_tier: str = "deployable",
    truth_name: str = "teacher_forced_kl",
    min_delta: float = 0.10,
    probe_stability: dict | None = None,
    max_rank_change: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    min_group: int = 5,
) -> GateResult:
    """Apply the plan's Stage 0 promotion rule, and say why it failed if it did.

    The rule, pre-registered and not negotiable after seeing numbers: the
    candidate must improve held-out rank correlation against exact damage by at
    least ``min_delta`` over the best cheap baseline, the paired 95 percent
    interval on that improvement must lie above zero, and the ordering must be
    stable between four and eight probes.

    Two implementation choices are worth stating because the plan does not fix
    them.  The correlation is the *blocked* one, computed inside matched-norm
    groups and averaged over them, because the plan's own bank exists so that
    error magnitude cannot win and a pooled correlation hands that channel
    straight back.  And the interval requires a lower bound above zero rather
    than merely excluding zero, since an interval lying entirely below zero
    also excludes it while meaning the opposite.

    Baselines are compared within a tier.  A candidate that needs information
    unavailable when a cache is written is judged against baselines allowed the
    same information, so neither side is credited for seeing the future.
    """
    rho = {
        name: blocked_spearman(vals, truth, groups, min_group)
        for name, vals in scores.items()
    }
    usable = [b for b in baselines if b in rho and not math.isnan(rho[b])]
    if not usable:
        raise ValueError(f"no usable baseline in tier {baseline_tier!r}")
    best = max(usable, key=lambda b: rho[b])

    delta, lo, hi, boots = cluster_bootstrap_blocked_delta(
        scores[candidate],
        scores[best],
        truth,
        groups,
        clusters,
        n_boot=n_boot,
        seed=seed,
        min_size=min_group,
        return_boots=True,
    )
    p_value = bootstrap_p_one_sided_positive(boots)

    stability = dict(probe_stability or {})
    reasons = []
    if delta < min_delta:
        reasons.append(
            f"delta rho {delta:.4f} below the pre-registered {min_delta:.2f}"
        )
    if math.isnan(lo) or lo <= 0.0:
        where = (
            "includes zero"
            if (math.isnan(hi) or hi >= 0.0)
            else "lies entirely below zero"
        )
        reasons.append(f"paired interval [{lo:.4f}, {hi:.4f}] {where}")
    md = stability.get("method_ranking_distance_4_to_8")
    if md is None or (isinstance(md, float) and math.isnan(md)):
        reasons.append("probe stability not measured")
    elif md > max_rank_change:
        reasons.append(
            f"method ordering moves {md:.3f} between 4 and 8 probes, "
            f"above {max_rank_change:.2f}"
        )

    n_groups = len({g for g in groups})
    n_clusters = len({c for c in clusters})
    return GateResult(
        model=model,
        kind=kind,
        n_items=len(truth),
        n_groups=n_groups,
        n_clusters=n_clusters,
        truth_name=truth_name,
        statistic="blocked_spearman_fisher_mean",
        rho=rho,
        baseline_tier=baseline_tier,
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
