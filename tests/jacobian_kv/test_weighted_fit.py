# SPDX-License-Identifier: GPL-2.0
"""The gate arithmetic, pinned where it is known to have failed before.

The headline regression is the resampler.  ``tools/kv/fp8_failure/statistics.py``
draws bootstrap indices with ``state = (1103515245*state + 12345) & 0x7FFFFFFF``
and then takes ``state % n``.  Taking ``n`` modulo a power of two reads only
the LCG's low bits, whose period is exactly ``2**k``, so each block of ``n``
draws is a *permutation* of ``0..n-1`` and every resample reproduces the
original sample.  Measured on this tree, the confidence interval it returns has
width exactly zero at n = 8, 16, 32, 64 and 128.

That is not a cosmetic flaw.  A zero-width interval "excludes zero" for any
nonzero point estimate, so a promotion rule written as "the paired bootstrap
95 percent interval for the improvement excludes zero" -- which is exactly how
the Stage 0 gate is written -- would pass on pure noise.  These tests pin the
replacement at the sizes where the original collapses, and check that it still
behaves like a bootstrap on data where the answer is known.

The rest of the file covers the statistics the gate composes: Spearman with
ties, the Kendall distance used for probe-count stability, Benjamini-Hochberg
against a worked example, and the weighted-fit closed forms.
"""

import math

import numpy as np
import pytest
import torch

from research.jacobian_kv.evaluate import (
    argmax_flip_rate,
    benjamini_hochberg,
    bootstrap_p_two_sided,
    delta_nll,
    evaluate_gate,
    kendall_distance,
    paired_bootstrap_spearman_delta,
    paired_bootstrap_statistic_delta,
    spearman,
    teacher_forced_kl,
)
from research.jacobian_kv.fit import fit_affine, weighted_sq_error


# ---------------------------------------------------------------------------
# the resampler regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128])
def test_bootstrap_is_not_degenerate_at_power_of_two_sizes(n):
    """The exact sizes where the inherited LCG resampler returns width zero."""
    rng = np.random.default_rng(n)
    truth = rng.normal(size=n)
    a = truth + 0.5 * rng.normal(size=n)
    b = truth + 1.5 * rng.normal(size=n)
    _, lo, hi = paired_bootstrap_spearman_delta(a, b, truth, n_boot=400, seed=0)
    assert hi > lo, f"degenerate interval at n={n}"


@pytest.mark.parametrize("n", [6, 17, 31, 48, 100])
def test_bootstrap_is_not_degenerate_at_other_sizes_either(n):
    rng = np.random.default_rng(n)
    truth = rng.normal(size=n)
    a = truth + 0.5 * rng.normal(size=n)
    b = truth + 1.5 * rng.normal(size=n)
    _, lo, hi = paired_bootstrap_spearman_delta(a, b, truth, n_boot=400, seed=0)
    assert hi > lo


def test_bootstrap_covers_zero_when_the_two_metrics_are_interchangeable():
    """No real difference in, no significant difference out."""
    rng = np.random.default_rng(7)
    n = 64
    truth = rng.normal(size=n)
    a = truth + rng.normal(size=n)
    b = truth + rng.normal(size=n)
    d, lo, hi = paired_bootstrap_spearman_delta(a, b, truth, n_boot=1000, seed=1)
    assert lo < 0 < hi, f"false positive: delta={d}, CI=[{lo},{hi}]"


def test_bootstrap_excludes_zero_when_one_metric_is_genuinely_better():
    rng = np.random.default_rng(8)
    n = 120
    truth = rng.normal(size=n)
    a = truth + 0.15 * rng.normal(size=n)
    b = truth + 3.0 * rng.normal(size=n)
    d, lo, hi = paired_bootstrap_spearman_delta(a, b, truth, n_boot=1000, seed=2)
    assert d > 0 and lo > 0


def test_bootstrap_is_reproducible_from_its_seed():
    rng = np.random.default_rng(9)
    truth = rng.normal(size=40)
    a, b = truth + rng.normal(size=40), truth + rng.normal(size=40)
    r1 = paired_bootstrap_spearman_delta(a, b, truth, n_boot=200, seed=3)
    r2 = paired_bootstrap_spearman_delta(a, b, truth, n_boot=200, seed=3)
    r3 = paired_bootstrap_spearman_delta(a, b, truth, n_boot=200, seed=4)
    assert r1 == r2
    assert r1[1] != r3[1]


def test_bootstrap_recovers_a_known_mean_interval():
    """Sanity against theory: the CI for a mean tracks 1.96 * SEM."""
    rng = np.random.default_rng(11)
    n = 500
    x = rng.normal(loc=1.0, scale=1.0, size=n)
    zero = np.zeros(n)

    def mean_stat(vals, _ignored):
        return float(np.mean(vals))

    d, lo, hi = paired_bootstrap_statistic_delta(
        x, zero, zero, mean_stat, n_boot=2000, seed=5
    )
    sem = float(np.std(x, ddof=1) / math.sqrt(n))
    assert abs(d - float(np.mean(x))) < 1e-12
    assert abs((hi - lo) - 2 * 1.96 * sem) < 0.4 * sem


def test_two_sided_p_is_floored_and_bounded():
    assert bootstrap_p_two_sided([1.0] * 100) == pytest.approx(0.01)
    assert bootstrap_p_two_sided([1.0] * 50 + [-1.0] * 50) == pytest.approx(1.0)
    assert math.isnan(bootstrap_p_two_sided([]))


# ---------------------------------------------------------------------------
# rank statistics
# ---------------------------------------------------------------------------


def test_spearman_is_one_on_a_monotone_transform():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert spearman(x, [math.exp(v) for v in x]) == pytest.approx(1.0)
    assert spearman(x, [-v for v in x]) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks():
    """A constant column has no ranking power and must come back as nan."""
    assert math.isnan(spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]))
    r = spearman([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert 0.8 < r < 1.0


def test_spearman_matches_scipy_when_it_is_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=50), rng.normal(size=50)
    assert spearman(x, y) == pytest.approx(scipy_stats.spearmanr(x, y).statistic)


def test_kendall_distance_bounds():
    x = [1.0, 2.0, 3.0, 4.0]
    assert kendall_distance(x, x) == 0.0
    assert kendall_distance(x, x[::-1]) == 1.0
    assert math.isnan(kendall_distance([1.0, 1.0], [1.0, 1.0]))


def test_benjamini_hochberg_worked_example():
    """The worked example from Benjamini and Hochberg (1995), Table 1.

    Fifteen hypotheses at q = 0.05; the published answer rejects exactly the
    four smallest, because the fifth p-value 0.0201 exceeds its threshold
    5/15 * 0.05 = 0.0167.
    """
    p = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
    ]
    reject, adj = benjamini_hochberg(p, q=0.05)
    assert sum(reject) == 4
    assert reject[:4] == [True, True, True, True]
    assert not any(reject[4:])
    assert adj == sorted(adj), "adjusted q-values must be monotone in sorted p"
    assert all(a >= b for a, b in zip(adj, p)), "adjustment must not shrink a p"


def test_benjamini_hochberg_is_stricter_for_a_lone_marginal_result():
    """One p of 0.04 among five clears per-test alpha but not BH.

    Its threshold at rank one is 1/5 * 0.05 = 0.01, so it is rejected -- which
    is the multiplicity control the gate family needs.  BH is *not* uniformly
    stricter than alpha: at the last rank the threshold is alpha itself, so a
    family where every p sits just under alpha is rejected whole.  Both
    behaviours are pinned so neither reads as a bug later.
    """
    lone = [0.04, 0.9, 0.9, 0.9, 0.9]
    reject, _ = benjamini_hochberg(lone, q=0.05)
    assert sum(reject) == 0

    all_marginal = [0.04, 0.045, 0.047, 0.048, 0.049]
    reject_all, _ = benjamini_hochberg(all_marginal, q=0.05)
    assert sum(reject_all) == 5


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _gate_inputs(n, candidate_noise, baseline_noise, seed):
    rng = np.random.default_rng(seed)
    truth = np.abs(rng.normal(size=n)) + 0.1
    scores = {
        "mse": truth + baseline_noise * rng.normal(size=n),
        "wo": truth + 2 * baseline_noise * rng.normal(size=n),
        "attn_local": truth + 1.5 * baseline_noise * rng.normal(size=n),
        "jtfj_p4": truth + candidate_noise * rng.normal(size=n),
        "jtfj_p8": truth + candidate_noise * rng.normal(size=n),
    }
    return scores, truth


def test_gate_passes_a_genuinely_better_candidate():
    scores, truth = _gate_inputs(200, candidate_noise=0.05, baseline_noise=1.2, seed=20)
    g = evaluate_gate(
        scores,
        truth,
        model="synthetic",
        kind="v",
        candidate="jtfj_p8",
        baselines=("mse", "wo", "attn_local"),
        probe_variants={4: scores["jtfj_p4"], 8: scores["jtfj_p8"]},
        n_boot=500,
        seed=0,
        max_rank_change=1.0,
    )
    assert g.passed, g.reasons
    assert g.delta >= 0.10 and g.ci_lo > 0


def test_gate_fails_a_candidate_that_only_ties():
    scores, truth = _gate_inputs(200, candidate_noise=0.5, baseline_noise=0.5, seed=21)
    g = evaluate_gate(
        scores,
        truth,
        model="synthetic",
        kind="v",
        candidate="jtfj_p8",
        baselines=("mse", "wo", "attn_local"),
        n_boot=500,
        seed=0,
    )
    assert not g.passed
    assert any("below the pre-registered" in r for r in g.reasons)


def test_gate_fails_an_unstable_candidate_even_when_it_is_better():
    """Probe instability is disqualifying on its own, by design."""
    scores, truth = _gate_inputs(200, candidate_noise=0.05, baseline_noise=1.2, seed=22)
    rng = np.random.default_rng(99)
    shuffled = rng.permutation(scores["jtfj_p4"])
    g = evaluate_gate(
        scores,
        truth,
        model="synthetic",
        kind="v",
        candidate="jtfj_p8",
        baselines=("mse", "wo", "attn_local"),
        probe_variants={4: shuffled, 8: scores["jtfj_p8"]},
        n_boot=500,
        seed=0,
    )
    assert not g.passed
    assert any("ordering moves" in r for r in g.reasons)


def test_gate_picks_the_strongest_baseline_not_a_convenient_one():
    scores, truth = _gate_inputs(150, candidate_noise=0.3, baseline_noise=1.0, seed=23)
    scores["attn_local"] = truth + 0.05 * np.random.default_rng(1).normal(size=150)
    g = evaluate_gate(
        scores,
        truth,
        model="synthetic",
        kind="k",
        candidate="jtfj_p8",
        baselines=("mse", "wo", "attn_local"),
        n_boot=300,
        seed=0,
    )
    assert g.best_baseline == "attn_local"


# ---------------------------------------------------------------------------
# damage metrics
# ---------------------------------------------------------------------------


def test_teacher_forced_kl_is_zero_on_identical_logits_and_positive_otherwise():
    torch.manual_seed(0)
    a = torch.randn(1, 5, 9)
    assert float(teacher_forced_kl(a, a).abs().max()) < 1e-6
    b = a + 0.5 * torch.randn_like(a)
    assert float(teacher_forced_kl(a, b).min()) > 0


def test_delta_nll_is_zero_on_identical_logits():
    torch.manual_seed(1)
    a = torch.randn(1, 6, 11)
    ids = torch.randint(0, 11, (1, 6))
    assert float(delta_nll(a, a, ids).abs().max()) < 1e-6


def test_argmax_flip_rate_bounds():
    a = torch.tensor([[[3.0, 1.0], [1.0, 3.0]]])
    assert argmax_flip_rate(a, a) == 0.0
    assert argmax_flip_rate(a, -a) == 1.0


# ---------------------------------------------------------------------------
# weighted fits used by the Stage 1 arms
# ---------------------------------------------------------------------------


def test_ridge_matches_its_closed_form():
    torch.manual_seed(2)
    n, ds, dt, lam = 40, 5, 3, 0.7
    C = torch.randn(n, ds, dtype=torch.float64)
    T = torch.randn(n, dt, dtype=torch.float64)
    got = fit_affine(C, T, ridge=lam, solver="kron", fit_intercept=False)
    want = torch.linalg.solve(
        C.T @ C + lam * torch.eye(ds, dtype=torch.float64), C.T @ T
    )
    assert torch.allclose(got.M, want, atol=1e-9, rtol=0)


def test_rank_constrained_fit_respects_its_rank_and_beats_nothing_lower():
    torch.manual_seed(3)
    n, ds, dt = 80, 6, 5
    C = torch.randn(n, ds, dtype=torch.float64)
    T = C @ torch.randn(ds, dt, dtype=torch.float64) + 0.2 * torch.randn(
        n, dt, dtype=torch.float64
    )
    errs = []
    for r in (1, 2, 4):
        m = fit_affine(C, T, rank=r, solver="whiten")
        assert torch.linalg.matrix_rank(m.M, tol=1e-8).item() <= r
        errs.append(float(weighted_sq_error(C, T, m)))
    assert errs[0] > errs[1] > errs[2], "more rank must not fit worse"


def test_per_sample_metric_fit_reaches_a_lower_weighted_objective():
    torch.manual_seed(4)
    n, ds, dt = 30, 4, 3
    C = torch.randn(n, ds, dtype=torch.float64)
    T = torch.randn(n, dt, dtype=torch.float64)
    g = torch.Generator().manual_seed(5)
    Gp = torch.stack(
        [
            (lambda A: A @ A.T + 0.1 * torch.eye(dt, dtype=torch.float64))(
                torch.randn(dt, dt, generator=g, dtype=torch.float64)
            )
            for _ in range(n)
        ]
    )
    plain = fit_affine(C, T, solver="kron")
    weighted = fit_affine(C, T, G_per_sample=Gp, solver="kron")
    assert weighted_sq_error(C, T, weighted, G_per_sample=Gp) < weighted_sq_error(
        C, T, plain, G_per_sample=Gp
    )


def test_alternating_low_rank_decreases_the_weighted_objective():
    torch.manual_seed(6)
    n, ds, dt, r = 24, 5, 4, 2
    C = torch.randn(n, ds, dtype=torch.float64)
    T = torch.randn(n, dt, dtype=torch.float64)
    g = torch.Generator().manual_seed(7)
    Gp = torch.stack(
        [
            (lambda A: A @ A.T + 0.2 * torch.eye(dt, dtype=torch.float64))(
                torch.randn(dt, dt, generator=g, dtype=torch.float64)
            )
            for _ in range(n)
        ]
    )
    unweighted = fit_affine(C, T, rank=r, solver="whiten")
    als = fit_affine(C, T, G_per_sample=Gp, rank=r, solver="als")
    assert torch.linalg.matrix_rank(als.M, tol=1e-8).item() <= r
    assert weighted_sq_error(C, T, als, G_per_sample=Gp) <= weighted_sq_error(
        C, T, unweighted, G_per_sample=Gp
    ) + 1e-9
