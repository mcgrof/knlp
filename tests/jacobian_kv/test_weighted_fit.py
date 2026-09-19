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
    blocked_spearman,
    bootstrap_p_one_sided_positive,
    bootstrap_p_two_sided,
    cluster_bootstrap_blocked_delta,
    delta_nll,
    evaluate_gate,
    kendall_distance,
    method_ranking_distance,
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
        0.0001,
        0.0004,
        0.0019,
        0.0095,
        0.0201,
        0.0278,
        0.0298,
        0.0344,
        0.0459,
        0.3240,
        0.4262,
        0.5719,
        0.6528,
        0.7590,
        1.0000,
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


def _blocked_inputs(n_ctx, n_groups_per_ctx, group_size, cand_noise, base_noise, seed):
    """Clustered, grouped synthetic data shaped like a real screen cell."""
    rng = np.random.default_rng(seed)
    truth, groups, clusters = [], [], []
    for c in range(n_ctx):
        for g in range(n_groups_per_ctx):
            for _ in range(group_size):
                truth.append(abs(rng.normal()) + 0.1)
                groups.append(f"c{c}|g{g}")
                clusters.append(c)
    truth = np.asarray(truth)
    n = truth.size
    scores = {
        "mse": truth + base_noise * rng.normal(size=n),
        "wo": truth + 2 * base_noise * rng.normal(size=n),
        "attn_prefill": truth + 1.5 * base_noise * rng.normal(size=n),
        "meanj": truth + 2.5 * base_noise * rng.normal(size=n),
        "jtfj_pos_p4": truth + cand_noise * rng.normal(size=n),
        "jtfj_pos_p8": truth + cand_noise * rng.normal(size=n),
    }
    return scores, truth, groups, clusters


BL = ("mse", "wo", "attn_prefill", "meanj")


def _gate(scores, truth, groups, clusters, candidate="jtfj_pos_p8", **kw):
    kw.setdefault("probe_stability", {"method_ranking_distance_4_to_8": 0.0})
    return evaluate_gate(
        scores,
        truth,
        groups,
        clusters,
        model="synthetic",
        kind="v",
        candidate=candidate,
        baselines=BL,
        n_boot=400,
        seed=0,
        **kw,
    )


# ---------------------------------------------------------------------------
# the blocked statistic
# ---------------------------------------------------------------------------


def test_blocked_correlation_ignores_between_group_scale():
    """A metric that only knows which group a row is in scores zero.

    This is the whole reason the gate uses the blocked statistic. Pooled across
    groups, a per-group constant correlates strongly with damage whenever
    damage differs by group -- which it does by layer and by error magnitude.
    Inside a group it carries nothing, and that is the honest reading.
    """
    rng = np.random.default_rng(3)
    truth, groups, group_mean = [], [], {}
    # enough groups that the Fisher mean of pure-noise correlations is tight:
    # each group's rho has a standard error near 1/sqrt(13), so 60 groups puts
    # the mean's standard error near 0.04
    for g in range(60):
        m = float(g)
        group_mean[f"g{g}"] = m
        for _ in range(14):
            truth.append(m + 0.3 * rng.normal())
            groups.append(f"g{g}")
    scale_only = [group_mean[g] for g in groups]
    assert spearman(scale_only, truth) > 0.9
    # constant inside every group, so its blocked correlation is undefined
    # rather than zero -- the stronger statement, and what the code returns
    assert math.isnan(blocked_spearman(scale_only, truth, groups))

    # the same metric with a little within-group jitter scores about nothing
    jitter = rng.normal(scale=1e-3, size=len(scale_only))
    noisy = [v + j for v, j in zip(scale_only, jitter)]
    assert spearman(noisy, truth) > 0.9
    assert abs(blocked_spearman(noisy, truth, groups)) < 0.12


def test_blocked_correlation_recovers_a_within_group_signal():
    rng = np.random.default_rng(4)
    truth, groups, good = [], [], []
    for g in range(12):
        for _ in range(14):
            t = abs(rng.normal()) + 0.1
            truth.append(t)
            groups.append(f"g{g}")
            good.append(t + 0.02 * rng.normal())
    assert blocked_spearman(good, truth, groups) > 0.9


def test_fisher_averaging_does_not_penalise_a_spread_out_arm():
    """Raw averaging compresses the arm with more across-group variance.

    Two arms with the same Fisher-scale mean but different spread must not be
    separated by the averaging rule, or the reported lead is an artefact of how
    the correlations were combined.
    """
    tight = [0.8, 0.8, 0.8, 0.8]
    spread = [0.2, 0.95, 0.99, 0.55]
    z = np.arctanh(np.array(spread))
    tight_matched = list(np.tanh(np.full(4, z.mean())))
    from research.jacobian_kv.evaluate import _fisher_mean

    assert _fisher_mean(spread) == pytest.approx(_fisher_mean(tight_matched), abs=1e-9)
    assert np.mean(spread) < np.mean(tight_matched) - 0.05


# ---------------------------------------------------------------------------
# the cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_is_wider_than_resampling_rows():
    """Rows inside a context are not independent, and the interval must say so."""
    scores, truth, groups, clusters = _blocked_inputs(12, 5, 14, 0.4, 0.9, seed=5)
    _, lo_c, hi_c = cluster_bootstrap_blocked_delta(
        scores["jtfj_pos_p8"],
        scores["mse"],
        truth,
        groups,
        clusters,
        n_boot=600,
        seed=0,
    )
    _, lo_r, hi_r = paired_bootstrap_spearman_delta(
        scores["jtfj_pos_p8"], scores["mse"], truth, n_boot=600, seed=0
    )
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_cluster_bootstrap_is_reproducible_and_non_degenerate():
    scores, truth, groups, clusters = _blocked_inputs(10, 4, 14, 0.3, 0.8, seed=6)
    a = cluster_bootstrap_blocked_delta(
        scores["jtfj_pos_p8"],
        scores["mse"],
        truth,
        groups,
        clusters,
        n_boot=300,
        seed=1,
    )
    b = cluster_bootstrap_blocked_delta(
        scores["jtfj_pos_p8"],
        scores["mse"],
        truth,
        groups,
        clusters,
        n_boot=300,
        seed=1,
    )
    assert a == b
    assert a[2] > a[1]


def test_one_sided_p_does_not_reward_being_worse():
    """A two-sided p gives a badly losing arm the smallest possible value."""
    losing = [-0.4] * 200
    winning = [0.4] * 200
    assert bootstrap_p_one_sided_positive(winning) == pytest.approx(1 / 200)
    assert bootstrap_p_one_sided_positive(losing) == pytest.approx(1.0)
    assert bootstrap_p_two_sided(losing) == pytest.approx(1 / 200)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def test_gate_passes_a_genuinely_better_candidate():
    scores, truth, groups, clusters = _blocked_inputs(14, 5, 14, 0.04, 1.4, seed=20)
    g = _gate(scores, truth, groups, clusters)
    assert g.passed, g.reasons
    assert g.delta >= 0.10 and g.ci_lo > 0
    assert g.statistic == "blocked_spearman_fisher_mean"


def test_gate_fails_a_candidate_that_only_ties():
    scores, truth, groups, clusters = _blocked_inputs(14, 5, 14, 0.6, 0.6, seed=21)
    g = _gate(scores, truth, groups, clusters)
    assert not g.passed
    assert any("below the pre-registered" in r for r in g.reasons)


def test_gate_reports_a_losing_candidate_as_below_zero_not_as_including_it():
    scores, truth, groups, clusters = _blocked_inputs(14, 5, 14, 3.0, 0.1, seed=24)
    g = _gate(scores, truth, groups, clusters)
    assert not g.passed
    assert any("entirely below zero" in r for r in g.reasons)


def test_gate_fails_an_unstable_candidate_even_when_it_is_better():
    scores, truth, groups, clusters = _blocked_inputs(14, 5, 14, 0.04, 1.4, seed=22)
    g = _gate(
        scores,
        truth,
        groups,
        clusters,
        probe_stability={"method_ranking_distance_4_to_8": 0.4},
    )
    assert not g.passed
    assert any("method ordering moves" in r for r in g.reasons)


def test_gate_fails_when_probe_stability_was_never_measured():
    scores, truth, groups, clusters = _blocked_inputs(14, 5, 14, 0.04, 1.4, seed=25)
    g = _gate(scores, truth, groups, clusters, probe_stability={})
    assert not g.passed
    assert any("not measured" in r for r in g.reasons)


def test_gate_picks_the_strongest_baseline_not_a_convenient_one():
    scores, truth, groups, clusters = _blocked_inputs(12, 5, 14, 0.5, 1.2, seed=23)
    scores["attn_prefill"] = np.asarray(truth) + 0.03 * np.random.default_rng(1).normal(
        size=len(truth)
    )
    g = _gate(scores, truth, groups, clusters)
    assert g.best_baseline == "attn_prefill"


def test_gate_requires_a_baseline_from_the_named_tier():
    scores, truth, groups, clusters = _blocked_inputs(8, 4, 14, 0.2, 0.8, seed=26)
    with pytest.raises(ValueError):
        evaluate_gate(
            scores,
            truth,
            groups,
            clusters,
            model="synthetic",
            kind="v",
            candidate="jtfj_pos_p8",
            baselines=("not_scored_here",),
            n_boot=100,
            seed=0,
        )


def test_method_ranking_distance_is_zero_when_probes_change_nothing():
    scores, truth, groups, clusters = _blocked_inputs(8, 4, 14, 0.2, 0.8, seed=27)
    fam = {"jtfj_pos_p8": scores["jtfj_pos_p8"]}
    d = method_ranking_distance(scores, truth, groups, fam, fam)
    assert d == pytest.approx(0.0)


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
    assert (
        weighted_sq_error(C, T, als, G_per_sample=Gp)
        <= weighted_sq_error(C, T, unweighted, G_per_sample=Gp) + 1e-9
    )
