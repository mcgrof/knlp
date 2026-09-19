"""Statistics helpers: deterministic bootstrap CIs that bracket the point estimate, BH-FDR that is
monotone and order-invariant, and LOFO attribution that ranks a hurting factor first. Cheap guards
against noise-mining the atlas's many cells."""

import random

import pytest

from tools.kv.fp8_failure import statistics as S


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    p1, lo1, hi1 = S.bootstrap_ci(vals, n_boot=500, seed=7)
    p2, lo2, hi2 = S.bootstrap_ci(vals, n_boot=500, seed=7)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)  # seed-deterministic
    assert abs(p1 - 3.0) < 1e-9
    assert lo1 <= p1 <= hi1
    assert hi1 > lo1  # an interval, not a point -- see the regression below


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128])
def test_bootstrap_ci_is_not_degenerate_at_power_of_two_sizes(n):
    """The exact sizes where the previous resampler returned zero width.

    It drew indices as `state % n` from a linear congruential generator. Taking a
    power-of-two modulus reads only the low bits, whose period is exactly that
    power of two, so each block of n draws was a permutation of the sample rather
    than a draw with replacement, every replicate reproduced the sample mean, and
    the interval collapsed to a point. Coverage of the true mean was 0%.

    This matters beyond tidiness: a zero-width interval "excludes zero" for any
    nonzero estimate, so a significance rule phrased that way passed on noise.
    """
    vals = [((i * 37) % 101) / 101.0 for i in range(n)]
    assert min(vals) != max(vals)
    _, lo, hi = S.bootstrap_ci(vals, n_boot=500, seed=0)
    assert hi > lo, f"degenerate interval at n={n}"


@pytest.mark.parametrize("n", [6, 17, 31, 48, 100])
def test_bootstrap_ci_is_not_degenerate_at_other_sizes(n):
    vals = [((i * 37) % 101) / 101.0 for i in range(n)]
    _, lo, hi = S.bootstrap_ci(vals, n_boot=500, seed=0)
    assert hi > lo


def test_bootstrap_ci_width_tracks_the_standard_error():
    """A correct interval is about 2 x 1.96 standard errors wide.

    The old resampler failed this badly even where it did not collapse: at n=48
    it returned about 0.81x the correct width. Checking against theory, rather
    than only against itself, is what catches that.
    """
    n = 200
    rng = random.Random(11)
    vals = [rng.gauss(1.0, 1.0) for _ in range(n)]
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sem = (var / n) ** 0.5
    _, lo, hi = S.bootstrap_ci(vals, n_boot=4000, seed=1)
    assert abs((hi - lo) - 2 * 1.96 * sem) < 0.35 * sem


def test_bootstrap_ci_covers_the_truth_at_about_the_nominal_rate():
    """Coverage, the property the interval actually claims.

    The old resampler covered the true mean 0% of the time at these sizes. A
    nominal 95% interval should land near 0.95; the tolerance here is loose
    enough for 200 trials and still nowhere near the old behaviour.
    """
    rng = random.Random(5)
    hits = 0
    trials = 200
    for t in range(trials):
        vals = [rng.gauss(1.0, 1.0) for _ in range(16)]
        _, lo, hi = S.bootstrap_ci(vals, n_boot=400, seed=t)
        hits += lo <= 1.0 <= hi
    assert 0.85 <= hits / trials <= 1.0, f"coverage {hits / trials}"


def test_bootstrap_ci_is_a_point_only_when_the_sample_is_constant():
    p, lo, hi = S.bootstrap_ci([2.0] * 32, n_boot=200, seed=0)
    assert p == lo == hi == 2.0


def test_paired_delta_sign():
    a = [1.0, 1.1, 0.9, 1.05]  # higher error (fp8)
    b = [0.5, 0.4, 0.6, 0.5]  # lower error (repair)
    d, lo, hi = S.paired_bootstrap_delta(a, b, n_boot=500, seed=3)
    assert d > 0 and lo > 0  # repair strictly helps
    assert hi > lo


@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_paired_delta_is_not_degenerate_at_power_of_two_sizes(n):
    """`paired_bootstrap_delta` resamples through `bootstrap_ci` and inherited
    the same collapse. It has no caller in the tree today, which is precisely
    why it needs pinning before one arrives."""
    rng = random.Random(n)
    a = [rng.gauss(1.0, 1.0) for _ in range(n)]
    b = [rng.gauss(0.5, 1.0) for _ in range(n)]
    _, lo, hi = S.paired_bootstrap_delta(a, b, n_boot=500, seed=0)
    assert hi > lo, f"degenerate paired interval at n={n}"


def test_paired_delta_covers_zero_when_the_two_arms_are_interchangeable():
    rng = random.Random(21)
    a = [rng.gauss(0.0, 1.0) for _ in range(64)]
    b = [rng.gauss(0.0, 1.0) for _ in range(64)]
    d, lo, hi = S.paired_bootstrap_delta(a, b, n_boot=2000, seed=2)
    assert lo < 0 < hi, f"false positive: delta={d}, CI=[{lo}, {hi}]"


def test_bh_fdr_monotone_and_order_invariant():
    pvals = [0.001, 0.04, 0.2, 0.5, 0.9]
    reject, adj = S.benjamini_hochberg(pvals, q=0.05)
    assert reject[0] is True and reject[-1] is False
    # order invariance: permuting input permutes output the same way
    perm = [2, 0, 4, 1, 3]
    rp, ap = S.benjamini_hochberg([pvals[i] for i in perm], q=0.05)
    for new_i, orig_i in enumerate(perm):
        assert rp[new_i] == reject[orig_i]
        assert abs(ap[new_i] - adj[orig_i]) < 1e-12


def test_lofo_ranks_hurting_factor_first():
    # full error 1.0; removing 'fp8_k' drops error to 0.2 (it was hurting most)
    imp = S.lofo_attribution(1.0, {"fp8_k": 0.2, "fp8_v": 0.8, "layout": 0.95})
    assert list(imp.keys())[0] == "fp8_k"
    assert imp["fp8_k"] > imp["fp8_v"] > imp["layout"]
