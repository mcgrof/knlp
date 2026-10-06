# SPDX-License-Identifier: GPL-2.0
"""CPU checks for the published-method reimplementations.

The plan names the risks these have to cover: weighted versus reference
solves, shared layer support, rotary round trips, affine intercepts,
serialization, incomplete outputs and wrong-checkpoint rejection. Each test
below is one of those, and several encode a convention that an independent
implementation of the same paper gets differently -- which is exactly where a
reproduction silently stops reproducing.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.kv_translate.published import (
    CACHE_BRIDGE,
    FULL_HEAD,
    method,
)
from research.kv_translate.published import attn_repair as ar
from research.kv_translate.published import ridge, support

RNG = np.random.default_rng(0)


# ---- the two methods are distinct, and stay distinct --------------------


def test_the_two_methods_differ_in_exactly_two_axes():
    """Everything else shared is what makes the paired comparison mean anything."""
    fh, cb = method(FULL_HEAD), method(CACHE_BRIDGE)
    assert fh.support != cb.support
    assert fh.row_weighting != cb.row_weighting
    for shared in ("selector", "solver", "rope", "ridge_lambda", "coefficient_dtype"):
        assert getattr(fh, shared) == getattr(cb, shared), shared


def test_head_local_alone_is_not_cachebridge():
    """The paper reports Head-Local separately and it scores lower.

    Implementing the cheap change and keeping the name is the specific error
    the plan warns against.
    """
    cb = method(CACHE_BRIDGE)
    assert "Attn-Repair" in cb.row_weighting
    assert any("Head-Local alone is not CacheBridge" in n for n in cb.notes)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="not a published method"):
        method("cachebridge_but_faster")


# ---- ridge conventions --------------------------------------------------


def test_lambda_is_added_to_the_unnormalised_gram():
    """An independent implementation of the baseline divides by n first.

    At the published 128,000 calibration rows that is a 128,000-fold
    difference in effective regularisation, so the two conventions are not
    variants of one method.
    """
    n, p, d = 400, 6, 3
    X = RNG.normal(size=(n, p))
    Y = X @ RNG.normal(size=(p, d)) + 0.05 * RNG.normal(size=(n, d))
    W, b, st = ridge.solve(X, Y, lam=1.0)
    Xc, Yc = X - st["xbar"], Y - st["ybar"]
    unnormalised = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(p), Xc.T @ Yc)
    normalised = np.linalg.solve(Xc.T @ Xc / n + 1.0 * np.eye(p), Xc.T @ Yc / n)
    assert np.allclose(W, unnormalised)
    assert not np.allclose(W, normalised)


def test_weighted_means_use_one_over_n_not_one_over_sum_w():
    """The paper writes xbar = n^-1 sum w x. The two agree only at mean-one w."""
    n, p = 50, 4
    X = RNG.normal(size=(n, p))
    Y = RNG.normal(size=(n, 2))
    w = ridge.rescale_to_mean_one(RNG.uniform(0.1, 3.0, size=n))
    xbar, _ = ridge.weighted_means(X, Y, w)
    assert np.allclose(xbar, (w @ X) / n)
    assert np.allclose(xbar, (w @ X) / w.sum())  # equal because w averages one
    w_unscaled = w * 7.0
    assert not np.allclose((w_unscaled @ X) / n, (w_unscaled @ X) / w_unscaled.sum())


def test_uniform_weights_reproduce_the_unweighted_solve_exactly():
    """The weighted path must contain the baseline as its own special case."""
    X = RNG.normal(size=(200, 5))
    Y = X @ RNG.normal(size=(5, 3)) + 0.01 * RNG.normal(size=(200, 3))
    W0, b0, _ = ridge.solve(X, Y)
    W1, b1, _ = ridge.solve(X, Y, w=np.ones(200))
    assert np.allclose(W0, W1) and np.allclose(b0, b1)
    W2, b2, _ = ridge.solve(X, Y, w=np.full(200, 17.0))  # rescaled to one
    assert np.allclose(W0, W2) and np.allclose(b0, b2)


def test_the_intercept_absorbs_an_offset_exactly():
    """An affine map with a real intercept; a linear-only fit would not recover it."""
    X = RNG.normal(size=(300, 4))
    Wtrue = RNG.normal(size=(4, 3))
    btrue = np.array([5.0, -2.0, 11.0])
    Y = X @ Wtrue + btrue
    W, b, _ = ridge.solve(X, Y, lam=1e-9)
    assert np.allclose(W, Wtrue, atol=1e-5)
    assert np.allclose(b, btrue, atol=1e-5)
    assert np.allclose(ridge.apply_map(X, W, b), Y, atol=1e-5)


def test_weighting_actually_moves_the_fit_toward_the_weighted_rows():
    """If weights changed nothing, Attn-Repair would be decoration."""
    X = np.concatenate([np.zeros((100, 1)), np.ones((100, 1))])
    Y = np.concatenate([np.zeros((100, 1)), np.ones((100, 1))])
    Y[:100] += 10.0  # the first group sits far off the second group's line
    w = np.concatenate([np.full(100, 0.01), np.full(100, 1.99)])
    W_u, b_u, _ = ridge.solve(X, Y, lam=1e-9)
    W_w, b_w, _ = ridge.solve(X, Y, w=w, lam=1e-9)
    assert abs(b_w[0]) < abs(b_u[0])


def test_mismatched_rows_are_refused():
    with pytest.raises(ValueError, match="row mismatch"):
        ridge.solve(RNG.normal(size=(10, 3)), RNG.normal(size=(9, 2)))


# ---- Attn-Repair --------------------------------------------------------


def test_the_effective_sample_size_floor_is_respected():
    """The floor is the whole safety argument for weighting at all."""
    n, width = 4000, 256
    r = RNG.pareto(1.2, size=n) + 0.01  # deliberately vicious dispersion
    w, info = ar.weights(r, feature_width=width)
    assert info["tau"] == 2 * width
    assert info["ess_after"] >= info["tau"] - 1e-6
    assert 0.0 <= info["alpha"] <= 1.0


def test_uniform_sensitivities_permit_full_weighting_and_change_nothing():
    r = np.ones(1000)
    w, info = ar.weights(r, feature_width=64)
    assert info["alpha"] == 1.0
    assert np.allclose(w, 1.0)


def test_an_unreachable_floor_degenerates_to_unweighted_and_says_so():
    """Silently weighting anyway would be the dangerous outcome."""
    r = RNG.pareto(1.1, size=100) + 0.01
    w, info = ar.weights(r, feature_width=1024)  # tau capped at n
    assert info["alpha"] == 0.0
    assert info["degenerated_to_unweighted"] is True
    assert np.allclose(w, 1.0)


def test_key_sensitivity_exceeds_value_sensitivity_when_values_spread():
    """Keys carry the extra distance and query-norm factors, values do not."""
    heads, pos, d = 4, 16, 8
    attn = RNG.dirichlet(np.ones(pos), size=heads)
    q = RNG.normal(size=(heads, d)) * 3.0
    v = RNG.normal(size=(pos, d)) * 5.0
    out = attn @ v
    r_k, r_v = ar.sensitivities(attn, q, v, out, head_dim=d)
    assert r_k.shape == r_v.shape == (pos,)
    assert np.all(r_k >= 0) and np.all(r_v >= 0)
    assert r_k.sum() > r_v.sum()


def test_boundaries_are_log_spaced_and_collisions_are_reported():
    b, info = ar.log_spaced_boundaries(12, 1023, 32)
    assert b[0] == 12 and b[-1] == 1023
    assert b == sorted(b) and len(set(b)) == len(b)
    assert info["collided"] == 32 - len(b)
    gaps = np.diff(b)
    assert gaps[-1] > gaps[0]  # spacing widens, as log spacing requires


def test_each_boundary_contributes_equal_mass():
    n = 50
    per = [
        (list(range(10)), RNG.uniform(size=10)),
        (list(range(40)), RNG.uniform(size=40)),
    ]
    total = ar.combine_boundaries(per, n)
    assert np.isclose(total.mean(), 1.0)
    # the short boundary's ten positions carry the same total mass as the long
    # boundary's forty, which is what normalising before summing buys
    raw_short = ar.normalise_to_unit_mass(per[0][1]).sum()
    raw_long = ar.normalise_to_unit_mass(per[1][1]).sum()
    assert np.isclose(raw_short, raw_long)


# ---- support ------------------------------------------------------------


def test_one_layer_set_serves_both_components_and_every_head():
    sk = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.4}
    sv = {0: 0.2, 1: 0.3, 2: 0.8, 3: 0.1}
    sel, combined = support.select_layers(sk, sv, k=2)
    assert sel == sorted(sel)
    assert len(sel) == 2
    # ranked on the mean of the two, not on either alone
    assert combined[1] == pytest.approx(0.6)
    assert combined[2] == pytest.approx(0.65)
    assert sel == [1, 2]


def test_probing_different_layer_sets_for_k_and_v_is_refused():
    with pytest.raises(ValueError, match="different source layers"):
        support.select_layers({0: 0.1}, {1: 0.2}, k=1)


def test_feature_widths_match_the_published_numbers_for_this_pair():
    """8192 versus 1024 per target head, stated directly in the paper."""
    kw = dict(k=8, n_source_kv_heads=8, head_dim=128)
    assert support.feature_width(FULL_HEAD, **kw) == 8192
    assert support.feature_width(CACHE_BRIDGE, **kw) == 1024


def test_coefficient_counts_and_bytes_match_the_published_sizes():
    """0.538 GB and 4.296 GB at float32, which is how the dtype was inferred."""
    kw = dict(
        n_target_layers=64, n_target_kv_heads=8, k=8, n_source_kv_heads=8, head_dim=128
    )
    cb = support.coefficient_count(CACHE_BRIDGE, **kw)
    fh = support.coefficient_count(FULL_HEAD, **kw)
    assert cb == 2 * 64 * 8 * (1024 * 128 + 128)
    assert fh == 8 * (cb - 2 * 64 * 8 * 128) + 2 * 64 * 8 * 128
    assert support.serialized_bytes(CACHE_BRIDGE, **kw) / 1e9 == pytest.approx(
        0.538, abs=0.002
    )
    assert support.serialized_bytes(FULL_HEAD, **kw) / 1e9 == pytest.approx(
        4.296, abs=0.002
    )


def test_the_design_matrix_width_follows_the_method():
    n, L, H, d = 20, 3, 8, 4
    src = {j: RNG.normal(size=(n, H, d)) for j in range(L)}
    assignment = support.identity_head_assignment(H, H)
    x_cb = support.build_features(CACHE_BRIDGE, src, [0, 2], 3, assignment)
    x_fh = support.build_features(FULL_HEAD, src, [0, 2], 3, assignment)
    assert x_cb.shape == (n, 2 * d)
    assert x_fh.shape == (n, 2 * H * d)
    # head-local really does read only its own assigned source head
    assert np.allclose(x_cb[:, :d], src[0][:, 3, :])


def test_mismatched_kv_head_counts_are_refused_rather_than_guessed():
    """The paper says the support rule is unestablished there."""
    with pytest.raises(ValueError, match="will not guess"):
        support.identity_head_assignment(8, 4)


# ---- end to end on a synthetic pair -------------------------------------


def test_both_arms_recover_a_planted_affine_relation():
    """A decisive case: when a true per-head affine map exists, both find it."""
    n, L, H, d = 300, 4, 3, 5
    src = {j: RNG.normal(size=(n, H, d)) for j in range(L)}
    assignment = support.identity_head_assignment(H, H)
    h = 1
    Xcb = support.build_features(CACHE_BRIDGE, src, [0, 1], h, assignment)
    Wtrue = RNG.normal(size=(Xcb.shape[1], d))
    btrue = RNG.normal(size=d)
    Y = Xcb @ Wtrue + btrue
    W, b, _ = ridge.solve(Xcb, Y, lam=1e-10)
    assert ridge.held_in_r2(Xcb, Y, W, b) > 0.999999
    assert np.allclose(ridge.apply_map(Xcb, W, b), Y, atol=1e-4)


def test_a_weighted_fit_still_recovers_an_exact_relation():
    """Weighting must not bias a fit that is exactly satisfiable."""
    n, p, d = 200, 6, 3
    X = RNG.normal(size=(n, p))
    Wtrue, btrue = RNG.normal(size=(p, d)), RNG.normal(size=d)
    Y = X @ Wtrue + btrue
    w, _ = ar.weights(RNG.pareto(2.0, size=n) + 0.01, feature_width=p)
    W, b, _ = ridge.solve(X, Y, w=w, lam=1e-10)
    assert np.allclose(W, Wtrue, atol=1e-4) and np.allclose(b, btrue, atol=1e-4)
