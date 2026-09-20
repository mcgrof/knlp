# SPDX-License-Identifier: GPL-2.0
"""The claims the weighted fit and the residual correction rest on.

Two of these are load-bearing for how results in this lane are reported.

That weights are normalised to mean one, which is what lets the same ridge grid
mean the same thing weighted and unweighted -- without it, a "weighting"
comparison would be partly a comparison of regularisation strengths.

And that the correction starts exactly at its affine baseline, which is what
makes any improvement attributable to training and makes a failed run degrade
to the baseline rather than to noise.

The rest pin the weighted statistics against a direct weighted fit, and pin the
participation ratio, the measure used to explain why the principled weighting
exponent loses to the linear one.
"""

import math

import pytest
import torch

from research.kv_translate.fit import SourceLayout, solve_ridge
from research.kv_translate.refine import (
    BlockResidual,
    behaviour_loss,
    kv_error_loss,
)
from research.kv_translate.weighted import (
    WeightedAccumulator,
    attention_mass,
    normalise,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# the weighting
# ---------------------------------------------------------------------------


def test_attention_mass_sums_what_each_position_receives():
    """Query axis is summed, not averaged: total attention received."""
    B, HQ, T = 1, 4, 6
    a = torch.zeros(B, HQ, T, T)
    a[:, :, :, 2] = 1.0  # every query attends entirely to position 2
    m = attention_mass(a, n_kv_heads=4, n_positions=T)
    assert m.shape == (4, T)
    assert torch.allclose(m[:, 2], torch.full((4,), float(T)))
    assert torch.allclose(m[:, 0], torch.zeros(4))


def test_attention_mass_folds_query_heads_into_their_group():
    B, HQ, T, HKV = 1, 6, 5, 2
    a = torch.rand(B, HQ, T, T)
    m = attention_mass(a, n_kv_heads=HKV, n_positions=T)
    assert m.shape == (HKV, T)
    grp = HQ // HKV
    expect = a[0].sum(dim=1).reshape(HKV, grp, T).mean(dim=1)
    assert torch.allclose(m, expect, atol=1e-6)


def test_normalise_gives_mean_one_per_head():
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, 0.0, 0.0]])
    n = normalise(w)
    assert torch.allclose(n.mean(dim=-1), torch.ones(2), atol=1e-6)


def test_normalise_keeps_the_ridge_grid_meaningful():
    """Weighted and unweighted Gram matrices must have comparable scale.

    The weighted fit reuses the unweighted fit's ridge grid. If weighting moved
    the Gram's magnitude, the same grid value would be a different amount of
    regularisation on each side and the comparison between them would be partly
    a comparison of regularisation.
    """
    g = torch.Generator().manual_seed(2)
    lay = SourceLayout(3, 1, 4)
    X = torch.randn(200, lay.n_features, generator=g, dtype=torch.float64)
    Y = torch.randn(200, 5, generator=g, dtype=torch.float64)
    cols = lay.columns_for([0, 1, 2], 0, False)

    flat = WeightedAccumulator(cols, 5)
    flat.add(X, Y, torch.ones(200, dtype=torch.float64))
    skew = WeightedAccumulator(cols, 5)
    w = normalise(torch.rand(200, generator=g, dtype=torch.float64).unsqueeze(0))[0]
    skew.add(X, Y, w)

    a = float(flat.xtx.diagonal().mean())
    b = float(skew.xtx.diagonal().mean())
    assert 0.5 < b / a < 2.0, f"weighting moved the Gram scale by {b / a:.2f}x"


def test_weighted_statistics_match_a_direct_weighted_fit():
    g = torch.Generator().manual_seed(3)
    lay = SourceLayout(2, 1, 4)
    n, d, dt = 150, lay.n_features, 3
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    Y = torch.randn(n, dt, generator=g, dtype=torch.float64)
    w = torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    cols = lay.columns_for([0, 1], 0, False)

    acc = WeightedAccumulator(cols, dt)
    for s, e in ((0, 60), (60, 120), (120, n)):
        acc.add(X[s:e], Y[s:e], w[s:e])
    G, C, xb, yb = acc.centred()

    sw = w.sum()
    xbar = (X * w.unsqueeze(1)).sum(0) / sw
    ybar = (Y * w.unsqueeze(1)).sum(0) / sw
    Xc, Yc = X - xbar, Y - ybar
    Gd = Xc.T @ (Xc * w.unsqueeze(1))
    Cd = (Xc * w.unsqueeze(1)).T @ Yc
    assert torch.allclose(G, Gd, atol=1e-8)
    assert torch.allclose(C, Cd, atol=1e-8)
    assert torch.allclose(xb, xbar, atol=1e-10)
    assert torch.allclose(yb, ybar, atol=1e-10)


def test_uniform_weights_reproduce_the_unweighted_fit():
    g = torch.Generator().manual_seed(4)
    lay = SourceLayout(2, 1, 4)
    n, dt = 120, 3
    X = torch.randn(n, lay.n_features, generator=g, dtype=torch.float64)
    Y = torch.randn(n, dt, generator=g, dtype=torch.float64)
    cols = lay.columns_for([0, 1], 0, False)
    acc = WeightedAccumulator(cols, dt)
    acc.add(X, Y, torch.ones(n, dtype=torch.float64))
    G, C, _, _ = acc.centred()
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    assert torch.allclose(G, Xc.T @ Xc, atol=1e-8)
    assert torch.allclose(C, Xc.T @ Yc, atol=1e-8)


def test_participation_ratio_reports_the_effective_width():
    """The measure used to explain the exponent result.

    Flat weights fit everything; a spike fits one position. This is why
    squaring a skewed attention mass is not a gentle reweighting.
    """
    from research.kv_translate.run_a2_probe import participation_ratio

    assert participation_ratio(torch.ones(512)) == pytest.approx(512.0)
    spike = torch.zeros(512)
    spike[3] = 1.0
    assert participation_ratio(spike) == pytest.approx(1.0)
    half = torch.zeros(512)
    half[:256] = 1.0
    assert participation_ratio(half) == pytest.approx(256.0)
    # squaring a skewed weight narrows it further
    skew = torch.rand(512, generator=torch.Generator().manual_seed(5)) ** 4
    assert participation_ratio(skew.pow(2)) < participation_ratio(skew)


# ---------------------------------------------------------------------------
# the correction
# ---------------------------------------------------------------------------


def test_correction_starts_at_exactly_zero():
    """Every arm begins at its affine baseline, so gains come from training."""
    m = BlockResidual(16, 8, hidden=4)
    x = torch.randn(7, 16)
    assert torch.equal(m(x), torch.zeros(7, 8))


def test_correction_can_move_once_trained():
    m = BlockResidual(16, 8, hidden=4)
    x = torch.randn(32, 16)
    target = torch.randn(32, 8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(200):
        loss = ((m(x) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert float(((m(x) - target) ** 2).mean()) < float((target**2).mean())


def test_correction_gradient_reaches_its_parameters():
    m = BlockResidual(12, 6, hidden=8)
    out = m(torch.randn(5, 12))
    out.sum().backward()
    assert m.fc1.weight.grad is not None
    assert m.fc2.weight.grad is not None
    assert float(m.fc2.weight.grad.abs().sum()) > 0


def test_kv_error_loss_is_zero_on_a_perfect_match_and_scale_free():
    a = [torch.randn(1, 2, 4, 3) for _ in range(3)]
    assert float(kv_error_loss(a, a)) == pytest.approx(0.0)
    b = [x + 1.0 for x in a]
    assert float(kv_error_loss(b, a)) == pytest.approx(1.0, abs=1e-6)


def test_behaviour_loss_is_zero_when_the_target_is_reproduced():
    logits = torch.randn(1, 6, 11)
    ref = torch.log_softmax(logits, dim=-1)
    assert float(behaviour_loss(ref, logits)) == pytest.approx(0.0, abs=1e-6)
    worse = logits + torch.randn_like(logits)
    assert float(behaviour_loss(ref, worse)) > 0


def test_behaviour_loss_carries_gradient_to_the_prediction():
    """The whole point: the target's divergence has to be differentiable."""
    ref = torch.log_softmax(torch.randn(1, 4, 9), dim=-1)
    logits = torch.randn(1, 4, 9, requires_grad=True)
    behaviour_loss(ref, logits).backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
