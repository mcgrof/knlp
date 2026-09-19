# SPDX-License-Identifier: GPL-2.0
"""The full-rank invariance trap, pinned before any GPU number exists.

The Jacobian-KV plan states the trap it is most likely to fall into: a fixed,
full-rank, positive output metric applied to every sample cannot change an
unconstrained full-rank least-squares affine solution.  An implementation that
reports a win under those conditions is measuring a bug, not a receiver metric.

These tests demonstrate the identity on real numerics with a solver that does
not assume it (``solver="kron"`` forms the weighted normal equations directly),
and then pin the four ways a receiver metric is *allowed* to matter:

  * the metric varies per sample;
  * the mapper is rank constrained;
  * a regularizer that does not commute with the metric is present;
  * the mapper class is otherwise restricted.

A fifth case is a trap of its own and is tested as such: when the metric is
singular -- and the categorical Fisher ``diag(p) - p p^T`` always is, it
annihilates the all-ones direction -- the weighted solution *does* move, but
only inside the metric's null space, where it changes no objective value.  A
"win" of that shape is not a win.

Everything here is CPU-only, float64, seeded, and runs in seconds.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv.fit import (  # noqa: E402
    fit_affine,
    psd_inv_sqrt,
    psd_sqrt,
    weighted_sq_error,
)

torch.manual_seed(0)

N, DS, DT = 64, 6, 5


def _data(seed=0, n=N, ds=DS, dt=DT, noise=0.4):
    """Paired source/target rows with a real affine signal plus noise."""
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(n, ds, generator=g, dtype=torch.float64)
    M_true = torch.randn(ds, dt, generator=g, dtype=torch.float64)
    b_true = torch.randn(dt, generator=g, dtype=torch.float64)
    T = C @ M_true + b_true + noise * torch.randn(
        n, dt, generator=g, dtype=torch.float64
    )
    return C, T


def _spd(dt=DT, seed=1, cond_boost=0.5):
    """A well-conditioned symmetric positive definite metric."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dt, dt, generator=g, dtype=torch.float64)
    return A @ A.transpose(0, 1) + cond_boost * torch.eye(dt, dtype=torch.float64)


def _categorical_fisher(logits):
    """``diag(p) - p p^T``.  Symmetric PSD, and singular by construction."""
    p = torch.softmax(logits, dim=-1)
    return torch.diag(p) - torch.outer(p, p)


# ---------------------------------------------------------------------------
# the identity itself
# ---------------------------------------------------------------------------


def test_fixed_full_rank_metric_cannot_move_an_unconstrained_fit():
    """The core trap: same solution with and without a fixed invertible metric.

    Solved through the general Kronecker normal equations, which never assume
    the collapse, so agreement is evidence rather than restatement.
    """
    C, T = _data()
    G = _spd()

    plain = fit_affine(C, T, solver="kron")
    weighted = fit_affine(C, T, G=G, solver="kron")

    assert torch.allclose(plain.M, weighted.M, atol=1e-9, rtol=0)
    assert torch.allclose(plain.b, weighted.b, atol=1e-9, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("cond_boost", [1e-3, 0.5, 10.0])
def test_invariance_holds_across_seeds_and_conditioning(seed, cond_boost):
    """The identity is not an artefact of one draw or one condition number."""
    C, T = _data(seed=seed)
    G = _spd(seed=seed + 100, cond_boost=cond_boost)
    plain = fit_affine(C, T, solver="kron")
    weighted = fit_affine(C, T, G=G, solver="kron")
    scale = plain.M.abs().max().clamp(min=1.0)
    assert (plain.M - weighted.M).abs().max() < 1e-7 * scale


def test_invariance_holds_without_an_intercept():
    """Centring is not what makes the metric vanish."""
    C, T = _data()
    G = _spd()
    plain = fit_affine(C, T, solver="kron", fit_intercept=False)
    weighted = fit_affine(C, T, G=G, solver="kron", fit_intercept=False)
    assert torch.allclose(plain.M, weighted.M, atol=1e-9, rtol=0)


def test_invariance_survives_scalar_sample_weights():
    """Scalar weights are shared by both arms, so they do not rescue the metric.

    This is the near miss worth pinning: attention-local *scalar* importance
    weighting changes the fit, but it changes both arms identically, so it is
    not evidence for a receiver metric.
    """
    C, T = _data()
    G = _spd()
    g = torch.Generator().manual_seed(7)
    w = torch.rand(N, generator=g, dtype=torch.float64) + 0.1

    plain = fit_affine(C, T, sample_weights=w, solver="kron")
    weighted = fit_affine(C, T, G=G, sample_weights=w, solver="kron")
    assert torch.allclose(plain.M, weighted.M, atol=1e-9, rtol=0)

    # ... and the weights really did do something on their own.
    unweighted = fit_affine(C, T, solver="kron")
    assert (unweighted.M - plain.M).abs().max() > 1e-6


def test_whiten_and_kron_solvers_agree_when_both_are_valid():
    """Two independent code paths, one answer: guards against a solver bug."""
    C, T = _data()
    G = _spd()
    kron = fit_affine(C, T, G=G, solver="kron")
    whiten = fit_affine(C, T, G=G, rank=min(DS, DT), solver="whiten")
    assert torch.allclose(kron.M, whiten.M, atol=1e-8, rtol=0)


# ---------------------------------------------------------------------------
# the legitimate escapes
# ---------------------------------------------------------------------------


def test_per_sample_metric_does_move_the_fit():
    """A sample-dependent metric is a real escape from the trap."""
    C, T = _data()
    g = torch.Generator().manual_seed(11)
    Gp = torch.stack(
        [_categorical_fisher(torch.randn(DT, generator=g, dtype=torch.float64) * 2.0)
         + 0.05 * torch.eye(DT, dtype=torch.float64) for _ in range(N)]
    )
    plain = fit_affine(C, T, solver="kron")
    weighted = fit_affine(C, T, G_per_sample=Gp, solver="kron")
    assert (plain.M - weighted.M).abs().max() > 1e-4

    # and it is a genuine optimum of its own objective, not noise
    obj_w = weighted_sq_error(C, T, weighted, G_per_sample=Gp)
    obj_p = weighted_sq_error(C, T, plain, G_per_sample=Gp)
    assert obj_w < obj_p


def test_rank_constraint_makes_a_fixed_metric_matter():
    """Under a rank cut the metric chooses which directions survive."""
    C, T = _data()
    G = _spd()
    r = 2
    plain = fit_affine(C, T, rank=r, solver="whiten")
    weighted = fit_affine(C, T, G=G, rank=r, solver="whiten")

    assert torch.linalg.matrix_rank(plain.M, tol=1e-8).item() <= r
    assert torch.linalg.matrix_rank(weighted.M, tol=1e-8).item() <= r
    assert (plain.M - weighted.M).abs().max() > 1e-4

    # each wins under its own metric -- the standard sanity pair
    assert weighted_sq_error(C, T, weighted, G=G) < weighted_sq_error(C, T, plain, G=G)
    assert weighted_sq_error(C, T, plain) < weighted_sq_error(C, T, weighted)


def test_ridge_makes_a_fixed_metric_matter():
    """An identity-metric penalty does not commute with a non-identity metric."""
    C, T = _data()
    G = _spd()
    plain = fit_affine(C, T, ridge=1.0, solver="kron")
    weighted = fit_affine(C, T, G=G, ridge=1.0, solver="kron")
    assert (plain.M - weighted.M).abs().max() > 1e-4


def test_metric_aligned_ridge_restores_the_invariance():
    """The converse, which is what makes the ridge escape a real one.

    Penalising ``tr(M G M')`` instead of ``||M||_F^2`` commutes with the metric,
    so the fit collapses back to the unweighted ridge solution.  Anyone who
    reports a ridge-driven win must therefore state which penalty they used.
    """
    C, T = _data()
    G = _spd()
    S = psd_sqrt(G)
    S_inv = psd_inv_sqrt(G)
    lam = 1.0

    plain = fit_affine(C, T, ridge=lam, solver="kron")

    # tr(M G M') penalty == identity-ridge on M@S, solved in whitened space
    Cc = C - C.mean(0)
    Tc = T - T.mean(0)
    gram = Cc.transpose(0, 1) @ Cc + lam * torch.eye(DS, dtype=torch.float64)
    M_tilde = torch.linalg.solve(gram, Cc.transpose(0, 1) @ (Tc @ S))
    M_aligned = M_tilde @ S_inv

    assert torch.allclose(M_aligned, plain.M, atol=1e-8, rtol=0)


# ---------------------------------------------------------------------------
# the null-space trap
# ---------------------------------------------------------------------------


def test_singular_metric_moves_the_fit_only_inside_its_null_space():
    """A rank-deficient metric is not an escape, it is a way to fool yourself.

    The categorical Fisher is singular.  Weighting by it leaves the fit
    underdetermined along its null direction, so the solver returns a different
    matrix -- but the difference costs nothing under the metric, and it makes
    the plain unweighted error strictly worse.  A reported improvement that
    lives here is measuring an arbitrary choice of solution, not a better map.
    """
    C, T = _data()
    logits = torch.linspace(-2.0, 2.0, DT, dtype=torch.float64)
    F = _categorical_fisher(logits)
    assert torch.linalg.matrix_rank(F, tol=1e-10).item() == DT - 1

    plain = fit_affine(C, T, solver="kron")
    weighted = fit_affine(C, T, G=F, solver="kron")

    delta = weighted.M - plain.M
    assert delta.abs().max() > 1e-6, "expected the singular metric to move the fit"

    # the movement is exactly rank one along the metric's null direction:
    # every column of delta is the same vector, so delta == u @ ones'
    assert torch.linalg.matrix_rank(delta, tol=1e-8).item() == 1
    col0 = delta[:, :1]
    assert torch.allclose(delta, col0.expand_as(delta), atol=1e-8, rtol=0)

    # therefore the metric annihilates the whole movement
    assert (delta @ F).abs().max() < 1e-9 * max(1.0, float(delta.abs().max()))

    # so it buys nothing under the metric, and costs under plain error
    obj_w = weighted_sq_error(C, T, weighted, G=F)
    obj_p = weighted_sq_error(C, T, plain, G=F)
    assert abs(float(obj_w - obj_p)) < 1e-8 * max(1.0, abs(float(obj_p)))
    assert weighted_sq_error(C, T, plain) <= weighted_sq_error(C, T, weighted) + 1e-9


def test_regularised_singular_metric_is_still_invariant():
    """Adding a floor to the Fisher makes it invertible, and the trap returns."""
    C, T = _data()
    logits = torch.linspace(-2.0, 2.0, DT, dtype=torch.float64)
    F = _categorical_fisher(logits) + 1e-3 * torch.eye(DT, dtype=torch.float64)
    plain = fit_affine(C, T, solver="kron")
    weighted = fit_affine(C, T, G=F, solver="kron")
    assert torch.allclose(plain.M, weighted.M, atol=1e-8, rtol=0)


# ---------------------------------------------------------------------------
# solver hygiene
# ---------------------------------------------------------------------------


def test_psd_sqrt_round_trips():
    G = _spd()
    S = psd_sqrt(G)
    assert torch.allclose(S @ S, G, atol=1e-9, rtol=0)
    assert torch.allclose(S, S.transpose(0, 1), atol=1e-12, rtol=0)
    assert torch.allclose(psd_inv_sqrt(G) @ S, torch.eye(DT, dtype=torch.float64), atol=1e-8)


def test_fit_recovers_a_noiseless_affine_map():
    """Floor check: with no noise the unconstrained fit is exact."""
    C, T = _data(noise=0.0)
    amap = fit_affine(C, T, solver="kron")
    assert torch.allclose(amap.apply(C), T, atol=1e-8, rtol=0)


def test_rejects_contradictory_arguments():
    C, T = _data()
    G = _spd()
    with pytest.raises(ValueError):
        fit_affine(C, T, G=G, G_per_sample=torch.zeros(N, DT, DT, dtype=torch.float64))
    with pytest.raises(ValueError):
        fit_affine(C, T, rank=2, solver="kron")
    with pytest.raises(ValueError):
        fit_affine(C, T[:-1], solver="kron")
