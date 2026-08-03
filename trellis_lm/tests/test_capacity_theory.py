"""Float64 tests for trellis_lm.capacity_theory.

Every correctness test uses asymmetric dimensions (m != d) so a swapped
axis cannot silently pass. Default shapes: d=5, m=7 unless a construction
forces otherwise (orthonormal keys need P <= d, etc.)."""

import math

import pytest
import torch

from trellis_lm.capacity_theory import (
    clustered_columns,
    delta_memory,
    delta_unrolled_closed_form,
    frame_metrics,
    hebbian_memory,
    least_squares_memory,
    loss_update_jacobian,
    low_coherence_frame,
    near_duplicate_columns,
    orthonormal_columns,
    random_unit_columns,
    range_matched_targets,
    readout_metrics,
    recall_metrics,
    rls_memory,
    score_matrix,
    simplex_frame,
    trellis_memory,
    update_jacobian_autograd,
    update_jacobian_kron,
    welch_bound,
)

D, M = 5, 7  # d=5 key dim, m=7 code dim — asymmetric everywhere


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Hebbian identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("P", [4, 9])  # P <= d and P > d
def test_hebbian_readout_identity(P):
    """M_H W == eta A G_W, exactly, for any P."""
    g = _gen(0)
    W = random_unit_columns(D, P, generator=g)
    A = random_unit_columns(M, P, generator=g)
    eta = 0.7
    MH = hebbian_memory(W, A, eta)
    assert MH.shape == (M, D)
    assert torch.allclose(MH @ W, eta * A @ (W.T @ W), atol=1e-13)


def test_hebbian_score_matrix_is_gram_product():
    """S = A^T (M_H W) == eta G_A G_W."""
    g = _gen(1)
    P = 6
    W = random_unit_columns(D, P, generator=g)
    A = random_unit_columns(M, P, generator=g)
    eta = 1.3
    S = score_matrix(hebbian_memory(W, A, eta), W, A)
    assert torch.allclose(S, eta * (A.T @ A) @ (W.T @ W), atol=1e-13)


def test_hebbian_bridge_m_equals_P_exposes_overlaps():
    """m = P, A = I: M_H = eta W^T and z = M_H q has coordinates
    z_i = eta <w_i, q> — genuine per-pattern overlaps."""
    g = _gen(2)
    P = 6
    W = random_unit_columns(D, P, generator=g)
    A = torch.eye(P, dtype=torch.float64)
    eta = 0.9
    MH = hebbian_memory(W, A, eta)
    assert torch.allclose(MH, eta * W.T, atol=0.0)
    q = random_unit_columns(D, 1, generator=g).squeeze(1)
    z = MH @ q
    for i in range(P):
        assert torch.allclose(z[i], eta * (W[:, i] @ q), atol=1e-15)


def test_hebbian_vector_snr_monte_carlo():
    """Random unit keys + isotropic unit codes: E||crosstalk_j||^2 =
    (P-1)/d exactly, so the RMS vector SNR is sqrt(d/(P-1))."""
    d, m, P = 64, 48, 33
    sq_norms = []
    for seed in range(50):
        g = _gen(1000 + seed)
        W = random_unit_columns(d, P, generator=g)
        A = random_unit_columns(m, P, generator=g)
        Z = hebbian_memory(W, A, 1.0) @ W
        sq_norms.append(((Z - A) ** 2).sum(dim=0))
    mean_sq = torch.stack(sq_norms).mean()
    expected = (P - 1) / d  # = 0.5
    assert abs(mean_sq - expected) / expected < 0.10
    snr = 1.0 / math.sqrt(mean_sq)
    assert abs(snr - math.sqrt(d / (P - 1))) / math.sqrt(d / (P - 1)) < 0.05


# ---------------------------------------------------------------------------
# delta recurrence vs exact closed-form unrolling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gamma,beta",
    [
        (0.3, 1.0),
        (1.0, 0.95),
        (torch.linspace(0.1, 0.9, 8, dtype=torch.float64), 1.0),
        (
            torch.linspace(0.2, 0.6, 8, dtype=torch.float64),
            torch.linspace(0.90, 1.0, 8, dtype=torch.float64),
        ),
    ],
)
def test_delta_recurrence_matches_closed_form(gamma, beta):
    """One-pass delta == M0 R_1..R_P + sum_i gamma_i a_i w_i^T R_{i+1}..R_P
    with R_t = beta_t I - gamma_t w_t w_t^T, including nonzero M0."""
    g = _gen(3)
    P = 8
    W = random_unit_columns(D, P, generator=g)
    A = random_unit_columns(M, P, generator=g)
    M0 = torch.randn(M, D, generator=g, dtype=torch.float64)
    M_rec = delta_memory(W, A, gamma, beta, passes=1, M0=M0)
    M_cf = delta_unrolled_closed_form(W, A, gamma, beta, M0=M0)
    assert torch.allclose(M_rec, M_cf, atol=1e-12)


def test_delta_cyclic_passes_match_tiled_closed_form():
    """K cyclic passes == one pass over K column-tiled copies of (W, A)
    with the schedule tiled the same way."""
    g = _gen(4)
    P, K = 5, 3
    W = random_unit_columns(D, P, generator=g)
    A = random_unit_columns(M, P, generator=g)
    M0 = torch.randn(M, D, generator=g, dtype=torch.float64)
    gamma = torch.linspace(0.2, 0.8, P, dtype=torch.float64)
    beta = torch.linspace(0.92, 1.0, P, dtype=torch.float64)
    M_rec = delta_memory(W, A, gamma, beta, passes=K, M0=M0)
    M_cf = delta_unrolled_closed_form(
        W.repeat(1, K), A.repeat(1, K), gamma.repeat(K), beta.repeat(K), M0=M0
    )
    assert torch.allclose(M_rec, M_cf, atol=1e-12)


def test_orthonormal_one_pass_delta_stores_exactly():
    """Orthonormal keys, gamma=1, beta=1: one-pass delta interpolates
    (M W == A) — each write lands in an untouched orthogonal direction."""
    g = _gen(5)
    d, P = 7, 7  # m=5 below keeps m != d
    m = 5
    W = orthonormal_columns(d, P, generator=g)
    A = random_unit_columns(m, P, generator=g)
    Mem = delta_memory(W, A, gamma=1.0, beta=1.0, passes=1)
    assert torch.allclose(Mem @ W, A, atol=1e-12)
    # with zero crosstalk the result is also exactly the Hebbian memory
    assert torch.allclose(Mem, hebbian_memory(W, A, 1.0), atol=1e-12)


# ---------------------------------------------------------------------------
# least squares / RLS oracles
# ---------------------------------------------------------------------------


def test_least_squares_exact_interpolation_P_le_d():
    g = _gen(6)
    d, m, P = 6, 4, 5
    W = random_unit_columns(d, P, generator=g)
    A = random_unit_columns(m, P, generator=g)
    MLS = least_squares_memory(W, A)
    assert torch.allclose(MLS @ W, A, atol=1e-9)


def test_least_squares_projection_P_gt_d():
    """P > d: M_LS W = A (W^+ W) (projection onto the row space of W) and
    the residual satisfies the normal equations (M_LS W - A) W^T = 0."""
    g = _gen(7)
    P = 12
    W = random_unit_columns(D, P, generator=g)
    A = random_unit_columns(M, P, generator=g)
    MLS = least_squares_memory(W, A)
    Z = MLS @ W
    proj = torch.linalg.pinv(W) @ W
    assert torch.allclose(Z, A @ proj, atol=1e-9)
    assert torch.allclose(
        (Z - A) @ W.T, torch.zeros(M, D, dtype=torch.float64), atol=1e-9
    )
    # genuinely not interpolating: the projected readout differs from A
    assert (Z - A).norm() > 1e-2


def test_rls_matches_ridge_solution():
    g = _gen(8)
    d, m, P = 6, 4, 5
    lam = 1e-10
    W = random_unit_columns(d, P, generator=g)
    A = random_unit_columns(m, P, generator=g)
    Mr, Pcov = rls_memory(W, A, lam=lam, return_state=True)
    assert Pcov.shape == (d, d)  # the extra d x d state RLS carries
    ridge = (
        A @ W.T @ torch.linalg.inv(W @ W.T + lam * torch.eye(d, dtype=torch.float64))
    )
    assert torch.allclose(Mr, ridge, atol=1e-7)
    assert torch.allclose(Mr @ W, A, atol=1e-4)  # near-interpolation, small lam


def test_converged_delta_approaches_least_squares():
    """Cyclic LMS to convergence, P <= d, beta=1: reaches the interpolating
    min-norm solution A W^+ (it starts at 0 and stays in the row space)."""
    g = _gen(9)
    d, m, P = 6, 4, 5
    W = random_unit_columns(d, P, generator=g)
    A = random_unit_columns(m, P, generator=g)
    M_conv = delta_memory(W, A, gamma=0.5, beta=1.0, passes=None, tol=1e-13)
    MLS = least_squares_memory(W, A)
    assert torch.allclose(M_conv, MLS, atol=1e-6)
    assert torch.allclose(M_conv @ W, A, atol=1e-8)


def test_one_pass_delta_is_not_the_pseudoinverse():
    """The explicit rejection: one-pass LMS leaves real residual where the
    pseudoinverse interpolates exactly."""
    g = _gen(10)
    P = 5
    W = random_unit_columns(D, P, generator=g)
    A = random_unit_columns(M, P, generator=g)
    one_pass = delta_memory(W, A, gamma=0.5, beta=1.0, passes=1)
    r_one = recall_metrics(one_pass, W, A)["normalized_mse"]
    r_ls = recall_metrics(least_squares_memory(W, A), W, A)["normalized_mse"]
    assert r_ls < 1e-18
    assert r_one > 1e-3


# ---------------------------------------------------------------------------
# nonlinear Trellis update
# ---------------------------------------------------------------------------


def test_trellis_identity_matches_delta():
    g = _gen(11)
    P = 8
    W = random_unit_columns(D, P, generator=g)
    A = random_unit_columns(M, P, generator=g)
    M0 = torch.randn(M, D, generator=g, dtype=torch.float64)
    gamma = torch.linspace(0.2, 0.7, P, dtype=torch.float64)
    beta = torch.linspace(0.9, 1.0, P, dtype=torch.float64)
    M_tr = trellis_memory(W, A, "identity", gamma, beta, passes=2, M0=M0)
    M_dl = delta_memory(W, A, gamma, beta, passes=2, M0=M0)
    assert torch.allclose(M_tr, M_dl, atol=1e-14)


@pytest.mark.parametrize("phi", ["silu", "ln_silu", "l2_silu"])
def test_trellis_nonlinear_runs_and_is_finite(phi):
    """Closed-form (silu/ln_silu) and generic-autograd (l2_silu) VJP paths
    produce finite states with range-matched targets."""
    g = _gen(12)
    P = 8
    W = random_unit_columns(D, P, generator=g)
    A = range_matched_targets(phi, M, P, generator=g)
    Mem = trellis_memory(W, A, phi, gamma=0.3, beta=0.99, passes=1)
    assert Mem.shape == (M, D)
    assert torch.isfinite(Mem).all()


def test_range_matched_targets_apply_phi_over_code_axis():
    """ln_silu range-matched targets must be normalized over m (the code
    axis), not over P: each column has zero mean and ~unit variance."""
    g = _gen(13)
    P = 9
    A = range_matched_targets("ln_silu", M, P, generator=g)
    assert A.shape == (M, P)
    assert torch.allclose(A.mean(dim=0), torch.zeros(P, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(
        A.var(dim=0, unbiased=False), torch.ones(P, dtype=torch.float64), atol=1e-4
    )


# ---------------------------------------------------------------------------
# frame / Welch metrics
# ---------------------------------------------------------------------------


def test_frame_metrics_orthonormal():
    g = _gen(14)
    X = orthonormal_columns(7, 7, generator=g)
    fm = frame_metrics(X)
    assert fm["max_coherence"] < 1e-12
    assert abs(fm["frame_potential"] - 7.0) < 1e-10
    assert fm["tight_frame_distance"] < 1e-10  # square orthogonal is tight
    assert fm["welch_bound"] == 0.0
    assert abs(fm["effective_rank"] - 7.0) < 1e-8
    # P < dim: still zero coherence, but no longer a tight frame
    fm2 = frame_metrics(orthonormal_columns(7, 4, generator=g))
    assert fm2["max_coherence"] < 1e-12
    assert fm2["tight_frame_distance"] > 0.5


def test_frame_metrics_simplex_hits_welch():
    d = 5
    X = simplex_frame(d)
    P = d + 1
    fm = frame_metrics(X)
    wb = welch_bound(d, P)
    assert abs(wb - 1.0 / d) < 1e-15  # sqrt((P-d)/(d(P-1))) = 1/d at P=d+1
    assert abs(fm["max_coherence"] - wb) < 1e-9
    assert abs(fm["mean_sq_off_coherence"] - wb**2) < 1e-12
    assert fm["tight_frame_distance"] < 1e-9  # the simplex is a UNTF


def test_ensemble_geometry_sanity():
    g = _gen(15)
    dim, P = 5, 10
    near = frame_metrics(near_duplicate_columns(dim, P, eps=1e-2, generator=g))
    assert near["max_coherence"] > 0.99
    rand = frame_metrics(random_unit_columns(dim, P, generator=g))
    clus = frame_metrics(
        clustered_columns(dim, P, n_clusters=3, spread=0.1, generator=g)
    )
    assert clus["mean_sq_off_coherence"] > rand["mean_sq_off_coherence"]
    opt = frame_metrics(low_coherence_frame(dim, P, generator=g))
    wb = welch_bound(dim, P)
    assert opt["max_coherence"] >= wb - 1e-9  # Welch is a hard floor
    # frame-potential minimization reaches a unit-norm tight frame: the
    # potential hits its bound P^2/dim and the MEAN-SQUARE off-diagonal
    # coherence attains its Welch level wb^2 (max coherence need not).
    assert abs(opt["frame_potential"] - opt["frame_potential_bound"]) < 1e-6
    assert opt["tight_frame_distance"] < 1e-6
    assert abs(opt["mean_sq_off_coherence"] - wb**2) < 1e-9
    assert opt["frame_potential"] < rand["frame_potential"]


# ---------------------------------------------------------------------------
# update-Jacobian: Kronecker form vs full autograd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi", ["identity", "silu", "ln_silu"])
def test_update_jacobian_kron_matches_autograd(phi):
    """beta I - gamma (w w^T (x) H_ell) in column-major vec == the full
    autograd Jacobian of F(M) = beta M - gamma u(M w, a) w^T."""
    g = _gen(16)
    gamma, beta = 0.37, 0.93
    Mem = torch.randn(M, D, generator=g, dtype=torch.float64)
    w = random_unit_columns(D, 1, generator=g).squeeze(1)
    a = range_matched_targets(phi, M, 1, generator=g).squeeze(1)
    H = loss_update_jacobian(Mem @ w, a, phi)
    assert H.shape == (M, M)
    assert torch.allclose(H, H.T, atol=1e-10)  # Hessian of the inner loss
    if phi == "identity":
        assert torch.allclose(H, torch.eye(M, dtype=torch.float64), atol=1e-12)
    Jk = update_jacobian_kron(w, H, gamma, beta)
    Ja = update_jacobian_autograd(Mem, w, a, phi, gamma, beta)
    assert torch.allclose(Jk, Ja, atol=1e-9)


def test_update_jacobian_kron_order_matters():
    """The row-major flattening (H (x) w w^T) is NOT the column-major
    Jacobian for a coupling phi — the convention trap is real."""
    g = _gen(17)
    Mem = torch.randn(M, D, generator=g, dtype=torch.float64)
    w = random_unit_columns(D, 1, generator=g).squeeze(1)
    a = range_matched_targets("ln_silu", M, 1, generator=g).squeeze(1)
    H = loss_update_jacobian(Mem @ w, a, "ln_silu")
    gamma, beta = 0.37, 0.93
    eye = torch.eye(M * D, dtype=torch.float64)
    J_wrong = beta * eye - gamma * torch.kron(H, torch.outer(w, w))
    Ja = update_jacobian_autograd(Mem, w, a, "ln_silu", gamma, beta)
    assert not torch.allclose(J_wrong, Ja, atol=1e-6)


# ---------------------------------------------------------------------------
# recall metrics
# ---------------------------------------------------------------------------


def test_readout_metrics_perfect_and_degraded():
    g = _gen(18)
    P = 6
    A = random_unit_columns(M, P, generator=g)
    perfect = readout_metrics(A.clone(), A)
    assert perfect["normalized_mse"] < 1e-30
    assert abs(perfect["mean_cosine"] - 1.0) < 1e-12
    assert perfect["top1"] == 1.0
    assert perfect["min_margin"] > 0.0
    noisy = readout_metrics(
        A + 0.5 * torch.randn(M, P, generator=g, dtype=torch.float64), A
    )
    assert noisy["normalized_mse"] > perfect["normalized_mse"]
    assert noisy["mean_cosine"] < 1.0
