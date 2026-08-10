"""Tests for the signed reciprocal credit spectrum utilities.

Covers the math-contract tests from the spectral_delta_ra plan:
planted-mode recovery, the synthetic cancellation case (trace 0 with
nonzero spectral mass), the exact first-order loss identity for a
linear loss, permutation-null behavior, split-half stability, basis
serialization round-trip, and the skew-symmetric Q/K asymmetry path.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fim.reciprocal_attention import spectral_credit as sc  # noqa: E402


def _orthonormal(d, r, seed=0):
    return sc.haar_random_basis(d, r, seed=seed)


def _planted_gr(n, d, u, scale=1.0, noise=0.0, seed=0):
    """Rows where the G/R pairing concentrates credit on direction u.

    R_i = s_i * u + noise, G_i = -s_i * u + noise, so
    M = G^T R / N ~= -E[s^2] u u^T and H = -sym(M) has a positive
    planted eigenvalue along u.
    """
    gen = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 1, generator=gen)
    g = -scale * s * u.T + noise * torch.randn(n, d, generator=gen)
    r = scale * s * u.T + noise * torch.randn(n, d, generator=gen)
    return g, r


def test_h_symmetric_real_eigenvalues():
    gen = torch.Generator().manual_seed(0)
    g = torch.randn(500, 16, generator=gen)
    r = torch.randn(500, 16, generator=gen)
    acc = sc.HeadCreditAccumulator(d=16)
    acc.update(g, r)
    h = acc.finalize()["H"]
    assert torch.allclose(h, h.T)
    lam, u = sc.sym_eig_by_abs(h)
    assert lam.dtype == torch.float64
    assert torch.isfinite(lam).all()
    # ordered by descending absolute value
    assert (lam.abs()[:-1] >= lam.abs()[1:] - 1e-12).all()


def test_accumulator_chunked_equals_single_shot():
    gen = torch.Generator().manual_seed(1)
    g = torch.randn(300, 8, generator=gen)
    r = torch.randn(300, 8, generator=gen)
    one = sc.HeadCreditAccumulator(d=8)
    one.update(g, r)
    chunked = sc.HeadCreditAccumulator(d=8)
    for lo in range(0, 300, 64):
        chunked.update(g[lo : lo + 64], r[lo : lo + 64])
    a, b = one.finalize(), chunked.finalize()
    for key in ("M", "H", "C_z", "C_r", "C_g"):
        assert torch.allclose(a[key], b[key], atol=1e-10)
    assert a["N"] == b["N"]


def test_planted_mode_recovery():
    d = 16
    u = _orthonormal(d, 1, seed=3)
    g, r = _planted_gr(4000, d, u, noise=0.05, seed=4)
    acc = sc.HeadCreditAccumulator(d=d)
    acc.update(g, r)
    h = acc.finalize()["H"]
    lam, vecs = sc.sym_eig_by_abs(h)
    # top mode is positive (the -s/+s pairing is loss-reducing under a
    # positive gate) and aligned with the planted direction
    assert lam[0] > 0
    assert abs(float(vecs[:, 0] @ u.squeeze())) > 0.99
    stats = sc.signed_spectrum_stats(lam)
    assert stats["top1_mass_fraction"] > 0.9


def test_synthetic_cancellation_case():
    """Eigenvalues [+4, -4, 0, ...]: scalar gate sees nothing, two mode
    gates see everything."""
    d = 8
    basis = _orthonormal(d, 2, seed=5)
    u1, u2 = basis[:, :1], basis[:, 1:2]
    h = 4.0 * (u1 @ u1.T) - 4.0 * (u2 @ u2.T)
    lam, vecs = sc.sym_eig_by_abs(h)
    stats = sc.signed_spectrum_stats(lam)
    assert abs(stats["trace"]) < 1e-9
    assert abs(stats["spectral_mass"] - 8.0) < 1e-9
    assert stats["cancellation_ratio"] > 0.999
    n = 1000
    # scalar gate: first-order signal -N * beta * trace(H) == 0
    beta_scalar = 0.1
    assert abs(-n * beta_scalar * stats["trace"]) < 1e-6
    # sign-matched mode gates: -N * sum beta_i lambda_i < 0
    beta_modes = 0.1 * torch.sign(lam[:2])
    first_order = -n * float((beta_modes * lam[:2]).sum())
    assert first_order < -100


def test_first_order_identity_linear_loss():
    """For a linear loss the first-order formula is exact:
    L(Y_std + R U diag(beta) U^T) - L(Y_std) == -N sum_i beta_i lam_i."""
    torch.manual_seed(6)
    n, d, r = 200, 12, 4
    g = torch.randn(n, d, dtype=torch.float64)
    rr = torch.randn(n, d, dtype=torch.float64)
    m = g.T @ rr / n
    h = sc.signed_credit_from_m(m)
    lam, u = sc.sym_eig_by_abs(h)
    u_r, lam_r = u[:, :r], lam[:r]
    beta = torch.tensor([0.3, -0.2, 0.05, 0.5], dtype=torch.float64)
    correction = rr @ u_r @ torch.diag(beta) @ u_r.T
    # linear loss L(Y) = sum_i g_i . y_i, so dL/dY == G exactly
    delta_l = float((g * correction).sum())
    predicted = -n * float((beta * lam_r).sum())
    assert abs(delta_l - predicted) < 1e-8


def test_permutation_null_planted_exceeds_random_does_not():
    d = 12
    u = _orthonormal(d, 1, seed=7)
    g, r = _planted_gr(1500, d, u, noise=0.1, seed=8)
    planted = sc.permutation_null(g, r, n_perm=60, seed=9)
    assert planted["exceeds_p95"]
    gen = torch.Generator().manual_seed(10)
    g2 = torch.randn(1500, d, generator=gen)
    r2 = torch.randn(1500, d, generator=gen)
    random_case = sc.permutation_null(g2, r2, n_perm=60, seed=11)
    # independent G/R should sit inside its own null distribution
    assert random_case["percentile_of_actual"] < 0.99


def test_split_half_overlap_planted_stable():
    d = 16
    u = _orthonormal(d, 1, seed=12)
    g, r = _planted_gr(4000, d, u, noise=0.05, seed=13)
    out = sc.split_half_overlap(g, r, ranks=(1, 2), seed=14)
    assert out["split_half_overlap_r1"] > 0.95
    assert 0.0 <= out["split_half_overlap_r2"] <= 1.0 + 1e-9


def test_subspace_overlap_bounds():
    u = _orthonormal(10, 3, seed=15)
    assert abs(sc.subspace_overlap(u, u) - 1.0) < 1e-9
    v = _orthonormal(10, 3, seed=16)
    val = sc.subspace_overlap(u, v)
    assert 0.0 <= val <= 1.0 + 1e-9
    # invariance to sign flips of basis columns
    flipped = v.clone()
    flipped[:, 0] = -flipped[:, 0]
    assert abs(sc.subspace_overlap(u, flipped) - val) < 1e-9


def test_haar_basis_orthonormal():
    u = sc.haar_random_basis(24, 8, seed=17)
    eye = torch.eye(8, dtype=torch.float64)
    assert torch.allclose(u.T @ u, eye, atol=1e-10)
    # deterministic under the same seed
    v = sc.haar_random_basis(24, 8, seed=17)
    assert torch.equal(u, v)


def test_s_pm_ratio_extremes():
    t = 6
    sym = torch.randn(t, t)
    sym = 0.5 * (sym + sym.T)
    assert sc.s_pm_ratio(sym) < 1e-6
    skew = torch.randn(t, t)
    skew = 0.5 * (skew - skew.T)
    assert sc.s_pm_ratio(skew) > 1e3


def test_qk_asymmetry_accumulator():
    d = 8
    acc = sc.QKAsymmetryAccumulator(d)
    gen = torch.Generator().manual_seed(18)
    q = torch.randn(2000, d, generator=gen)
    acc.update(q, q.clone())  # K == Q: no asymmetry
    out = acc.finalize()
    assert out["rho_asym"] < 1e-6
    # planted rotation between two features produces skew mass
    acc2 = sc.QKAsymmetryAccumulator(d)
    k = q.clone()
    k[:, 0], k[:, 1] = -q[:, 1], q[:, 0]
    acc2.update(q, k)
    out2 = acc2.finalize()
    assert out2["rho_asym"] > 0.5
    # skew modes pair up: top two singular values are equal
    sv = out2["singular_values"]
    assert abs(float(sv[0] - sv[1])) < 1e-6 * max(1.0, float(sv[0]))


def test_psd_stats_identity_matrix():
    c = torch.eye(16, dtype=torch.float64)
    stats = sc.psd_spectrum_stats(c)
    assert abs(stats["trace"] - 16.0) < 1e-9
    assert abs(stats["effective_rank"] - 16.0) < 1e-6
    assert abs(stats["top1_explained_variance"] - 1.0 / 16.0) < 1e-9


def test_diag_ridge_recorded():
    c = torch.eye(4, dtype=torch.float64) * 2.0
    ridged, ridge = sc.diag_ridge(c, eps=1e-8)
    assert abs(ridge - 1e-8 * 2.0) < 1e-15
    assert torch.allclose(ridged, c + ridge * torch.eye(4, dtype=torch.float64))


def test_basis_save_load_roundtrip(tmp_path):
    u = {"L3H4": sc.haar_random_basis(16, 4, seed=19)}
    lam = {"L3H4": torch.tensor([0.5, -0.3, 0.1, -0.05], dtype=torch.float64)}
    meta = {
        "basis_source": "signed_credit",
        "model_commit": "deadbeef",
        "calibration_seed": 0,
    }
    paths = sc.save_basis(tmp_path, u, lam, meta)
    loaded = sc.load_basis(tmp_path)
    assert torch.equal(loaded["U_by_layer_head"]["L3H4"], u["L3H4"])
    assert torch.equal(loaded["eigenvalues_by_layer_head"]["L3H4"], lam["L3H4"])
    assert loaded["meta"]["basis_source"] == "signed_credit"
    assert Path(paths["basis_pt"]).exists()


def test_signed_stats_normalization():
    lam = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    stats = sc.signed_spectrum_stats(lam, gr_norm_mean=7.0)
    assert abs(stats["spectral_mass"] - 3.5) < 1e-12
    assert abs(stats["normalized_spectral_mass"] - 0.5) < 1e-9
    assert abs(stats["positive_mass"] - 2.5) < 1e-12
    assert abs(stats["negative_mass"] - 1.0) < 1e-12
