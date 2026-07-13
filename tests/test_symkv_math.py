# SPDX-License-Identifier: GPL-2.0
"""SymKV Gate-0: CPU FP32/FP64 math-core unit tests. No GPU, no model.

These lock the algebra before any GPU/model lane runs. The load-bearing one is
test_pca_lower_bounds_symkv: PCA is the raw-reconstruction optimum (Eckart-Young),
so the symmetry-constrained SymKV basis can NEVER beat PCA on raw KV MSE. That is
not a bug -- it is the premise. The whole line is the bet that SymKV nonetheless
wins on PREDICTIVE quality (measured later, on a model), and if it cannot beat even
random_sym there, the hypothesis is dead.
"""

import math

import torch

from symkv import (SymKVConfig, HeadCovariance, consensus_mode, perp_projector,
                   build_basis, encode, decode, reconstruct, recon_mse,
                   byte_accounting)

torch.manual_seed(0)
H, D = 8, 16  # Llama-3.2-3B head layout scale (num_key_value_heads=8, small D for speed)


def _fake_cov(n_tokens=200):
    """A head covariance with a real consensus component plus anisotropic structure."""
    g = torch.Generator().manual_seed(1)
    acc = HeadCovariance(H)
    shared = torch.randn(n_tokens, 1, D, generator=g)          # consensus signal
    perhead = 0.5 * torch.randn(n_tokens, H, D, generator=g)   # per-head variation
    X = shared + perhead                                       # (T, H, D)
    for t in range(n_tokens):
        acc.update(X[t])
    return acc.covariance(), X


# 1
def test_consensus_unit_norm():
    u0 = consensus_mode(H)
    assert torch.allclose(u0, torch.ones(H, dtype=u0.dtype) / math.sqrt(H))
    assert abs(float(torch.linalg.norm(u0)) - 1.0) < 1e-12


# 2
def test_perp_is_projector():
    P = perp_projector(consensus_mode(H))
    assert torch.allclose(P, P.T, atol=1e-12)
    assert torch.allclose(P @ P, P, atol=1e-10)


# 3
def test_perp_kills_consensus():
    u0 = consensus_mode(H)
    assert float(torch.linalg.norm(perp_projector(u0) @ u0)) < 1e-10


# 4
def test_symkv_basis_orthonormal():
    C, _ = _fake_cov()
    for m in (1, 2, 4, H):
        B = build_basis("symkv_raw", H, m, C=C)
        assert torch.allclose(B.T @ B, torch.eye(m, dtype=B.dtype), atol=1e-8), m


# 5
def test_symkv_forces_consensus_mode0():
    C, _ = _fake_cov()
    B = build_basis("symkv_raw", H, 4, C=C)
    u0 = consensus_mode(H)
    # column 0 is +/- u0
    assert min(float(torch.linalg.norm(B[:, 0] - u0)),
               float(torch.linalg.norm(B[:, 0] + u0))) < 1e-8


# 6
def test_pca_basis_orthonormal():
    C, _ = _fake_cov()
    for m in (1, 2, 4, H):
        B = build_basis("pca_head", H, m, C=C)
        assert torch.allclose(B.T @ B, torch.eye(m, dtype=B.dtype), atol=1e-8), m


# 7
def test_full_basis_exact():
    _, X = _fake_cov()
    B = build_basis("full", H, H)
    assert recon_mse(X, B) < 1e-18


# 8
def test_codec_roundtrip_full():
    _, X = _fake_cov()
    B = build_basis("full", H, H)
    assert torch.allclose(reconstruct(X.double(), B), X.double(), atol=1e-10)


# 9
def test_mse_monotone_in_modes():
    C, X = _fake_cov()
    prev = float("inf")
    for m in (1, 2, 3, 4, 6, 8):
        e = recon_mse(X, build_basis("pca_head", H, m, C=C))
        assert e <= prev + 1e-12, (m, e, prev)
        prev = e


# 10
def test_byte_accounting_matched_and_ratio():
    a = byte_accounting(H, D, n_modes=2, n_tokens=100000)
    b = byte_accounting(H, D, n_modes=4, n_tokens=100000)
    # more modes -> more stored bytes -> lower ratio
    assert a["stored_bytes_per_token"] < b["stored_bytes_per_token"]
    assert a["compression_ratio"] > b["compression_ratio"]
    # full rank (m=H), no basis amortization and no metadata -> exactly no
    # compression; with metadata the stored side is strictly larger (ratio < 1).
    full = byte_accounting(H, D, n_modes=H, n_tokens=10**9, meta_bytes=0)
    assert abs(full["compression_ratio"] - 1.0) < 1e-3
    full_meta = byte_accounting(H, D, n_modes=H, n_tokens=10**9, meta_bytes=16)
    assert full_meta["compression_ratio"] < full["compression_ratio"]


# 11  -- the premise: PCA lower-bounds every rank-m basis on raw reconstruction
def test_pca_lower_bounds_symkv():
    C, X = _fake_cov()
    for m in (1, 2, 3, 4):
        pca = recon_mse(X, build_basis("pca_head", H, m, C=C))
        sym = recon_mse(X, build_basis("symkv_raw", H, m, C=C))
        rnd = recon_mse(X, build_basis("random_sym", H, m, C=C, seed=0))
        # PCA is optimal; sym is constrained so cannot beat it (allow fp slack)
        assert pca <= sym + 1e-9, (m, pca, sym)
        # but a covariance-fit sym basis should beat a random symmetric one
        assert sym <= rnd + 1e-9, (m, sym, rnd)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} SymKV Gate-0 math tests passed")
