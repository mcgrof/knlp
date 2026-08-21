# SPDX-License-Identifier: GPL-2.0
"""Gate-0 Householder construction properties. CPU, no GPU, no model."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from kv_house.transforms import (
    apply_householder,
    householder_matrix,
    householder_vector,
)

BLOCK_SIZES = (4, 8, 16, 32, 64)


def _unit(b, seed):
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(b, generator=gen, dtype=torch.float64)
    return u / u.norm()


def test_householder_maps_direction_to_e0():
    for b in BLOCK_SIZES:
        for seed in range(5):
            u = _unit(b, seed)
            v, sign = householder_vector(u)
            h = householder_matrix(v)
            e0 = torch.zeros(b, dtype=torch.float64)
            e0[0] = sign
            assert torch.allclose(h @ u, e0, atol=1e-12)


def test_householder_orthogonal_involution():
    for b in BLOCK_SIZES:
        v, _ = householder_vector(_unit(b, 42))
        h = householder_matrix(v)
        eye = torch.eye(b, dtype=torch.float64)
        assert torch.allclose(h.T @ h, eye, atol=1e-12)
        assert torch.allclose(h @ h, eye, atol=1e-12)


def test_householder_preserves_frobenius_norm():
    for b in BLOCK_SIZES:
        v, _ = householder_vector(_unit(b, 7))
        x = torch.randn(b, 64, dtype=torch.float64)
        assert torch.allclose(
            apply_householder(x, v).norm(), x.norm(), atol=1e-10
        )


def test_degenerate_directions_are_stable():
    """u aligned with +-e0 must not cancel; the LAPACK sign choice
    guarantees |v[0]| >= 1."""
    for b in BLOCK_SIZES:
        for s in (1.0, -1.0):
            u = torch.zeros(b, dtype=torch.float64)
            u[0] = s
            v, sign = householder_vector(u)
            assert float(v.abs()[0]) >= 1.0
            h = householder_matrix(v)
            got = h @ u
            e0 = torch.zeros(b, dtype=torch.float64)
            e0[0] = sign
            assert torch.allclose(got, e0, atol=1e-12)


def test_rank_one_apply_matches_dense():
    for b in BLOCK_SIZES:
        v, _ = householder_vector(_unit(b, 3))
        x = torch.randn(b, 32, dtype=torch.float64)
        assert torch.allclose(apply_householder(x, v), householder_matrix(v) @ x, atol=1e-12)
