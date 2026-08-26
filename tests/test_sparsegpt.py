"""Tests for the SparseGPT baseline (fim/fisher_pruning/sparsegpt.py).

The point of SparseGPT over a scoring rule is that surviving weights
are UPDATED to absorb the pruned ones, so the tests check for that
explicitly rather than only checking that zeros appeared.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.sparsegpt import (  # noqa: E402
    _HessianAccumulator,
    _prune_linear,
)


def test_hessian_matches_direct_computation():
    torch.manual_seed(0)
    lin = nn.Linear(6, 4, bias=False)
    acc = _HessianAccumulator(lin)
    xs = [torch.randn(2, 5, 6) for _ in range(3)]
    for x in xs:
        acc.add(x)
    flat = torch.cat([x.reshape(-1, 6) for x in xs]).float()
    expected = 2.0 / flat.shape[0] * (flat.T @ flat)
    torch.testing.assert_close(acc.h, expected, rtol=1e-4, atol=1e-5)


def test_prune_reaches_target_sparsity():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=False)
    x = torch.randn(512, 64)
    h = 2.0 / 512 * (x.T @ x)
    _prune_linear(lin, h, sparsity=0.5)
    got = (lin.weight == 0).float().mean().item()
    assert abs(got - 0.5) < 0.02, got


def test_survivors_are_updated_not_just_masked():
    """The defining property: unpruned weights must MOVE."""
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=False)
    before = lin.weight.data.clone()
    x = torch.randn(512, 64)
    h = 2.0 / 512 * (x.T @ x)
    _prune_linear(lin, h, sparsity=0.5)
    after = lin.weight.data
    survivors = after != 0
    moved = (after[survivors] - before[survivors]).abs()
    assert moved.max() > 1e-6, "survivors unchanged: this is masking, not SparseGPT"
    assert torch.isfinite(after).all()


def test_reconstruction_beats_plain_masking_on_the_layer():
    """Against the same mask budget, reconstruction must lower the
    layer's output error on held-out inputs.

    Inputs must be CORRELATED for this to mean anything: with
    isotropic inputs the Hessian is a multiple of the identity, its
    inverse has no off-diagonal mass, and SparseGPT correctly
    degenerates to magnitude pruning with nothing to reconstruct.
    Real activations are strongly correlated.
    """
    torch.manual_seed(0)
    lin = nn.Linear(96, 48, bias=False)
    w0 = lin.weight.data.clone()
    mix = torch.randn(96, 96) / math.sqrt(96) + 0.3 * torch.eye(96)
    xcal = torch.randn(4096, 96) @ mix
    xtest = torch.randn(512, 96) @ mix
    ref = xtest @ w0.T
    h = 2.0 / xcal.shape[0] * (xcal.T @ xcal)

    _prune_linear(lin, h, sparsity=0.5)
    err_sgpt = ((xtest @ lin.weight.data.T) - ref).pow(2).mean().item()

    # same sparsity by plain magnitude masking
    k = int(w0.numel() * 0.5)
    thresh = w0.abs().flatten().sort()[0][k]
    masked = w0 * (w0.abs() > thresh)
    err_mask = ((xtest @ masked.T) - ref).pow(2).mean().item()

    assert err_sgpt < err_mask, f"sparsegpt {err_sgpt:.4e} !< magnitude {err_mask:.4e}"


def test_zero_sparsity_is_a_near_noop():
    torch.manual_seed(0)
    lin = nn.Linear(32, 16, bias=False)
    before = lin.weight.data.clone()
    x = torch.randn(256, 32)
    h = 2.0 / 256 * (x.T @ x)
    _prune_linear(lin, h, sparsity=0.0)
    torch.testing.assert_close(lin.weight.data, before, rtol=1e-3, atol=1e-4)
