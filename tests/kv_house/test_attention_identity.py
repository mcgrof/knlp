# SPDX-License-Identifier: GPL-2.0
"""Gate-0 transformed-attention identity. CPU, no GPU, no model.

The whole point of keeping the temporal transform orthogonal is that
attention can consume transformed blocks without reconstructing the
original token vectors:

    q @ K.T == (q @ K_tilde.T) @ R      with K_tilde = R @ K
    a @ V   == (a @ R.T) @ V_tilde      with V_tilde = R @ V

This must hold for every orthogonal transform before any
quantization experiment is allowed to run.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from kv_house.transforms import TRANSFORMS

ORTHOGONAL = [n for n, c in TRANSFORMS.items() if c.is_orthogonal]


def _fitted(name, b, x):
    t = TRANSFORMS[name](b)
    t.fit(x)
    return t


def test_qk_logit_identity():
    torch.manual_seed(0)
    for name in ORTHOGONAL:
        for b in (8, 16, 32):
            k = torch.randn(b, 64, dtype=torch.float64)
            q = torch.randn(4, 64, dtype=torch.float64)
            t = _fitted(name, b, k)
            r = t.matrix(dtype=torch.float64)
            if r is None:
                continue
            k_tilde = t.forward(k)
            ref = q @ k.T
            got = (q @ k_tilde.T) @ r
            assert torch.allclose(ref, got, atol=1e-9), (name, b)


def test_av_identity():
    torch.manual_seed(1)
    for name in ORTHOGONAL:
        for b in (8, 16, 32):
            v = torch.randn(b, 64, dtype=torch.float64)
            a = torch.softmax(torch.randn(4, b, dtype=torch.float64), dim=-1)
            t = _fitted(name, b, v)
            r = t.matrix(dtype=torch.float64)
            if r is None:
                continue
            v_tilde = t.forward(v)
            ref = a @ v
            got = (a @ r.T) @ v_tilde
            assert torch.allclose(ref, got, atol=1e-9), (name, b)


def test_forward_equals_matrix_action():
    torch.manual_seed(2)
    for name in ORTHOGONAL:
        for b in (8, 16, 32):
            x = torch.randn(b, 48, dtype=torch.float64)
            t = _fitted(name, b, x)
            r = t.matrix(dtype=torch.float64)
            if r is None:
                continue
            assert torch.allclose(t.forward(x), r @ x, atol=1e-9), (name, b)
            assert torch.allclose(t.inverse(t.forward(x)), x, atol=1e-9), (name, b)


def test_orthonormal_matrices():
    for name in ORTHOGONAL:
        for b in (8, 16, 32):
            t = TRANSFORMS[name](b)
            t.fit(torch.randn(b, 32, dtype=torch.float64))
            r = t.matrix(dtype=torch.float64)
            if r is None:
                continue
            eye = torch.eye(b, dtype=torch.float64)
            assert torch.allclose(r.T @ r, eye, atol=1e-9), name
