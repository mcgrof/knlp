"""Intermediate activations phi / f for the Trellis memory.

All operate over the last (slot, M) dimension. ln_silu is the paper default for
the intermediate phi; l2_silu and softmax are ablations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def ln_silu(x: torch.Tensor) -> torch.Tensor:
    """LayerNorm(SiLU(x)) over the last dim (no affine — keep it parameter-free
    so the same fn is reusable as phi and f without owning weights)."""
    s = F.silu(x)
    mean = s.mean(dim=-1, keepdim=True)
    var = s.var(dim=-1, keepdim=True, unbiased=False)
    return (s - mean) / torch.sqrt(var + _EPS)


def ln_silu_vjp(z: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
    """Closed-form J_phi(z)^T @ err for phi = ln_silu, over the last dim.

    Equals torch.autograd.grad(ln_silu(z), z, grad_outputs=err) but with no
    autograd graph. The Trellis inner step computes exactly this VJP once per
    chunk; the per-chunk autograd.grad was the kernel's dominant overhead.
    `err` carries the alpha dependence (err = phi(z) - alpha with z treated as a
    constant), and the result is linear in err so the outer backward to alpha
    still flows.
    """
    sig = torch.sigmoid(z)
    s = z * sig  # silu(z)
    silu_grad = sig + s * (1.0 - sig)  # d silu / d z
    mean = s.mean(dim=-1, keepdim=True)
    var = s.var(dim=-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + _EPS)
    y = (s - mean) / std  # = ln_silu(z)
    ds = (
        err - err.mean(dim=-1, keepdim=True) - y * (err * y).mean(dim=-1, keepdim=True)
    ) / std
    return silu_grad * ds


def l2_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) / (||SiLU(x)|| + eps) over the last dim."""
    s = F.silu(x)
    return s / (s.norm(dim=-1, keepdim=True) + _EPS)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)


_ACT = {"ln_silu": ln_silu, "l2_silu": l2_silu, "softmax": softmax}


def get_activation(name: str):
    if name == "linear":
        return lambda x: x
    if name not in _ACT:
        raise ValueError(f"unknown activation {name}")
    return _ACT[name]
