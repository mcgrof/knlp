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


def ln_silu_alpha_adjoint(z: torch.Tensor, bar_u: torch.Tensor) -> torch.Tensor:
    """Adjoint of u = ln_silu_vjp_from_alpha(z, alpha) w.r.t. alpha (z fixed).

    Since u = J_phi(z)^T (phi(z) - alpha) is linear in alpha with z constant,
    the cotangent for alpha is bar_alpha = -(J_phi(z) @ bar_u) = -LNop(silu'(z) *
    bar_u). Used by the fused state-evolution kernel's backward, where z is
    detached so the only gradient from u flows to alpha.
    """
    work = torch.float32 if z.dtype in (torch.bfloat16, torch.float16) else z.dtype
    z32 = z.to(work)
    sig = torch.sigmoid(z32)
    s = z32 * sig
    silu_grad = sig + s * (1.0 - sig)
    mean = s.mean(dim=-1, keepdim=True)
    var = s.var(dim=-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + _EPS)
    y = (s - mean) / std
    h = silu_grad * bar_u.to(work)
    lnop = (
        h - h.mean(dim=-1, keepdim=True) - y * (h * y).mean(dim=-1, keepdim=True)
    ) / std
    return (-lnop).to(z.dtype)


def ln_silu_vjp_from_alpha(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Fused phi(z)=ln_silu(z) AND u = J_phi^T(phi(z)-alpha) in one pass.

    The Trellis inner step needs both phi(z) (to form the error) and the VJP;
    computing them separately recomputes SiLU/mean/var/y twice. This shares that
    work. Reductions are done in fp32 for bf16 stability, then cast back.
    """
    # upcast only low precision (bf16/fp16) to fp32; keep fp32/fp64 as given
    work = torch.float32 if z.dtype in (torch.bfloat16, torch.float16) else z.dtype
    z32 = z.to(work)
    sig = torch.sigmoid(z32)
    s = z32 * sig  # silu(z)
    silu_grad = sig + s * (1.0 - sig)  # d silu / d z
    mean = s.mean(dim=-1, keepdim=True)
    var = s.var(dim=-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + _EPS)
    y = (s - mean) / std  # = ln_silu(z)
    err = y - alpha.to(work)  # alpha kept in graph -> u differentiable in alpha
    ds = (
        err - err.mean(dim=-1, keepdim=True) - y * (err * y).mean(dim=-1, keepdim=True)
    ) / std
    return (silu_grad * ds).to(z.dtype)


def l2_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) / (||SiLU(x)|| + eps) over the last dim."""
    s = F.silu(x)
    return s / (s.norm(dim=-1, keepdim=True) + _EPS)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)


def identity(x: torch.Tensor) -> torch.Tensor:
    """phi = identity: the inner objective becomes 1/2||M w - alpha||^2, whose
    VJP is u = M w - alpha, so the gated update M <- beta M - gamma outer(u, w)
    is exactly the (gated) linear delta rule. This is the same-shell control the
    paper runs (its 11.65-vs-10.87 ablation): identical mixer, projections,
    two-pass shell and parameter budget, with the nonlinear write removed --
    isolating "does the nonlinear write help?" cleanly (unlike external DeltaNet,
    which also differs in shell/norm/gating/conv). The generic autograd VJP in
    trellis_memory._trellis_vjp handles it (grad(z, z, err) = err)."""
    return x


def scaled_identity(x: torch.Tensor) -> torch.Tensor:
    """phi = a*x diagnostic: a fixed-scale linear map. Tests whether a nonlinear
    arm loses to target-SCALE mismatch (alpha unconstrained vs a normalized phi)
    rather than to the nonlinear memory management itself. Not a faithful Trellis
    candidate -- a control. Fixed a=1 here is just identity; the learned-scale
    variant lives in the mixer (needs a parameter)."""
    return x


_ACT = {
    "silu": silu,                 # plain SiLU -- unconstrained nonlinear phi candidate
    "ln_silu": ln_silu,           # LayerNorm-SiLU (param-free), the current default
    "l2_silu": l2_silu,
    "softmax": softmax,
    "identity": identity,
    "scaled_identity": scaled_identity,
}


def get_activation(name: str):
    if name in ("linear", "identity"):
        return identity
    if name not in _ACT:
        raise ValueError(f"unknown activation {name}")
    return _ACT[name]
