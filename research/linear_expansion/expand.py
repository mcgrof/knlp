"""Function-preserving reparameterizations of selected attention matrices.

Each selected nn.Linear weight W is replaced by a trainable form whose
effective matrix equals W at initialization and which folds back into an
ordinary nn.Linear for export:

  D   original W
  F1  W = B A with inner width r = 128 (square factors)
  F2  W = B A with inner width r = 256 (wider factors)
  A4  W = (U1 + U2 + U3 + U4) / 2, each Uj = W / 2

The factor product runs as two linear operations during training; it is
never materialized in the forward pass.
"""

import hashlib

import torch
import torch.nn.functional as F
from torch import nn

from research.tinystories_depth.model import digest, generator

ARMS = ("D", "F1", "F2", "A4")
WIDTH = {"F1": 128, "F2": 256}
COPIES = 4
TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")


class Factored(nn.Module):
    """y = B (A x) + bias, with no operation between the two products."""

    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = None if bias is None else nn.Parameter(bias)

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B, self.bias)

    def effective(self):
        return self.B @ self.A


class Additive(nn.Module):
    """One linear operation on the differentiable sum of redundant copies."""

    def __init__(self, W, copies=COPIES, bias=None):
        super().__init__()
        self.U = nn.ParameterList(
            [nn.Parameter(W.detach().clone() * (2.0 / copies)) for _ in range(copies)]
        )
        self.bias = None if bias is None else nn.Parameter(bias)

    def forward(self, x):
        return F.linear(x, self.effective(), self.bias)

    def effective(self):
        return sum(self.U) / 2


def effective(module):
    return module.weight if isinstance(module, nn.Linear) else module.effective()


def target_names(model, shape=(128, 128)):
    """Resolve the 16 attention projections and fail on any surprise."""
    names = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in TARGET_LEAVES and ".self_attn." in name:
            assert isinstance(module, nn.Linear), name
            assert tuple(module.weight.shape) == shape, (name, module.weight.shape)
            assert module.bias is None, name + " has a bias"
            names.append(name)
    return names


def orthonormal(rows, cols, gen):
    """Named-seed Gaussian QR with the sign of R's diagonal made positive."""
    q, r = torch.linalg.qr(torch.randn(rows, cols, generator=gen, dtype=torch.float64))
    sign = torch.sign(torch.diagonal(r))
    sign[sign == 0] = 1
    return q * sign


def balanced_factors(W, width, gen):
    """W = U diag(s) V^T; A = R diag(sqrt s) V^T, B = U diag(sqrt s) R^T."""
    W64 = W.detach().to(torch.float64)
    U, s, Vh = torch.linalg.svd(W64, full_matrices=False)
    assert width >= s.numel(), "inner width below rank would truncate W"
    R = orthonormal(width, s.numel(), gen)
    root = s.sqrt()
    A = R @ (root[:, None] * Vh)
    B = (U * root) @ R.T
    return A, B, s, R


def _parent(root, name):
    head, _, leaf = name.rpartition(".")
    return root.get_submodule(head), leaf


def expand(model, arm, seed, names=None):
    """Replace target modules in place; return a reconstruction receipt."""
    assert arm in ARMS
    names = target_names(model) if names is None else names
    receipt = {"arm": arm, "targets": names, "matrices": {}}
    if arm == "D":
        return receipt
    for name in names:
        parent, leaf = _parent(model, name)
        linear = getattr(parent, leaf)
        W = linear.weight.detach()
        dtype, device = W.dtype, W.device
        bias = None if linear.bias is None else linear.bias.detach().clone()
        record = {}
        if arm == "A4":
            module = Additive(W, bias=bias)
        else:
            gen = generator(seed, f"{name}:R{WIDTH[arm]}")
            A, B, s, R = balanced_factors(W, WIDTH[arm], gen)
            record = {
                "singular_max": s.max().item(),
                "singular_min": s.min().item(),
                "R_sha256": digest(R),
                "A_norm": A.norm().item(),
                "B_norm": B.norm().item(),
                "reconstruction_max_abs_float64": (B @ A - W.to(torch.float64))
                .abs()
                .max()
                .item(),
            }
            module = Factored(A.to(dtype), B.to(dtype), bias)
        module.to(device)
        with torch.no_grad():
            folded = effective(module).to(torch.float64)
        record["reconstruction_max_abs_master"] = (
            (folded - W.to(torch.float64)).abs().max().item()
        )
        setattr(parent, leaf, module)
        receipt["matrices"][name] = record
    return receipt


@torch.no_grad()
def folded_state(model):
    """State dict of the ordinary model: each wrapper becomes '<name>.weight'.

    Products are formed in float64 from the master values, then cast to the
    master dtype, so no factor tensor remains in the exported checkpoint.
    """
    state = {}
    wrappers = {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, (Factored, Additive))
    }
    for key, value in model.state_dict().items():
        owner = next((n for n in wrappers if key.startswith(n + ".")), None)
        if owner is None:
            state[key] = value.detach().clone()
    for name, module in wrappers.items():
        if isinstance(module, Factored):
            W = module.B.to(torch.float64) @ module.A.to(torch.float64)
            dtype = module.A.dtype
        else:
            W = sum(u.to(torch.float64) for u in module.U) / 2
            dtype = module.U[0].dtype
        state[name + ".weight"] = W.to(dtype)
        if module.bias is not None:
            state[name + ".bias"] = module.bias.detach().clone()
    return state


def selected_parameters(model, names):
    """Parameters that implement the selected matrices, in any arm."""
    out = []
    for name in names:
        module = model.get_submodule(name)
        if isinstance(module, nn.Linear):
            out.append(module.weight)
            if module.bias is not None:
                out.append(module.bias)
        else:
            out.extend(module.parameters())
    return out


def param_groups(model, names, lr, multiplier):
    """Selected matrices get lr * multiplier; everything else gets lr."""
    selected = selected_parameters(model, names)
    ids = {id(p) for p in selected}
    common = [p for p in model.parameters() if id(p) not in ids]
    assert len(ids) == len(selected)
    return [
        {"params": selected, "lr": lr * multiplier, "base_lr": lr * multiplier},
        {"params": common, "lr": lr, "base_lr": lr},
    ]


def counts(model, names):
    selected = selected_parameters(model, names)
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selected": sum(p.numel() for p in selected),
    }


@torch.no_grad()
def matrix_diagnostics(model, names):
    """Norms, singular extremes and factor balance for every selected matrix."""
    out = {}
    for name in names:
        module = model.get_submodule(name)
        W = effective(module).float()
        s = torch.linalg.svdvals(W)
        row = {
            "W_norm": W.norm().item(),
            "singular_max": s.max().item(),
            "singular_min": s.min().item(),
        }
        if isinstance(module, Factored):
            A, B = module.A.float(), module.B.float()
            gap = A @ A.T - B.T @ B
            row.update(
                {
                    "A_norm": A.norm().item(),
                    "B_norm": B.norm().item(),
                    "balance": (
                        gap.norm() / ((A @ A.T).norm() + (B.T @ B).norm())
                    ).item(),
                }
            )
        out[name] = row
    return out


def effective_weights(model, names):
    with torch.no_grad():
        return {
            n: effective(model.get_submodule(n)).detach().float().clone() for n in names
        }


def json_hash(value):
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
