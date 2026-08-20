"""K-FAC factor capture for linear layers via hooks.

For a linear layer y = W x + b with W [out, in], accumulates over
tokens (all leading dimensions flattened):

    A = E[x x^T]            [in, in]   input second moment
    G = E[g g^T]            [out, out] output-gradient second moment
    D = E[(g x^T)^2]        [out, in]  exact elementwise diagonal of
                                       the per-token empirical Fisher,
                                       D_ij = E[g_i^2 x_j^2]

where g is the gradient of the PER-TOKEN loss w.r.t. the layer output.
Autograd delivers the gradient of the MEAN loss, so captured grads are
rescaled by the token count of the batch before accumulation.

The Kronecker approximation of the Fisher block is F ~= G (x) A, and
diag(G (x) A)_ij = G_ii * A_jj — a rank-1 approximation of the exact
elementwise diagonal D. Both are captured so diagonal-vs-factored can
be compared at identical statistics.

All accumulation is in fp32 on the compute device; finalized factors
are returned on CPU. Run the model in eval mode (no dropout) — the
Fisher of the deployed function is what pruning cares about.
"""

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class FactorAccumulator:
    """Attach to named nn.Linear modules; accumulate A, G, D."""

    def __init__(self, model: nn.Module, target_names: Iterable[str]):
        self.targets: Dict[str, nn.Linear] = {}
        named = dict(model.named_modules())
        for name in target_names:
            mod = named.get(name)
            if mod is None:
                raise KeyError(f"module {name!r} not found")
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"module {name!r} is {type(mod).__name__}")
            self.targets[name] = mod
        self._a: Dict[str, torch.Tensor] = {}
        self._g: Dict[str, torch.Tensor] = {}
        self._d: Dict[str, torch.Tensor] = {}
        self._x_pending: Dict[str, torch.Tensor] = {}
        self.n_tokens = 0
        self._handles: List = []
        self._enabled = False

    def attach(self) -> None:
        for name, mod in self.targets.items():
            self._handles.append(mod.register_forward_hook(self._make_fwd_hook(name)))
        self._enabled = True

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._enabled = False

    def _make_fwd_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._enabled:
                return
            x = inputs[0].detach()
            x2d = x.reshape(-1, x.shape[-1]).float()
            a = self._a.get(name)
            prod = x2d.T @ x2d
            self._a[name] = prod if a is None else a + prod
            self._x_pending[name] = x2d
            if output.requires_grad:
                output.register_hook(self._make_grad_hook(name))

        return hook

    def _make_grad_hook(self, name: str):
        def hook(grad):
            g2d = grad.detach().reshape(-1, grad.shape[-1]).float()
            n_tok = g2d.shape[0]
            # autograd carries d(mean over tokens)/dy; per-token loss
            # gradients are n_tok times larger.
            g2d = g2d * n_tok
            g = self._g.get(name)
            prod = g2d.T @ g2d
            self._g[name] = prod if g is None else g + prod
            x2d = self._x_pending.pop(name, None)
            if x2d is None:
                raise RuntimeError(f"grad for {name} with no pending input")
            if x2d.shape[0] != n_tok:
                raise RuntimeError(f"{name}: token mismatch x={x2d.shape[0]} g={n_tok}")
            dprod = (g2d**2).T @ (x2d**2)
            d = self._d.get(name)
            self._d[name] = dprod if d is None else d + dprod

        return hook

    def count_batch(self, n_tokens: int) -> None:
        """Record the token count of a completed fwd+bwd batch."""
        self.n_tokens += n_tokens

    def finalize(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.n_tokens == 0:
            raise RuntimeError("no tokens accumulated")
        if self._x_pending:
            raise RuntimeError(
                f"pending inputs without grads: {sorted(self._x_pending)}"
            )
        out = {}
        for name in self.targets:
            if name not in self._g:
                raise RuntimeError(f"no gradients captured for {name}")
            out[name] = {
                "A": (self._a[name] / self.n_tokens).cpu(),
                "G": (self._g[name] / self.n_tokens).cpu(),
                "D": (self._d[name] / self.n_tokens).cpu(),
            }
        return out


def damped_inverse_diag(factor: torch.Tensor, kappa: float) -> torch.Tensor:
    """diag((M + lambda I)^-1) with lambda = kappa * tr(M)/dim(M).

    Computed in fp64. M must be symmetric PSD (a second-moment
    matrix); the damped matrix is PD, so Cholesky applies.
    """
    m = factor.double()
    dim = m.shape[0]
    lam = kappa * torch.clamp(torch.diagonal(m).sum() / dim, min=1e-30)
    damped = m + lam * torch.eye(dim, dtype=m.dtype)
    chol = torch.linalg.cholesky(damped)
    inv = torch.cholesky_inverse(chol)
    return torch.diagonal(inv).clone()


def default_target_names(n_layer: int) -> List[str]:
    names = []
    for i in range(n_layer):
        names += [
            f"transformer.h.{i}.attn.c_attn",
            f"transformer.h.{i}.attn.c_proj",
            f"transformer.h.{i}.mlp.c_fc",
            f"transformer.h.{i}.mlp.c_proj",
        ]
    return names


def per_layer_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask at the given sparsity within one matrix.

    Ties are broken by flattened index order (deterministic).
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity {sparsity} out of range")
    flat = score.reshape(-1)
    k = int(round(sparsity * flat.numel()))
    if k == 0:
        return torch.ones_like(flat, dtype=torch.bool).reshape(score.shape)
    order = torch.argsort(flat, stable=True)
    mask = torch.ones_like(flat, dtype=torch.bool)
    mask[order[:k]] = False
    return mask.reshape(score.shape)


def global_masks(
    scores: Dict[str, torch.Tensor], sparsity: float
) -> Dict[str, torch.Tensor]:
    """Keep-masks from one global threshold across all matrices."""
    flat = torch.cat([s.reshape(-1) for s in scores.values()])
    k = int(round(sparsity * flat.numel()))
    if k == 0:
        thresh = -float("inf")
    else:
        thresh = torch.kthvalue(flat, k).values.item()
    # Ties at the threshold can over/under-shoot slightly; accepted
    # (recorded via actual sparsity in the results log).
    return {name: s > thresh for name, s in scores.items()}
