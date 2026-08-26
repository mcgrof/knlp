"""SparseGPT: one-shot pruning with weight reconstruction.

Frantar and Alistarh, arXiv:2301.00774. For a linear layer with
weight W [out, in] and calibration inputs X, the layer Hessian is
H = 2 X X^T. Columns are pruned left to right; each time a weight
is zeroed the error it introduces is pushed onto the not-yet-visited
columns of the same row, scaled by the inverse Hessian, so the
surviving weights absorb what the pruned ones were doing:

    d      = Hinv[j, j]
    err    = (w_j - q_j) / d
    W[:, j:] -= err (outer) Hinv[j, j:]

This is what separates SparseGPT from a scoring rule: the mask is
chosen by the Optimal-Brain-Surgeon criterion w^2 / Hinv_jj^2, but
the surviving weights are then UPDATED. Our other arms only mask.

The pass is sequential over transformer blocks, as published: each
block is pruned using activations produced by the ALREADY PRUNED
blocks before it, so earlier reconstruction error is visible
downstream. Running every block off dense-model activations instead
would weaken the baseline, which is the wrong direction for a method
we are comparing ourselves against.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class _Catcher(nn.Module):
    """Steals the inputs of the first block and aborts the forward."""

    class Stop(Exception):
        pass

    _DROP = ("past_key_value", "past_key_values", "layer_past", "cache_position")

    def __init__(self, module, store):
        super().__init__()
        self.module = module
        self.store = store

    def forward(self, hidden_states, **kwargs):
        # host-resident: the whole calibration set of hidden
        # states does not fit beside the model and the Hessians
        self.store["hidden"].append(hidden_states.detach().cpu())
        # Anything carrying generation state must not be replayed: a
        # cache grows on every block call, so the second replay sees
        # twice the sequence length it was built for. Masks and
        # position embeddings are identical across same-shape batches,
        # so one captured copy drives every replay.
        kept = {k: v for k, v in kwargs.items() if k not in self._DROP}
        kept["use_cache"] = False
        self.store["kwargs"] = kept
        raise _Catcher.Stop


def _block_list(model):
    """The transformer block stack and its dotted name prefix."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            first = mod[0]
            if any(isinstance(m, nn.Linear) for m in first.modules()):
                return list(mod), name
    raise RuntimeError("no transformer block stack found")


class _HessianAccumulator:
    """H = 2 X X^T accumulated over calibration tokens, in fp32."""

    def __init__(self, linear: nn.Linear):
        self.columns = linear.weight.shape[1]
        self.h = torch.zeros(
            (self.columns, self.columns),
            device=linear.weight.device,
            dtype=torch.float32,
        )
        self.n = 0

    def add(self, inp: torch.Tensor) -> None:
        x = inp.reshape(-1, inp.shape[-1]).float()
        self.h *= self.n / (self.n + x.shape[0])
        self.n += x.shape[0]
        x = x * math.sqrt(2.0 / self.n)
        self.h += x.T @ x


def _prune_linear(
    linear: nn.Linear,
    hess: torch.Tensor,
    sparsity: float,
    blocksize: int = 128,
    percdamp: float = 0.01,
    importance: Optional[torch.Tensor] = None,
) -> None:
    """Prune one linear in place, reconstructing survivors.

    importance, when given, replaces the diagonal Optimal-Brain-Surgeon
    criterion for CHOOSING the mask (higher means keep); the
    reconstruction is unchanged. That isolates the mask from the weight
    update, which is the only way to tell whether a better scoring rule
    still buys anything once survivors are reconstructed.
    """
    w = linear.weight.data.clone().float()
    h = hess.clone()
    dead = torch.diagonal(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diagonal(h))
    h[range(h.shape[0]), range(h.shape[0])] += damp
    h = torch.linalg.cholesky(h)
    h = torch.cholesky_inverse(h)
    h = torch.linalg.cholesky(h, upper=True)
    hinv = h

    losses = torch.zeros_like(w)
    for i1 in range(0, w.shape[1], blocksize):
        i2 = min(i1 + blocksize, w.shape[1])
        count = i2 - i1
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        # Mask chosen per block from the deletion cost, exactly as
        # published: the diagonal of the inverse Hessian, squared.
        if importance is None:
            tmp = w1**2 / (torch.diagonal(hinv1).reshape(1, -1)) ** 2
        else:
            # externally scored: same orientation, low means prune
            tmp = importance[:, i1:i2].to(w1.device).float()
        k = int(tmp.numel() * sparsity)
        if k <= 0:
            mask1 = torch.zeros_like(tmp, dtype=torch.bool)
        else:
            thresh = torch.sort(tmp.flatten())[0][k - 1]
            mask1 = tmp <= thresh

        for i in range(count):
            wcol = w1[:, i]
            d = hinv1[i, i]
            q = wcol.clone()
            q[mask1[:, i]] = 0
            q1[:, i] = q
            losses[:, i1 + i] = (wcol - q) ** 2 / d**2
            e = (wcol - q) / d
            w1[:, i:] -= e.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
            err1[:, i] = e

        w[:, i1:i2] = q1
        # push this block's accumulated error onto the columns ahead
        w[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])

    linear.weight.data.copy_(w.to(linear.weight.dtype))


@torch.no_grad()
def sparsegpt_prune(
    model,
    batches,
    sparsity: float,
    target_names: Optional[List[str]] = None,
    blocksize: int = 128,
    percdamp: float = 0.01,
    device: str = "cuda",
    mask_scores: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Prune the model in place. Returns per-layer achieved sparsity.

    batches is a list of [B, T] token tensors; they are pushed through
    the block stack once, then replayed block by block.
    """
    blocks, prefix = _block_list(model)
    store = {"hidden": [], "kwargs": {}}
    blocks_parent = None
    for mod in model.modules():
        if isinstance(mod, nn.ModuleList) and list(mod) == blocks:
            blocks_parent = mod
            break

    blocks_parent[0] = _Catcher(blocks[0], store)
    for batch in batches:
        try:
            model(batch.to(device))
        except _Catcher.Stop:
            pass
    blocks_parent[0] = blocks[0]

    kwargs = store["kwargs"]
    hiddens = store["hidden"]
    achieved: Dict[str, float] = {}

    for bi, block in enumerate(blocks):
        linears = {
            n: m
            for n, m in block.named_modules()
            if isinstance(m, nn.Linear)
            and (target_names is None or f"blocks.{bi}.{n}" in target_names or True)
        }
        # One linear at a time: a block's Hessians together are
        # several gigabytes at 1B scale (an 8192-wide layer alone is
        # 268 MB in fp32), which is what exhausted a 45 GiB card.
        for n, m in linears.items():
            acc = _HessianAccumulator(m)
            handle = m.register_forward_pre_hook(lambda mod, inp, a=acc: a.add(inp[0]))
            for h in hiddens:
                block(h.to(device), **kwargs)
            handle.remove()
            imp = None
            if mask_scores is not None:
                full = f"{prefix}.{bi}.{n}" if prefix else f"{bi}.{n}"
                imp = mask_scores.get(full)
                if imp is None:
                    raise KeyError(f"no mask score supplied for {full}")
            _prune_linear(m, acc.h, sparsity, blocksize, percdamp, imp)
            achieved[f"block{bi}.{n}"] = float((m.weight == 0).float().mean())
            del acc
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        torch.cuda.empty_cache() if device.startswith("cuda") else None

        # propagate through the now-pruned block for the next one
        for i in range(len(hiddens)):
            out = block(hiddens[i].to(device), **kwargs)
            out = out[0] if isinstance(out, tuple) else out
            hiddens[i] = out.detach().cpu()

    return achieved
