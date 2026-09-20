# SPDX-License-Identifier: GPL-2.0
"""A small nonlinear correction on top of the affine map, trained two ways.

The affine baseline leaves structured error. A3 asks whether a small residual
network removes enough of it to be worth its parameters, and -- the more
interesting question -- whether it matters what that network is trained
against.

Two objectives, same architecture, same initialisation, same step budget.

Trained on target cache error, the correction learns to reproduce tensors. That
is the obvious objective and the one the prior lane in this program showed
ranks real damage poorly: a cache can be close in norm and wrong in the
directions the receiver reads.

Trained on the target's own continuation divergence, the correction learns to
reproduce *behaviour*. Gradients flow from the frozen target's logits, back
through attention, through the re-rotation, into the correction. That path
already carries the receiver's sensitivity for each individual example, exactly
and nonlinearly, without any averaged Jacobian attached -- which is the whole
reason the roadmap puts behavioural training ahead of a global receiver metric
rather than beside it.

Both base models stay frozen throughout. Only the correction has parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


class BlockResidual(nn.Module):
    """A per-block correction added to the affine prediction.

    Deliberately small. The affine map it corrects already holds most of the
    signal, so a correction with comparable capacity would be a different
    method rather than a refinement of this one, and its byte count would stop
    being comparable.

    The final layer starts at zero, so the module is exactly the affine map at
    step zero. Any improvement is then attributable to training rather than to
    a lucky initialisation, and a failed run degrades to the baseline instead
    of to noise.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class ResidualMapper(nn.Module):
    """A frozen affine mapper plus a trainable per-block correction."""

    def __init__(self, affine, layout, geom, kind: str, hidden: int = 32):
        super().__init__()
        self.affine = affine
        self.layout = layout
        self.geom = geom
        self.kind = kind
        self.cols = {}
        mods = {}
        for (li, h), m in affine.maps.items():
            cols = layout.columns_for(m.layers, h, m.head_local)
            self.cols[(li, h)] = cols
            mods[f"{li}_{h}"] = BlockResidual(int(cols.numel()), m.M.shape[1], hidden)
        self.blocks = nn.ModuleDict(mods)
        for p in self.parameters():
            p.requires_grad_(True)

    @property
    def n_residual_params(self) -> int:
        return sum(p.numel() for p in self.blocks.parameters())

    def forward(self, X: torch.Tensor) -> list:
        """``[T, d_full]`` -> ``[L][1, H, T, D]``, affine plus correction."""
        out = []
        for li in range(self.geom.n_layers):
            heads = []
            for h in range(self.geom.n_kv_heads):
                m = self.affine.maps[(li, h)]
                cols = self.cols[(li, h)].to(X.device)
                x = X[:, cols]
                base = x.to(m.M.dtype) @ m.M + m.b
                heads.append(base + self.blocks[f"{li}_{h}"](x.float()).to(base.dtype))
            out.append(torch.stack(heads, 0).unsqueeze(0))
        return out

    def freeze_affine(self) -> None:
        for m in self.affine.maps.values():
            m.M = m.M.detach()
            m.b = m.b.detach()


@dataclass
class TrainReport:
    """What a training arm cost as well as what it achieved."""

    objective: str
    steps: int
    lr: float
    residual_params: int
    residual_bytes: float
    train_seconds: float
    peak_mem_bytes: int
    first_loss: float
    last_loss: float
    dev_loss: float = float("nan")

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def kv_error_loss(
    pred: Sequence[torch.Tensor], target: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Mean squared error against the target's own cache blocks."""
    num = sum(((p - t.to(p.dtype)) ** 2).sum() for p, t in zip(pred, target))
    den = sum(t.numel() for t in target)
    return num / den


def behaviour_loss(ref_logprobs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Divergence of the target's continuation from what it would have said.

    The reference is stored as log-probabilities so it can be cached in half
    precision without a second forward pass per step; the live side is kept in
    float32 because the gradient flows through it.
    """
    lq = torch.log_softmax(logits.float(), dim=-1)
    p = ref_logprobs.to(logits.device).float().exp()
    return (p * (ref_logprobs.to(logits.device).float() - lq)).sum(-1).mean()
