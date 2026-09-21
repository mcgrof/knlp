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

    Setting ``linear=True`` removes the activation and nothing else. The two
    variants then have identical parameter counts, identical shapes and
    identical initialisation, so a comparison between them isolates the
    nonlinearity rather than capacity. The linear form is also mergeable: a
    composition of two affine maps is one affine map, so a trained linear
    correction folds into the frozen map it corrects and costs nothing at all
    online. :meth:`merged_delta` returns that fold.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 32, linear: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)
        self.linear = linear
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.fc1(x)
        if not self.linear:
            h = torch.nn.functional.gelu(h)
        return self.fc2(h)

    @torch.no_grad()
    def merged_delta(self):
        """``(dM, db)`` such that ``x @ (M + dM) + (b + db)`` is the corrected map.

        Only defined without the activation. Raises otherwise rather than
        returning a silently wrong linearisation, which would understate the
        nonlinear arm's online cost by making it look free.
        """
        if not self.linear:
            raise ValueError("a nonlinear correction does not fold into an affine map")
        W1, b1 = self.fc1.weight, self.fc1.bias  # [h, d_in], [h]
        W2, b2 = self.fc2.weight, self.fc2.bias  # [d_out, h], [d_out]
        return (W1.t() @ W2.t()), (b1 @ W2.t() + b2)


class ResidualMapper(nn.Module):
    """A frozen affine mapper plus a trainable per-block correction."""

    def __init__(
        self, affine, layout, geom, kind: str, hidden: int = 32, linear: bool = False
    ):
        super().__init__()
        self.affine = affine
        self.layout = layout
        self.geom = geom
        self.kind = kind
        self.linear = linear
        self.cols = {}
        mods = {}
        for (li, h), m in affine.maps.items():
            cols = layout.columns_for(m.layers, m.head, m.head_local)
            self.cols[(li, h)] = cols
            mods[f"{li}_{h}"] = BlockResidual(
                int(cols.numel()), m.M.shape[1], hidden, linear=linear
            )
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

    def save_init(self, path: str) -> str:
        """Serialise the untrained correction so other arms can start from it.

        Zeroing the last layer makes every arm's *initial prediction* equal,
        which is why an unmatched first layer is easy to miss: the arms agree
        at step zero and diverge afterwards. They are drawing their first-layer
        weights from the global generator, so two arms built in sequence begin
        optimisation from different points and a comparison between them
        carries that difference along with whatever was meant to be varied.

        Writing one initialisation and loading it into every matched arm makes
        the starting point an input with a hash rather than a side effect of
        construction order.
        """
        import hashlib

        state = {k: v.detach().cpu() for k, v in self.blocks.state_dict().items()}
        torch.save({"kind": self.kind, "linear": self.linear, "blocks": state}, path)
        h = hashlib.sha256()
        for k in sorted(state):
            h.update(k.encode())
            h.update(state[k].to(torch.float64).numpy().tobytes())
        return h.hexdigest()[:32]

    def load_init(self, path: str) -> str:
        """Start from a serialised initialisation, refusing a mismatched one."""
        import hashlib

        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("linear") != self.linear:
            raise ValueError(
                f"initialisation is linear={blob.get('linear')} but this "
                f"correction is linear={self.linear}"
            )
        self.blocks.load_state_dict(blob["blocks"])
        h = hashlib.sha256()
        for k in sorted(blob["blocks"]):
            h.update(k.encode())
            h.update(blob["blocks"][k].to(torch.float64).numpy().tobytes())
        return h.hexdigest()[:32]

    @torch.no_grad()
    def fold_parity(self, X: torch.Tensor) -> dict:
        """Compare every block against its folded self, and say so numerically.

        The earlier check looked at one key block of one layer and printed a
        number without asserting on it. One block agreeing is consistent with
        the fold being wrong everywhere else, and a printed number that nothing
        reads is not a check.
        """
        if not self.linear:
            raise ValueError("only a linear correction folds")
        worst, worst_block, n = 0.0, None, 0
        for (li, h), m in self.affine.maps.items():
            cols = self.cols[(li, h)].to(X.device)
            x = X[:, cols]
            before = x.to(m.M.dtype) @ m.M + m.b
            before = before + self.blocks[f"{li}_{h}"](x.float()).to(before.dtype)
            dM, db = self.blocks[f"{li}_{h}"].merged_delta()
            after = x.to(m.M.dtype) @ (m.M + dM.to(m.M.dtype)) + (
                m.b + db.to(m.b.dtype)
            )
            gap = float((before - after).abs().max())
            scale = float(before.abs().max().clamp(min=1e-30))
            rel = gap / scale
            n += 1
            if rel > worst:
                worst, worst_block = rel, (li, h, self.kind)
        return {
            "kind": self.kind,
            "blocks_checked": n,
            "worst_relative_gap": worst,
            "worst_block": worst_block,
        }

    def cast(self, dtype: torch.dtype) -> "ResidualMapper":
        """Apply the frozen map at serving precision, as the plain map does.

        The ridge fit is solved in double because a Gram matrix is
        ill-conditioned, and the solution is stored at the precision it was
        solved in. :meth:`Mapper.cast` exists so a plain map is never applied
        there; without the same call here a corrected arm runs its dominant
        matmul in double while the arm it is being compared against runs it in
        float32, and on hardware with no double-precision matrix path that
        difference is most of the measured cost. It then looks like the
        correction is expensive when what is expensive is the precision.
        """
        for m in self.affine.maps.values():
            m.to(dtype)
        return self

    def freeze_affine(self) -> None:
        for m in self.affine.maps.values():
            m.M = m.M.detach()
            m.b = m.b.detach()

    @torch.no_grad()
    def merge(self):
        """Fold a trained linear correction into the affine map it corrects.

        After this the mapper is an ordinary affine map again: same shapes,
        same byte count, same online operator, no residual to evaluate. That
        is the whole argument for the linear variant, so it is exercised here
        rather than asserted in a report.
        """
        for (li, h), m in self.affine.maps.items():
            dM, db = self.blocks[f"{li}_{h}"].merged_delta()
            m.M = m.M + dM.to(m.M.dtype)
            m.b = m.b + db.to(m.b.dtype)
        return self.affine


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
