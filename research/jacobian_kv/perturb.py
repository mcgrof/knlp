# SPDX-License-Identifier: GPL-2.0
"""A controlled bank of KV-cache errors, built so magnitude alone cannot win.

Stage 0 asks whether a receiver-sensitive metric ranks *real* cache damage
better than cheap incumbents.  That question is only meaningful against a bank
of errors whose damage varies for reasons other than size, so the bank here is
organised into **matched-norm groups**: within a group every perturbation has
the same Frobenius norm to floating-point tolerance, and only its direction
differs.  Inside such a group raw cache MSE is constant and therefore has
exactly zero ranking power, which turns the pooled comparison into a real test
instead of a size-detector contest.

Both views are reported.  The pooled correlation across all groups is the
headline the gate is written against; the within-group correlation is the
scientific question -- given a byte budget already spent, does the metric know
which direction to spend it in.

The error kinds cover what the plan asks for: truncation of the block's own
spectrum, random subspaces, fake-quantisation round-trips, head and position
subsets, and the residual of a cheap affine map, which is the only kind that
resembles a deployed codec's actual error.

Convention: blocks are ``[B, n_kv_heads, T, head_dim]``, matching the cache
layout used everywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Perturbation:
    """One cache error, with everything needed to reproduce and to audit it."""

    name: str
    kind: str  # "k" or "v"
    layer: int
    family: str
    delta: torch.Tensor
    norm_group: Optional[str] = None
    params: dict = field(default_factory=dict)

    @property
    def frob(self) -> float:
        return float(self.delta.float().norm())


def _unit(t: torch.Tensor) -> torch.Tensor:
    n = t.float().norm()
    return t / n.clamp(min=1e-30)


def _randn_like(block: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        block.shape, generator=generator, device=block.device, dtype=torch.float32
    )


# ---------------------------------------------------------------------------
# error families
# ---------------------------------------------------------------------------


def err_isotropic(block, generator, **_):
    """Direction-free noise.  The null hypothesis of the whole screen."""
    return _randn_like(block, generator)


def err_random_subspace(block, generator, *, rank: int = 8, **_):
    """Noise confined to a random ``rank``-dimensional subspace of head_dim."""
    d = block.shape[-1]
    q, _ = torch.linalg.qr(
        torch.randn(
            d, rank, generator=generator, device=block.device, dtype=torch.float32
        )
    )
    noise = _randn_like(block, generator)
    return noise @ q @ q.transpose(0, 1)


def err_svd_truncation(block, generator, *, rank: int = 16, **_):
    """The block's own tail spectrum: what plain per-head SVD would throw away.

    This is the error a rank-``r`` PCA/SVD codec actually makes, which is why
    the plan makes unweighted SVD a mandatory baseline rather than a straw man.
    """
    b, h, t, d = block.shape
    x = block.float().reshape(b * h, t, d)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    r = int(min(rank, s.shape[-1]))
    s_tail = s.clone()
    s_tail[..., :r] = 0.0
    recon_tail = (u * s_tail.unsqueeze(-2)) @ vh
    return recon_tail.reshape(b, h, t, d)


def err_fake_quant(block, generator, *, bits: int = 4, per: str = "head", **_):
    """Round-trip error of symmetric fake quantisation at ``bits``.

    ``per="head"`` uses one scale per (head, feature) column, ``per="token"``
    one per (head, position) row.  Both appear in deployed codecs and they fail
    differently, which is the point of having both in the bank.
    """
    x = block.float()
    dim = -2 if per == "head" else -1
    amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12)
    qmax = 2 ** (bits - 1) - 1
    scale = amax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale - x


def err_head_subset(block, generator, *, frac: float = 0.25, **_):
    """Noise on a random subset of kv heads only."""
    b, h, t, d = block.shape
    n = max(1, int(round(frac * h)))
    perm = torch.randperm(h, generator=generator, device=block.device)[:n]
    mask = torch.zeros(h, device=block.device, dtype=torch.float32)
    mask[perm] = 1.0
    return _randn_like(block, generator) * mask.view(1, h, 1, 1)


def err_position_slice(
    block, generator, *, where: str = "recent", frac: float = 0.25, **_
):
    """Noise on the most recent, oldest, or a middle band of cached positions.

    Position matters a great deal to a receiver and not at all to raw MSE, so
    this family is where a purely size-based metric should lose first.
    """
    b, h, t, d = block.shape
    n = max(1, int(round(frac * t)))
    mask = torch.zeros(t, device=block.device, dtype=torch.float32)
    if where == "recent":
        mask[t - n :] = 1.0
    elif where == "oldest":
        mask[:n] = 1.0
    else:
        start = max(0, (t - n) // 2)
        mask[start : start + n] = 1.0
    return _randn_like(block, generator) * mask.view(1, 1, t, 1)


def err_affine_residual(block, generator, *, rank: int = 16, **_):
    """Residual of a cheap per-head linear map fitted on the block itself.

    Stands in for a deployed codec's structured error: a low-rank predictor
    that is good on average and wrong in a correlated way.
    """
    b, h, t, d = block.shape
    x = block.float().reshape(b * h, t, d)
    mean = x.mean(dim=-2, keepdim=True)
    xc = x - mean
    u, s, vh = torch.linalg.svd(xc, full_matrices=False)
    r = int(min(rank, s.shape[-1]))
    recon = (u[..., :r] * s[..., :r].unsqueeze(-2)) @ vh[..., :r, :] + mean
    return (recon - x).reshape(b, h, t, d)


FAMILIES = {
    "isotropic": err_isotropic,
    "random_subspace": err_random_subspace,
    "svd_truncation": err_svd_truncation,
    "fake_quant": err_fake_quant,
    "head_subset": err_head_subset,
    "position_slice": err_position_slice,
    "affine_residual": err_affine_residual,
}


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------

DEFAULT_SPECS = [
    ("isotropic", {}),
    ("random_subspace", {"rank": 4}),
    ("random_subspace", {"rank": 16}),
    ("svd_truncation", {"rank": 8}),
    ("svd_truncation", {"rank": 32}),
    ("fake_quant", {"bits": 4, "per": "head"}),
    ("fake_quant", {"bits": 3, "per": "token"}),
    ("head_subset", {"frac": 0.25}),
    ("head_subset", {"frac": 0.5}),
    ("position_slice", {"where": "recent", "frac": 0.25}),
    ("position_slice", {"where": "oldest", "frac": 0.25}),
    ("position_slice", {"where": "middle", "frac": 0.25}),
    ("affine_residual", {"rank": 8}),
    ("affine_residual", {"rank": 32}),
]

DEFAULT_LEVELS = (0.005, 0.02, 0.08)


def build_bank(
    block: torch.Tensor,
    *,
    kind: str,
    layer: int,
    generator: torch.Generator,
    specs=DEFAULT_SPECS,
    levels=DEFAULT_LEVELS,
    context_tag: str = "",
) -> list:
    """Every spec at every level, each group renormalised to a common norm.

    The level is relative to the block's own Frobenius norm, so a "2 percent"
    perturbation means the same physical thing on a 64-dim Llama head and a
    128-dim Qwen head.  Renormalising *within* a level is what creates the
    matched-norm groups, and it is done after the family has chosen a direction
    so the direction is never distorted by the rescale.
    """
    out: list = []
    ref = float(block.float().norm())
    for level in levels:
        target = ref * level
        group = f"{context_tag}|L{layer}|{kind}|lvl{level}"
        for family, params in specs:
            raw = FAMILIES[family](block, generator, **params)
            if raw.float().norm() < 1e-20:
                continue  # a degenerate direction carries no information
            delta = (_unit(raw) * target).to(block.dtype)
            out.append(
                Perturbation(
                    name=f"{group}|{family}|{_ptag(params)}",
                    kind=kind,
                    layer=layer,
                    family=family,
                    delta=delta,
                    norm_group=group,
                    params=dict(params, level=level),
                )
            )
    return out


def _ptag(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items())) or "default"


def matched_norm_report(bank) -> dict:
    """Confirm every group really is norm-matched before anything is believed.

    If this drifts, raw MSE regains ranking power inside a group and the
    within-group comparison silently stops being a direction-only test.
    """
    groups: dict = {}
    for p in bank:
        groups.setdefault(p.norm_group, []).append(p.frob)
    out = {}
    for g, norms in groups.items():
        lo, hi = min(norms), max(norms)
        out[g] = {
            "n": len(norms),
            "norm": lo,
            "max_rel_spread": (hi - lo) / max(lo, 1e-30),
        }
    return out
