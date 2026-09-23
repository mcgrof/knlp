# SPDX-License-Identifier: GPL-2.0
"""Selective repair of a translated cache, and the selectors that choose what.

The mapper produces a cache the receiver can read but reads badly. This asks
whether repairing a small fraction of its positions -- with receiver-native
values, or later by actually recomputing them -- recovers enough to be worth
the work. It is the CacheBlend idea moved across a model boundary, where the
error being repaired is a mapping error rather than a missing cross-chunk
interaction.

Two contracts are kept separate throughout, because conflating them is how
this kind of experiment goes wrong.

A repair *replaces* the contents of positions that already exist. It never
appends, never reorders, and never changes how many positions the cache holds.
A cache of T positions is still a cache of T positions afterwards, and
position i is still position i. That is what makes the result comparable with
the unrepaired cache at all.

A *selector* decides which positions to repair. Whether it is allowed to see
the native cache is what separates an oracle from something deployable, and
the two are never mixed silently: a selector that reads native values is
oracle data even when its rule is one a deployment could run.

The unit is a block of contiguous positions, because a repair that recomputes
tokens will need contiguity and because per-position selection would let a
selector smuggle in information a real one could not use.
"""

from __future__ import annotations

import torch

BLOCK = 16


def n_blocks(n_positions: int, block: int = BLOCK) -> int:
    """Whole blocks only. A partial trailing block is not selectable."""
    return n_positions // block


def blocks_for_budget(n_positions: int, frac: float, block: int = BLOCK) -> int:
    """How many blocks a budget buys, rounded down.

    Rounded down rather than to nearest, so a stated budget is never exceeded
    by the rounding itself.
    """
    return int(n_blocks(n_positions, block) * frac)


def mask_from_blocks(n_positions: int, blocks, block: int = BLOCK) -> torch.Tensor:
    """A boolean position mask from selected block indices."""
    m = torch.zeros(n_positions, dtype=torch.bool)
    for b in blocks:
        lo = b * block
        m[lo : lo + block] = True
    return m


def uniform_blocks(n_positions: int, k: int, block: int = BLOCK):
    """Evenly spaced blocks: the position-coverage control.

    Deterministic, and independent of anything measured. If this matches or
    beats a ranked selector then adaptivity has not earned its cost.
    """
    total = n_blocks(n_positions, block)
    if k <= 0 or total <= 0:
        return []
    if k >= total:
        return list(range(total))
    # Spread across the whole span, first block included.
    step = total / k
    return sorted({min(total - 1, int(i * step)) for i in range(k)})


def rank_blocks_by_attention(attn, n_positions: int, k: int, block: int = BLOCK):
    """Blocks ranked by summed attention mass, ties broken by block index.

    ``attn`` is a 1-D tensor of per-position attention already reduced over
    layers, heads and query tokens. Summing within a block rather than taking
    a maximum keeps a block of consistently-attended positions ahead of one
    with a single spike, which is the quantity a budget actually spends on.
    """
    total = n_blocks(n_positions, block)
    if k <= 0 or total <= 0:
        return []
    a = attn[: total * block].reshape(total, block).sum(dim=1)
    # Stable ordering: sort by score descending, then by index ascending. A
    # plain topk leaves ties to kernel order, which is not reproducible.
    order = sorted(range(total), key=lambda i: (-float(a[i]), i))
    return sorted(order[:k])


class CacheContract:
    """What a cache pair must satisfy before anything is copied into it.

    Checked once, against the receiver's own geometry, and before allocation
    rather than after. A shape check alone does not establish that axis -2 is
    positions: the serving path this mirrors passes a tensor whose head axis
    lands where the helper expects tokens, and every shape in it is valid.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, n_positions, dtype, device):
        self.n_layers = int(n_layers)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.n_positions = int(n_positions)
        self.dtype = dtype
        self.device = torch.device(device)

    def expected(self):
        return (1, self.n_kv_heads, self.n_positions, self.head_dim)

    def check_cache(self, cache, name):
        if len(cache) != self.n_layers:
            raise ValueError(
                f"{name}: {len(cache)} layers, receiver has {self.n_layers}"
            )
        want = self.expected()
        for li, t in enumerate(cache):
            if tuple(t.shape) != want:
                raise ValueError(
                    f"{name} layer {li}: shape {tuple(t.shape)} != {want}. Axis "
                    "order is batch, kv-head, position, channel; a permuted "
                    "tensor of the same volume is not the same cache."
                )
            if t.dtype != self.dtype:
                raise ValueError(f"{name} layer {li}: dtype {t.dtype} != {self.dtype}")
            if t.device.type != self.device.type:
                raise ValueError(
                    f"{name} layer {li}: device {t.device} != {self.device}"
                )

    def check_pair(self, keys, values, name):
        self.check_cache(keys, f"{name} keys")
        self.check_cache(values, f"{name} values")

    def check_mask(self, mask):
        if not isinstance(mask, torch.Tensor):
            raise ValueError("mask must be a tensor")
        if mask.dtype != torch.bool:
            raise ValueError(f"mask dtype {mask.dtype} is not bool")
        if mask.dim() != 1:
            raise ValueError(f"mask has {mask.dim()} dimensions, expected 1")
        if mask.shape[0] != self.n_positions:
            raise ValueError(
                f"mask covers {mask.shape[0]} positions, cache holds "
                f"{self.n_positions}"
            )

    def check_blocks(self, blocks, block=BLOCK):
        total = n_blocks(self.n_positions, block)
        if len(set(blocks)) != len(blocks):
            raise ValueError(f"block indices repeat: {blocks}")
        bad = [b for b in blocks if not (0 <= b < total)]
        if bad:
            raise ValueError(f"block indices out of range 0..{total - 1}: {bad}")

    def check_fraction(self, frac):
        if not (0.0 <= float(frac) <= 1.0):
            raise ValueError(f"budget fraction {frac} is outside 0..1")

    def check_scores(self, scores):
        if scores.shape[0] != self.n_positions:
            raise ValueError(
                f"scores cover {scores.shape[0]} positions, cache holds "
                f"{self.n_positions}"
            )
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("selection scores contain non-finite values")


def rank_blocks_on_device(attn, n_positions: int, k: int, block: int = BLOCK):
    """Block ranking without a synchronisation per block.

    The readable version indexes a device tensor once per block, and each of
    those is a synchronisation. Here the whole reduction happens on device and
    exactly one transfer follows. Ties are broken by block index, which is
    reproducible, rather than by whatever order a sort kernel produced.
    """
    total = n_blocks(n_positions, block)
    if k <= 0 or total <= 0:
        return []
    a = attn[: total * block].reshape(total, block).sum(dim=1)
    idx = torch.arange(total, device=a.device, dtype=a.dtype)
    # Subtract a strictly increasing tiebreak smaller than any real gap, so a
    # descending sort puts the lower index first among equals.
    span = (a.max() - a.min()).clamp(min=1.0)
    keyed = a - idx * (span * 1e-9)
    sel = torch.topk(keyed, k=min(k, total)).indices
    return sorted(int(x) for x in sel.cpu().tolist())


def repair_cache(mapped, native, mask):
    """Replace masked positions of every layer with the donor's values.

    ``mapped`` and ``native`` are lists of [1, H, T, D] tensors, one per layer,
    and must agree in length, shape and dtype: a repair that silently upcasts
    or reshapes is measuring something other than the repair. The mask is over
    positions and is applied identically in every layer, because introducing a
    position in one layer and not another would require hidden states the
    handoff does not carry.

    Returns new tensors. The inputs are not modified, so a caller can compare
    against them afterwards.
    """
    if len(mapped) != len(native):
        raise ValueError(f"layer count {len(mapped)} != {len(native)}")
    out = []
    for li, (m, n) in enumerate(zip(mapped, native)):
        if m.shape != n.shape:
            raise ValueError(f"layer {li}: shape {tuple(m.shape)} != {tuple(n.shape)}")
        if m.dtype != n.dtype:
            raise ValueError(f"layer {li}: dtype {m.dtype} != {n.dtype}")
        if mask.shape[0] != m.shape[2]:
            raise ValueError(
                f"layer {li}: mask covers {mask.shape[0]} positions, cache holds "
                f"{m.shape[2]}"
            )
        r = m.clone()
        idx = mask.to(m.device)
        r[:, :, idx, :] = n[:, :, idx, :]
        out.append(r)
    return out


def positions_changed(before, after):
    """Which positions actually differ, as a sanity check on a repair.

    A repair that changed nothing, or changed more than it was asked to, is a
    different experiment from the one being reported.
    """
    T = before[0].shape[2]
    changed = torch.zeros(T, dtype=torch.bool)
    for b, a in zip(before, after):
        d = (b != a).any(dim=-1).any(dim=1).squeeze(0)
        changed |= d.cpu()
    return changed
