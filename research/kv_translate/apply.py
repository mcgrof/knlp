# SPDX-License-Identifier: GPL-2.0
"""Applying a whole cache map in a few large multiplies instead of many small ones.

The fitted mapper is one small affine map per target block. Applying it a block
at a time means, on a 28-layer four-head target, 224 separate gather-plus-
multiply pairs of shape [tokens, few-thousand] by [few-thousand, 128]. That is
far too little arithmetic per launch to reach a GPU's throughput, and a
measurement of it reports the cost of dispatching work rather than of doing it.

Two ways to fix that, with different byte costs, so both are built and the
measurement chooses:

``grouped``
    Blocks whose source-layer selection is identical share a column gather and
    one multiply, their weights concatenated along the output axis. Storage is
    unchanged. How much this helps depends entirely on how many *distinct*
    selections the fit produced, which is a property of the data and not
    knowable in advance -- hence :func:`selection_groups` to look before
    choosing.

``dense``
    One multiply against the whole source feature space, with each block's
    weights scattered into the rows it selected and zeros elsewhere. Always
    exactly one multiply per tensor kind, at the cost of storing the zeros:
    bytes grow by the ratio of full features to selected ones.

Neither changes what the map computes. A test pins the two against the
block-at-a-time reference.

One hard rule lives here: online weights are never float64. Double is correct
for an ill-conditioned solve that happens once and wrong for a multiply that
happens on every prefix, and applying a fitted map in double once made this
path 3.15 times the prefill it exists to replace. The constructor refuses it
rather than trusting a caller to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch


def assert_serving_dtype(
    dtype: torch.dtype, what: str = "online mapper weights"
) -> None:
    """Refuse double precision on the serving path.

    Not a style preference. The fit is solved in float64 because a Gram matrix
    is ill-conditioned; carrying that dtype into the apply costs roughly thirty
    times the arithmetic for a difference measured at four parts in ten
    million, and it silently turned a method that beat re-prefilling into one
    that lost to it.
    """
    if dtype in (torch.float64, torch.complex128):
        raise ValueError(
            f"{what} must not be float64: the solve needs double, the apply "
            "never does, and it costs about 30x the throughput for no "
            "measurable accuracy"
        )


def selection_groups(mapper, layout) -> dict:
    """Blocks bucketed by the source columns they read.

    The key is what determines whether grouping helps: identical selections
    share a gather and a multiply, distinct ones cannot.
    """
    groups: dict = {}
    for (li, h), m in mapper.maps.items():
        key = (tuple(m.layers), bool(m.head_local), h if m.head_local else -1)
        groups.setdefault(key, []).append((li, h))
    return groups


@dataclass
class ApplyStats:
    """What the chosen packing costs, so a latency result can be attributed."""

    strategy: str
    n_blocks: int
    n_groups: int
    n_matmuls: int
    weight_bytes: float
    dtype: str
    d_full: int
    d_selected_max: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class BatchedMapper:
    """A fitted mapper prepacked for application.

    Column indices are resolved once at construction rather than rebuilt per
    call; on the block-at-a-time path they were recomputed on every prefix.
    """

    def __init__(
        self,
        mapper,
        layout,
        geom,
        *,
        dtype: torch.dtype = torch.float32,
        strategy: str = "grouped",
        device=None,
    ):
        assert_serving_dtype(dtype)
        self.geom = geom
        self.layout = layout
        self.dtype = dtype
        self.strategy = strategy
        self.head_dim = next(iter(mapper.maps.values())).M.shape[1]
        dev = device or next(iter(mapper.maps.values())).M.device
        self.device = dev

        groups = selection_groups(mapper, layout)
        self.n_blocks = len(mapper.maps)
        self.n_groups = len(groups)

        if strategy == "dense":
            self._build_dense(mapper, layout, dev)
        elif strategy == "grouped":
            self._build_grouped(mapper, layout, groups, dev)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

    # -- packing ---------------------------------------------------------

    def _build_grouped(self, mapper, layout, groups, dev) -> None:
        self.groups = []
        for key, blocks in groups.items():
            layers, head_local, head = key
            cols = layout.columns_for(
                list(layers), head if head_local else 0, head_local
            ).to(dev)
            W = torch.cat(
                [mapper.maps[b].M.to(self.dtype) for b in blocks], dim=1
            ).contiguous()
            b = torch.cat(
                [mapper.maps[b].b.to(self.dtype) for b in blocks], dim=0
            ).contiguous()
            self.groups.append({"cols": cols, "W": W, "b": b, "blocks": blocks})
        self._weight_bytes = sum(
            g["W"].numel() * g["W"].element_size()
            + g["b"].numel() * g["b"].element_size()
            for g in self.groups
        )
        self._d_sel_max = max(int(g["cols"].numel()) for g in self.groups)
        self._n_matmuls = len(self.groups)

    def _build_dense(self, mapper, layout, dev) -> None:
        d_full = layout.n_features
        order = sorted(mapper.maps.keys())
        self.order = order
        W = torch.zeros(
            d_full, len(order) * self.head_dim, dtype=self.dtype, device=dev
        )
        b = torch.zeros(len(order) * self.head_dim, dtype=self.dtype, device=dev)
        for i, key in enumerate(order):
            m = mapper.maps[key]
            cols = layout.columns_for(
                list(m.layers), key[1] if m.head_local else 0, m.head_local
            ).to(dev)
            sl = slice(i * self.head_dim, (i + 1) * self.head_dim)
            W[cols, sl] = m.M.to(self.dtype)
            b[sl] = m.b.to(self.dtype)
        self.W, self.b = W.contiguous(), b.contiguous()
        self._weight_bytes = W.numel() * W.element_size() + b.numel() * b.element_size()
        self._d_sel_max = d_full
        self._n_matmuls = 1

    # -- application -----------------------------------------------------

    @torch.no_grad()
    def apply(self, X: torch.Tensor) -> list:
        """``[T, d_full]`` source features -> ``[L][1, H, T, D]`` target blocks."""
        Xc = X.to(self.dtype)
        H, D = self.geom.n_kv_heads, self.head_dim
        out = torch.empty(
            self.geom.n_layers, H, Xc.shape[0], D, dtype=self.dtype, device=Xc.device
        )
        if self.strategy == "dense":
            flat = torch.addmm(self.b, Xc, self.W)  # [T, n_blocks*D]
            flat = flat.view(Xc.shape[0], len(self.order), D)
            for i, (li, h) in enumerate(self.order):
                out[li, h] = flat[:, i]
        else:
            for g in self.groups:
                y = torch.addmm(g["b"], Xc[:, g["cols"]], g["W"])
                y = y.view(Xc.shape[0], len(g["blocks"]), D)
                for i, (li, h) in enumerate(g["blocks"]):
                    out[li, h] = y[:, i]
        return [out[li].unsqueeze(0) for li in range(self.geom.n_layers)]

    @property
    def stats(self) -> ApplyStats:
        return ApplyStats(
            strategy=self.strategy,
            n_blocks=self.n_blocks,
            n_groups=self.n_groups,
            n_matmuls=self._n_matmuls,
            weight_bytes=float(self._weight_bytes),
            dtype=str(self.dtype),
            d_full=self.layout.n_features,
            d_selected_max=int(self._d_sel_max),
        )


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


@torch.no_grad()
def time_callable(fn, *, warmup: int = 10, reps: int = 100, device="cuda") -> dict:
    """Median and 95th percentile of ``fn``, measured with device events.

    Wall-clock around an asynchronous launch measures the launch, not the work.
    Events are recorded on the stream and read after a synchronise, and the
    warmup is discarded so one-off allocation and autotuning do not land in the
    distribution.
    """
    import numpy as np

    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(reps)])
    else:
        import time as _t

        ms = []
        for _ in range(reps):
            t0 = _t.perf_counter()
            fn()
            ms.append(1000 * (_t.perf_counter() - t0))
        ms = np.array(ms)
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "reps": reps,
    }
