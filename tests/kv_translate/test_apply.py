# SPDX-License-Identifier: GPL-2.0
"""The batched apply must compute what the block-at-a-time apply computed.

Packing a mapper into a few large multiplies is an optimisation, so the only
thing that makes it safe is an equivalence check against the path it replaces.
Both packings are checked, on a mapper whose blocks deliberately do *not* all
select the same source layers -- the case where grouping has to reassemble
outputs in the right order and is therefore able to get it wrong.

The dtype guard is tested as behaviour rather than documented as advice: a
float64 serving weight is the bug that made this path slower than the prefill
it replaces, and it should be impossible to reintroduce by accident.
"""

import pytest
import torch

from research.kv_translate.apply import (
    BatchedMapper,
    assert_serving_dtype,
    selection_groups,
)
from research.kv_translate.fit import AffineMap, SourceLayout

L_SRC, H_SRC, D_SRC = 6, 2, 8
L_TGT, H_TGT, D_TGT = 4, 3, 5


class _Geom:
    n_layers = L_TGT
    n_kv_heads = H_TGT


class _Mapper:
    """Minimal stand-in carrying the fields the packer reads."""

    def __init__(self, maps):
        self.maps = maps

    def apply_reference(self, X, layout):
        """The block-at-a-time path, kept here as the thing to match."""
        out = []
        for li in range(L_TGT):
            heads = []
            for h in range(H_TGT):
                m = self.maps[(li, h)]
                cols = layout.columns_for(list(m.layers), h, m.head_local)
                heads.append(X[:, cols].to(m.M.dtype) @ m.M + m.b)
            out.append(torch.stack(heads, 0).unsqueeze(0))
        return out


def _make(seed=0, varied=True, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    lay = SourceLayout(L_SRC, H_SRC, D_SRC)
    maps = {}
    for li in range(L_TGT):
        for h in range(H_TGT):
            layers = (
                tuple(sorted(((li + h) % L_SRC, (li + h + 1) % L_SRC)))
                if varied
                else (0, 1)
            )
            d_sel = len(layers) * H_SRC * D_SRC
            maps[(li, h)] = AffineMap(
                M=torch.randn(d_sel, D_TGT, generator=g, dtype=dtype),
                b=torch.randn(D_TGT, generator=g, dtype=dtype),
                layers=layers,
                head=h,
                kind="k",
                target_layer=li,
                ridge=0.0,
                head_local=False,
                n_calib_tokens=1,
            )
    return lay, _Mapper(maps)


@pytest.mark.parametrize("strategy", ["grouped", "dense"])
def test_batched_apply_matches_the_reference(strategy):
    lay, mp = _make(varied=True)
    X = torch.randn(17, lay.n_features, dtype=torch.float64)
    ref = mp.apply_reference(X, lay)
    bm = BatchedMapper(mp, lay, _Geom(), dtype=torch.float32, strategy=strategy)
    got = bm.apply(X)
    assert len(got) == len(ref) == L_TGT
    for a, b in zip(got, ref):
        assert a.shape == b.shape
        assert torch.allclose(a.double(), b, atol=1e-5, rtol=0)


@pytest.mark.parametrize("strategy", ["grouped", "dense"])
def test_batched_apply_matches_when_every_block_shares_a_selection(strategy):
    """The degenerate grouping: one group holding every block."""
    lay, mp = _make(varied=False)
    X = torch.randn(9, lay.n_features, dtype=torch.float64)
    ref = mp.apply_reference(X, lay)
    bm = BatchedMapper(mp, lay, _Geom(), dtype=torch.float32, strategy=strategy)
    for a, b in zip(bm.apply(X), ref):
        assert torch.allclose(a.double(), b, atol=1e-5, rtol=0)


def test_grouping_collapses_shared_selections_and_not_distinct_ones():
    lay, shared = _make(varied=False)
    lay2, varied = _make(varied=True)
    assert len(selection_groups(shared, lay)) == 1
    assert len(selection_groups(varied, lay2)) > 1
    assert BatchedMapper(shared, lay, _Geom(), strategy="grouped").stats.n_matmuls == 1


def test_dense_is_always_one_multiply_and_grouped_is_never_more_than_blocks():
    lay, mp = _make(varied=True)
    dense = BatchedMapper(mp, lay, _Geom(), strategy="dense").stats
    grouped = BatchedMapper(mp, lay, _Geom(), strategy="grouped").stats
    assert dense.n_matmuls == 1
    assert 1 <= grouped.n_matmuls <= grouped.n_blocks
    # dense buys its single multiply with stored zeros
    assert dense.weight_bytes > grouped.weight_bytes


def test_serving_dtype_guard_refuses_double():
    """The exact bug that made the apply slower than the prefill it replaces."""
    with pytest.raises(ValueError, match="must not be float64"):
        assert_serving_dtype(torch.float64)
    assert_serving_dtype(torch.float32)
    assert_serving_dtype(torch.bfloat16)
    lay, mp = _make()
    with pytest.raises(ValueError, match="must not be float64"):
        BatchedMapper(mp, lay, _Geom(), dtype=torch.float64)


def test_packed_weights_carry_the_serving_dtype():
    lay, mp = _make()
    for strategy in ("grouped", "dense"):
        bm = BatchedMapper(mp, lay, _Geom(), dtype=torch.float32, strategy=strategy)
        out = bm.apply(torch.randn(5, lay.n_features))
        assert out[0].dtype == torch.float32
        assert bm.stats.dtype == str(torch.float32)


def test_column_indices_are_resolved_once():
    """They were rebuilt on every prefix on the path this replaces."""
    lay, mp = _make()
    bm = BatchedMapper(mp, lay, _Geom(), strategy="grouped")
    assert all(isinstance(g["cols"], torch.Tensor) for g in bm.groups)
    assert all(g["W"].is_contiguous() for g in bm.groups)
