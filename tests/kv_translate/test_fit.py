# SPDX-License-Identifier: GPL-2.0
"""The fitter, pinned on problems whose answer is known in advance.

Three things have to be true before a translation number means anything: the
streamed statistics have to give the same map a direct fit would, the source
layer a target block actually came from has to be the one selection picks, and
the head arms have to differ only in which columns they read.

There is also a property of the evaluation itself worth pinning here, because
it decides whether one of the plan's controls can ever fail: attention over a
cache is a sum over slots, so permuting cached key and value pairs together
cannot change the result. A control built on that permutation is vacuous, and
the audit reports it as a zero-by-construction check rather than as evidence.
"""

import math

import pytest
import torch

from research.kv_translate.fit import (
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
    solve_ridge,
    source_layer_scores,
)

torch.manual_seed(0)

L, H, D = 6, 2, 8
TD = 5


def _layout():
    return SourceLayout(n_layers=L, n_kv_heads=H, head_dim=D)


def _synthetic(n_tokens=400, source_layer=3, head=0, noise=0.05, seed=0):
    """Target blocks generated from one known source layer, plus noise."""
    g = torch.Generator().manual_seed(seed)
    lay = _layout()
    X = torch.randn(n_tokens, lay.n_features, generator=g, dtype=torch.float64)
    W = torch.randn(D, TD, generator=g, dtype=torch.float64)
    b = torch.randn(TD, generator=g, dtype=torch.float64)
    sl = lay.feature_slice(source_layer, head)
    Y = (
        X[:, sl] @ W
        + b
        + noise * torch.randn(n_tokens, TD, generator=g, dtype=torch.float64)
    )
    return lay, X, Y, W, b


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def test_head_local_reads_only_the_matching_head():
    lay = _layout()
    cols = lay.columns_for([2], head=1, head_local=True)
    assert cols.numel() == D
    expected = lay.feature_slice(2, 1)
    assert cols.tolist() == list(range(expected.start, expected.stop))


def test_full_fan_in_reads_every_head():
    lay = _layout()
    cols = lay.columns_for([2], head=1, head_local=False)
    assert cols.numel() == H * D
    assert cols.tolist()[:D] == list(range(*_span(lay.feature_slice(2, 0))))


def _span(sl):
    return sl.start, sl.stop


def test_column_count_scales_with_selected_layers():
    lay = _layout()
    for k in (1, 3, L):
        layers = list(range(k))
        assert lay.columns_for(layers, 0, True).numel() == k * D
        assert lay.columns_for(layers, 0, False).numel() == k * H * D
    assert lay.n_features == L * H * D


def test_selected_columns_are_distinct():
    lay = _layout()
    cols = lay.columns_for([0, 1, 2], head=0, head_local=False)
    assert len(set(cols.tolist())) == cols.numel()


# ---------------------------------------------------------------------------
# streamed statistics
# ---------------------------------------------------------------------------


def test_streamed_fit_matches_a_direct_one():
    """The whole point of sufficient statistics: same answer, no tokens kept."""
    lay, X, Y, _, _ = _synthetic(n_tokens=300)
    # fed in several chunks, because that is how the real pass arrives
    acc = Accumulator(lay, n_targets=1, target_dim=TD)
    start = 0
    for size in (50, 120, 130):
        acc.add(X[start : start + size], Y[start : start + size].unsqueeze(0))
        start += size
    assert acc.n == 300

    layers = list(range(L))
    amap = fit_block(
        acc,
        target=0,
        target_layer=0,
        head=0,
        kind="k",
        layers=layers,
        ridge=1e-6,
        head_local=False,
    )
    cols = lay.columns_for(layers, 0, False)
    Xc = X[:300, cols]
    Xm, Ym = Xc.mean(0), Y[:300].mean(0)
    Xd, Yd = Xc - Xm, Y[:300] - Ym
    direct = solve_ridge(Xd.T @ Xd, Xd.T @ Yd, 1e-6)
    assert torch.allclose(amap.M, direct, atol=1e-6, rtol=0)
    assert torch.allclose(amap.b, Ym - direct.T @ Xm, atol=1e-6, rtol=0)


def test_accumulation_is_order_independent():
    lay, X, Y, _, _ = _synthetic(n_tokens=200)
    a = Accumulator(lay, 1, TD)
    a.add(X, Y.unsqueeze(0))
    b = Accumulator(lay, 1, TD)
    for s, e in ((0, 80), (80, 160), (160, 200)):
        b.add(X[s:e], Y[s:e].unsqueeze(0))
    assert torch.allclose(a.xtx, b.xtx, atol=1e-8)
    assert torch.allclose(a.xty, b.xty, atol=1e-8)
    assert a.n == b.n == 200


def test_fit_recovers_a_noiseless_map():
    lay, X, Y, W, b = _synthetic(n_tokens=300, source_layer=2, noise=0.0)
    acc = Accumulator(lay, 1, TD)
    acc.add(X, Y.unsqueeze(0))
    amap = fit_block(
        acc,
        target=0,
        target_layer=0,
        head=0,
        kind="k",
        layers=[2],
        ridge=1e-10,
        head_local=True,
    )
    assert torch.allclose(amap.M, W, atol=1e-5, rtol=0)
    assert torch.allclose(amap.b, b, atol=1e-5, rtol=0)


def test_ridge_shrinks_the_map():
    lay, X, Y, _, _ = _synthetic(n_tokens=300)
    acc = Accumulator(lay, 1, TD)
    acc.add(X, Y.unsqueeze(0))
    norms = []
    for r in (1e-8, 1e2, 1e5):
        m = fit_block(
            acc,
            target=0,
            target_layer=0,
            head=0,
            kind="k",
            layers=list(range(L)),
            ridge=r,
            head_local=False,
        )
        norms.append(float(m.M.norm()))
    assert norms[0] > norms[1] > norms[2]


# ---------------------------------------------------------------------------
# source-layer selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_layer", [0, 3, 5])
def test_selection_finds_the_layer_the_target_came_from(true_layer):
    lay, X, Y, _, _ = _synthetic(n_tokens=500, source_layer=true_layer, noise=0.05)
    acc = Accumulator(lay, 1, TD)
    acc.add(X, Y.unsqueeze(0))
    ranked = source_layer_scores(acc, 0, 0, ridge=1e-6, head_local=True)
    assert ranked[0][0] == true_layer
    assert select_layers(acc, 0, 0, 1e-6, 1, True) == (true_layer,)


def test_selection_returns_layers_in_depth_order():
    lay, X, Y, _, _ = _synthetic(n_tokens=400, source_layer=4)
    acc = Accumulator(lay, 1, TD)
    acc.add(X, Y.unsqueeze(0))
    sel = select_layers(acc, 0, 0, 1e-6, 3, True)
    assert list(sel) == sorted(sel)
    assert len(set(sel)) == 3


def test_more_layers_never_fit_worse():
    lay, X, Y, _, _ = _synthetic(n_tokens=500, source_layer=2)
    acc = Accumulator(lay, 1, TD)
    acc.add(X, Y.unsqueeze(0))
    prev = None
    for k in (1, 2, 4, L):
        layers = select_layers(acc, 0, 0, 1e-8, k, True)
        m = fit_block(
            acc,
            target=0,
            target_layer=0,
            head=0,
            kind="k",
            layers=layers,
            ridge=1e-8,
            head_local=True,
        )
        cols = lay.columns_for(layers, 0, True)
        G, C, _, _ = acc.centred(cols, 0)
        rss = -2 * float((m.M * C).sum()) + float((m.M.T @ G @ m.M).diagonal().sum())
        if prev is not None:
            assert rss <= prev + 1e-6
        prev = rss


# ---------------------------------------------------------------------------
# the property that makes one control vacuous
# ---------------------------------------------------------------------------


def test_attention_over_a_cache_ignores_slot_order():
    """Permuting cached key/value pairs together cannot change attention.

    This is why the plan's "shuffled token positions" control is vacuous as
    stated: position is carried by the rotation already applied to the stored
    keys, not by the slot index, so the audit substitutes a control that
    re-rotates keys at the wrong positions instead.
    """
    g = torch.Generator().manual_seed(3)
    T, d = 32, 16
    q = torch.randn(1, 1, d, generator=g, dtype=torch.float64)
    k = torch.randn(1, T, d, generator=g, dtype=torch.float64)
    v = torch.randn(1, T, d, generator=g, dtype=torch.float64)

    def attend(k, v):
        w = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(d), dim=-1)
        return w @ v

    perm = torch.randperm(T, generator=g)
    assert torch.allclose(attend(k, v), attend(k[:, perm], v[:, perm]), atol=1e-12)
    # permuting the keys alone is *not* a no-op, which is what makes the
    # substituted control meaningful
    assert not torch.allclose(attend(k, v), attend(k[:, perm], v), atol=1e-6)


def test_head_local_refuses_an_undefined_correspondence():
    """Pairing target head h with source head h needs both to have head h.

    Model families do not guarantee it: Qwen2.5 carries two key/value heads up
    to 1.5B and four at 7B. Silently taking the wrong columns produced an
    out-of-bounds index that surfaced as a device-side assert several frames
    away, at the start of a paid run.
    """
    lay = _layout()  # two kv heads
    lay.columns_for([0], head=1, head_local=True)  # defined
    with pytest.raises(ValueError, match="only 2 key/value heads"):
        lay.columns_for([0], head=3, head_local=True)
    # full fan-in does not use the head index and stays defined
    assert lay.columns_for([0], head=3, head_local=False).numel() == H * D
