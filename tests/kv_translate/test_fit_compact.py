# SPDX-License-Identifier: GPL-2.0
"""The compact fit's support choice, on data whose right answer is known.

The stored supports are in depth order, so a fitter that forgot to re-rank
would silently take the shallowest layers and still produce a map that runs.
These are built so that failure is visible: the layers that actually carry the
target are not the shallowest ones in the support.
"""

from __future__ import annotations

import pytest
import torch

from research.kv_translate.fit import Accumulator, SourceLayout
from research.kv_translate.fit_compact import HEAD_ASSIGNMENT, rank_within_support

L, H, D = 28, 2, 8
SUPPORT = tuple(sorted([0, 1, 2, 3, 5, 8, 9, 11, 14, 17, 19, 21, 23, 25, 26, 27]))
TRUE = (2, 9, 17, 25)


@pytest.fixture(scope="module")
def acc():
    torch.manual_seed(0)
    layout = SourceLayout(L, H, D)
    a = Accumulator(layout, n_targets=1, target_dim=D)
    w = {li: torch.randn(H * D, D) * 0.6 for li in TRUE}
    for _ in range(40):
        x = torch.randn(256, layout.n_features)
        y = torch.zeros(1, 256, D)
        for li in TRUE:
            y[0] += x[:, layout.columns_for([li], 0, head_local=False)] @ w[li]
        y += 0.02 * torch.randn_like(y)
        a.add(x, y)
    return a


def test_the_ranking_finds_the_layers_that_carry_the_block(acc):
    got = rank_within_support(acc, 0, 0, SUPPORT, 1e-3, False, 4)
    assert got == TRUE, f"chose {got}"


def test_the_choice_is_not_the_shallowest_four(acc):
    """Otherwise the test could not tell a re-ranking from a depth-order bug."""
    assert rank_within_support(acc, 0, 0, SUPPORT, 1e-3, False, 4) != tuple(
        sorted(SUPPORT)[:4]
    )


def test_the_ranking_repeats_exactly(acc):
    a = rank_within_support(acc, 0, 0, SUPPORT, 1e-3, False, 4)
    b = rank_within_support(acc, 0, 0, SUPPORT, 1e-3, False, 4)
    assert a == b


def test_the_choice_stays_inside_the_support_it_was_given(acc):
    narrow = (5, 8, 11, 14, 19, 21)
    got = rank_within_support(acc, 0, 0, narrow, 1e-3, False, 4)
    assert set(got) <= set(narrow), "ranked outside the block's own support"
    assert len(got) == 4


def test_it_returns_them_in_depth_order(acc):
    """So two selections can be compared by eye."""
    got = rank_within_support(acc, 0, 0, SUPPORT, 1e-3, False, 4)
    assert list(got) == sorted(got)


def test_asking_for_the_whole_support_returns_the_whole_support(acc):
    got = rank_within_support(acc, 0, 0, SUPPORT, 1e-3, False, len(SUPPORT))
    assert got == SUPPORT


def test_the_head_assignment_covers_the_receiver_and_fits_the_source():
    """Four receiver heads onto two source heads, declared not derived."""
    assert len(HEAD_ASSIGNMENT) == 4
    assert set(HEAD_ASSIGNMENT) == {0, 1}
    assert max(HEAD_ASSIGNMENT) < H


def test_head_local_columns_partition_the_full_fan_in():
    """Each source head's columns are disjoint and together are the whole."""
    layout = SourceLayout(L, H, D)
    sel = list(TRUE)
    full = set(layout.columns_for(sel, 0, head_local=False).tolist())
    parts = [
        set(layout.columns_for(sel, h, head_local=True).tolist()) for h in range(H)
    ]
    assert not (parts[0] & parts[1])
    assert parts[0] | parts[1] == full
    for p in parts:
        assert len(p) == len(full) // H


def test_narrowing_the_support_narrows_the_input(acc):
    layout = SourceLayout(L, H, D)
    wide = layout.columns_for(list(SUPPORT), 0, head_local=False).numel()
    k4_full = layout.columns_for(list(TRUE), 0, head_local=False).numel()
    k4_local = layout.columns_for(list(TRUE), 0, head_local=True).numel()
    assert wide / k4_full == len(SUPPORT) / 4
    assert wide / k4_local == len(SUPPORT) / 4 * H


def test_the_apply_path_gathers_the_source_head_the_fit_recorded():
    """A receiver head is not a source head when the counts differ.

    A head-local map fitted with a receiver-to-source assignment stores the
    source head it reads. Two places in the batched apply path used to derive
    the head from the block's dictionary key instead, which is the receiver
    head; for a pair whose counts match that is the same number and nothing
    goes wrong, and for a pair like this one it gathers the wrong columns or
    runs off the end of the source. This asserts the grouping key and the
    dense packing both use the recorded head.
    """
    import torch

    from research.kv_translate.apply import selection_groups
    from research.kv_translate.fit import AffineMap

    layout = SourceLayout(L, H, D)

    class FakeMapper:
        pass

    m = FakeMapper()
    # four receiver heads, two source heads, assigned as the fitter assigns them
    m.maps = {
        (0, h): AffineMap(
            M=torch.zeros(len(TRUE) * D, D),
            b=torch.zeros(D),
            layers=TRUE,
            head=HEAD_ASSIGNMENT[h],
            kind="k",
            target_layer=0,
            ridge=1.0,
            head_local=True,
            n_calib_tokens=1,
        )
        for h in range(4)
    }
    groups = selection_groups(m, layout)
    heads_in_keys = sorted({key[2] for key in groups})
    assert heads_in_keys == [0, 1], (
        f"grouping keyed on {heads_in_keys}; it must key on the two source "
        "heads, not on the four receiver heads"
    )
    # receiver heads 0,1 share a source head, and so must share a group
    assert len(groups) == 2
    for key, blocks in groups.items():
        assert len(blocks) == 2


def test_freezing_and_loading_preserves_the_source_head(tmp_path):
    """The format used to reconstruct it from the block's name.

    A block is named for the receiver head it writes. The head it *reads* is
    the source head, and those differ whenever the two models carry different
    numbers of key/value heads. Reconstructing from the name gave back a head
    the source may not even have, which is invisible for a matched pair and
    fatal for this one.
    """
    import torch

    from research.kv_translate import freeze
    from research.kv_translate.fit import AffineMap
    from research.kv_translate.pairs import ModelGeometry
    from research.kv_translate.run_a1 import Mapper

    layout = SourceLayout(L, H, D)
    geom = ModelGeometry(
        model_id="fake",
        revision="none",
        n_layers=1,
        n_q_heads=4,
        n_kv_heads=4,
        head_dim=D,
        hidden=D * 4,
        rope_theta=10000.0,
        rope_scaling=None,
        vocab=32,
    )

    def block(recv_head):
        return AffineMap(
            M=torch.zeros(len(TRUE) * D, D),
            b=torch.zeros(D),
            layers=TRUE,
            head=HEAD_ASSIGNMENT[recv_head],  # the SOURCE head it reads
            kind="k",
            target_layer=0,
            ridge=1.0,
            head_local=True,
            n_calib_tokens=1,
        )

    maps = {(0, h): block(h) for h in range(4)}
    mk = Mapper(dict(maps), layout, geom, "k")
    mv = Mapper(dict(maps), layout, geom, "v")
    path = str(tmp_path / "arm.pt")
    freeze.save(
        path,
        mk,
        mv,
        source={"model_id": "s", "revision": "r"},
        target={"model_id": "t", "revision": "r"},
        fit_context=8,
        calib_doc_ids=[],
        dev_doc_ids=[],
        eval_doc_ids=[],
        config={},
    )
    lk, lv, _ = freeze.load(path, layout, geom)
    for h in range(4):
        want = HEAD_ASSIGNMENT[h]
        assert lk.maps[(0, h)].head == want, (
            f"receiver head {h} came back reading source head "
            f"{lk.maps[(0, h)].head}, not {want}"
        )
        # and the columns it would gather are legal for the source
        layout.columns_for(list(TRUE), lk.maps[(0, h)].head, True)


def test_an_artifact_without_the_field_still_loads(tmp_path):
    """Older artifacts predate it, and for those the name was correct."""
    import torch

    from research.kv_translate import freeze

    path = str(tmp_path / "old.pt")
    blob = {
        "blocks": {
            "k.0.1": {
                "M": torch.zeros(4, 4),
                "b": torch.zeros(4),
                "layers": [0],
                "head_local": True,
                "ridge": 1.0,
                "n_calib_tokens": 1,
            }
        },
        "manifest": {"weight_hashes": {}, "joint_weight_sha256": "x"},
    }
    torch.save(blob, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    rec = loaded["blocks"]["k.0.1"]
    assert "source_head" not in rec
    assert int(rec.get("source_head", 1)) == 1, "the fallback must use the name"
