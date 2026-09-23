# SPDX-License-Identifier: GPL-2.0
"""The focused checks the recovery plan requires before any rental.

Each of these is cheap and each corresponds to a way the experiment could
report a repair it did not perform, or perform one it did not report. The
serving path this mirrors already has a live axis defect — a helper that
treats head indices as token slots — so none of the shape contracts here are
hypothetical.
"""

from __future__ import annotations

import pytest
import torch

from research.kv_translate.recovery import (
    BLOCK,
    blocks_for_budget,
    mask_from_blocks,
    n_blocks,
    positions_changed,
    rank_blocks_by_attention,
    repair_cache,
    uniform_blocks,
)

L, H, T, D = 4, 3, 256, 8


def _cache(fill):
    return [torch.full((1, H, T, D), float(fill + i)) for i in range(L)]


def test_canonical_axes_are_enforced():
    """Positions are axis -2. The serving bug confused it with heads."""
    mapped, native = _cache(0), _cache(100)
    # A mask sized to the head axis rather than the position axis must be
    # refused, not broadcast into something plausible.
    with pytest.raises(ValueError, match="mask covers"):
        repair_cache(mapped, native, torch.ones(H, dtype=torch.bool))
    ok = repair_cache(mapped, native, torch.ones(T, dtype=torch.bool))
    assert ok[0].shape == (1, H, T, D)


def test_zero_repair_is_a_true_no_op():
    mapped, native = _cache(0), _cache(100)
    out = repair_cache(mapped, native, torch.zeros(T, dtype=torch.bool))
    for a, b in zip(mapped, out):
        assert torch.equal(a, b)
    assert positions_changed(mapped, out).sum() == 0


def test_full_repair_reproduces_the_donor_exactly():
    """The 100% replacement parity control."""
    mapped, native = _cache(0), _cache(100)
    out = repair_cache(mapped, native, torch.ones(T, dtype=torch.bool))
    for n, o in zip(native, out):
        assert torch.equal(n, o)


def test_one_token_corruption_changes_exactly_that_token():
    mapped, native = _cache(0), _cache(100)
    m = torch.zeros(T, dtype=torch.bool)
    m[123] = True
    out = repair_cache(mapped, native, m)
    ch = positions_changed(mapped, out)
    assert ch.sum() == 1 and bool(ch[123])


def test_same_model_identity_repair_changes_nothing():
    """Repairing a cache with itself must be indistinguishable from not.

    If this fails, the repair path is perturbing values on its own and every
    downstream difference is partly its own doing.
    """
    mapped = _cache(0)
    same = [t.clone() for t in mapped]
    out = repair_cache(mapped, same, torch.ones(T, dtype=torch.bool))
    for a, b in zip(mapped, out):
        assert torch.equal(a, b)
    assert positions_changed(mapped, out).sum() == 0


def test_positions_are_preserved_under_sparse_repair():
    """Unselected positions keep their exact original values, in every layer."""
    mapped, native = _cache(0), _cache(100)
    blocks = [0, 5, 9]
    m = mask_from_blocks(T, blocks)
    out = repair_cache(mapped, native, m)
    for li in range(L):
        keep = ~m
        assert torch.equal(out[li][:, :, keep, :], mapped[li][:, :, keep, :])
        assert torch.equal(out[li][:, :, m, :], native[li][:, :, m, :])


def test_mask_and_slot_intersection_is_exactly_the_blocks():
    blocks = [1, 4]
    m = mask_from_blocks(T, blocks)
    assert int(m.sum()) == len(blocks) * BLOCK
    expect = set()
    for b in blocks:
        expect |= set(range(b * BLOCK, b * BLOCK + BLOCK))
    assert set(torch.nonzero(m).flatten().tolist()) == expect


def test_the_cache_length_never_changes():
    """A repair replaces; it does not append. A longer cache would mean the
    query attends to duplicated copies of repaired tokens."""
    mapped, native = _cache(0), _cache(100)
    out = repair_cache(mapped, native, mask_from_blocks(T, [0, 1, 2]))
    for a, b in zip(mapped, out):
        assert a.shape[2] == b.shape[2] == T


def test_a_budget_is_never_exceeded_by_rounding():
    for frac in (0.10, 0.25, 0.333):
        k = blocks_for_budget(T, frac)
        assert k <= n_blocks(T) * frac + 1e-9
        assert (
            int(mask_from_blocks(T, uniform_blocks(T, k)).sum())
            <= int(T * frac) + BLOCK
        )


def test_uniform_selection_is_deterministic_and_spread():
    a = uniform_blocks(T, 4)
    b = uniform_blocks(T, 4)
    assert a == b
    assert len(a) == 4
    assert a[0] == 0 and a[-1] < n_blocks(T)
    # Spread, not clustered at one end.
    assert a[-1] - a[0] > n_blocks(T) // 2


def test_attention_ranking_picks_the_attended_blocks_and_breaks_ties_by_index():
    attn = torch.zeros(T)
    attn[BLOCK * 7 : BLOCK * 8] = 1.0  # block 7 clearly attended
    attn[BLOCK * 2 : BLOCK * 3] = 2.0  # block 2 more so
    got = rank_blocks_by_attention(attn, T, 2)
    assert got == [2, 7]
    # All-equal attention must fall back to the lowest indices, reproducibly.
    flat = torch.ones(T)
    assert rank_blocks_by_attention(flat, T, 3) == [0, 1, 2]
    assert rank_blocks_by_attention(flat, T, 3) == rank_blocks_by_attention(flat, T, 3)


def test_a_ranked_selector_never_exceeds_its_budget():
    attn = torch.rand(T, generator=torch.Generator().manual_seed(0))
    for k in (1, 4, 9):
        sel = rank_blocks_by_attention(attn, T, k)
        assert len(sel) == k
        assert int(mask_from_blocks(T, sel).sum()) == k * BLOCK


def test_mismatched_shapes_and_dtypes_are_refused():
    mapped, native = _cache(0), _cache(100)
    native[1] = native[1].to(torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        repair_cache(mapped, native, torch.ones(T, dtype=torch.bool))
    mapped2, native2 = _cache(0), _cache(100)
    native2[2] = native2[2][:, :, : T - 1, :]
    with pytest.raises(ValueError, match="shape"):
        repair_cache(mapped2, native2, torch.ones(T, dtype=torch.bool))


def test_repair_does_not_mutate_its_inputs():
    """The unrepaired arm must still be the unrepaired arm afterwards."""
    mapped, native = _cache(0), _cache(100)
    snapshot = [t.clone() for t in mapped]
    repair_cache(mapped, native, mask_from_blocks(T, [0, 3]))
    for a, b in zip(snapshot, mapped):
        assert torch.equal(a, b)
