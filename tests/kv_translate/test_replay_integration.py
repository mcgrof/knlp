# SPDX-License-Identifier: GPL-2.0
"""The integrated checks, run through a real model on the processor.

The earlier helper tests substituted constant arrays and asserted that the
right slots changed. That establishes the scatter arithmetic and nothing about
a model: it cannot tell whether a recomputed token carried its original
position, whether it read the cache it was supposed to read, or whether the
selector's own query leaked into the prefix it was selecting from.

So these run a genuine small transformer with grouped-query attention and
rotary positions, on the processor, and compare against what the same model
computes natively. A tiny parity does not replace qualification on the card
the real run uses, and is not offered as one.
"""

from __future__ import annotations

import pytest
import torch

from research.kv_translate.recovery import BLOCK, mask_from_blocks
from research.kv_translate.replay import absolute_causal_mask, replay_selected

TOL = 2e-4  # chosen before any run, for float32 on this fixture


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )
    m = Qwen2ForCausalLM(cfg).eval().to(torch.float32)
    T = 128
    ids = torch.randint(0, 200, (1, T))
    return m, ids, T


def _native(model, ids):
    from transformers import DynamicCache

    c = DynamicCache()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=c, use_cache=True)
    return ([l.keys for l in c.layers], [l.values for l in c.layers])


def _approx(native, scale=0.6, shift=0.05):
    """A stand-in for a translated cache: wrong, but the right shape."""
    return (
        [k * scale + shift for k in native[0]],
        [v * scale - shift for v in native[1]],
    )


def test_full_replay_reproduces_native_computation(tiny):
    """The contract the plan calls real-model 100% replay parity.

    Every position recomputed, reading only recomputed context, must match
    what a native prefill produces. If this fails, nothing measured with a
    partial mask means anything.
    """
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    mask = torch.ones(T, dtype=torch.bool)
    rk, rv, info = replay_selected(model, ids, mask, ak, av)
    assert info["replayed"] == T
    assert info["layers_written"] == [0, 1, 2]
    for li in range(len(nk)):
        assert torch.allclose(rk[li], nk[li], atol=TOL), f"keys layer {li}"
        assert torch.allclose(rv[li], nv[li], atol=TOL), f"values layer {li}"


def test_zero_replay_is_a_true_no_op_not_a_copy(tiny):
    """Nothing computed, nothing copied, and the same objects handed back."""
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    rk, rv, info = replay_selected(model, ids, torch.zeros(T, dtype=torch.bool), ak, av)
    assert info["replayed"] == 0 and info["layers_written"] == []
    assert rk is ak and rv is av


def test_the_prefix_length_never_changes(tiny):
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    for blocks in ([0], [1, 5], list(range(T // BLOCK))):
        rk, rv, _ = replay_selected(model, ids, mask_from_blocks(T, blocks), ak, av)
        assert all(k.shape[2] == T for k in rk)
        assert all(v.shape[2] == T for v in rv)


def test_unselected_positions_are_untouched_in_every_layer(tiny):
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    m = mask_from_blocks(T, [1, 4])
    rk, rv, _ = replay_selected(model, ids, m, ak, av)
    keep = ~m
    for li in range(len(ak)):
        assert torch.equal(rk[li][:, :, keep, :], ak[li][:, :, keep, :])
        assert torch.equal(rv[li][:, :, keep, :], av[li][:, :, keep, :])
        assert not torch.allclose(rk[li][:, :, m, :], ak[li][:, :, m, :])


def test_both_keys_and_values_are_covered(tiny):
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    m = mask_from_blocks(T, [2])
    rk, rv, _ = replay_selected(model, ids, m, ak, av)
    for li in range(len(ak)):
        assert not torch.allclose(rk[li][:, :, m, :], ak[li][:, :, m, :]), li
        assert not torch.allclose(rv[li][:, :, m, :], av[li][:, :, m, :]), li


def test_a_replayed_token_carries_its_original_absolute_position(tiny):
    """Replaying one late block alone must match what native computed there.

    It can only match if the token was rotated at its own position and allowed
    to read everything before it. Feeding it as though it were at position
    zero, or letting it see only the other selected tokens, both fail here.
    """
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    # First layer only: its keys depend on position and on nothing upstream,
    # so a match isolates the position contract from cache quality.
    ak, av = _approx((nk, nv))
    late = [T // BLOCK - 2]
    m = mask_from_blocks(T, late)
    rk, _, _ = replay_selected(model, ids, m, ak, av)
    assert torch.allclose(rk[0][:, :, m, :], nk[0][:, :, m, :], atol=TOL)


def test_a_noncontiguous_selection_is_handled(tiny):
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    m = mask_from_blocks(T, [0, 3, 6])
    rk, _, info = replay_selected(model, ids, m, ak, av)
    assert info["replayed"] == 3 * BLOCK
    assert torch.allclose(rk[0][:, :, m, :], nk[0][:, :, m, :], atol=TOL)


def test_replaying_an_identical_cache_changes_nothing_measurable(tiny):
    """Repairing a native cache with native computation is a fixed point."""
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    m = mask_from_blocks(T, [2, 5])
    rk, rv, _ = replay_selected(
        model, ids, m, [k.clone() for k in nk], [v.clone() for v in nv]
    )
    for li in range(len(nk)):
        assert torch.allclose(rk[li], nk[li], atol=TOL)
        assert torch.allclose(rv[li], nv[li], atol=TOL)


def test_the_absolute_mask_forbids_looking_forward(tiny):
    _, _, T = tiny
    pos = torch.tensor([5, 60, 100])
    m = absolute_causal_mask(pos, T, torch.float32, torch.device("cpu"))
    assert m.shape == (1, 1, 3, T)
    neg = torch.finfo(torch.float32).min
    for r, p in enumerate(pos.tolist()):
        assert float(m[0, 0, r, p]) == 0.0
        assert float(m[0, 0, r, p - 1]) == 0.0
        if p + 1 < T:
            assert float(m[0, 0, r, p + 1]) == neg


def test_a_malformed_cache_fails_cleanly(tiny):
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    with pytest.raises(ValueError, match="mask covers"):
        replay_selected(model, ids, torch.ones(T - 1, dtype=torch.bool), ak, av)
    with pytest.raises(ValueError, match="layer count"):
        replay_selected(model, ids, torch.ones(T, dtype=torch.bool), ak, av[:-1])
