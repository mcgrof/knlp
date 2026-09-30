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

from research.kv_translate.blend_select import (
    qwen3_layer0_keys,
    rank_tokens_by_key_deviation,
    select_qwen3_layer0_key_deviation,
)
from research.kv_translate.kri_cost import measure_kri_d_kv_blocks
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


@pytest.fixture(scope="module")
def tiny_qwen3():
    """The receiving architecture in the 14B -> 32B recovery run.

    Unlike Qwen2, Qwen3 normalises projected queries and keys before RoPE and
    may use a head width that is not hidden_size / attention_heads.  A replay
    that reconstructs either detail outside the model can pass the Qwen2
    fixture and still write the wrong cache for the actual receiver.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(1)
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        rope_theta=1_000_000.0,
        tie_word_embeddings=False,
    )
    m = Qwen3ForCausalLM(cfg).eval().to(torch.float32)
    T = 128
    ids = torch.randint(0, 200, (1, T))
    return m, ids, T


def _native(model, ids):
    from transformers import DynamicCache

    c = DynamicCache()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=c, use_cache=True)
    return (
        [layer.keys for layer in c.layers],
        [layer.values for layer in c.layers],
    )


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


def test_qwen3_full_replay_includes_qk_norm_and_rope(tiny_qwen3):
    """The real receiver's attention path, not a hand-built approximation.

    Calling the model for replay makes Qwen3 apply q_norm, k_norm and RoPE in
    their native order.  Full replay must therefore reproduce its native cache
    in both components and every layer.
    """
    model, ids, T = tiny_qwen3
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    rk, rv, info = replay_selected(model, ids, torch.ones(T, dtype=torch.bool), ak, av)
    assert info == {"replayed": T, "layers_written": [0, 1, 2]}
    for li in range(len(nk)):
        assert torch.allclose(rk[li], nk[li], atol=TOL), f"keys layer {li}"
        assert torch.allclose(rv[li], nv[li], atol=TOL), f"values layer {li}"


def test_qwen3_sparse_replay_uses_absolute_cache_slots(tiny_qwen3):
    """Position ids, cache positions and scatter slots name the same tokens."""
    model, ids, T = tiny_qwen3
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    mask = mask_from_blocks(T, [1, 5])
    slots = torch.nonzero(mask, as_tuple=False).flatten()
    seen = {}
    orig = model.forward

    def spy(*args, **kwargs):
        seen["position_ids"] = kwargs["position_ids"].detach().clone()
        seen["cache_position"] = kwargs["cache_position"].detach().clone()
        return orig(*args, **kwargs)

    model.forward = spy
    try:
        rk, rv, info = replay_selected(model, ids, mask, ak, av)
    finally:
        model.forward = orig

    assert torch.equal(seen["position_ids"], slots.reshape(1, -1))
    assert torch.equal(seen["cache_position"], slots)
    assert info["replayed"] == int(mask.sum())
    keep = ~mask
    for li in range(len(nk)):
        assert torch.equal(rk[li][:, :, keep], ak[li][:, :, keep])
        assert torch.equal(rv[li][:, :, keep], av[li][:, :, keep])
    # Layer zero reads no cache: exact equality isolates Q/K normalisation,
    # RoPE, token order and the destination slots from mixed-cache effects.
    assert torch.equal(rk[0][:, :, mask], nk[0][:, :, mask])
    assert torch.equal(rv[0][:, :, mask], nv[0][:, :, mask])


def test_qwen3_zero_replay_bypasses_the_model(tiny_qwen3):
    model, ids, T = tiny_qwen3
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    called = False
    orig = model.forward

    def spy(*args, **kwargs):
        nonlocal called
        called = True
        return orig(*args, **kwargs)

    model.forward = spy
    try:
        rk, rv, info = replay_selected(
            model, ids, torch.zeros(T, dtype=torch.bool), ak, av
        )
    finally:
        model.forward = orig
    assert not called
    assert rk is ak and rv is av
    assert info == {"replayed": 0, "layers_written": []}


def test_qwen3_selector_keys_match_the_model_cache(tiny_qwen3):
    """The selector applies Qwen3 K normalisation and RoPE exactly once."""
    model, ids, _ = tiny_qwen3
    nk, _ = _native(model, ids)
    fresh = qwen3_layer0_keys(model, ids)
    assert torch.equal(fresh, nk[0])


def test_key_deviation_ranks_tokens_not_heads(tiny_qwen3):
    """A planted change identifies one token regardless of its head."""
    model, ids, T = tiny_qwen3
    nk, _ = _native(model, ids)
    mapped = nk[0].clone()
    mapped[:, 1, 73, 4] += 10
    selected, scores = rank_tokens_by_key_deviation(nk[0], mapped, 1)
    assert selected.tolist() == [73]
    assert scores.shape == (T,)
    assert int(scores.argmax()) == 73


def test_qwen3_deviation_selection_replays_exact_budget(tiny_qwen3):
    model, ids, T = tiny_qwen3
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    count = T // 4
    selected, scores, fresh = select_qwen3_layer0_key_deviation(
        model, ids, ak[0], count
    )
    assert selected.numel() == count
    assert selected.tolist() == sorted(selected.tolist())
    assert scores.shape == (T,)
    assert torch.equal(fresh, nk[0])

    mask = torch.zeros(T, dtype=torch.bool)
    mask[selected.cpu()] = True
    rk, rv, info = replay_selected(model, ids, mask, ak, av)
    assert info["replayed"] == count
    assert torch.equal(rk[0][:, :, mask], nk[0][:, :, mask])
    assert torch.equal(rv[0][:, :, mask], nv[0][:, :, mask])
    assert torch.equal(rk[0][:, :, ~mask], ak[0][:, :, ~mask])
    assert torch.equal(rv[0][:, :, ~mask], av[0][:, :, ~mask])


def test_kri_cost_probe_ablates_the_token_axis_without_mutating_input(tiny_qwen3):
    model, ids, T = tiny_qwen3
    nk, nv = _native(model, ids)
    before_k = [tensor.clone() for tensor in nk]
    before_v = [tensor.clone() for tensor in nv]
    probe = torch.tensor([[3, 7, 11]])
    result = measure_kri_d_kv_blocks(
        model,
        nk,
        nv,
        probe,
        prefix_tokens=T,
        block_size=BLOCK,
        blocks=[0, T // BLOCK - 1],
    )
    assert result["total_blocks"] == T // BLOCK
    assert result["sampled_blocks"] == [0, T // BLOCK - 1]
    assert result["reference_seconds"] > 0
    assert len(result["samples"]) == 2
    for sample in result["samples"]:
        assert sample["seconds"] > 0
        assert sample["hidden_l2_sum"] >= 0
        assert sample["retained_cache_tokens"] == T + probe.shape[1] - BLOCK - 1
    for before, after in zip(before_k, nk):
        assert torch.equal(before, after)
    for before, after in zip(before_v, nv):
        assert torch.equal(before, after)


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


def test_asking_for_one_logit_does_not_change_the_repaired_cache(tiny):
    """The repair wants keys and values; the vocabulary projection is waste.

    Dropping it is only an optimization if the cache it leaves is the same
    cache. Checked bit-for-bit rather than approximately: the change removes
    work after the layers, so anything but equality would mean it removed
    something else too.
    """
    import research.kv_translate.replay as R

    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    m = mask_from_blocks(T, [1, 4])

    seen = {}
    orig = model.forward

    def spy(*a, **kw):
        seen["logits_to_keep"] = kw.get("logits_to_keep")
        return orig(*a, **kw)

    model.forward = spy
    try:
        k1, v1, _ = R.replay_selected(model, ids, m, ak, av)
    finally:
        model.forward = orig
    assert seen["logits_to_keep"] == 1, "replay still asks for every position"

    # and the result is what the unrestricted call produces
    src = R.replay_selected.__doc__
    assert src is not None
    k2, v2, _ = _replay_without_logit_limit(model, ids, m, ak, av)
    for a, b in zip(k1, k2):
        assert torch.equal(a, b), "restricting the logits changed the keys"
    for a, b in zip(v1, v2):
        assert torch.equal(a, b), "restricting the logits changed the values"


def _replay_without_logit_limit(model, input_ids, mask, keys, values):
    """The previous behaviour, kept only so the change can be compared to it."""
    from research.kv_translate.replay import ScatterCache, absolute_causal_mask

    device = keys[0].device
    T = keys[0].shape[2]
    slots = torch.nonzero(mask, as_tuple=False).flatten().to(device)
    cache = ScatterCache(keys, values, slots)
    with torch.no_grad():
        model(
            input_ids=input_ids[0, slots].reshape(1, -1).to(device),
            attention_mask=absolute_causal_mask(slots, T, keys[0].dtype, device),
            position_ids=slots.reshape(1, -1),
            past_key_values=cache,
            use_cache=True,
        )
    return cache.keys, cache.values, None


def test_repairing_in_place_gives_the_same_cache_as_repairing_a_copy(tiny):
    """The copy is a safety property for one caller, not part of the result."""
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    m = mask_from_blocks(T, [2, 5])

    a_k, a_v = _approx((nk, nv))
    copied_k, copied_v, _ = replay_selected(model, ids, m, a_k, a_v)

    b_k, b_v = _approx((nk, nv))
    before = [t.clone() for t in b_k]
    inplace_k, inplace_v, _ = replay_selected(model, ids, m, b_k, b_v, in_place=True)
    for x, y in zip(copied_k, inplace_k):
        assert torch.equal(x, y)
    for x, y in zip(copied_v, inplace_v):
        assert torch.equal(x, y)
    # and in-place really did write through, which is the point of the flag
    assert any(not torch.equal(x, y) for x, y in zip(before, b_k))


def test_the_default_still_leaves_the_caller_s_cache_untouched(tiny):
    """Scoring reuses one mapped cache across conditions and depends on this."""
    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    before_k = [t.clone() for t in ak]
    before_v = [t.clone() for t in av]
    replay_selected(model, ids, mask_from_blocks(T, [0, 3]), ak, av)
    for x, y in zip(before_k, ak):
        assert torch.equal(x, y), "the default path mutated its input"
    for x, y in zip(before_v, av):
        assert torch.equal(x, y)


def test_the_qualification_checks_compose_against_a_real_replay(tiny):
    """The RQ contract's checks, run together through an actual model.

    The unit tests show each check rejecting its own defect on constructed
    tensors. This shows them all passing on a replay that is genuinely
    correct, which is the other half: a suite that only ever fails is as
    useless as one that only ever passes.

    The independence control is the one to watch. A full replay recomputes
    every position, so the cache it started from must leave no trace, and here
    the translated cache, a cache of zeros and a deliberately wrong cache all
    produce the same result exactly.
    """
    from research.kv_translate.replay_checks import (
        check_initial_cache_independence,
        check_layer0_exact,
        check_native_repeat,
        check_ownership_modes,
        check_partial_mask_per_layer,
        per_layer_profile,
    )

    model, ids, T = tiny
    nk, nv = _native(model, ids)
    ak, av = _approx((nk, nv))
    full = torch.ones(T, dtype=torch.bool)

    assert check_native_repeat(lambda: _native(model, ids))["bitwise_identical"]

    def replay_from(initial):
        k, v = initial
        rk, rv, _ = replay_selected(
            model, ids, full, [t.clone() for t in k], [t.clone() for t in v]
        )
        return rk, rv

    rk, rv = replay_from((ak, av))
    l0 = check_layer0_exact(rk, rv, nk, nv)
    assert l0["keys_exact"] and l0["values_exact"]

    g = torch.Generator().manual_seed(7)
    ind = check_initial_cache_independence(
        replay_from,
        (ak, av),
        ([torch.zeros_like(t) for t in ak], [torch.zeros_like(t) for t in av]),
        (
            [torch.randn(t.shape, generator=g) for t in ak],
            [torch.randn(t.shape, generator=g) for t in av],
        ),
        tol=1e-6,
    )
    assert ind["passed"], ind
    assert ind["worst"] == 0.0, "a full replay should leave nothing of its input"

    mask = mask_from_blocks(T, [1, 4])
    pk, pv, _ = replay_selected(model, ids, mask, ak, av)
    pm = check_partial_mask_per_layer(ak, av, pk, pv, mask)
    assert pm["all_unselected_preserved"] and pm["all_selected_written"]

    own = check_ownership_modes(
        lambda: replay_selected(model, ids, mask, ak, av)[:2],
        lambda: replay_selected(
            model,
            ids,
            mask,
            [t.clone() for t in ak],
            [t.clone() for t in av],
            in_place=True,
        )[:2],
        ak,
        av,
    )
    assert own["passed"], own

    prof = per_layer_profile(rk, rv, nk, nv)
    assert len(prof) == len(nk)
    assert all("values_rel_frobenius" in p for p in prof)
