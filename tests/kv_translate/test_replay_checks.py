# SPDX-License-Identifier: GPL-2.0
"""Each check, run against an implementation that is wrong in the way it targets.

A qualification is only worth the failures it can produce. Every check here is
shown rejecting a specific defect, and the defects are the ones the previous
qualification would have passed: a replay that leaks its input cache, one that
writes keys and forgets values, one that writes some layers, and a copying mode
that disagrees with the in-place mode it was supposed to stand in for.
"""

from __future__ import annotations

import pytest
import torch

from research.kv_translate.replay_checks import (
    check_initial_cache_independence,
    check_layer0_exact,
    check_native_repeat,
    check_ownership_modes,
    check_partial_mask_per_layer,
    error_quantiles,
    per_layer_profile,
    rel_frobenius,
)

L, H, T, D = 4, 2, 32, 8


def cache(seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    k = [torch.randn(1, H, T, D, generator=g) * scale for _ in range(L)]
    v = [torch.randn(1, H, T, D, generator=g) * scale for _ in range(L)]
    return k, v


# -- layer zero -------------------------------------------------------------


def test_layer0_check_reports_keys_and_values_separately():
    nk, nv = cache(0)
    rk, rv = [t.clone() for t in nk], [t.clone() for t in nv]
    r = check_layer0_exact(rk, rv, nk, nv)
    assert r["keys_exact"] and r["values_exact"]


def test_layer0_check_catches_a_replay_that_writes_keys_but_not_values():
    """The earlier check compared keys only and would have passed this."""
    nk, nv = cache(0)
    rk = [t.clone() for t in nk]
    rv = [t.clone() for t in nv]
    rv[0] = rv[0] + 1.0
    r = check_layer0_exact(rk, rv, nk, nv)
    assert r["keys_exact"] is True
    assert r["values_exact"] is False


def test_layer0_check_says_what_it_cannot_see():
    nk, nv = cache(0)
    r = check_layer0_exact([t.clone() for t in nk], [t.clone() for t in nv], nk, nv)
    assert "cannot detect a wrong attention mask" in r["scope"]


# -- independence from the starting cache -----------------------------------


def test_independence_passes_when_the_replay_really_recomputes():
    """A correct full replay ignores what it started from."""
    target_k, target_v = cache(99)

    def replay(_initial):
        return [t.clone() for t in target_k], [t.clone() for t in target_v]

    r = check_initial_cache_independence(replay, cache(1), cache(2), cache(3), tol=1e-6)
    assert r["passed"] and r["worst"] == 0.0


def test_independence_catches_a_replay_that_leaks_its_input():
    """The defect no agreement check can see: some of the input survives."""
    target_k, target_v = cache(99)

    def leaky(initial):
        ik, iv = initial
        k = [0.5 * a + 0.5 * b for a, b in zip(target_k, ik)]
        v = [0.5 * a + 0.5 * b for a, b in zip(target_v, iv)]
        return k, v

    r = check_initial_cache_independence(leaky, cache(1), cache(2), cache(3), tol=1e-6)
    assert r["passed"] is False
    assert r["worst"] > 0.1


def test_independence_catches_a_replay_that_skips_a_layer():
    target_k, target_v = cache(99)

    def skips_last(initial):
        ik, iv = initial
        k = [t.clone() for t in target_k]
        v = [t.clone() for t in target_v]
        k[-1], v[-1] = ik[-1].clone(), iv[-1].clone()  # never written
        return k, v

    r = check_initial_cache_independence(
        skips_last, cache(1), cache(2), cache(3), tol=1e-6
    )
    assert r["passed"] is False


# -- partial masks ----------------------------------------------------------


def make_mask(idx):
    m = torch.zeros(T, dtype=torch.bool)
    m[idx] = True
    return m


def test_partial_mask_check_passes_a_correct_partial_repair():
    bk, bv = cache(0)
    ak = [t.clone() for t in bk]
    av = [t.clone() for t in bv]
    m = make_mask([3, 4, 5])
    for li in range(L):
        ak[li][:, :, m, :] += 1.0
        av[li][:, :, m, :] += 1.0
    r = check_partial_mask_per_layer(bk, bv, ak, av, m)
    assert r["all_unselected_preserved"] and r["all_selected_written"]


def test_partial_mask_check_catches_values_left_unwritten():
    """Keys repaired, values not -- invisible to a keys-only check."""
    bk, bv = cache(0)
    ak = [t.clone() for t in bk]
    av = [t.clone() for t in bv]
    m = make_mask([3, 4, 5])
    for li in range(L):
        ak[li][:, :, m, :] += 1.0
    r = check_partial_mask_per_layer(bk, bv, ak, av, m)
    assert r["all_unselected_preserved"] is True
    assert r["all_selected_written"] is False


def test_partial_mask_check_catches_one_layer_left_unwritten():
    """An or across layers hides this; a per-layer report does not."""
    bk, bv = cache(0)
    ak = [t.clone() for t in bk]
    av = [t.clone() for t in bv]
    m = make_mask([7])
    for li in range(L - 1):
        ak[li][:, :, m, :] += 1.0
        av[li][:, :, m, :] += 1.0
    r = check_partial_mask_per_layer(bk, bv, ak, av, m)
    assert r["all_selected_written"] is False
    assert r["per_layer"][-1]["selected_keys_changed"] is False


def test_partial_mask_check_catches_a_scatter_that_overruns():
    bk, bv = cache(0)
    ak = [t.clone() for t in bk]
    av = [t.clone() for t in bv]
    m = make_mask([3])
    for li in range(L):
        ak[li][:, :, 3:5, :] += 1.0  # writes one position too many
        av[li][:, :, 3:5, :] += 1.0
    r = check_partial_mask_per_layer(bk, bv, ak, av, m)
    assert r["all_unselected_preserved"] is False


# -- ownership --------------------------------------------------------------


def test_ownership_check_passes_when_the_two_modes_agree():
    k, v = cache(0)
    out_k, out_v = cache(5)

    def clone_mode():
        return [t.clone() for t in out_k], [t.clone() for t in out_v]

    def in_place_mode():
        return [t.clone() for t in out_k], [t.clone() for t in out_v]

    r = check_ownership_modes(clone_mode, in_place_mode, k, v)
    assert r["passed"]


def test_ownership_check_catches_the_two_modes_disagreeing():
    """The gap that left every timing figure produced by unqualified code."""
    k, v = cache(0)
    out_k, out_v = cache(5)

    def clone_mode():
        return [t.clone() for t in out_k], [t.clone() for t in out_v]

    def in_place_mode():
        return [t + 0.01 for t in out_k], [t.clone() for t in out_v]

    r = check_ownership_modes(clone_mode, in_place_mode, k, v)
    assert r["passed"] is False
    assert r["clone_and_in_place_keys_identical"] is False


def test_ownership_check_catches_a_clone_mode_that_mutates_its_input():
    k, v = cache(0)
    out_k, out_v = cache(5)

    def bad_clone():
        k[0] += 1.0  # the scoring loop depends on this not happening
        return [t.clone() for t in out_k], [t.clone() for t in out_v]

    r = check_ownership_modes(
        bad_clone,
        lambda: ([t.clone() for t in out_k], [t.clone() for t in out_v]),
        k,
        v,
    )
    assert r["clone_left_inputs_unchanged"] is False
    assert r["passed"] is False


# -- native repeat ----------------------------------------------------------


def test_native_repeat_detects_a_deterministic_prefill():
    k, v = cache(0)
    r = check_native_repeat(lambda: ([t.clone() for t in k], [t.clone() for t in v]))
    assert r["bitwise_identical"] is True


def test_native_repeat_detects_a_nondeterministic_prefill():
    """If this fires, it is the floor under every agreement figure."""
    state = {"n": 0}

    def wobbly():
        state["n"] += 1
        k, v = cache(0)
        return [t + state["n"] * 1e-6 for t in k], v

    r = check_native_repeat(wobbly)
    assert r["bitwise_identical"] is False
    assert r["keys_rel_frobenius"] > 0


# -- diagnostics ------------------------------------------------------------


def test_the_per_layer_profile_records_every_layer():
    nk, nv = cache(0)
    rk = [t + 1e-3 * (i + 1) for i, t in enumerate(nk)]
    rv = [t + 1e-3 * (i + 1) for i, t in enumerate(nv)]
    prof = per_layer_profile(rk, rv, nk, nv)
    assert len(prof) == L
    assert [p["layer"] for p in prof] == list(range(L))
    errs = [p["keys_rel_frobenius"] for p in prof]
    assert errs == sorted(errs), "a growing error should read as growing"


def test_quantiles_show_the_bulk_not_the_tail():
    """The statistic the earlier report quoted from a comment."""
    nk, _ = cache(0)
    rk = [t.clone() for t in nk]
    flat = rk[0].reshape(-1)
    flat[0] += 10.0  # one outlier
    q = error_quantiles(rk[0], nk[0])
    assert q["q500"] == 0.0
    assert q["q1000"] > 0.0, "the maximum should see the outlier"
    assert q["n_total"] == nk[0].numel()


def test_relative_frobenius_is_not_dominated_by_one_element():
    a, _ = cache(0)
    b = [t.clone() for t in a]
    b[0].reshape(-1)[0] += 5.0
    assert rel_frobenius(b[0], a[0]) < 0.35
