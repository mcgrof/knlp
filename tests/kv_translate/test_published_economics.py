# SPDX-License-Identifier: GPL-2.0
"""The byte arithmetic of sharing one cache, checked against known sizes."""

from __future__ import annotations

import pytest

from research.kv_translate.published import economics as eco

# Qwen3-14B and Qwen3-32B: 40 and 64 layers, 8 key-value heads of width 128
SRC = eco.cache_bytes_per_token(40, 8, 128)
TGT = eco.cache_bytes_per_token(64, 8, 128)
SMALL_MAPPER = 537_397_592
LARGE_MAPPER = 4_295_493_984


def test_a_token_costs_keys_and_values_in_every_layer():
    assert SRC == 40 * 2 * 8 * 128 * 2 == 163_840
    assert TGT == 262_144
    assert eco.cache_bytes_per_token(1, 1, 1, bytes_per_value=4) == 8
    assert (
        eco.geometry_bytes_per_token({"layers": 40, "kv_heads": 8, "head_dim": 128})
        == SRC
    )


def test_nonsense_geometry_is_refused():
    with pytest.raises(ValueError, match="layers"):
        eco.cache_bytes_per_token(0, 8, 128)
    with pytest.raises(ValueError, match="negative"):
        eco.stored_bytes(-1, SRC, TGT, 0)


def test_both_caches_of_one_long_prefix_match_the_known_size():
    got = eco.stored_bytes(16_384, SRC, TGT, SMALL_MAPPER)
    assert got["both_caches"] == 6_979_321_856
    assert got["shared_source"] == 16_384 * SRC + SMALL_MAPPER
    assert got["saving_fraction"] == pytest.approx(0.5384, abs=1e-4)


def test_an_empty_store_has_no_saving_to_report():
    assert eco.stored_bytes(0, SRC, TGT, SMALL_MAPPER)["saving_fraction"] is None


def test_the_saving_approaches_the_share_of_the_cache_dropped():
    limit = eco.saving_limit(SRC, TGT)
    assert limit == pytest.approx(64 / 104)
    big = eco.stored_bytes(10**9, SRC, TGT, LARGE_MAPPER)["saving_fraction"]
    assert limit - 1e-4 < big < limit


def test_break_even_is_the_first_store_that_meets_the_requirement():
    for mapper in (SMALL_MAPPER, LARGE_MAPPER):
        n = eco.break_even_tokens(0.25, SRC, TGT, mapper)
        assert eco.stored_bytes(n, SRC, TGT, mapper)["saving_fraction"] >= 0.25
        assert eco.stored_bytes(n - 1, SRC, TGT, mapper)["saving_fraction"] < 0.25
    assert eco.break_even_tokens(0.25, SRC, TGT, SMALL_MAPPER) == 3_453
    assert eco.break_even_tokens(0.25, SRC, TGT, LARGE_MAPPER) == 27_598
    assert eco.break_even_tokens(0.25, SRC, TGT, 0) == 0


def test_a_saving_beyond_the_limit_has_no_break_even():
    assert eco.break_even_tokens(0.62, SRC, TGT, SMALL_MAPPER) is None
    assert eco.break_even_tokens(0.61, SRC, TGT, SMALL_MAPPER) is not None
    with pytest.raises(ValueError, match="not a fraction"):
        eco.break_even_tokens(1.0, SRC, TGT, SMALL_MAPPER)
