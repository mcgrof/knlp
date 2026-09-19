# SPDX-License-Identifier: GPL-2.0
"""Byte accounting that cannot flatter a method by accident.

Each test pins one way a compression claim gets inflated: charging the baseline
for a full final block, forgetting that quantisation scales are stored, mixing
transfer bytes with resident bytes, or omitting the mapper from the ledger it
has to be amortised against.
"""

import math

import pytest

from research.jacobian_kv.accounting import (
    BF16,
    FP8,
    INT8_ASYM,
    KVGeometry,
    QuantSpec,
    blocked_allocation_tokens,
    blocked_token_count,
    break_even_prefixes,
    cache_bytes,
    mapper_bytes,
    savings_fraction,
)

GEOM = KVGeometry(n_layers=28, n_kv_heads=8, head_dim=128, n_tokens=1000)


def test_bf16_cache_bytes_match_the_hand_calculation():
    r = cache_bytes(GEOM, BF16, BF16, label="bf16")
    expect = 28 * 8 * 128 * 1000 * 2
    assert r.k_bytes == pytest.approx(expect)
    assert r.total == pytest.approx(2 * expect)


def test_quantisation_scales_are_charged():
    """FP8 at group 128 is more than half of bf16, and the gap is the scales."""
    base = cache_bytes(GEOM, BF16, BF16)
    q = cache_bytes(GEOM, FP8, FP8)
    assert q.total > base.total * 0.5
    n = GEOM.elements
    expected_overhead = (n / 128) * 16 / 8
    assert q.k_bytes - n == pytest.approx(expected_overhead)


def test_asymmetric_scheme_charges_the_zero_point_too():
    sym = cache_bytes(GEOM, FP8, FP8)
    asym = cache_bytes(GEOM, INT8_ASYM, INT8_ASYM)
    assert asym.total > sym.total


def test_smaller_groups_cost_more_metadata():
    coarse = QuantSpec(bits=4.0, group_size=256)
    fine = QuantSpec(bits=4.0, group_size=32)
    assert cache_bytes(GEOM, fine, fine).total > cache_bytes(GEOM, coarse, coarse).total


def test_a_four_bit_format_with_tiny_groups_can_lose_to_eight_bit():
    """The failure mode that makes unaccounted metadata dangerous."""
    absurd = QuantSpec(bits=4.0, group_size=4, scale_bits=16.0, zero_point_bits=8.0)
    assert cache_bytes(GEOM, absurd, absurd).total > cache_bytes(GEOM, FP8, FP8).total


def test_partial_final_block_is_not_charged_as_a_whole_block():
    """1000 tokens in blocks of 256 occupy 1024 slots but hold 1000 tokens."""
    assert blocked_token_count(1000, 256) == 1000
    assert blocked_allocation_tokens(1000, 256) == 1024
    assert blocked_allocation_tokens(1024, 256) == 1024
    assert blocked_allocation_tokens(1, 256) == 256
    assert blocked_allocation_tokens(1000, 0) == 1000


def test_asymmetric_k16_v8_sits_between_bf16_and_all_fp8():
    """The deployed incumbent every compression claim is measured against."""
    bf = cache_bytes(GEOM, BF16, BF16, label="bf16")
    k16v8 = cache_bytes(GEOM, BF16, FP8, label="K16/V8")
    allfp8 = cache_bytes(GEOM, FP8, FP8, label="fp8")
    assert allfp8.total < k16v8.total < bf.total
    assert savings_fraction(bf, k16v8) == pytest.approx(0.2461, abs=1e-3)


def test_savings_fraction_is_negative_when_a_method_grows_the_cache():
    bf = cache_bytes(GEOM, BF16, BF16)
    worse = cache_bytes(GEOM, BF16, BF16, metadata_bytes=1e6)
    assert savings_fraction(bf, worse) < 0


def test_wire_and_resident_bytes_are_kept_apart():
    """A receiver that reconstructs full K/V saved transfer, not residency."""
    wire = cache_bytes(GEOM, FP8, FP8, label="transfer", resident=False)
    resident = cache_bytes(GEOM, BF16, BF16, label="resident", resident=True)
    assert wire.to_dict()["measures"] == "wire"
    assert resident.to_dict()["measures"] == "resident"
    assert wire.to_dict()["measures"] != resident.to_dict()["measures"]


def test_break_even_is_infinite_when_a_prefix_saves_nothing():
    assert break_even_prefixes(1e6, 0.0) == math.inf
    assert break_even_prefixes(1e6, -5.0) == math.inf


def test_break_even_counts_the_mapper_against_the_saving():
    """A per-head affine map between two real geometries is not small."""
    mb = mapper_bytes(n_maps=24 * 2, in_dim=64, out_dim=128, dtype_bits=16.0)
    bf = cache_bytes(GEOM, BF16, BF16)
    half = cache_bytes(GEOM, FP8, FP8)
    saved = bf.total - half.total
    n = break_even_prefixes(mb, saved)
    assert 0 < n < 1.0, "a 1000-token prefix already outweighs this mapper"

    tiny = KVGeometry(n_layers=28, n_kv_heads=8, head_dim=128, n_tokens=4)
    saved_tiny = cache_bytes(tiny, BF16, BF16).total - cache_bytes(tiny, FP8, FP8).total
    assert break_even_prefixes(mb, saved_tiny) > 1.0


def test_geometry_element_count_is_the_obvious_product():
    g = KVGeometry(n_layers=2, n_kv_heads=3, head_dim=4, n_tokens=5, batch=6)
    assert g.elements == 2 * 3 * 4 * 5 * 6
