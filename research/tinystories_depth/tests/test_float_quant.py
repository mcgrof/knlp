# SPDX-License-Identifier: MIT
"""Check exact floating-format levels, ties and independent row scales."""

import os

import torch

from research.tinystories_depth.float_quant import scaled_float
from research.tinystories_depth.kv_cache import fakequant, bytes_per_token

DEVICE = os.environ.get("KNLP_BETA_TEST_DEVICE", "cpu")


def test_fp4_levels_and_even_ties():
    values = torch.tensor(
        [[0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.75, -5]], device=DEVICE
    )
    expected = torch.tensor([[0, 0, 1, 1, 2, 2, 4, 4, 6, -1, -4]], device=DEVICE)
    assert torch.equal(scaled_float(values, 4), expected)
    assert torch.equal(scaled_float(torch.zeros(2, 8), 4), torch.zeros(2, 8))


def test_fp8_matches_native_conversion_with_its_recorded_scale():
    values = torch.tensor(
        [[0.0, 0.1, -0.35, 3.0, 448.0], [1.0, 2, 3, 4, 5]], device=DEVICE
    )
    scale = values.abs().amax(-1, keepdim=True) / 448
    expected = (values / scale).to(torch.float8_e4m3fn).float() * scale
    assert torch.equal(scaled_float(values, 8), expected)


def test_cache_format_dispatch_counts_one_fp32_scale_per_vector():
    values = torch.tensor([[0.0, 0.25, 0.75, 6.0]], device=DEVICE)
    for bits, method, expected_bytes in (
        (4, "fp4_e2m1", 4608),
        (8, "fp8_e4m3fn", 8704),
    ):
        assert torch.equal(fakequant(values, bits, method), scaled_float(values, bits))
        case = dict(name=method, k_bits=bits, v_bits=bits, method=method)
        assert bytes_per_token(case, depth=32)["ideal_total_bytes"] == expected_bytes
