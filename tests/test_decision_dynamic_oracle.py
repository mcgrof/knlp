# SPDX-License-Identifier: MIT
"""Keep scale-construction rounding distinct from operand normalization."""

import numpy as np
import pytest
import torch

from tools.kv.decision.production_dynamic_oracle import dynamic_bytes


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unit_scale_ties_and_signed_zero_use_independent_e4m3_codes(dtype):
    values = torch.tensor(
        [[448.0, -448.0, 1.0625, 1.1875, -1.0625, -1.1875, 0.0, -0.0]],
        dtype=dtype,
    )
    scales, codes = dynamic_bytes(values)
    assert scales.dtype == np.float32 and scales.shape == (1, 1)
    assert scales.view(np.uint32).tolist() == [[0x3F800000]]
    assert codes.dtype == np.uint8
    assert codes.tolist() == [[0x7E, 0xFE, 0x38, 0x3A, 0xB8, 0xBA, 0x00, 0x80]]


def test_scale_creation_preserves_the_cuda_rounded_reciprocal_step():
    values = torch.tensor([[1.5, -1.5], [65504.0, -65504.0]], dtype=torch.float16)
    scales, codes = dynamic_bytes(values)
    assert scales.view(np.uint32).ravel().tolist() == [0x3B5B6DB8, 0x431236DC]
    # Correctly rounded maximum/448 would differ by one ULP in these two cases.
    assert scales.view(np.uint32).ravel().tolist() != [0x3B5B6DB7, 0x431236DB]
    assert codes.tolist() == [[0x7E, 0xFE], [0x7E, 0xFE]]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_heads_use_the_declared_scale_floor_without_losing_zero_sign(dtype):
    scales, codes = dynamic_bytes(torch.tensor([[0.0, -0.0]], dtype=dtype))
    assert scales.view(np.uint32).tolist() == [[0x2720D7C5]]
    assert codes.tolist() == [[0, 128]]


def test_subnormal_native_values_follow_the_same_floor_rule():
    scales, codes = dynamic_bytes(torch.tensor([[2.0**-24]], dtype=torch.float16))
    assert scales.view(np.uint32).tolist() == [[0x2F124925]]
    assert codes.tolist() == [[0x7E]]
    scales, codes = dynamic_bytes(torch.tensor([[2.0**-133]], dtype=torch.bfloat16))
    assert scales.view(np.uint32).tolist() == [[0x2720D7C5]]
    assert codes.tolist() == [[0]]


def test_token_head_scales_are_independent_for_noncontiguous_inputs():
    source = torch.tensor(
        [[[448.0, -448.0], [1.5, -1.5]], [[0.0, -0.0], [1.0, -1.0]]],
        dtype=torch.float16,
    )
    values = source.transpose(0, 1)
    before = source.view(torch.uint8).clone()
    scales, codes = dynamic_bytes(values)
    assert scales.shape == (2, 2, 1)
    assert scales.view(np.uint32).squeeze(-1).tolist() == [
        [0x3F800000, 0x2720D7C5],
        [0x3B5B6DB8, 0x3B124925],
    ]
    assert codes.tolist() == [[[0x7E, 0xFE], [0, 128]], [[0x7E, 0xFE], [0x7E, 0xFE]]]
    assert torch.equal(source.view(torch.uint8), before)


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_operands_are_rejected(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        dynamic_bytes(torch.tensor([[value]], dtype=torch.float16))


@pytest.mark.parametrize(
    "operand",
    [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((0, 2), dtype=torch.float16),
        torch.tensor(1.0, dtype=torch.float16),
        torch.empty((1, 2), dtype=torch.float16, device="meta"),
        np.zeros((1, 2), dtype=np.float16),
    ],
)
def test_dtype_device_and_empty_geometry_cannot_change_the_oracle(operand):
    with pytest.raises(ValueError, match="native-16 CPU"):
        dynamic_bytes(operand)
