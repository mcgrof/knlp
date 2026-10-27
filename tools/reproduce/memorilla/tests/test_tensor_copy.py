# SPDX-License-Identifier: MIT
"""The bounded copy must preserve every FP16 value and special bit pattern."""

import pytest

torch = pytest.importorskip("torch")
from tools.reproduce.memorilla.tensor_copy import copy_fp16_cpu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scratch", [4096, 1024 * 1024])
def test_all_fp16_bit_patterns_match_direct_torch_conversion(dtype, scratch):
    source = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    source = source.reshape(8192, 8)
    destination = torch.empty(source.shape, dtype=dtype)
    expected = source.to(dtype)
    copy_fp16_cpu(source, destination, scratch_bytes=scratch)
    assert torch.equal(destination.view(torch.uint8), expected.view(torch.uint8))


def test_multiple_finite_chunks_and_partial_tail_are_exact():
    source = torch.linspace(-20, 20, 10003).to(torch.float16).reshape(1, -1)
    destination = torch.empty(source.shape, dtype=torch.bfloat16)
    copy_fp16_cpu(source, destination, scratch_bytes=4096)
    assert torch.equal(
        destination.view(torch.uint8), source.to(torch.bfloat16).view(torch.uint8)
    )


def test_invalid_shape_or_scratch_refuses_before_writing():
    source = torch.ones(2, 8, dtype=torch.float16)
    destination = torch.zeros(2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="scratch"):
        copy_fp16_cpu(source, destination, scratch_bytes=0)
    assert not destination.any()
    with pytest.raises(ValueError, match="matching"):
        copy_fp16_cpu(source, destination[:, :4])
