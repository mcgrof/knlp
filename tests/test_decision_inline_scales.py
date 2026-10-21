# SPDX-License-Identifier: MIT
"""Reject scale copies, wrong offsets and NHD/FP8 representation mistakes."""

import pytest
import torch

from tools.kv.decision.inline_scales import (
    inline_fp8_views,
    verify_inline_scale_aliases,
)


GEOMETRY = {"block_size": 16, "kv_heads": 2, "head_dim": 8}


def backing():
    return torch.zeros((2, 2, 16, 2, 12), dtype=torch.uint8)


@pytest.mark.parametrize("typed", [False, True])
def test_code_and_scale_writes_use_disjoint_bytes_of_one_backing(typed):
    raw = backing()
    cache = raw.view(torch.float8_e4m3fn) if typed else raw
    codes, scales = inline_fp8_views(cache, GEOMETRY)
    codes[0].view(torch.uint8).fill_(0x38)
    codes[1].view(torch.uint8).fill_(0xB8)
    values = torch.arange(64, dtype=torch.float32).reshape(2, 16, 2) + 1
    scales[0].copy_(values)
    scales[1].copy_(values + 100)
    assert torch.equal(
        raw[:, 0, ..., :8], torch.full((2, 16, 2, 8), 0x38, dtype=torch.uint8)
    )
    assert torch.equal(
        raw[:, 1, ..., :8], torch.full((2, 16, 2, 8), 0xB8, dtype=torch.uint8)
    )
    # Independently read each physical 12-byte head as 8 codes plus one FP32.
    flat = raw.flatten()
    for page in range(2):
        for side in range(2):
            for token in range(16):
                for head in range(2):
                    offset = (((page * 2 + side) * 16 + token) * 2 + head) * 12
                    actual = flat[offset + 8 : offset + 12].view(torch.float32).item()
                    assert actual == values[page, token, head].item() + side * 100
    assert (
        sum(t.numel() * t.element_size() for t in (*codes, *scales))
        == raw.numel()
        == 1536
    )
    assert {t.untyped_storage().data_ptr() for t in (*codes, *scales)} == {
        raw.data_ptr()
    }
    verify_inline_scale_aliases(cache, GEOMETRY, scales)


def test_aligned_subview_preserves_prefix_and_suffix_and_scale_base_offset():
    parent = torch.full((1568,), 0xAA, dtype=torch.uint8)
    raw = parent[16:-16].reshape(2, 2, 16, 2, 12)
    codes, scales = inline_fp8_views(raw, GEOMETRY)
    for code in codes:
        code.view(torch.uint8).zero_()
    for scale in scales:
        scale.fill_(1)
    assert scales[0].storage_offset() == 6  # 16 prefix bytes + 8 code bytes.
    assert torch.equal(parent[:16], torch.full((16,), 0xAA, dtype=torch.uint8))
    assert torch.equal(parent[-16:], torch.full((16,), 0xAA, dtype=torch.uint8))
    verify_inline_scale_aliases(raw, GEOMETRY, scales)


@pytest.mark.parametrize(
    "defect", ["dtype", "padding", "stride", "alignment", "empty", "geometry"]
)
def test_nonconforming_storage_is_rejected(defect):
    cache, geometry = backing(), dict(GEOMETRY)
    if defect == "dtype":
        cache = cache.view(torch.float8_e5m2)
    elif defect == "padding":
        cache = torch.zeros((2, 2, 16, 2, 16), dtype=torch.uint8)
    elif defect == "stride":
        cache = cache.transpose(0, 1)  # Same shape, different physical plane order.
    elif defect == "alignment":
        cache = torch.zeros(1537, dtype=torch.uint8)[1:].reshape(2, 2, 16, 2, 12)
    elif defect == "empty":
        cache = cache[:0]
    else:
        geometry["head_dim"] = 7
    with pytest.raises(ValueError):
        inline_fp8_views(cache, geometry)


@pytest.mark.parametrize(
    "defect", ["copy", "swap", "offset", "stride", "dtype", "missing"]
)
def test_backend_scale_views_must_alias_the_exact_inline_bytes(defect):
    cache = backing()
    _, original = inline_fp8_views(cache, GEOMETRY)
    actual = list(original)
    if defect == "copy":
        actual[0] = actual[0].clone()
    elif defect == "swap":
        actual.reverse()
    elif defect == "offset":
        actual[0] = torch.as_strided(
            actual[0],
            actual[0].shape,
            actual[0].stride(),
            actual[0].storage_offset() + 1,
        )
    elif defect == "stride":
        actual[0] = torch.as_strided(actual[0], actual[0].shape, (96, 6, 3))
    elif defect == "dtype":
        actual[0] = actual[0].view(torch.int32)
    else:
        actual.pop()
    with pytest.raises(ValueError):
        verify_inline_scale_aliases(cache, GEOMETRY, actual)
