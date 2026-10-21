# SPDX-License-Identifier: MIT
"""Read literal packed bytes, including scales, through causal page prefixes."""

import struct

import pytest
import torch

from tools.kv.decision.inline_scales import decode_inline_fp8_snapshot
from tools.kv.decision.production_numerics import attention


GEOMETRY = {"block_size": 2, "kv_heads": 2, "head_dim": 4}


def snapshot(reverse=False):
    # Literal E4M3 codebook values, independent of Torch's FP8 conversion.
    codes = [0x00, 0x80, 0x01, 0x7E] if reverse else [0x38, 0xB8, 0x40, 0xC0]
    values = [0.0, -0.0, 2**-9, 448.0] if reverse else [1.0, -1.0, 2.0, -2.0]
    scales = [1.5, 0.03125, 4.0, 0.5, 2.0, 8.0]
    packed = bytearray()
    for scale in scales:
        packed.extend(codes)
        packed.extend(struct.pack("<f", scale))
    # The unused final token deliberately contains invalid codes and scales.
    packed.extend([0xFF] * 16)
    raw = torch.tensor(list(packed), dtype=torch.uint8).reshape(2, 2, 2, 8)
    expected = torch.tensor(
        [[value * scale for value in values] for scale in scales],
        dtype=torch.float32,
    ).reshape(3, 2, 4)
    return raw, expected


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_literal_bytes_preserve_each_token_head_scale_and_ignore_page_tail(
    typed, reverse
):
    raw, expected = snapshot(reverse)
    before = raw.clone()
    supplied = raw.view(torch.float8_e4m3fn) if typed else raw
    actual = decode_inline_fp8_snapshot(supplied, GEOMETRY, 3)
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    if reverse:
        assert torch.signbit(actual[..., 1]).all()
        assert not torch.signbit(actual[..., 0]).any()
    assert torch.equal(raw, before)
    with pytest.raises(ValueError, match="valid cache prefix"):
        decode_inline_fp8_snapshot(supplied, GEOMETRY, 4)


def test_decoded_prefix_attention_matches_an_independent_gqa_head_loop():
    raw_k, expected_k = snapshot(False)
    raw_v, expected_v = snapshot(True)
    keys = decode_inline_fp8_snapshot(raw_k, GEOMETRY, 3)
    values = decode_inline_fp8_snapshot(raw_v, GEOMETRY, 3)
    query = torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.4],
            [0.2, 0.1, -0.1, 0.3],
            [-0.1, 0.3, 0.2, 0.1],
            [0.2, -0.2, 0.3, -0.1],
        ]
    )
    expected = torch.stack(
        [
            ((expected_k[:, h // 2].double() @ query[h].double()) * 0.5).softmax(0)
            @ expected_v[:, h // 2].double()
            for h in range(4)
        ]
    )
    actual = attention(query, keys, values, softmax_scale=0.5)
    assert torch.allclose(actual.double(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_used_scale_is_rejected(value):
    raw, _ = snapshot()
    raw.flatten()[4:8] = torch.tensor(list(struct.pack("<f", value)), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Invalid scale"):
        decode_inline_fp8_snapshot(raw, GEOMETRY, 3)


def test_nonfinite_used_code_and_reconstruction_overflow_are_rejected():
    raw, _ = snapshot()
    raw.flatten()[0] = 0x7F
    with pytest.raises(ValueError, match="Nonfinite decoded"):
        decode_inline_fp8_snapshot(raw, GEOMETRY, 3)
    raw, _ = snapshot(True)
    raw.flatten()[4:8] = torch.tensor(list(struct.pack("<f", 3e38)), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Nonfinite decoded"):
        decode_inline_fp8_snapshot(raw, GEOMETRY, 3)


@pytest.mark.parametrize("length", [0, -1, 5, 1.0, True])
def test_causal_length_cannot_escape_the_snapshot(length):
    raw, _ = snapshot()
    with pytest.raises(ValueError, match="Causal prefix"):
        decode_inline_fp8_snapshot(raw, GEOMETRY, length)


@pytest.mark.parametrize(
    "bad", ["dtype", "stride", "shape", "alignment", "empty", "geometry"]
)
def test_invalid_snapshot_layout_is_rejected(bad):
    raw, _ = snapshot()
    geometry = dict(GEOMETRY)
    if bad == "dtype":
        raw = raw.view(torch.float8_e5m2)
    elif bad == "stride":
        raw = raw.transpose(0, 1)
    elif bad == "shape":
        raw = raw[..., :4].contiguous()
    elif bad == "alignment":
        raw = torch.cat([torch.zeros(1, dtype=torch.uint8), raw.flatten()])[
            1:
        ].reshape_as(raw)
    elif bad == "empty":
        raw = raw[:0]
    else:
        geometry["head_dim"] = 3
    with pytest.raises(ValueError):
        decode_inline_fp8_snapshot(raw, geometry, 3)
