# SPDX-License-Identifier: GPL-2.0
"""Gate-0 per-mode quantizer behavior and byte accounting."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from kv_house.quant import (
    HEADER_BYTES,
    SCALE_BYTES,
    payload_bytes,
    quantize_rows,
    standard_allocations,
    uniform_allocation,
)


def test_error_monotone_in_bits():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(16, 64, generator=gen)
    errs = []
    for bits in (2, 4, 8, 16):
        y = quantize_rows(x, (bits,) * 16)
        errs.append(float((y - x).norm()))
    assert errs == sorted(errs, reverse=True)


def test_zero_bits_drops_row():
    x = torch.randn(8, 32)
    y = quantize_rows(x, (0,) * 4 + (8,) * 4)
    assert torch.equal(y[:4], torch.zeros(4, 32))
    assert (y[4:] - x[4:]).abs().max() < 0.1


def test_payload_bytes_accounting():
    d = 64
    b = payload_bytes((8,) * 16, d)
    assert b == 16 * d + 16 * SCALE_BYTES + HEADER_BYTES
    b16 = payload_bytes((16,) * 16, d)
    assert b16 == 16 * d * 2 + HEADER_BYTES
    b0 = payload_bytes((0,) * 16, d)
    assert b0 == HEADER_BYTES


def test_standard_allocations_have_distinct_names():
    allocs = standard_allocations(16)
    names = [a.name for a in allocs]
    assert len(names) == len(set(names))
    for a in allocs:
        assert len(a.bits) == 16
        assert a.total_bytes(64) > 0


def test_uniform_allocation_bytes_order():
    d = 64
    u8 = uniform_allocation(32, 8).total_bytes(d)
    u4 = uniform_allocation(32, 4).total_bytes(d)
    u2 = uniform_allocation(32, 2).total_bytes(d)
    assert u8 > u4 > u2
