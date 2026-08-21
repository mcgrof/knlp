# SPDX-License-Identifier: GPL-2.0
"""Per-mode quantization of temporally transformed KV blocks.

An orthogonal transform is lossless; compression happens when the
transformed temporal modes (rows of Y = R @ X) receive different
representation budgets. This module implements symmetric uniform
fake-quantization with a per-row absmax scale, plus byte accounting
that charges everything a real artifact would carry: payload bits,
per-row scales, per-block transform metadata, and a fixed header.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SCALE_BYTES = 2
HEADER_BYTES = 16


def quantize_rows(x, bits_per_row):
    """Symmetric uniform fake-quant, one absmax scale per row.

    x: [B, d]; bits_per_row: sequence of B ints (0 means the row is
    dropped and reconstructs as zero; 16 means the row is kept in
    fp16). Returns the dequantized tensor.
    """
    y = torch.empty_like(x)
    for i, bits in enumerate(bits_per_row):
        row = x[i]
        if bits <= 0:
            y[i] = torch.zeros_like(row)
        elif bits >= 16:
            y[i] = row.to(torch.float16).to(row.dtype)
        else:
            qmax = float(2 ** (bits - 1) - 1)
            scale = row.abs().max().clamp_min(1e-12) / qmax
            q = torch.clamp(torch.round(row / scale), -qmax - 1, qmax)
            y[i] = q * scale
    return y


def payload_bytes(bits_per_row, d):
    """Payload plus per-row-scale plus header bytes for one [B, d]
    block quantized with the given per-row bit widths."""
    bits = sum(int(b) * d for b in bits_per_row if b > 0)
    scales = sum(1 for b in bits_per_row if 0 < b < 16) * SCALE_BYTES
    return (bits + 7) // 8 + scales + HEADER_BYTES


@dataclass(frozen=True)
class Allocation:
    """A named per-row bit schedule for a block of size B."""

    name: str
    bits: tuple

    def total_bytes(self, d, metadata_floats=0):
        return payload_bytes(self.bits, d) + metadata_floats * SCALE_BYTES


def uniform_allocation(block_size, bits):
    return Allocation(f"uniform{bits}", (bits,) * block_size)


def front_loaded_allocation(block_size, head_bits, head_rows, tail_bits):
    """head_rows leading modes at head_bits, the rest at tail_bits.

    Meaningful only after a transform that concentrates energy into
    the leading rows; under identity it just protects the first
    tokens and serves as a control.
    """
    head = min(head_rows, block_size)
    bits = (head_bits,) * head + (tail_bits,) * (block_size - head)
    return Allocation(f"front{head_bits}x{head}tail{tail_bits}", bits)


def standard_allocations(block_size):
    """The milestone sweep: matched-average-rate pairs so transforms
    compete at equal payload budgets, plus corner points."""
    allocs = [
        uniform_allocation(block_size, 8),
        uniform_allocation(block_size, 4),
        uniform_allocation(block_size, 2),
        front_loaded_allocation(block_size, 8, max(1, block_size // 8), 4),
        front_loaded_allocation(block_size, 8, max(1, block_size // 8), 2),
        front_loaded_allocation(block_size, 8, max(1, block_size // 4), 2),
        front_loaded_allocation(block_size, 16, 1, 4),
        front_loaded_allocation(block_size, 16, 1, 2),
        # FreqKV-style truncation equivalents: keep leading modes,
        # drop the tail to zero bits. Only sensible after an
        # energy-compacting transform; under identity they discard
        # trailing tokens and act as an honesty control.
        front_loaded_allocation(block_size, 16, max(1, block_size // 4), 0),
        front_loaded_allocation(block_size, 8, max(1, block_size // 2), 0),
    ]
    return allocs
