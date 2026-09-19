# SPDX-License-Identifier: GPL-2.0
"""Honest KV byte accounting, including the parts that are easy to leave out.

Three corrections to the pattern in ``kri_tierkv/block_index.py``, which the
plan points at as the accounting model to copy.

First, a partial final block holds the tokens it actually holds.  Charging a
full block for it inflates the baseline, which flatters any method measured
against it.

Second, quantisation is not free.  Every group of values carries a scale, and
an asymmetric scheme carries a zero point too.  At the group sizes real codecs
use these are a few percent of the payload, which is the same order as the
savings some of the methods in this program were claiming.

Third, wire bytes and resident bytes are different quantities and are kept in
different fields.  A receiver that reconstructs full K and V before decoding
has saved transfer, not residency, and the plan is explicit that calling one
the other is not allowed.

There is also the amortisation question, which decides deployability
independently of quality: a mapper that costs more bytes than it saves is not
a compressor at the prefix population it will actually see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVGeometry:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_tokens: int
    batch: int = 1

    @property
    def elements(self) -> int:
        """Elements in K (or in V); the cache holds both."""
        return (
            self.batch * self.n_layers * self.n_kv_heads * self.head_dim * self.n_tokens
        )


@dataclass
class QuantSpec:
    """A per-tensor format.  ``group_size=0`` means one scale per tensor."""

    bits: float = 16.0
    group_size: int = 0
    scale_bits: float = 16.0
    zero_point_bits: float = 0.0

    def overhead_bits(self, n_elements: int) -> float:
        if self.group_size <= 0:
            return self.scale_bits + self.zero_point_bits
        n_groups = -(-n_elements // self.group_size)  # ceil
        return n_groups * (self.scale_bits + self.zero_point_bits)

    def bytes_for(self, n_elements: int) -> float:
        payload = n_elements * self.bits
        return (payload + self.overhead_bits(n_elements)) / 8.0


BF16 = QuantSpec(bits=16.0)
FP8 = QuantSpec(bits=8.0, group_size=128, scale_bits=16.0)
INT8_ASYM = QuantSpec(bits=8.0, group_size=128, scale_bits=16.0, zero_point_bits=8.0)


@dataclass
class ByteReport:
    k_bytes: float
    v_bytes: float
    metadata_bytes: float = 0.0
    label: str = ""
    resident: bool = True

    @property
    def total(self) -> float:
        return self.k_bytes + self.v_bytes + self.metadata_bytes

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "k_bytes": self.k_bytes,
            "v_bytes": self.v_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total,
            "measures": "resident" if self.resident else "wire",
        }


def cache_bytes(
    geom: KVGeometry,
    k_spec: QuantSpec = BF16,
    v_spec: QuantSpec = BF16,
    *,
    metadata_bytes: float = 0.0,
    label: str = "",
    resident: bool = True,
) -> ByteReport:
    """Bytes for one cache under a (possibly asymmetric) K/V format."""
    n = geom.elements
    return ByteReport(
        k_bytes=k_spec.bytes_for(n),
        v_bytes=v_spec.bytes_for(n),
        metadata_bytes=metadata_bytes,
        label=label,
        resident=resident,
    )


def blocked_token_count(n_tokens: int, block_size: int) -> int:
    """Tokens actually stored when a cache is blocked.

    A blocked cache still stores ``n_tokens`` tokens; it is the *allocation*
    that rounds up.  Both are returned by the caller's choice of function so
    that an accounting bug cannot hide behind the ambiguity.
    """
    return n_tokens


def blocked_allocation_tokens(n_tokens: int, block_size: int) -> int:
    """Token slots allocated, including the partial final block."""
    if block_size <= 0:
        return n_tokens
    return -(-n_tokens // block_size) * block_size


def savings_fraction(baseline: ByteReport, candidate: ByteReport) -> float:
    """Fraction of ``baseline`` bytes removed.  Negative means it grew."""
    if baseline.total <= 0:
        raise ValueError("baseline has no bytes")
    return (baseline.total - candidate.total) / baseline.total


def break_even_prefixes(mapper_bytes: float, bytes_saved_per_prefix: float) -> float:
    """How many cached prefixes a mapper must serve before it pays for itself.

    ``inf`` when a prefix saves nothing, which is the honest answer: no prefix
    population makes that mapper worth storing.
    """
    if bytes_saved_per_prefix <= 0:
        return float("inf")
    return mapper_bytes / bytes_saved_per_prefix


def mapper_bytes(
    n_maps: int,
    in_dim: int,
    out_dim: int,
    *,
    dtype_bits: float = 16.0,
    bias: bool = True,
) -> float:
    """Storage for a set of affine maps, counting the bias."""
    per = in_dim * out_dim + (out_dim if bias else 0)
    return n_maps * per * dtype_bits / 8.0
