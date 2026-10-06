#!/usr/bin/env python3
"""What sharing one stored cache between two models saves in bytes.

A deployment that serves a prefix from two models can keep a cache for each,
or keep the source's cache alone and translate it for the receiver on demand.
The second stores less per token and pays for a mapper once. This module is
the arithmetic of that trade, kept apart from any measurement so a forecast
and a measured run are held to the same formula.

Sizes are exact counts of stored values. They say nothing about latency,
which has to be measured, and nothing about compression, which would change
both sides of the comparison.
"""

from __future__ import annotations

import math

BF16_BYTES = 2


def cache_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=BF16_BYTES):
    """Keys and values for one token across every layer of one model."""
    for name, v in (("layers", layers), ("kv_heads", kv_heads), ("head_dim", head_dim)):
        if v <= 0:
            raise ValueError(f"{name} must be positive, got {v}")
    return layers * 2 * kv_heads * head_dim * bytes_per_value


def geometry_bytes_per_token(geometry, bytes_per_value=BF16_BYTES):
    """The same, read from the geometry record a capture writes."""
    return cache_bytes_per_token(
        geometry["layers"], geometry["kv_heads"], geometry["head_dim"], bytes_per_value
    )


def stored_bytes(tokens, source_per_token, target_per_token, mapper_bytes):
    """Persistent bytes under each policy for a store holding this many tokens."""
    if tokens < 0:
        raise ValueError("a store cannot hold a negative number of tokens")
    both = tokens * (source_per_token + target_per_token)
    shared = tokens * source_per_token + mapper_bytes
    return {
        "both_caches": both,
        "shared_source": shared,
        "saving_fraction": (1.0 - shared / both) if both else None,
    }


def saving_limit(source_per_token, target_per_token):
    """The saving a store approaches as it grows and the mapper amortises."""
    return 1.0 - source_per_token / (source_per_token + target_per_token)


def break_even_tokens(required, source_per_token, target_per_token, mapper_bytes):
    """Fewest stored tokens at which sharing saves at least ``required``.

    Returns None when no store is large enough, which happens when the
    required saving exceeds what dropping the receiver's cache can give.
    """
    if not 0.0 <= required < 1.0:
        raise ValueError(f"a required saving of {required} is not a fraction")
    gain = (1.0 - required) * (source_per_token + target_per_token) - source_per_token
    if gain <= 0:
        return None
    return math.ceil(mapper_bytes / gain)
