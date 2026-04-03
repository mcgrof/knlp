# SPDX-License-Identifier: GPL-2.0
"""Backward-compatibility shim — autotune kernel is now in fused_routed_attention.py.

Existing callers that import ``fused_routed_decode_autotune`` from this
module will continue to work. New code should use::

    from routing.fused_routed_attention import fused_routed_decode
    fused_routed_decode(..., autotune=True)
"""

from routing.fused_routed_attention import (  # noqa: F401
    fused_routed_decode,
    reference_routed_decode,
    select_top_k_blocks,
)


def fused_routed_decode_autotune(
    q, k_cache, v_cache, block_tables, block_counts, scale=None
):
    """Thin wrapper — calls fused_routed_decode(autotune=True)."""
    return fused_routed_decode(
        q,
        k_cache,
        v_cache,
        block_tables,
        block_counts,
        scale=scale,
        autotune=True,
    )
