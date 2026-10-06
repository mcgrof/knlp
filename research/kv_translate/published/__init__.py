"""Reference implementations of published cross-model KV cache transfer.

Two methods live here, kept deliberately separate from this repository's own
historical mappers so that a reproduction claim is never confused with an
in-house result:

``FULL_HEAD``   Heo et al., arXiv:2608.03893v1. Per-target-head closed-form
                ridge in RoPE-stripped content space, with each target head
                reading every source KV head of the selected source layers.
                CacheBridge coined the name "Full-Head Mapping" for it; the
                paper itself does not use that phrase.

``CACHE_BRIDGE`` Qu et al., arXiv:2609.00891v1. The same selector, interface
                and solver, with three offline changes: Head-Local support,
                Attn-Repair row weighting, and Fused-Fit, which is a speed
                rewrite that the authors state changes no coefficient.

Neither has a verified author release, so both are reimplementations from the
published text. Every value the papers did not pin is recorded in the gap
register rather than chosen silently.
"""

from .methods import (  # noqa: F401
    CACHE_BRIDGE,
    FULL_HEAD,
    MethodSpec,
    method,
)
