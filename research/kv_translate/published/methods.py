#!/usr/bin/env python3
"""What each published method is, stated so the two cannot be conflated.

CacheBridge differs from Full-Head Mapping in exactly three offline places and
nowhere at inference. Writing that down as data rather than as prose matters
because the easy mistake is to implement the cheap part -- Head-Local support,
which is a one-line change to how the design matrix is built -- and call the
result CacheBridge. Attn-Repair is the part that costs work, and the plan says
in as many words not to omit it and keep the name.

Fused-Fit is included as a declared no-op on coefficients. The authors state
the optimised path "preserves mapper support, attention weights,
regularization, solver policy, coefficient dtype, and serialized mapper
schema", so a reference implementation that skips it is still the same method
producing the same numbers, only slower. Claiming the authors' construction
speed without it would not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FULL_HEAD = "full_head_mapping"
CACHE_BRIDGE = "cache_bridge"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    paper: str
    arxiv: str
    # How the design matrix is built for one target (layer, head).
    support: str
    # Whether calibration rows are reweighted before the solve.
    row_weighting: str
    # Everything below is shared and must stay shared: the two arms are only
    # comparable because they differ in these two axes and nothing else.
    selector: str = (
        "per target layer, rank source layers by held-in R-squared of centered "
        "single-source-layer OLS probes averaged over KV heads and over K and "
        "V, take the top k; one layer set serves every head of that target "
        "layer and both components"
    )
    solver: str = (
        "centered affine ridge in closed form, W = (Xc^T Dw Xc + lambda I)^-1 "
        "Xc^T Dw Yc, b = ybar_w - xbar_w W"
    )
    rope: str = (
        "keys are fitted in content space on both sides: source rotary is "
        "inverted before the regression and target rotary applied after "
        "mapping; values carry no rotation and are mapped directly"
    )
    ridge_lambda: float = 0.01
    coefficient_dtype: str = "float32"
    notes: tuple = field(default_factory=tuple)


_FULL_HEAD = MethodSpec(
    method_id=FULL_HEAD,
    paper="Cross-Model KV Cache Transfer in LLM Families",
    arxiv="2608.03893v1",
    support=(
        "every source KV head of every selected source layer, concatenated: "
        "width k * H_source * d_source"
    ),
    row_weighting="none; every calibration row carries equal weight",
    notes=(
        "the paper never uses the name Full-Head Mapping; CacheBridge coined "
        "it for this baseline",
        "one design matrix per (target layer, component) is shared by all "
        "heads of that layer, which is what makes the Gram reusable",
    ),
)

_CACHE_BRIDGE = MethodSpec(
    method_id=CACHE_BRIDGE,
    paper="CacheBridge",
    arxiv="2609.00891v1",
    support=(
        "one architecture-indexed source head per target head, concatenated "
        "over selected source layers: width k * d_source. The assignment a(h) "
        "is a fixed prior from architecture metadata, not learned, and is the "
        "identity when source and target expose aligned KV groups"
    ),
    row_weighting=(
        "Attn-Repair: rows reweighted by a first-order surrogate of receiver "
        "attention drift, shrunk toward uniform by the largest factor that "
        "keeps the Kish effective sample size at or above 2x the feature width"
    ),
    notes=(
        "Fused-Fit is a construction-speed rewrite the authors state changes "
        "no coefficient; omitting it reproduces the numbers but not the "
        "reported build time, and that distinction must survive into any claim",
        "Head-Local alone is not CacheBridge; the paper reports it separately "
        "as an ablation and it scores lower than the full method",
    ),
)

_BY_ID = {FULL_HEAD: _FULL_HEAD, CACHE_BRIDGE: _CACHE_BRIDGE}


def method(method_id):
    try:
        return _BY_ID[method_id]
    except KeyError:
        raise ValueError(
            f"{method_id!r} is not a published method implemented here; "
            f"choose one of {sorted(_BY_id_keys())}"
        ) from None


def _by_id_keys():
    return _BY_ID.keys()


# kept as a separate name so the error message above cannot drift from the map
_BY_id_keys = _by_id_keys


def shared_axes():
    """What the two arms must hold identical for the comparison to mean anything.

    The papers are explicit that the paired comparison shares calibration rows,
    selected layers, ridge strength, centering, token sampling and evaluation
    examples. If any of these differ between arms, the measured difference is
    not attributable to the method.
    """
    return (
        "calibration rows",
        "selected source layers",
        "ridge lambda",
        "centering convention",
        "token sampling stride",
        "evaluation examples",
        "decoding settings",
        "scorer",
    )


def differing_axes():
    """The only two things allowed to differ between the arms."""
    return ("support", "row_weighting")
