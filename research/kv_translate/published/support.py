#!/usr/bin/env python3
"""Which source layers feed a target layer, and which source heads feed a head.

Two separate choices that are easy to conflate.

Layer selection is shared by both methods and is the reason a 40-layer source
can feed a 64-layer target at all: each target layer independently ranks the
source layers by how well a single one of them predicts it, and keeps the best
k. One layer set serves every head of that target layer and both components,
which is what lets the baseline reuse a single Gram per layer.

Head selection is the one place the two methods differ in their design matrix.
The baseline concatenates every source KV head of the selected layers;
CacheBridge takes one source head per target head under a fixed
architecture-indexed assignment. On a pair where both sides expose the same
number of KV heads that assignment is the identity, which is the only case
either paper establishes.
"""

from __future__ import annotations

import numpy as np

from .methods import CACHE_BRIDGE, FULL_HEAD
from .ridge import held_in_r2, solve


def identity_head_assignment(n_source_kv_heads, n_target_kv_heads):
    """a(h) = h, valid only when the two sides expose aligned KV groups.

    Refusing the mismatched case is deliberate. CacheBridge says the support
    rule is unestablished for mismatched head counts, so inventing an
    assignment here would be inventing a method the paper does not describe
    and reporting it under the paper's name.
    """
    if n_source_kv_heads != n_target_kv_heads:
        raise ValueError(
            f"source has {n_source_kv_heads} KV heads and target has "
            f"{n_target_kv_heads}; the published head assignment is only "
            "established for aligned KV groups, and this implementation will "
            "not guess one"
        )
    return {h: h for h in range(n_target_kv_heads)}


def rank_source_layers(source_by_layer, target, *, weights=None, lam=0.01):
    """Rank source layers for one target layer by held-in R-squared.

    source_by_layer maps a source layer index to [n, heads, dim]; target is
    [n, heads, dim] for the target layer. The probe is a centered
    single-source-layer fit under the head-index-aligned assignment, and the
    score is averaged across heads -- and, by the caller, across K and V,
    because the selected set must serve both.
    """
    scores = {}
    for j, src in sorted(source_by_layer.items()):
        src = np.asarray(src, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        if src.shape[0] != tgt.shape[0]:
            raise ValueError("probe row counts differ between source and target")
        per_head = []
        for h in range(tgt.shape[1]):
            X = src[:, min(h, src.shape[1] - 1), :]
            Y = tgt[:, h, :]
            W, b, _ = solve(X, Y, w=weights, lam=lam)
            per_head.append(held_in_r2(X, Y, W, b, w=weights))
        scores[j] = float(np.mean(per_head))
    return scores


def select_layers(scores_k, scores_v, k):
    """Top-k source layers by the mean of the K and V probe scores.

    Averaging before ranking, rather than ranking each component and merging,
    is what makes one set serve both -- and the papers are explicit that one
    set does.
    """
    if set(scores_k) != set(scores_v):
        raise ValueError("K and V probes covered different source layers")
    combined = {j: (scores_k[j] + scores_v[j]) / 2.0 for j in scores_k}
    if k > len(combined):
        raise ValueError(f"asked for {k} source layers out of {len(combined)}")
    ranked = sorted(combined, key=lambda j: (-combined[j], j))
    return sorted(ranked[:k]), combined


def build_features(method_id, source_by_layer, selected, target_head, assignment):
    """The design matrix for one target (layer, head), per the method.

    Returned width is the published one: k*d for CacheBridge, k*H*d for the
    baseline. Getting this wrong is silent -- both produce a solvable system --
    so the width is asserted by the caller against the manifest.
    """
    blocks = []
    for j in selected:
        src = np.asarray(source_by_layer[j], dtype=np.float64)
        if method_id == CACHE_BRIDGE:
            blocks.append(src[:, assignment[target_head], :])
        elif method_id == FULL_HEAD:
            blocks.append(src.reshape(src.shape[0], -1))
        else:
            raise ValueError(f"unknown method {method_id!r}")
    return np.concatenate(blocks, axis=1)


def feature_width(method_id, k, n_source_kv_heads, head_dim):
    if method_id == CACHE_BRIDGE:
        return k * head_dim
    if method_id == FULL_HEAD:
        return k * n_source_kv_heads * head_dim
    raise ValueError(f"unknown method {method_id!r}")


def coefficient_count(
    method_id, *, n_target_layers, n_target_kv_heads, k, n_source_kv_heads, head_dim
):
    """Weights plus biases over both components, which is what gets serialized."""
    p = feature_width(method_id, k, n_source_kv_heads, head_dim)
    per_map = p * head_dim + head_dim
    return 2 * n_target_layers * n_target_kv_heads * per_map


def serialized_bytes(method_id, *, dtype_bytes=4, **kw):
    return coefficient_count(method_id, **kw) * dtype_bytes
