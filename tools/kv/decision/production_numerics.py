# SPDX-License-Identifier: MIT
"""Untimed numerical helpers for native-query paged worker qualification."""

import math


def slots(query_offsets, lengths, pages, positions, page_size=16):
    """Derive writes from original per-request page IDs, never candidate slots."""
    if (
        not lengths
        or len(query_offsets) != len(lengths) + 1
        or len(pages) != len(lengths)
        or query_offsets[0] != 0
        or query_offsets[-1] != len(positions)
        or page_size <= 0
    ):
        raise ValueError("Incomplete raw request geometry")
    result = []
    for i, length in enumerate(lengths):
        start, end = query_offsets[i : i + 2]
        if not 0 <= start < end <= len(positions) or length < end - start:
            raise ValueError("Invalid query/sequence lengths")
        wanted = list(range(length - (end - start), length))
        if positions[start:end] != wanted:
            raise ValueError("Absolute positions differ from raw request suffix")
        if len(pages[i]) < (length + page_size - 1) // page_size:
            raise ValueError("Raw page table is too short")
        for position in wanted:
            page = pages[i][position // page_size]
            if type(page) is not int or page < 0:
                raise ValueError("Invalid physical page ID")
            result.append(page * page_size + position % page_size)
    if len(set(result)) != len(result):
        raise ValueError("Independent requests alias a write slot")
    return result


def comparison(actual, expected):
    """Return full-tensor metrics without tuning a tolerance from observations."""
    import torch

    if actual.shape != expected.shape or not actual.numel():
        raise ValueError("Oracle output shapes differ or are empty")
    a, b = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Nonfinite numerical oracle output")
    delta = a - b
    norm = float(torch.linalg.vector_norm(b))
    maximum = float(b.abs().max())
    error_norm = float(torch.linalg.vector_norm(delta))
    error_max = float(delta.abs().max())
    return {
        "exact": bool(torch.equal(actual, expected)),
        "relative_l2": error_norm / max(norm, 1e-12),
        "maximum_absolute_error": error_max,
        "normalized_maximum_error": error_max / max(maximum, 1e-12),
        "reference_norm": norm,
        "reference_maximum": maximum,
        "elements": actual.numel(),
    }


def within(row, bounds):
    if set(bounds) != {
        "relative_l2",
        "normalized_maximum_error",
        "maximum_absolute_error",
    }:
        raise ValueError("Numerical bounds must name all three error limits")
    if any(not math.isfinite(v) or v < 0 for v in bounds.values()):
        raise ValueError("Invalid numerical tolerance")
    # Absolute tolerance protects near-zero outputs; otherwise both global
    # normalized metrics must pass. Preserve all three observed values.
    return row["maximum_absolute_error"] <= bounds["maximum_absolute_error"] or (
        row["relative_l2"] <= bounds["relative_l2"]
        and row["normalized_maximum_error"] <= bounds["normalized_maximum_error"]
    )


def planes(cache, implementation):
    import torch

    if isinstance(cache, tuple) and len(cache) == 2:
        return cache
    if not isinstance(cache, torch.Tensor) or cache.ndim != 5:
        raise ValueError("Unsupported paged cache layout")
    if implementation == "FlashInferImpl" and cache.shape[1] == 2:
        return cache[:, 0], cache[:, 1]
    if implementation == "FlashAttentionImpl" and cache.shape[0] == 2:
        return cache[0], cache[1]
    raise ValueError("Backend layout differs from the frozen NHD contract")


def decoded(plane, scale):
    import torch

    if plane.dtype == torch.uint8:
        plane = plane.view(torch.float8_e4m3fn)
    if plane.dtype not in (torch.float16, torch.float8_e4m3fn):
        raise ValueError("Unsupported cache operand dtype")
    return plane.float() * scale


def attention(query, keys, values, *, softmax_scale):
    """FP32 scalar-scale attention, with TF32 explicitly disabled.

    The caller gathers original page IDs and truncates at each query's
    absolute causal position. This does not use candidate routing metadata.
    K/V are already decoded in FP32. A second diagnostic may explicitly
    round them to FP16 to measure reference reconstruction order separately.
    """
    import torch

    if (
        query.ndim != 2
        or keys.ndim != 3
        or keys.shape != values.shape
        or query.shape[1] != keys.shape[2]
        or query.shape[0] % keys.shape[1]
    ):
        raise ValueError("Unsupported query/cache geometry")
    groups = query.shape[0] // keys.shape[1]
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q = query.float().reshape(keys.shape[1], groups, keys.shape[2])
        k = keys.float().permute(1, 2, 0)
        v = values.float().permute(1, 0, 2)
        logits = torch.bmm(q, k) * softmax_scale
        result = torch.bmm(logits.softmax(-1), v).reshape_as(query)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    return result
