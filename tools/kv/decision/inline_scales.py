# SPDX-License-Identifier: MIT
"""Read-only views for NHD E4M3 caches with inline token/head FP32 scales.

These helpers check representation and aliases. They do not qualify a cache
writer, attention kernel, allocator budget or production recipe.
"""


def inline_fp8_views(cache, geometry):
    """Separate code bytes and scales without copying their common backing.

    Each physical head has head_dim E4M3 bytes followed by four scale bytes.
    An aligned subview of a larger backing is accepted; the caller must audit
    that backing's complete ownership and allocator size separately.
    """
    import torch

    block, heads, width = (geometry[k] for k in ("block_size", "kv_heads", "head_dim"))
    if any(type(n) is not int or n <= 0 for n in (block, heads, width)) or width % 4:
        raise ValueError("Inline FP32 scales require positive aligned geometry")
    if (
        not isinstance(cache, torch.Tensor)
        or cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or cache.ndim != 5
        or cache.shape[0] <= 0
        or tuple(cache.shape[1:]) != (2, block, heads, width + 4)
    ):
        raise ValueError("Expected two NHD E4M3 planes with four inline scale bytes")
    row = width + 4
    if cache.stride() != (
        2 * block * heads * row,
        block * heads * row,
        heads * row,
        row,
        1,
    ):
        raise ValueError("Inline cache strides differ from the declared NHD layout")
    if cache.storage_offset() % 4:
        raise ValueError("Inline scale backing is not FP32 aligned")
    raw = cache.view(torch.uint8)
    codes = tuple(
        raw[:, side, ..., :width].view(torch.float8_e4m3fn) for side in (0, 1)
    )
    scales = tuple(
        raw[:, side, ..., width:].view(torch.float32).squeeze(-1) for side in (0, 1)
    )
    return codes, scales


def verify_inline_scale_aliases(cache, geometry, actual):
    """Match backend scale views to independently decoded physical offsets."""
    import torch

    _, expected = inline_fp8_views(cache, geometry)
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise ValueError("Expected the backend's K and V scale views")
    for observed, wanted in zip(actual, expected):
        if (
            not isinstance(observed, torch.Tensor)
            or observed.dtype != torch.float32
            or observed.device != wanted.device
            or observed.shape != wanted.shape
            or observed.stride() != wanted.stride()
            or observed.storage_offset() != wanted.storage_offset()
            or observed.untyped_storage().data_ptr()
            != wanted.untyped_storage().data_ptr()
            or observed.untyped_storage().nbytes() != wanted.untyped_storage().nbytes()
        ):
            raise ValueError("Backend scale view differs from the inline byte layout")
    return expected


def decode_inline_fp8_snapshot(raw, geometry, length):
    """Decode an untimed, page-ordered byte snapshot through a causal prefix.

    The caller gathers original page IDs into a contiguous four-dimensional
    snapshot that includes all code and scale bytes. Only ``length`` valid
    tokens are decoded; the unused tail of the final page is not evidence.
    FP32 reconstruction here is an oracle operand, not a serving path.
    """
    import torch

    block, heads, width = (geometry[k] for k in ("block_size", "kv_heads", "head_dim"))
    if any(type(n) is not int or n <= 0 for n in (block, heads, width)) or width % 4:
        raise ValueError("Inline snapshot requires positive aligned geometry")
    if (
        not isinstance(raw, torch.Tensor)
        or raw.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or raw.ndim != 4
        or raw.shape[0] <= 0
        or tuple(raw.shape[1:]) != (block, heads, width + 4)
        or not raw.is_contiguous()
        or raw.storage_offset() % 4
    ):
        raise ValueError("Expected a contiguous page snapshot of inline FP8 bytes")
    if type(length) is not int or not 0 < length <= raw.shape[0] * block:
        raise ValueError("Causal prefix is outside the gathered page snapshot")
    prefix = raw.view(torch.uint8).reshape(-1, heads, width + 4)[:length]
    scales = prefix[..., width:].view(torch.float32)
    if not bool(torch.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError("Invalid scale in the valid cache prefix")
    values = prefix[..., :width].view(torch.float8_e4m3fn).float() * scales
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite decoded value in the valid cache prefix")
    return values
