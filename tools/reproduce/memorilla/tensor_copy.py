# SPDX-License-Identifier: MIT
"""Bounded, exact CPU copies for FP16 document banks."""

import math

import torch


def copy_fp16_cpu(source, destination, *, scratch_bytes=1024 * 1024):
    """Copy CPU FP16 values into an owned FP16 or BF16 destination.

    Large BF16 copies use a bounded FP32 staging buffer. Every finite FP16
    value is represented exactly in FP32, so only the final BF16 conversion
    rounds. Some Torch kernels handle this pair faster than direct FP16 to
    BF16. NaN chunks use the direct kernel to preserve its exact NaN bits.
    Small banks that fit one scratch chunk retain the direct copy path.

    Scratch uses at most scratch_bytes plus a scalar reduction result. The
    caller retains its source lease until this returns.
    """
    if (
        source.device.type != "cpu"
        or destination.device.type != "cpu"
        or source.dtype != torch.float16
        or destination.dtype not in (torch.float16, torch.bfloat16)
        or source.shape != destination.shape
        or not source.is_contiguous()
        or not destination.is_contiguous()
    ):
        raise ValueError("copy requires matching contiguous CPU FP16/BF16 tensors")
    if type(scratch_bytes) is not int or scratch_bytes < 4:
        raise ValueError("scratch byte limit must hold at least one float32 value")
    chunk = scratch_bytes // 4
    if destination.dtype == torch.float16 or source.numel() <= chunk:
        destination.copy_(source)
        return
    source_flat = source.view(-1)
    target_flat = destination.view(-1)
    scratch = torch.empty(chunk, dtype=torch.float32, device="cpu")
    for start in range(0, source.numel(), chunk):
        values = source_flat[start : start + chunk]
        target = target_flat[start : start + chunk]
        temporary = scratch[: values.numel()]
        temporary.copy_(values)
        # max propagates a NaN without allocating a full-size bool mask.
        if math.isnan(temporary.max().item()):
            target.copy_(values)
        else:
            target.copy_(temporary)
