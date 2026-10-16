# SPDX-License-Identifier: MIT
"""Simulate scaled E2M1/E4M3FN values with one FP32 scale per final axis."""

import torch


def scaled_float(x, bits):
    if bits not in (4, 8):
        raise ValueError("Floating quantization supports E2M1 FP4 or E4M3FN FP8")
    w = x.float()
    limit = 6.0 if bits == 4 else 448.0
    scale = w.abs().amax(-1, keepdim=True) / limit
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = (w / safe).clamp(-limit, limit)
    if bits == 8:
        rounded = normalized.to(torch.float8_e4m3fn).float()
    else:
        levels = w.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        midpoints = (levels[:-1] + levels[1:]) / 2
        absolute = normalized.abs().contiguous()
        lower = torch.bucketize(absolute, midpoints)
        # At an exact midpoint, select the code with an even low bit.
        tie = (lower < 7) & (absolute == midpoints[lower.clamp_max(6)])
        indices = lower + (tie & (lower % 2 == 1)).long()
        rounded = levels[indices] * normalized.sign()
    return (rounded * scale).to(x.dtype)
