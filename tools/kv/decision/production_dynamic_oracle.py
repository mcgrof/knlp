# SPDX-License-Identifier: MIT
"""CPU scale-bit and E4M3 code oracle for a prospective dynamic worker.

Scale creation matches the reference CUDA scalar-division operation: clamp
the FP32 maximum, then multiply by the rounded FP32 reciprocal of 448.
Normalization uses FP32 division in the independent NumPy codebook oracle.
This module does not change or qualify a GPU writer.
"""

import numpy as np

from .oracle import oracle


def dynamic_bytes(operand):
    """Return FP32 scales and uint8 code bits from retained native CPU operands."""
    import torch

    if (
        not isinstance(operand, torch.Tensor)
        or operand.device.type != "cpu"
        or operand.dtype not in (torch.float16, torch.bfloat16)
        or operand.ndim < 1
        or not operand.numel()
    ):
        raise ValueError("Expected a nonempty retained native-16 CPU operand")
    original = operand.detach()
    values = original.float().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite native cache-write operand")
    maximum = np.maximum(
        np.max(np.abs(values), axis=-1, keepdims=True), np.float32(1e-12)
    )
    scales = np.multiply(maximum, np.float32(1.0 / 448.0), dtype=np.float32)
    return scales, oracle(original, scales)
