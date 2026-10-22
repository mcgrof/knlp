# SPDX-License-Identifier: MIT
"""Reject corrupted reductions while allowing legitimate FP32 sum orders."""

import math

import numpy as np
import pytest

from tools.kv.decision.analyze import validate_nll


def row(values, mean=None):
    values = np.asarray(values, dtype=np.float32)
    return {
        "nll": values.astype(float).tolist(),
        "mean_nll": float(values.mean()) if mean is None else mean,
    }


def test_rounding_depends_on_order_and_magnitude():
    values = np.array([1000.0] + [0.1] * 127, dtype=np.float32)
    exact = math.fsum(values) / 128
    recorded = []
    for operands in (values, values[::-1]):
        accumulator = np.float32(0)
        for operand in operands:
            accumulator = np.float32(accumulator + operand)
        mean = float(np.float32(accumulator / np.float32(128)))
        recorded.append(mean)
        evidence = row(values, mean)
        validate_nll(evidence)
        assert evidence["mean_nll"] == mean
    assert recorded[0] != recorded[1]
    assert abs(recorded[0] - exact) > 1e-6


@pytest.mark.parametrize("scale", [0.0, 1e-6, 1.0, 32.0, 1000.0])
def test_valid_reductions_at_different_magnitudes(scale):
    validate_nll(row(np.linspace(0, scale, 128)))


@pytest.mark.parametrize("replacement", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_operands(replacement):
    evidence = row([1.0] * 128)
    evidence["nll"][0] = replacement
    with pytest.raises(ValueError, match="NLL row invalid"):
        validate_nll(evidence)


@pytest.mark.parametrize("mean", [float("nan"), float("inf"), -1.0, 1.01])
def test_invalid_means(mean):
    with pytest.raises(ValueError):
        validate_nll(row([1.0] * 128, mean))


def test_rejects_corruption_beyond_rounding_and_wrong_length():
    with pytest.raises(ValueError, match="rounding bound"):
        validate_nll(row([8.0] * 128, float(np.float32(8.001))))
    with pytest.raises(ValueError, match="NLL row invalid"):
        validate_nll(row([8.0] * 127))


def test_rejects_non_fp32_serialization():
    evidence = row([1.0] * 128)
    evidence["nll"][0] = 1.0 + 2**-30
    with pytest.raises(ValueError, match="FP32 operands"):
        validate_nll(evidence)
    with pytest.raises(ValueError, match="serialized FP32"):
        validate_nll(row([1.0] * 128, 1.0 + 2**-30))


def test_rejects_overflowing_reduction():
    with pytest.raises(ValueError, match="FP32 operands"):
        validate_nll(row([float(np.finfo(np.float32).max)] * 128, 0.0))
