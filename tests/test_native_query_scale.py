# SPDX-License-Identifier: MIT
"""Check query scale encoding without requiring a serving engine."""

import pytest
import torch

from tools.kv.native.instrument import query_scale


def test_query_scale_has_one_fp32_encoding():
    value = 0.000123456789
    assert query_scale({"5.q": value}, 5, "k8_v8") == float(
        torch.tensor(value, dtype=torch.float32)
    )


@pytest.mark.parametrize("arm", ["k16_v16", "k16_v8"])
def test_native_queries_ignore_symmetric_scale(arm):
    assert query_scale({"5.q": float("nan")}, 5, arm) == 1.0


def test_unspecified_query_scale_preserves_unit_default():
    assert query_scale({}, 5, "k8_v8") == 1.0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 1e40, 1e-50])
def test_unusable_query_scale_is_rejected(value):
    with pytest.raises(ValueError):
        query_scale({"5.q": value}, 5, "k8_v8")
