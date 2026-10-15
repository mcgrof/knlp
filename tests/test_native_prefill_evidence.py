# SPDX-License-Identifier: MIT
"""Check synchronized trace joins and bounded scratch views on CPU."""

import pytest
import torch

from tools.kv.native.prefill_evidence import scratch_view, split_dispatch_trace


def test_dispatch_join_requires_complete_kernel_inside_its_region():
    trace = {
        "traceEvents": [
            {"name": "case", "cat": "user_annotation", "ph": "X", "ts": 100, "dur": 30},
            {
                "name": "other",
                "cat": "user_annotation",
                "ph": "X",
                "ts": 150,
                "dur": 30,
            },
            {
                "name": "BatchPrefillWithPagedKVCacheKernel<int>",
                "cat": "kernel",
                "ts": 105,
                "dur": 10,
            },
            {
                "name": "BatchPrefillWithPagedKVCacheKernel<int>",
                "cat": "kernel",
                "ts": 95,
                "dur": 10,
            },
            {
                "name": "BatchPrefillWithPagedKVCacheKernel<int>",
                "cat": "kernel",
                "ts": 125,
                "dur": 10,
            },
        ]
    }
    result = split_dispatch_trace(
        trace, [dict(region="case", backend="fa2"), dict(region="other", backend="fa3")]
    )
    assert result[0]["passed"] and result[0]["attention_kernel_count"] == 1
    assert len(result[0]["trace"]["traceEvents"]) == 2
    assert not result[1]["passed"]


def test_duplicate_or_missing_region_cannot_prove_dispatch():
    with pytest.raises(ValueError, match="expected one synchronized region"):
        split_dispatch_trace(
            {"traceEvents": []}, [dict(region="absent", backend="fa2")]
        )
    region = dict(name="duplicate", cat="user_annotation", ph="X", ts=0, dur=10)
    with pytest.raises(ValueError, match="expected one synchronized region"):
        split_dispatch_trace(
            {"traceEvents": [region, region]}, [dict(region="duplicate", backend="fa2")]
        )


def test_scratch_views_reuse_storage_and_preserve_finite_fp8_values():
    storage = torch.empty(24, dtype=torch.float16)
    pointer = storage.untyped_storage().data_ptr()
    for layer in range(3):
        values = (
            (torch.arange(24).float() - 12 + layer)
            .to(torch.float8_e4m3fn)
            .reshape(3, 2, 4)
        )
        view = scratch_view(storage, values)
        view.copy_(values)
        assert view.untyped_storage().data_ptr() == pointer
        assert view.shape == values.shape
        assert torch.equal(view, values.half())
    assert storage.numel() == 24


def test_scratch_rejects_capacity_or_dtype_mismatch():
    with pytest.raises(ValueError, match="exceed scratch capacity"):
        scratch_view(torch.empty(23, dtype=torch.float16), torch.empty(24))
    with pytest.raises(ValueError, match="flat FP16 allocation"):
        scratch_view(torch.empty(24), torch.empty(24))


def test_gpu_annotation_with_same_name_is_not_a_second_cpu_region():
    trace = {
        "traceEvents": [
            dict(name="call", cat="user_annotation", ph="X", ts=100, dur=30),
            dict(name="call", cat="gpu_user_annotation", ph="X", ts=105, dur=10),
            dict(
                name="PrefillWithKVCacheKernel<half>",
                cat="kernel",
                ph="X",
                ts=105,
                dur=10,
            ),
        ]
    }
    result = split_dispatch_trace(trace, [dict(region="call", backend="fa3")])
    assert result[0]["passed"] and result[0]["attention_kernel_count"] == 1
