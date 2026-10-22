# SPDX-License-Identifier: MIT
"""Replay real CPU view metadata as synthetic CUDA allocation records."""

import copy

import pytest
import torch
from decision_storage_fixture import memory_fixture

from tools.kv.decision.inline_scales import inline_fp8_views
from tools.kv.decision.paged_memory import audit_record
from tools.kv.native.instrument import tensor_inventory


def fixture():
    geometry, record = memory_fixture("SF")
    geometry.update(head_dim=4, query_heads=4)
    cache = torch.zeros((2, 2, 16, 1, 8), dtype=torch.uint8)
    codes, scales = inline_fp8_views(cache, geometry)

    def described(tensor):
        result = tensor_inventory(tensor)[0]
        result["device"] = "cuda:0"
        return result

    layer = record["layers"][0]
    layer.update(
        implementation="TritonAttentionImpl",
        K=described(codes[0]),
        V=described(codes[1]),
        inline_scales={"K": described(scales[0]), "V": described(scales[1])},
    )

    def allocations(tensors):
        values = {r["storage_pointer"]: r["storage_bytes"] for r in tensors}
        rows = [
            {
                "device": 0,
                "address": address,
                "requested_bytes": size,
                "allocated_block_bytes": ((size + 511) // 512) * 512,
                "rounding_bytes": (-size) % 512,
            }
            for address, size in values.items()
        ]
        return {
            "allocations": rows,
            "allocation_count": len(rows),
            **{
                key: sum(r[key] for r in rows)
                for key in (
                    "requested_bytes",
                    "allocated_block_bytes",
                    "rounding_bytes",
                )
            },
        }

    # Keep buffers alive while constructing address metadata.
    scratch = [torch.empty(shape) for shape in [(8, 4, 16, 4), (8, 4, 16), (8, 4, 16)]]
    scratch_records = [described(t) for t in scratch]
    record.update(
        arm="SD",
        K_payload_bytes=128,
        V_payload_bytes=128,
        inline_scale_logical_bytes=256,
        KV=allocations([layer["K"], layer["V"]]),
        persistent_combined=allocations([layer["K"], layer["V"], *layer["scales"]]),
        triton_scratch={
            **allocations(scratch_records),
            "builders": [
                {"sequence_threshold": 8, "segments": 16, "tensors": scratch_records}
            ],
        },
    )
    record["budget_slack_bytes"] = (
        4096 - record["persistent_combined"]["allocated_block_bytes"]
    )
    return geometry, record


def test_inline_scales_count_once_and_scratch_stays_outside_cache_budget():
    geometry, record = fixture()
    result = audit_record(record, "SD", geometry, 4096, equal_blocks=2)
    assert record["KV"]["requested_bytes"] == 512
    assert result["inline_scale_logical_bytes"] == 256
    assert result["persistent_allocated_bytes"] == 2048
    assert result["scale_allocated_bytes"] == 1536
    assert result["triton_scratch_allocated_bytes"] == 12288


@pytest.mark.parametrize(
    "defect",
    [
        "offset",
        "stride",
        "dtype",
        "backing",
        "device",
        "missing",
        "double_count",
        "scratch_shape",
        "scratch_threshold",
        "scratch_alias",
        "scratch_internal_alias",
        "scratch_stride",
    ],
)
def test_wrong_inline_alias_or_scratch_metadata_fails(defect):
    geometry, record = fixture()
    scale = record["layers"][0]["inline_scales"]["V"]
    if defect == "offset":
        scale["storage_offset"] -= 1
    elif defect == "stride":
        scale["strides"][0] += 1
    elif defect == "dtype":
        scale["dtype"] = "torch.float16"
    elif defect == "backing":
        scale["storage_pointer"] += 1024
    elif defect == "device":
        scale["device"] = "cuda:1"
    elif defect == "missing":
        del record["layers"][0]["inline_scales"]["K"]
    elif defect == "double_count":
        record["persistent_combined"]["requested_bytes"] += 256
    elif defect == "scratch_shape":
        record["triton_scratch"]["builders"][0]["tensors"][0]["shape"][0] = 16
    elif defect == "scratch_threshold":
        record["triton_scratch"]["builders"][0]["sequence_threshold"] = 16
    elif defect == "scratch_internal_alias":
        tensors = record["triton_scratch"]["builders"][0]["tensors"]
        tensors[2] = copy.deepcopy(tensors[1])
    elif defect == "scratch_stride":
        record["triton_scratch"]["builders"][0]["tensors"][0]["strides"][0] = 0
    else:
        record["triton_scratch"]["builders"][0]["tensors"][0] = copy.deepcopy(
            record["layers"][0]["K"]
        )
    with pytest.raises(ValueError):
        audit_record(record, "SD", geometry, 4096)
