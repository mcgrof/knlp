# SPDX-License-Identifier: MIT
"""Check physical KV backing and separate payload from staging allocations."""

import math

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def storage(s, mixed, shape):
    assert (
        len(s["layers"]) == shape["layers"]
        and len({x["layer"] for x in s["layers"]}) == shape["layers"]
    )
    backing = {}
    kp = vp = 0
    for row in s["layers"]:
        k, v = row["K"], row["V"]
        n = row["physical_blocks"]
        assert k["shape"] == v["shape"] == [n, 16, shape["kv_heads"], shape["head_dim"]]
        assert k["dtype"] == "torch.float16" and v["dtype"] == (
            "torch.float8_e4m3fn" if mixed else "torch.float16"
        )
        assert k["element_size"] == 2 and v["element_size"] == (1 if mixed else 2)
        assert n == s["physical_blocks_per_layer"]
        assert s["logical_KV_tokens_per_layer"] == n * 16
        assert (
            k["storage_pointer"] == v["storage_pointer"]
            and k["storage_bytes"] == v["storage_bytes"]
        )
        assert k["storage_bytes"] == k["logical_bytes"] + v["logical_bytes"]
        assert (
            k["logical_bytes"] == math.prod(k["shape"]) * k["element_size"]
            and v["logical_bytes"] == math.prod(v["shape"]) * v["element_size"]
        )
        assert k["storage_pointer"] not in backing
        backing[k["storage_pointer"]] = k["storage_bytes"]
        kp += k["logical_bytes"]
        vp += v["logical_bytes"]
        if mixed:
            assert (
                k["contiguous"]
                and v["contiguous"]
                and k["tensor_pointer"] == k["storage_pointer"]
                and v["tensor_pointer"] == k["tensor_pointer"] + k["logical_bytes"]
            )
    assert (
        sum(backing.values()) == s["unique_KV_backing_bytes"]
        and kp == s["K_payload_bytes"]
        and vp == s["V_payload_bytes"]
    )
    assert s["unique_KV_allocations"] == shape["layers"]
    assert s["serving_workspace_payload_bytes"] == (
        4096 * 16 * shape["kv_heads"] * shape["head_dim"] * 4 if mixed else 0
    )
    return {
        k: s[k]
        for k in [
            "physical_blocks_per_layer",
            "logical_KV_tokens_per_layer",
            "K_payload_bytes",
            "V_payload_bytes",
            "unique_KV_backing_bytes",
            "unique_scale_backing_bytes",
            "unique_metadata_backing_bytes",
            "serving_workspace_payload_bytes",
            "allocator_peak_bytes",
            "sampled_driver_used_bytes",
        ]
    }
