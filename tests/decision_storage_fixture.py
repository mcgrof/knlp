# SPDX-License-Identifier: MIT
"""Small retained CUDA-layout metadata; these fixtures allocate no GPU memory."""

import math


def memory_fixture(arm="AF"):
    geometry = {"layers": 1, "block_size": 16, "kv_heads": 1, "head_dim": 2}

    def tensor(shape, dtype, size, address, backing, offset=0, strides=None):
        strides = strides if strides is not None else [32, 2, 2, 1]
        return {
            "shape": shape,
            "dtype": dtype,
            "element_size": size,
            "storage_pointer": address,
            "storage_bytes": backing,
            "storage_offset": offset,
            "tensor_pointer": address + offset * size,
            "strides": strides,
            "device": "cuda:0",
            "logical_bytes": math.prod(shape) * size,
        }

    if arm == "AF":
        k = tensor([2, 16, 1, 2], "torch.float16", 2, 4096, 128)
        v = tensor([2, 16, 1, 2], "torch.uint8", 1, 8192, 64)
    else:
        size, dtype = (1, "torch.uint8") if arm == "SF" else (2, "torch.float16")
        k = tensor([2, 16, 1, 2], dtype, size, 4096, 128 * size, strides=[64, 2, 2, 1])
        v = tensor(
            [2, 16, 1, 2],
            dtype,
            size,
            4096,
            128 * size,
            offset=32,
            strides=[64, 2, 2, 1],
        )
    scales = [
        tensor([], "torch.float32", 4, 16384 + 512 * i, 4, strides=[]) for i in range(3)
    ]

    def allocation(tensors):
        unique = {t["storage_pointer"]: t["storage_bytes"] for t in tensors}
        rows = [
            {
                "device": 0,
                "address": a,
                "requested_bytes": n,
                "allocated_block_bytes": 512,
                "rounding_bytes": 512 - n,
            }
            for a, n in unique.items()
        ]
        return {
            "allocations": rows,
            "allocation_count": len(rows),
            **{
                k: sum(r[k] for r in rows)
                for k in ("requested_bytes", "allocated_block_bytes", "rounding_bytes")
            },
        }

    combined = allocation([k, v, *scales])
    return geometry, {
        "arm": arm,
        "physical_blocks_per_layer": 2,
        "token_slots_per_layer": 32,
        "layers": [{"layer": "model.layers.0.attn", "K": k, "V": v, "scales": scales}],
        "K_payload_bytes": k["logical_bytes"],
        "V_payload_bytes": v["logical_bytes"],
        "KV": allocation([k, v]),
        "scales_including_native_Q_constant": allocation(scales),
        "persistent_combined": combined,
        "persistent_budget_bytes": 4096,
        "budget_slack_bytes": 4096 - combined["allocated_block_bytes"],
        "workspace": None,
        "whole_process": {"scope": "synthetic CPU fixture"},
    }
