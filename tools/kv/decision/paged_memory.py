# SPDX-License-Identifier: MIT
"""Count live paged-cache storage and CUDA allocator rounding separately.

The caller must also verify actual worker read/write formats. An allocation
inventory alone cannot establish that an attention kernel consumed the cache.
"""

import math


def audit_record(record, arm, geometry, persistent_budget_bytes, *, equal_blocks=None):
    """Recompute retained allocation totals on CPU without inferring allocator pools."""
    if (
        record["arm"] != arm
        or record["persistent_budget_bytes"] != persistent_budget_bytes
    ):
        raise ValueError("Retained storage identity or budget differs")
    layers = record["layers"]
    if len(layers) != geometry["layers"] or len({r["layer"] for r in layers}) != len(
        layers
    ):
        raise ValueError("Missing or repeated attention layer storage")
    blocks = record["physical_blocks_per_layer"]
    if (
        type(blocks) is not int
        or blocks <= 0
        or equal_blocks is not None
        and blocks != equal_blocks
    ):
        raise ValueError("Physical block geometry differs")
    if record["token_slots_per_layer"] != blocks * geometry["block_size"]:
        raise ValueError("Token slots differ from actual blocks")
    shape = [blocks, geometry["block_size"], geometry["kv_heads"], geometry["head_dim"]]
    payload, scales = [], []
    for layer in layers:
        for side in ("K", "V"):
            tensor = layer[side]
            compressed = arm == "SF" or arm == "AF" and side == "V"
            dtypes = (
                {"torch.float8_e4m3fn", "torch.uint8"}
                if compressed
                else {"torch.float16"}
            )
            if (
                tensor["shape"] != shape
                or tensor["dtype"] not in dtypes
                or tensor["element_size"] != (1 if compressed else 2)
            ):
                raise ValueError("Physical plane dtype or geometry differs")
            payload.append(tensor)
        if len(layer["scales"]) != 3 or any(
            s["dtype"] != "torch.float32"
            or s["element_size"] != 4
            or s["logical_bytes"] != 4
            for s in layer["scales"]
        ):
            raise ValueError("Missing native-Q or component FP32 scale")
        scales.extend(layer["scales"])
    for tensor in payload + scales:
        if (
            not tensor["device"].startswith("cuda:")
            or tensor["logical_bytes"]
            != math.prod(tensor["shape"]) * tensor["element_size"]
            or tensor["tensor_pointer"]
            != tensor["storage_pointer"]
            + tensor["storage_offset"] * tensor["element_size"]
            or len(tensor["shape"]) != len(tensor["strides"])
            or any(x < 0 for x in tensor["strides"])
            or tensor["storage_offset"] < 0
        ):
            raise ValueError("Invalid retained tensor storage geometry")
        last = tensor["storage_offset"] + sum(
            (n - 1) * stride for n, stride in zip(tensor["shape"], tensor["strides"])
        )
        if (last + 1) * tensor["element_size"] > tensor["storage_bytes"]:
            raise ValueError("Tensor view exceeds its backing allocation")
    for key, tensors in (
        ("KV", payload),
        ("scales_including_native_Q_constant", scales),
        ("persistent_combined", payload + scales),
    ):
        expected = {}
        for tensor in tensors:
            address = (int(tensor["device"].split(":")[1]), tensor["storage_pointer"])
            if address in expected and expected[address] != tensor["storage_bytes"]:
                raise ValueError("Aliased backing sizes disagree")
            expected[address] = tensor["storage_bytes"]
        rows = record[key]["allocations"]
        observed = {(r["device"], r["address"]): r["requested_bytes"] for r in rows}
        if len(observed) != len(rows) or observed != expected:
            raise ValueError("Allocation inventory differs from unique backing storage")
        if any(
            r["allocated_block_bytes"] < r["requested_bytes"]
            or r["rounding_bytes"] != r["allocated_block_bytes"] - r["requested_bytes"]
            for r in rows
        ):
            raise ValueError("Invalid CUDA block rounding")
        if record[key]["allocation_count"] != len(rows) or any(
            record[key][field] != sum(r[field] for r in rows)
            for field in ("requested_bytes", "allocated_block_bytes", "rounding_bytes")
        ):
            raise ValueError("Retained allocation totals differ")
    if record["KV"]["requested_bytes"] != sum(t["logical_bytes"] for t in payload):
        raise ValueError("KV backing includes oversized persistent storage")
    if (
        record["persistent_combined"]["allocation_count"]
        != record["KV"]["allocation_count"]
        + record["scales_including_native_Q_constant"]["allocation_count"]
    ):
        raise ValueError("Scales and KV unexpectedly alias")
    for side in ("K", "V"):
        if record[side + "_payload_bytes"] != sum(
            r[side]["logical_bytes"] for r in layers
        ):
            raise ValueError("Plane payload total differs")
    used = record["persistent_combined"]["allocated_block_bytes"]
    if (
        used > persistent_budget_bytes
        or record["budget_slack_bytes"] != persistent_budget_bytes - used
    ):
        raise ValueError("Persistent allocation exceeds budget or slack differs")
    if arm != "AF" and record["workspace"] is not None:
        raise ValueError("Control unexpectedly retains asymmetric staging")
    return {
        "arm": arm,
        "physical_blocks_per_layer": blocks,
        "token_slots_per_layer": blocks * geometry["block_size"],
        "K_payload_bytes": record["K_payload_bytes"],
        "V_payload_bytes": record["V_payload_bytes"],
        "persistent_allocated_bytes": used,
        "budget_slack_bytes": persistent_budget_bytes - used,
        "scale_allocated_bytes": record["scales_including_native_Q_constant"][
            "allocated_block_bytes"
        ],
        "workspace": record["workspace"],
        "whole_process": record["whole_process"],
    }


def allocation_sizes(storages, segments):
    """Match unique live storage pointers to pinned Torch allocator blocks.

    ``memory_snapshot`` block sizes include allocator rounding. Segment sizes
    include inactive space and must not be attributed wholesale to one tensor.
    Fail closed for storage outside the captured allocator (for example custom
    CUDA allocations), rather than reporting requested bytes as allocated bytes.
    """
    blocks = {}
    for segment in segments:
        address = segment["address"]
        for block in segment["blocks"]:
            if block.get("address", address) != address:
                raise ValueError("Allocator block address disagrees with segment")
            if block["state"] == "active_allocated":
                key = (segment["device"], address)
                if key in blocks:
                    raise ValueError("Repeated allocator block")
                blocks[key] = block
            address += block["size"]
        if address - segment["address"] != segment["total_size"]:
            raise ValueError("Allocator blocks do not span their segment")
    unique = {}
    for storage in storages:
        if not storage["device"].startswith("cuda:"):
            raise ValueError("Expected CUDA storage")
        key = (int(storage["device"].split(":")[1]), storage["storage_pointer"])
        requested = storage["storage_bytes"]
        if key in unique:
            if unique[key]["requested_bytes"] != requested:
                raise ValueError("Aliased storage sizes disagree")
            continue
        if key not in blocks:
            raise ValueError("Storage is absent from live CUDA allocator snapshot")
        block = blocks[key]
        if block["requested_size"] != requested or block["size"] < requested:
            raise ValueError("Storage bytes differ from allocator request")
        unique[key] = {
            "device": key[0],
            "address": key[1],
            "requested_bytes": requested,
            "allocated_block_bytes": block["size"],
            "rounding_bytes": block["size"] - requested,
        }
    return {
        "allocations": list(unique.values()),
        "allocation_count": len(unique),
        "requested_bytes": sum(row["requested_bytes"] for row in unique.values()),
        "allocated_block_bytes": sum(
            row["allocated_block_bytes"] for row in unique.values()
        ),
        "rounding_bytes": sum(row["rounding_bytes"] for row in unique.values()),
    }


def inventory(worker, arm, *, geometry, persistent_budget_bytes):
    import torch

    from tools.kv.native.instrument import native_memory_state, tensor_inventory

    if arm not in ["AF", "SF", "native_fi", "native_fa3"]:
        raise ValueError("Unsupported paged-worker recipe")
    model = worker.model_runner.get_model()
    layers, payload, scales = [], [], []
    for name, module in model.named_modules():
        if not hasattr(module, "_k_scale") or not hasattr(module, "kv_cache"):
            continue
        cache = module.kv_cache
        impl = type(module.impl).__name__
        if arm == "AF":
            if not isinstance(cache, tuple) or len(cache) != 2:
                raise ValueError("Asymmetric cache must expose two physical planes")
            k, v = cache
        else:
            if not isinstance(cache, torch.Tensor) or cache.ndim != 5:
                raise ValueError(
                    "Symmetric/native cache must be a five-dimensional tensor"
                )
            if impl == "FlashInferImpl" and cache.shape[1] == 2:
                k, v = cache[:, 0], cache[:, 1]
            elif impl == "FlashAttentionImpl" and cache.shape[0] == 2:
                k, v = cache[0], cache[1]
            else:
                raise ValueError("Unqualified backend cache layout")
        ki, vi = tensor_inventory(k)[0], tensor_inventory(v)[0]
        native = {"torch.float16"}
        fp8 = {"torch.float8_e4m3fn", "torch.uint8"}
        if ki["dtype"] not in (fp8 if arm == "SF" else native) or vi["dtype"] not in (
            fp8 if arm in ["AF", "SF"] else native
        ):
            raise ValueError("Physical K/V dtype differs from selected recipe")
        if ki["shape"] != vi["shape"] or ki["shape"][1:] != [
            geometry["block_size"],
            geometry["kv_heads"],
            geometry["head_dim"],
        ]:
            raise ValueError("Physical K/V geometry differs")
        side_scales = tensor_inventory(
            (module._q_scale, module._k_scale, module._v_scale)
        )
        if len(side_scales) != 3 or any(
            s["dtype"] != "torch.float32" or s["logical_bytes"] != 4
            for s in side_scales
        ):
            raise ValueError("Expected three scalar FP32 scale allocations per layer")
        layers.append(
            {
                "layer": name,
                "implementation": impl,
                "K": ki,
                "V": vi,
                "scales": side_scales,
            }
        )
        payload.extend([ki, vi])
        scales.extend(side_scales)
    if len(layers) != geometry["layers"]:
        raise ValueError("Attention layer inventory is incomplete")
    block_counts = {row["K"]["shape"][0] for row in layers}
    if len(block_counts) != 1:
        raise ValueError("Layer cache block counts differ")
    # Synchronize only outside measured request intervals.
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory_snapshot()
    kv = allocation_sizes(payload, snapshot)
    sf = allocation_sizes(scales, snapshot)
    combined = allocation_sizes([*payload, *scales], snapshot)
    if kv["requested_bytes"] != sum(r["logical_bytes"] for r in payload):
        raise ValueError("Paged-cache backing includes unused or oversized storage")
    if combined["allocation_count"] != kv["allocation_count"] + sf["allocation_count"]:
        raise ValueError("KV and scale allocations unexpectedly alias")
    if combined["allocated_block_bytes"] > persistent_budget_bytes:
        raise ValueError("Live KV plus scales exceeds the common persistent budget")
    workspace = getattr(worker.vllm_config, "_compact_prefill_workspace", None)
    if arm != "AF" and workspace is not None:
        raise ValueError("Unexpected asymmetric staging workspace in control")
    blocks = block_counts.pop()
    return {
        "arm": arm,
        "layers": layers,
        "physical_blocks_per_layer": blocks,
        "token_slots_per_layer": blocks * geometry["block_size"],
        "K_payload_bytes": sum(r["K"]["logical_bytes"] for r in layers),
        "V_payload_bytes": sum(r["V"]["logical_bytes"] for r in layers),
        "KV": kv,
        "scales_including_native_Q_constant": sf,
        "persistent_combined": combined,
        "persistent_budget_bytes": persistent_budget_bytes,
        "budget_slack_bytes": persistent_budget_bytes
        - combined["allocated_block_bytes"],
        "workspace": None if workspace is None else workspace.receipt(),
        "whole_process": native_memory_state(worker),
        "scope": "Live unique KV and scale allocator blocks; inactive allocator pools, model weights, graph pools and staging are outside this persistent-cache budget and reported separately. Actual kernel read/write qualification is required independently.",
    }
