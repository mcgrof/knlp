# SPDX-License-Identifier: MIT
"""Count live paged-cache storage and CUDA allocator rounding separately.

The caller must also verify actual worker read/write formats. An allocation
inventory alone cannot establish that an attention kernel consumed the cache.
"""

import math

from .production_numerics import planes


def audit_inline_layer(layer, geometry, blocks):
    """Check byte-addressed inline scales independently of a backend view."""
    page, heads, width = (geometry[k] for k in ("block_size", "kv_heads", "head_dim"))
    if width % 4 or layer.get("implementation") != "TritonAttentionImpl":
        raise ValueError("Dynamic cache requires the aligned Triton NHD layout")
    inline = layer.get("inline_scales", {})
    if set(inline) != {"K", "V"}:
        raise ValueError("Both dynamic scale planes are required")
    row = width + 4
    backing = 2 * blocks * page * heads * row
    base = layer["K"]["storage_pointer"]
    records = []
    for index, side in enumerate(("K", "V")):
        code, scale = layer[side], inline[side]
        code_offset = index * page * heads * row
        code_strides = [2 * page * heads * row, heads * row, row, 1]
        if (
            code["storage_offset"] != code_offset
            or code["strides"] != code_strides
            or scale["shape"] != [blocks, page, heads]
            or scale["dtype"] != "torch.float32"
            or scale["element_size"] != 4
            or scale["storage_offset"] != (code_offset + width) // 4
            or scale["strides"] != [s // 4 for s in code_strides[:3]]
            or any(
                t["storage_pointer"] != base
                or t["storage_bytes"] != backing
                or t["device"] != layer["K"]["device"]
                for t in (code, scale)
            )
            or scale["device"] != code["device"]
        ):
            raise ValueError(
                "Dynamic code/scale views do not span the declared backing"
            )
        records.append(scale)
    return records


def scratch_tensors(record, geometry):
    """Audit the separate segmented-softmax buffers, outside the KV budget."""
    builders = record["builders"]
    if len(builders) != 1:
        raise ValueError("Expected one Triton metadata builder for this cache group")
    builder = builders[0]
    if builder["sequence_threshold"] != 8 or builder["segments"] != 16:
        raise ValueError("Triton scratch threshold differs from frozen graph shapes")
    tensors = builder["tensors"]
    expected = [
        [8, geometry["query_heads"], 16, geometry["head_dim"]],
        [8, geometry["query_heads"], 16],
        [8, geometry["query_heads"], 16],
    ]
    if len(tensors) != 3 or any(
        tensor["shape"] != shape
        or tensor["dtype"] != "torch.float32"
        or tensor["element_size"] != 4
        or tensor["storage_offset"] != 0
        or tensor["strides"] != [math.prod(shape[i + 1 :]) for i in range(len(shape))]
        or tensor["logical_bytes"] != tensor["storage_bytes"]
        for tensor, shape in zip(tensors, expected)
    ):
        raise ValueError("Unexpected Triton softmax scratch allocation geometry")
    if len({(t["device"], t["storage_pointer"]) for t in tensors}) != 3:
        raise ValueError("Segmented-softmax buffers unexpectedly alias one another")
    return tensors


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
    payload, scales, inline = [], [], []
    for layer in layers:
        for side in ("K", "V"):
            tensor = layer[side]
            compressed = arm in ("SF", "SD") or arm == "AF" and side == "V"
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
        if arm == "SD":
            inline.extend(audit_inline_layer(layer, geometry, blocks))
        elif layer.get("inline_scales"):
            raise ValueError("Fixed/native cache unexpectedly contains inline scales")
    scratch = scratch_tensors(record["triton_scratch"], geometry) if arm == "SD" else []
    if arm != "SD" and record.get("triton_scratch") is not None:
        raise ValueError("Non-Triton worker unexpectedly reports Triton scratch")
    for tensor in payload + scales + inline + scratch:
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
    inventories = [
        ("KV", payload),
        ("scales_including_native_Q_constant", scales),
        ("persistent_combined", payload + scales),
    ]
    if scratch:
        if {t["device"] for t in scratch + payload + scales} != {payload[0]["device"]}:
            raise ValueError(
                "Attention storage spans more than the single worker device"
            )
        inventories.append(("triton_scratch", scratch))
        if {t["storage_pointer"] for t in scratch} & {
            t["storage_pointer"] for t in payload + scales
        }:
            raise ValueError("Attention scratch unexpectedly aliases persistent cache")
    for key, tensors in inventories:
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
    inline_bytes = sum(t["logical_bytes"] for t in inline)
    if record.get("inline_scale_logical_bytes", 0) != inline_bytes:
        raise ValueError("Inline scale payload total differs")
    if (
        record["KV"]["requested_bytes"]
        != sum(t["logical_bytes"] for t in payload) + inline_bytes
    ):
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
        "inline_scale_logical_bytes": inline_bytes,
        "triton_scratch_allocated_bytes": (
            record["triton_scratch"]["allocated_block_bytes"] if scratch else 0
        ),
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

    from .inline_scales import inline_fp8_views, verify_inline_scale_aliases

    if arm not in ["AF", "SF", "SD", "native_fi", "native_fa3"]:
        raise ValueError("Unsupported paged-worker recipe")
    model = worker.model_runner.get_model()
    layers, payload, scales, inline = [], [], [], []
    for name, module in model.named_modules():
        if not hasattr(module, "_k_scale") or not hasattr(module, "kv_cache"):
            continue
        cache = module.kv_cache
        impl = type(module.impl).__name__
        inline_record = {}
        if arm == "AF":
            if not isinstance(cache, tuple) or len(cache) != 2:
                raise ValueError("Asymmetric cache must expose two physical planes")
            k, v = cache
        elif arm == "SD":
            if impl != "TritonAttentionImpl":
                raise ValueError("Dynamic worker must use the frozen Triton backend")
            (k, v), dynamic = inline_fp8_views(cache, geometry)
            verify_inline_scale_aliases(
                cache,
                geometry,
                (module.impl._k_scale_cache, module.impl._v_scale_cache),
            )
            inline_record = dict(zip(("K", "V"), tensor_inventory(dynamic)))
            inline.extend(inline_record.values())
        else:
            if not isinstance(cache, torch.Tensor) or cache.ndim != 5:
                raise ValueError(
                    "Symmetric/native cache must be a five-dimensional tensor"
                )
            if impl not in ("FlashInferImpl", "FlashAttentionImpl"):
                raise ValueError("Unqualified backend cache layout")
            k, v = planes(cache, impl)
        ki, vi = tensor_inventory(k)[0], tensor_inventory(v)[0]
        native = {"torch.float16"}
        fp8 = {"torch.float8_e4m3fn", "torch.uint8"}
        if ki["dtype"] not in (fp8 if arm in ("SF", "SD") else native) or vi[
            "dtype"
        ] not in (fp8 if arm in ["AF", "SF", "SD"] else native):
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
                "inline_scales": inline_record,
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
    scratch = None
    if arm == "SD":
        builders = [
            builder
            for groups in worker.model_runner.attn_groups
            for group in groups
            for builder in group.metadata_builders
        ]
        builder_records = [
            {
                "sequence_threshold": b.seq_threshold_3D,
                "segments": b.num_par_softmax_segments,
                "tensors": tensor_inventory(
                    (b.softmax_segm_output, b.softmax_segm_max, b.softmax_segm_expsum)
                ),
            }
            for b in builders
        ]
        scratch = {"builders": builder_records}
        scratch.update(allocation_sizes(scratch_tensors(scratch, geometry), snapshot))
    kv = allocation_sizes(payload, snapshot)
    sf = allocation_sizes(scales, snapshot)
    combined = allocation_sizes([*payload, *scales], snapshot)
    inline_bytes = sum(r["logical_bytes"] for r in inline)
    if kv["requested_bytes"] != sum(r["logical_bytes"] for r in payload) + inline_bytes:
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
        "inline_scale_logical_bytes": inline_bytes,
        "triton_scratch": scratch,
        "KV": kv,
        "scales_including_native_Q_constant": sf,
        "persistent_combined": combined,
        "persistent_budget_bytes": persistent_budget_bytes,
        "budget_slack_bytes": persistent_budget_bytes
        - combined["allocated_block_bytes"],
        "workspace": None if workspace is None else workspace.receipt(),
        "whole_process": native_memory_state(worker),
        "scope": "Live unique KV and scale allocator blocks. Dynamic inline scales share KV backing and are counted once; scalar constants are separate. Inactive allocator pools, weights, graph pools and staging are outside this cache budget and reported separately. Actual kernel read/write qualification is required independently.",
    }
