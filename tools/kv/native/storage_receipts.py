# SPDX-License-Identifier: MIT
"""Report physical K/V planes and deduplicate backing allocations per arm."""

from .instrument import tensor_inventory, native_memory_state


def inventory(worker, arm, *, geometry=None):
    # The default preserves the established Qwen3 recipe. Transfer callers must
    # supply their independently pinned conventional-attention geometry.
    geometry = geometry or {"layers": 36, "kv_heads": 8, "head_dim": 128}
    model = worker.model_runner.get_model()
    layers = []
    storages = {}
    scales = {}
    for name, module in model.named_modules():
        if not hasattr(module, "_k_scale") or not hasattr(module, "kv_cache"):
            continue
        cache = module.kv_cache
        if isinstance(cache, tuple):
            k, v = cache
        elif cache.shape[0] == 2:
            k, v = cache[0], cache[1]
        else:
            assert cache.shape[1] == 2
            k, v = cache[:, 0], cache[:, 1]
        ki, vi = tensor_inventory(k)[0], tensor_inventory(v)[0]
        assert ki["dtype"] == "torch.float16" and vi["dtype"] == (
            "torch.float16" if arm == "k16_v16" else "torch.float8_e4m3fn"
        )
        assert ki["shape"] == vi["shape"] and ki["shape"][1:] == [
            16,
            geometry["kv_heads"],
            geometry["head_dim"],
        ]
        for plane in [ki, vi]:
            key = (plane["device"], plane["storage_pointer"])
            assert key not in storages or storages[key] == plane["storage_bytes"]
            storages[key] = plane["storage_bytes"]
        scale = tensor_inventory((module._q_scale, module._k_scale, module._v_scale))
        for tensor in scale:
            scales[(tensor["device"], tensor["storage_pointer"])] = tensor[
                "storage_bytes"
            ]
        layers.append(
            dict(
                layer=name,
                K=ki,
                V=vi,
                physical_blocks=ki["shape"][0],
                logical_KV_tokens=16 * ki["shape"][0],
                scale_buffers=scale,
            )
        )
    assert len(layers) == geometry["layers"]
    state = native_memory_state(worker)
    metadata = {}

    def collect(value):
        if isinstance(value, dict):
            if all(k in value for k in ["device", "storage_pointer", "storage_bytes"]):
                metadata[(value["device"], value["storage_pointer"])] = value[
                    "storage_bytes"
                ]
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(state["metadata_tensors"])
    workspace = getattr(worker.vllm_config, "_compact_prefill_workspace", None)
    workspace_receipt = None if workspace is None else workspace.receipt()
    if arm == "k16_v16":
        assert workspace is None
    else:
        assert workspace_receipt["payload_capacity_bytes"] == (
            4096 * 16 * geometry["kv_heads"] * geometry["head_dim"] * 4
        )
    kv_total = sum(storages.values())
    K = sum(r["K"]["logical_bytes"] for r in layers)
    V = sum(r["V"]["logical_bytes"] for r in layers)
    assert kv_total == K + V
    return dict(
        arm=arm,
        layers=layers,
        physical_blocks_per_layer=layers[0]["physical_blocks"],
        logical_KV_tokens_per_layer=layers[0]["logical_KV_tokens"],
        K_payload_bytes=K,
        V_payload_bytes=V,
        unique_KV_backing_bytes=kv_total,
        unique_KV_allocations=len(storages),
        unique_scale_backing_bytes=sum(scales.values()),
        unique_metadata_backing_bytes=sum(metadata.values()),
        serving_workspace_payload_bytes=(
            0 if workspace is None else workspace_receipt["payload_capacity_bytes"]
        ),
        workspace=workspace_receipt,
        allocator_peak_bytes=state["peak_reserved_since_reset"],
        sampled_driver_used_bytes=state["cuda_device_used_bytes"],
        state=state,
        deduplicated_by="device and untyped storage pointer; K/V aliases counted once",
        same_pool_does_not_imply_total_allocation_reduction=True,
        not_a_capacity_sweep=True,
    )
