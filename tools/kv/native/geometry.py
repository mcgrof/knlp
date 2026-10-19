# SPDX-License-Identifier: MIT
"""Exercise the selected allocator and production writer before model scoring."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from .instrument import save, tensor_inventory
from .qualification import record_attention_check


def qualify(inputs, output, *, dtype=torch.float16, geometry=None):
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    import flashinfer

    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("unsupported native geometry dtype")
    bf16 = dtype == torch.bfloat16
    heads, query_heads, layers = (4, 28, 28) if bf16 else (8, 32, 36)
    dim = 128
    if geometry is not None:
        layers = geometry["layers"]
        query_heads = geometry["query_heads"]
        heads = geometry["kv_heads"]
        dim = geometry["head_dim"]
        if (
            min(layers, query_heads, heads) <= 0
            or query_heads % heads
            or dim not in (64, 128, 256)
            or (
                dim == 256
                and (dtype != torch.float16 or (query_heads, heads) != (16, 4))
            )
        ):
            raise ValueError("unsupported qualification geometry")
    tests = Path(inputs) / "vllm/tests/v1/worker/test_asym_kv_cache_allocation.py"
    spec = importlib.util.spec_from_file_location("candidate_allocation_test", tests)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    cpu_checks = []
    for name in sorted(n for n in vars(helpers) if n.startswith("test_")):
        getattr(helpers, name)()
        cpu_checks.append(name)
    # Use production allocator with multiple pages and a remapped partial tail.
    helper_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=heads,
        head_size=dim,
        dtype=dtype,
        v_dtype=torch.float8_e4m3fn,
    )
    cfg, raw = helpers._build_kv_cache_config(helper_spec)
    raw = {name: t.to("cuda") for name, t in raw.items()}
    cache = helpers._reshape(cfg, helper_spec, raw)[helpers.LAYER]
    k, v = cache
    slots = torch.tensor(
        list(range(48, 64)) + list(range(0, 16)) + [32],
        dtype=torch.int64,
        device="cuda",
    )
    values = torch.arange(33 * heads * dim, device="cuda", dtype=torch.float32).reshape(
        33, heads, dim
    )
    key = torch.sin(values / 91).to(dtype)
    value = (torch.cos(values / 79) / 4).to(dtype)
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1 / 64, device="cuda"),
    )
    impl = SimpleNamespace(kv_sharing_target_layer_name=None, _v_cache_str="fp8_e4m3")
    expected_v = (value.float() * 64).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    observed = []
    layer_caches = []
    for layer_index in range(layers):
        if layer_index:
            fresh = {name: torch.zeros_like(tensor) for name, tensor in raw.items()}
            cache = helpers._reshape(cfg, helper_spec, fresh)[helpers.LAYER]
            k, v = cache
        layer_caches.append(cache)
        key = torch.sin((values + 17 * layer_index) / 91).to(dtype)
        value = (torch.cos((values + 11 * layer_index) / 79) / 4).to(dtype)
        expected_v = (
            (value.float() * 64).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        )
        k.zero_()
        v.zero_()
        FlashInferImpl.do_kv_cache_update(impl, layer, key, value, cache, slots)
        torch.cuda.synchronize()
        actual_k = k.flatten(0, 1)[slots]
        actual_v = v.flatten(0, 1)[slots].float()
        if not torch.equal(actual_k, key) or not torch.equal(actual_v, expected_v):
            raise RuntimeError("production remapped K/V writer changed another plane")
        untouched = torch.ones(64, dtype=torch.bool, device="cuda")
        untouched[slots] = False
        if torch.count_nonzero(k.flatten(0, 1)[untouched]) or torch.count_nonzero(
            v.flatten(0, 1)[untouched].float()
        ):
            raise RuntimeError("writer changed an unassigned cache slot")
        observed.append(
            {
                "layer_instance": layer_index,
                "k_exact": True,
                "v_quantized_exact": True,
                "unassigned_slots_unchanged": True,
            }
        )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True, backend="auto"
    )
    wrapper.plan(
        torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        torch.tensor([3, 0, 2], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        query_heads,
        heads,
        dim,
        16,
        q_data_type=dtype,
        k_data_type=dtype,
        v_data_type=torch.float8_e4m3fn,
    )
    q = (
        torch.sin(
            torch.arange(query_heads * dim, device="cuda", dtype=torch.float32) / 23
        )
        .reshape(1, query_heads, dim)
        .to(dtype)
    )
    result = wrapper.run(q, cache, k_scale=1.0, v_scale=1 / 64)
    keys = key.float().repeat_interleave(query_heads // heads, dim=1)
    vals = (expected_v / 64).repeat_interleave(query_heads // heads, dim=1)
    weights = torch.einsum("bhd,thd->bht", q.float(), keys) / (dim**0.5)
    reference = torch.einsum("bht,thd->bhd", weights.softmax(-1), vals)
    error = (result.float() - reference).abs()
    fixture = {
        "query": q,
        "key": key,
        "value": value,
        "page_indptr": torch.tensor([0, 3], dtype=torch.int32),
        "page_indices": torch.tensor([3, 0, 2], dtype=torch.int32),
        "last_page_len": torch.tensor([1], dtype=torch.int32),
    }
    configuration = {
        "layers": layers,
        "layer_instance": layers - 1,
        "query_heads": query_heads,
        "kv_heads": heads,
        "head_dim": dim,
        "page_size": 16,
        "stored_tokens": 33,
        "k_scale": 1.0,
        "v_scale": 1 / 64,
        "dtype": str(dtype),
        "input_source": str(inputs),
    }
    record_attention_check(
        output,
        "geometry-decode",
        wrapper=wrapper,
        result=result,
        reference=reference,
        tensors=fixture,
        cache=cache,
        configuration=configuration,
    )
    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    prefill.plan(
        torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        torch.tensor([3, 0, 2], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        query_heads,
        heads,
        dim,
        16,
        causal=True,
        q_data_type=dtype,
        k_data_type=dtype,
        v_data_type=torch.float8_e4m3fn,
    )
    queries = (
        torch.sin(
            torch.arange(3 * query_heads * dim, device="cuda", dtype=torch.float32) / 23
        )
        .reshape(3, query_heads, dim)
        .to(dtype)
    )
    pref_result = prefill.run(queries, cache, k_scale=1.0, v_scale=1 / 64)
    logits = torch.einsum("qhd,thd->hqt", queries.float(), keys) / (dim**0.5)
    allowed = (
        torch.arange(33, device="cuda")[None, :]
        <= torch.arange(30, 33, device="cuda")[:, None]
    )
    logits.masked_fill_(~allowed[None, :, :], -torch.inf)
    pref_reference = torch.einsum("hqt,thd->qhd", logits.softmax(-1), vals)
    pref_error = (pref_result.float() - pref_reference).abs()
    record_attention_check(
        output,
        "geometry-prefill",
        wrapper=prefill,
        result=pref_result,
        reference=pref_reference,
        tensors={**fixture, "query": queries},
        cache=cache,
        configuration=configuration,
    )
    info = tensor_inventory(cache)
    if info[0]["storage_pointer"] != info[1]["storage_pointer"]:
        raise RuntimeError("candidate allocator copied the cache planes")
    save(
        Path(output) / "geometry.json",
        {
            "passed": True,
            "geometry": {
                "layers": layers,
                "query_heads": query_heads,
                "kv_heads": heads,
                "head_dim": dim,
                "block_size": 16,
            },
            "storage": info,
            "all_layer_storage": [tensor_inventory(cache) for cache in layer_caches],
            "block_indices": [3, 0, 2],
            "last_page_len": 1,
            "stored_tokens": 33,
            "k_scale": 1.0,
            "v_scale": 1 / 64,
            "production_writer_checks": observed,
            "candidate_cpu_allocation_checks": cpu_checks,
            "fp32_attention_max_absolute_error": float(error.max()),
            "fp32_attention_mean_absolute_error": float(error.mean()),
            "backend": str(getattr(wrapper, "_backend", None)),
            "prefill_backend": str(getattr(prefill, "_backend", None)),
            "prefill_fp32_max_absolute_error": float(pref_error.max()),
            "prefill_fp32_mean_absolute_error": float(pref_error.mean()),
        },
    )
