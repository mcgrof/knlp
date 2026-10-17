# SPDX-License-Identifier: MIT
"""Tie attention CUDA events to actual scheduler batches and model layers."""

import json
from contextlib import contextmanager
from pathlib import Path


def _union(intervals):
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def join_cuda_events(trace):
    """Use launch correlation for identity and GPU events for device duration."""
    events = trace["traceEvents"]
    ranges = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("name", "").startswith("knlp|")
    ]
    launches = {}
    groups = {}
    for event in events:
        if event.get("cat") != "cuda_runtime" or "correlation" not in event.get(
            "args", {}
        ):
            continue
        found = [
            r
            for r in ranges
            if r.get("pid") == event.get("pid")
            and r.get("tid") == event.get("tid")
            and r["ts"] <= event["ts"] < r["ts"] + r["dur"]
        ]
        if not found:
            continue
        selected = min(found, key=lambda x: x["dur"])
        launches[event["args"]["correlation"]] = (event, selected)
    unmatched = 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        identity = event.get("args", {}).get("correlation")
        if identity not in launches:
            unmatched += 1
            continue
        launch, selected = launches[identity]
        tag = selected["name"]
        group = groups.setdefault(
            tag,
            {
                "tag": tag,
                "host_range_us": selected["dur"],
                "kernels": [],
                "launches": {},
            },
        )
        group["kernels"].append(
            {
                "name": event["name"],
                "correlation": identity,
                "device_start_us": event["ts"],
                "device_duration_us": event["dur"],
            }
        )
        group["launches"][identity] = {
            "name": launch["name"],
            "host_duration_us": launch["dur"],
        }
    for group in groups.values():
        kernels = group["kernels"]
        group["device_kernel_sum_us"] = sum(e["device_duration_us"] for e in kernels)
        group["device_kernel_union_us"] = _union(
            [
                (e["device_start_us"], e["device_start_us"] + e["device_duration_us"])
                for e in kernels
            ]
        )
        group["host_cuda_runtime_us"] = sum(
            e["host_duration_us"] for e in group["launches"].values()
        )
        group["launches"] = list(group["launches"].values())
    return {
        "groups": list(groups.values()),
        "unmatched_cuda_kernels": unmatched,
        "matched_cuda_kernels": sum(len(x["kernels"]) for x in groups.values()),
        "host_ranges_are_not_device_durations": True,
        "component_results_descriptive": True,
    }


@contextmanager
def trace_attention_routes(path):
    """Add profiling ranges outside serving timers; restore every wrapped API."""
    import flashinfer
    import torch
    from vllm.v1.attention.backends.compact_prefill import CompactPrefillWorkspace
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = []
    layers = []
    records = []
    original_execute = GPUModelRunner.execute_model
    original_prepare = GPUModelRunner.prepare_inputs
    original_forward = FlashInferImpl.forward
    original_gather = CompactPrefillWorkspace.gather
    prefill_type = flashinfer.BatchPrefillWithPagedKVCacheWrapper
    decode_type = flashinfer.BatchDecodeWithPagedKVCacheWrapper
    original_prefill = prefill_type.run
    original_decode = decode_type.run

    def execute(runner, output, *args, **kwargs):
        bid = getattr(output, "_knlp_batch_id", None)
        if bid is None:
            raise RuntimeError("profile batch lacks scheduler identity")
        stack.append(
            {
                "batch_id": bid,
                "requests": None,
                "scheduled": dict(output.num_scheduled_tokens),
            }
        )
        try:
            with torch.profiler.record_function(f"knlp|batch|{bid}"):
                return original_execute(runner, output, *args, **kwargs)
        finally:
            stack.pop()

    def prepare(runner, *args, **kwargs):
        batch = original_prepare(runner, *args, **kwargs)
        stack[-1]["requests"] = list(batch.req_ids)
        return batch

    def forward(impl, layer, *args, **kwargs):
        layers.append(layer.layer_name)
        try:
            with torch.profiler.record_function(
                f'knlp|layer|{stack[-1]["batch_id"]}|{layer.layer_name}'
            ):
                return original_forward(impl, layer, *args, **kwargs)
        finally:
            layers.pop()

    def operation(kind, call, owner, query, cache, *args, **kwargs):
        if not stack or not layers:
            return call(owner, query, cache, *args, **kwargs)
        context = stack[-1]
        tag = f'knlp|{kind}|{context["batch_id"]}|{layers[-1]}'
        dtypes = (
            [str(x.dtype) for x in cache]
            if isinstance(cache, tuple)
            else [str(cache.dtype)] * 2
        )
        records.append(
            dict(
                tag=tag,
                **context,
                layer=layers[-1],
                query_tokens=len(query),
                query_dtype=str(query.dtype),
                cache_dtypes=dtypes,
                backend=str(getattr(owner, "_backend", None)),
            )
        )
        with torch.profiler.record_function(tag):
            return call(owner, query, cache, *args, **kwargs)

    def prefill(owner, query, cache, *args, **kwargs):
        return operation(
            "prefill", original_prefill, owner, query, cache, *args, **kwargs
        )

    def decode(owner, query, cache, *args, **kwargs):
        return operation(
            "decode", original_decode, owner, query, cache, *args, **kwargs
        )

    def gather(owner, cache, ids):
        if not stack or not layers:
            return original_gather(owner, cache, ids)
        context = stack[-1]
        tag = f'knlp|gather|{context["batch_id"]}|{layers[-1]}'
        records.append(dict(tag=tag, **context, layer=layers[-1], pages=ids.numel()))
        with torch.profiler.record_function(tag):
            return original_gather(owner, cache, ids)

    GPUModelRunner.execute_model = execute
    GPUModelRunner.prepare_inputs = prepare
    FlashInferImpl.forward = forward
    CompactPrefillWorkspace.gather = gather
    prefill_type.run = prefill
    decode_type.run = decode
    try:
        yield records
    finally:
        GPUModelRunner.execute_model = original_execute
        GPUModelRunner.prepare_inputs = original_prepare
        FlashInferImpl.forward = original_forward
        CompactPrefillWorkspace.gather = original_gather
        prefill_type.run = original_prefill
        decode_type.run = original_decode
        path.write_text(json.dumps(records, indent=2) + "\n")
