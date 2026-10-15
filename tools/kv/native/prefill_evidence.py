# SPDX-License-Identifier: MIT
"""Capture paged prefill dispatch and bound transient expansion storage."""

import gc
import hashlib
import json
from pathlib import Path

import torch

from .prefill import build, compare_close, fixture, measure, save, tensor_info


def split_dispatch_trace(trace, specs):
    """Join synchronized operation regions to their complete device kernels."""
    result = []
    for spec in specs:
        regions = [
            e
            for e in trace["traceEvents"]
            if e.get("name") == spec["region"]
            and e.get("cat") == "user_annotation"
            and e.get("ph") == "X"
        ]
        if len(regions) != 1:
            raise ValueError("expected one synchronized region: " + spec["region"])
        region = regions[0]
        start, end = region["ts"], region["ts"] + region["dur"]
        kernels = [
            e
            for e in trace["traceEvents"]
            if e.get("cat") == "kernel"
            and start <= e["ts"]
            and e["ts"] + e.get("dur", 0) <= end
        ]
        family = (
            "BatchPrefillWithPagedKVCacheKernel<"
            if spec["backend"] == "fa2"
            else "PrefillWithKVCacheKernel<"
        )
        attention = [e for e in kernels if family in e["name"]]
        result.append(
            dict(
                **spec,
                kernel_count=len(kernels),
                attention_kernel_count=len(attention),
                expected_family=family,
                passed=bool(attention),
                kernel_names=[e["name"] for e in kernels],
                trace={**trace, "traceEvents": [region] + kernels},
            )
        )
    return result


def _load(fixtures, dtype, layer):
    path = Path(fixtures) / dtype / f"layer-{layer}.pt"
    metadata = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("fixture hash differs: " + str(path))
    return torch.load(path, weights_only=True, map_location="cpu"), metadata


def profile_dispatch(fixtures, destination, config):
    """Profile explicit backends in one session after unprofiled measurements."""
    import flashinfer

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    specs = []
    # One CUPTI session avoids relying on repeated profiler activation.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        acc_events=True,
    ) as prof:
        for dtype in ["mixed", "native"]:
            for layer in config["layers"]:
                data, meta = _load(fixtures, dtype, layer)
                for batch in [1, 8]:
                    q, k, v, metadata = fixture(data, batch)
                    for backend in ["fa3", "fa2"]:
                        wrapper, plan, out, lse, setup_seconds = build(
                            q, k, v, metadata, backend
                        )
                        for _ in range(config["warmup"]):
                            wrapper.run(
                                q,
                                (k, v),
                                v_scale=meta["scale"],
                                out=out,
                                lse=lse,
                                return_lse=True,
                            )
                        torch.cuda.synchronize()
                        name = f"paged-prefill::{dtype}::{layer}::{batch}::{backend}"
                        with torch.profiler.record_function(name):
                            wrapper.run(
                                q,
                                (k, v),
                                v_scale=meta["scale"],
                                out=out,
                                lse=lse,
                                return_lse=True,
                            )
                            torch.cuda.synchronize()
                        specs.append(
                            dict(
                                region=name,
                                dtype=dtype,
                                layer=layer,
                                batch=batch,
                                backend=backend,
                                selected_backend=wrapper._backend,
                                fixture_sha256=meta["sha256"],
                                q=tensor_info(q),
                                k=tensor_info(k),
                                v=tensor_info(v),
                                scale=meta["scale"],
                                first_plan_seconds=setup_seconds,
                                module=repr(wrapper._cached_module),
                                float_workspace=tensor_info(
                                    wrapper._float_workspace_buffer
                                ),
                                int_workspace=tensor_info(
                                    wrapper._int_workspace_buffer
                                ),
                                flashinfer_version=flashinfer.__version__,
                            )
                        )
                        del wrapper, plan, out, lse
                    del q, k, v, metadata
                    gc.collect()
                    torch.cuda.empty_cache()
                del data
    full = destination / "dispatch-full.json"
    prof.export_chrome_trace(str(full))
    save(destination / "dispatch-specs.json", specs)
    records = split_dispatch_trace(json.loads(full.read_text()), specs)
    for record in records:
        name = "trace-{dtype}-{layer}-{batch}-{backend}.json".format(**record)
        save(destination / name, record.pop("trace"))
        record["trace_path"] = name
        record["trace_sha256"] = hashlib.sha256(
            (destination / name).read_bytes()
        ).hexdigest()
    result = dict(
        passed=len(records) == len(config["layers"]) * 8
        and all(r["passed"] and r["backend"] == r["selected_backend"] for r in records),
        profiler_sessions=1,
        cells=records,
        full_trace_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
        scope="Separate synchronized dispatch calls; no profiled duration enters a timing verdict.",
    )
    save(destination / "dispatch-evidence.json", result)
    return result


def scratch_view(storage, value):
    """Expose actual read pages from a bounded FP16 scratch allocation."""
    if storage.dtype != torch.float16 or storage.ndim != 1:
        raise ValueError("scratch must be a flat FP16 allocation")
    if value.numel() > storage.numel():
        raise ValueError("read pages exceed scratch capacity")
    return storage[: value.numel()].view(value.shape)


def _expansion_case(fixtures, destination, config, layer, batch, storage):
    data, meta = _load(fixtures, "mixed", layer)
    q, k, v, metadata = fixture(data, batch)
    scratch = scratch_view(storage, v)
    torch.cuda.reset_peak_memory_stats()
    mixed, mplan, mout, mlse, mixed_setup = build(q, k, v, metadata, "fa3")
    native, nplan, nout, nlse, native_setup = build(q, k, scratch, metadata, "fa3")

    def baseline():
        mixed.run(q, (k, v), v_scale=meta["scale"], out=mout, lse=mlse, return_lse=True)

    def expanded():
        scratch.copy_(v)
        native.run(q, (k, scratch), out=nout, lse=nlse, return_lse=True)
        nout.mul_(meta["scale"])

    baseline()
    expanded()
    torch.cuda.synchronize()
    numerical = dict(output=compare_close(nout, mout), lse=compare_close(nlse, mlse))
    cpu = {"mixed": (mout.cpu(), mlse.cpu()), "expanded": (nout.cpu(), nlse.cpu())}
    cpu_numerical = dict(
        output=compare_close(cpu["expanded"][0], cpu["mixed"][0]),
        lse=compare_close(cpu["expanded"][1], cpu["mixed"][1]),
    )
    if numerical != cpu_numerical:
        raise ValueError("full GPU and CPU numerical counters differ")
    positions = [
        offset + t
        for offset in range(0, len(q), 8192)
        for t in [0, 1, 15, 16, 4095, 4096, 8190, 8191]
    ]
    samples = {
        name: tuple(t[positions] for t in tensors) for name, tensors in cpu.items()
    }
    torch.save(
        dict(positions=positions, outputs=samples),
        destination / f"reuse-samples-{layer}-{batch}.pt",
    )
    hashes = {
        name: [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in tensors]
        for name, tensors in cpu.items()
    }
    del cpu, samples
    records = []
    for block, order in enumerate(config["component_orders"]):
        for choice in order:
            operation, plan = (
                (baseline, mplan) if choice == "fa3" else (expanded, nplan)
            )
            rows = measure(operation, plan, config["warmup"], config["repetitions"])
            records.append(
                dict(
                    layer=layer,
                    batch=batch,
                    block=block,
                    operation=(
                        "mixed-fa3" if choice == "fa3" else "expand-native-fa3-scale"
                    ),
                    rows=rows,
                    numerical=numerical,
                    cpu_numerical=cpu_numerical,
                    full_tensor_sha256=hashes,
                    fixture_sha256=meta["sha256"],
                    scratch_bytes=scratch.numel() * scratch.element_size(),
                    scratch_storage_pointer=storage.untyped_storage().data_ptr(),
                    scratch_storage_bytes=storage.untyped_storage().nbytes(),
                    scratch_view_pointer=scratch.data_ptr(),
                    scratch_storage_offset=scratch.storage_offset(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    device_used_bytes=torch.cuda.device_memory_used(),
                    first_mixed_plan_seconds=mixed_setup,
                    first_native_plan_seconds=native_setup,
                    mixed_workspace=tensor_info(mixed._float_workspace_buffer),
                    native_workspace=tensor_info(native._float_workspace_buffer),
                )
            )
    return records


def expansion_reuse(fixtures, destination, config):
    """Reuse one actual-read V buffer across layers for each fixed batch."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    first, _ = _load(fixtures, "mixed", config["layers"][0])
    values_per_request = first["v"].numel()
    del first
    rows, lifetime = [], []
    for batch in [1, 8]:
        storage = torch.empty(
            batch * values_per_request, dtype=torch.float16, device="cuda"
        )
        pointer = storage.untyped_storage().data_ptr()
        for layer in config["layers"]:
            rows.extend(
                _expansion_case(fixtures, destination, config, layer, batch, storage)
            )
            # Case-local closures and wrappers have died before collecting this state.
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            lifetime.append(
                dict(
                    batch=batch,
                    layer=layer,
                    scratch_storage_pointer=pointer,
                    scratch_storage_bytes=storage.untyped_storage().nbytes(),
                    retained_allocated_bytes=torch.cuda.memory_allocated(),
                    retained_reserved_bytes=torch.cuda.memory_reserved(),
                    device_used_bytes=torch.cuda.device_memory_used(),
                )
            )
            save(destination / "expansion-reuse.json", rows)
            save(destination / "scratch-lifetime.json", lifetime)
        del storage
        gc.collect()
        torch.cuda.empty_cache()
    return rows
