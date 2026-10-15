# SPDX-License-Identifier: MIT
"""Retain model-derived paged prefill inputs and compare explicit backends."""

import hashlib
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_info(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        bytes=tensor.numel() * tensor.element_size(),
    )


def select_pages(tensor, indices):
    """Preserve encoded FP8 bytes through indexing kernels with byte support."""
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= tensor.shape[0]
    ):
        raise ValueError("page index is outside the cache page dimension")
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8).index_select(0, indices).view(tensor.dtype)
    return tensor.index_select(0, indices)


def unpack_cache(cache):
    """Expose NHD K/V planes without changing their storage or page strides."""
    if isinstance(cache, tuple) and len(cache) == 2:
        planes = cache
    elif isinstance(cache, torch.Tensor) and cache.ndim == 5 and cache.shape[1] == 2:
        planes = cache.unbind(1)
    else:
        raise ValueError("expected a K/V pair or a [pages, 2, page, heads, dim] cache")
    if any(plane.ndim != 4 for plane in planes) or planes[0].shape != planes[1].shape:
        raise ValueError("paged K/V planes must have the same four-dimensional shape")
    return planes


def install_fixture_capture(model, destination, layers=(0, 17, 35)):
    """Copy selected full single-request prefills before the attention call."""
    import flashinfer

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    backing = {
        unpack_cache(module.kv_cache)[0]
        .untyped_storage()
        .data_ptr(): int(name.split("layers.")[1].split(".")[0])
        for name, module in model.named_modules()
        if hasattr(module, "_v_scale") and hasattr(module, "kv_cache")
    }
    cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
    original = cls.run
    captured = set()

    def observed(wrapper, q, cache, *args, **kwargs):
        planes = unpack_cache(cache)
        layer = backing[planes[0].untyped_storage().data_ptr()]
        if layer in layers and layer not in captured and q.shape[0] == 8192:
            qo = wrapper._qo_indptr_buf.cpu().clone()
            ptr = wrapper._paged_kv_indptr_buf.cpu().clone()
            last = wrapper._paged_kv_last_page_len_buf.cpu().clone()
            if qo.tolist() != [0, 8192] or len(ptr) != 2 or last.tolist() != [16]:
                raise RuntimeError("capture requires a full 8192-token single prefill")
            indices = wrapper._paged_kv_indices_buf[: int(ptr[-1])].long()
            data = dict(
                q=q.detach().cpu().contiguous(),
                k=select_pages(planes[0], indices).detach().cpu().contiguous(),
                v=select_pages(planes[1], indices).detach().cpu().contiguous(),
            )
            target = destination / f"layer-{layer}.pt"
            torch.save(data, target)
            save(
                target.with_suffix(".json"),
                dict(
                    layer=layer,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    tensors={key: tensor_info(value) for key, value in data.items()},
                    original_cache={
                        str(i): tensor_info(t) for i, t in enumerate(planes)
                    },
                    original_page_indices=indices.cpu().tolist(),
                    scale=kwargs.get("v_scale"),
                    causal=wrapper._causal,
                    sm_scale=wrapper._sm_scale,
                    backend=wrapper._backend,
                    compact_fixture=True,
                    model_derived=True,
                    note="Captured after cache write; no fixture capture enters timing.",
                ),
            )
            captured.add(layer)
        return original(wrapper, q, cache, *args, **kwargs)

    cls.run = observed

    def restore():
        cls.run = original
        if captured != set(layers):
            raise RuntimeError(f"missing model fixtures: {set(layers) - captured}")

    return restore


def compare_close(actual, reference, rtol=1e-3, atol=1e-3):
    difference = (actual.float() - reference.float()).abs()
    allowed = atol + rtol * reference.float().abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return dict(
        passed=finite and bool((difference <= allowed).all()),
        finite=finite,
        rtol=rtol,
        atol=atol,
        max_absolute_error=float(difference.max()) if finite else None,
        violating_elements=int((difference > allowed).sum()),
        elements=actual.numel(),
    )


def fixture(data, batch, tail=False):
    """Replicate a captured request outside timing, with disjoint paged storage."""
    lengths = ([8192, 8191, 8187] * 3)[:batch] if tail else [8192] * batch
    if tail and batch == 1:
        lengths = [8191]
    q = torch.cat([data["q"][:length] for length in lengths]).cuda()
    k = data["k"].repeat(batch, 1, 1, 1).cuda()
    v = data["v"].view(torch.uint8).repeat(batch, 1, 1, 1).view(data["v"].dtype).cuda()
    qo = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int32)
    ptr = torch.arange(batch + 1, dtype=torch.int32) * 512
    # A deterministic permutation exercises page-index indirection in both arms.
    generator = torch.Generator().manual_seed(61015)
    order = torch.randperm(len(k), generator=generator)
    inverse = torch.argsort(order).to(torch.int32).cuda()
    k, v = select_pages(k, order.cuda()), select_pages(v, order.cuda())
    last = torch.tensor(
        [((length - 1) % 16) + 1 for length in lengths], dtype=torch.int32
    )
    return (
        q,
        k,
        v,
        dict(
            qo_indptr=qo,
            paged_kv_indptr=ptr,
            paged_kv_indices=inverse,
            paged_kv_last_page_len=last,
        ),
    )


def build(q, k, v, metadata, backend):
    import flashinfer

    workspace = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )

    def plan():
        wrapper.plan(
            **metadata,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim_qk=128,
            page_size=16,
            causal=True,
            sm_scale=1 / math.sqrt(128),
            q_data_type=q.dtype,
            k_data_type=k.dtype,
            v_data_type=v.dtype,
            o_data_type=torch.float16,
        )

    started = time.perf_counter()
    plan()
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    return wrapper, plan, out, lse, time.perf_counter() - started


def measure(operation, plan, warmup, repetitions):
    for _ in range(warmup):
        plan()
        operation()
    torch.cuda.synchronize()
    rows = []
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repetitions)
    ]
    for first, last in events:
        torch.cuda.synchronize()
        start = time.perf_counter()
        plan_start = time.perf_counter()
        plan()
        plan_ms = (time.perf_counter() - plan_start) * 1000
        first.record()
        operation()
        last.record()
        last.synchronize()
        rows.append(
            dict(
                inclusive_ms=(time.perf_counter() - start) * 1000,
                host_plan_ms=plan_ms,
                run_cuda_ms=first.elapsed_time(last),
            )
        )
    return rows


def screen(fixtures, destination, config):
    """Run a frozen component matrix; correctness failures cannot promote."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    results, numerical, compilation, dispatch = [], [], [], []
    for dtype in ["mixed", "native"]:
        for layer in config["layers"]:
            path = Path(fixtures) / dtype / f"layer-{layer}.pt"
            meta = json.loads(path.with_suffix(".json").read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
            data = torch.load(path, weights_only=True, map_location="cpu")
            scale = meta["scale"]
            for batch in [1, 8]:
                for tail in [False, True]:
                    q, k, v, metadata = fixture(data, batch, tail)
                    operations, plans, outputs = {}, {}, {}
                    keep = []
                    for backend in ["fa3", "fa2"]:
                        wrapper, plan, out, lse, compile_s = build(
                            q, k, v, metadata, backend
                        )
                        keep.append(wrapper)
                        operation = lambda w=wrapper, o=out, l=lse, q=q, k=k, v=v, scale=scale: w.run(
                            q, (k, v), v_scale=scale, out=o, lse=l, return_lse=True
                        )
                        operation()
                        torch.cuda.synchronize()
                        assert wrapper._backend == backend
                        operations[backend], plans[backend], outputs[backend] = (
                            operation,
                            plan,
                            (out, lse),
                        )
                        compilation.append(
                            dict(
                                dtype=dtype,
                                layer=layer,
                                batch=batch,
                                tail=tail,
                                backend=backend,
                                first_plan_seconds=compile_s,
                            )
                        )
                        dispatch.append(
                            dict(
                                dtype=dtype,
                                layer=layer,
                                batch=batch,
                                tail=tail,
                                backend=wrapper._backend,
                                module=repr(wrapper._cached_module),
                                q=tensor_info(q),
                                k=tensor_info(k),
                                v=tensor_info(v),
                                scale=scale,
                                float_workspace=tensor_info(
                                    wrapper._float_workspace_buffer
                                ),
                                int_workspace=tensor_info(
                                    wrapper._int_workspace_buffer
                                ),
                            )
                        )
                    checks = dict(
                        dtype=dtype,
                        layer=layer,
                        batch=batch,
                        tail=tail,
                        output=compare_close(outputs["fa2"][0], outputs["fa3"][0]),
                        lse=compare_close(outputs["fa2"][1], outputs["fa3"][1]),
                    )
                    cpu_outputs = {
                        name: tuple(t.cpu() for t in output)
                        for name, output in outputs.items()
                    }
                    checks["cpu_output"] = compare_close(
                        cpu_outputs["fa2"][0], cpu_outputs["fa3"][0]
                    )
                    checks["cpu_lse"] = compare_close(
                        cpu_outputs["fa2"][1], cpu_outputs["fa3"][1]
                    )
                    assert (
                        checks["output"] == checks["cpu_output"]
                        and checks["lse"] == checks["cpu_lse"]
                    )
                    checks["full_tensor_sha256"] = {
                        name: [
                            hashlib.sha256(t.numpy().tobytes()).hexdigest()
                            for t in output
                        ]
                        for name, output in cpu_outputs.items()
                    }
                    positions = []
                    for start, end in zip(
                        metadata["qo_indptr"][:-1].tolist(),
                        metadata["qo_indptr"][1:].tolist(),
                    ):
                        positions.extend(
                            start + x
                            for x in [
                                0,
                                1,
                                15,
                                16,
                                4095,
                                4096,
                                end - start - 2,
                                end - start - 1,
                            ]
                        )
                    torch.save(
                        dict(
                            positions=positions,
                            outputs={
                                name: tuple(t[positions].clone() for t in output)
                                for name, output in cpu_outputs.items()
                            },
                        ),
                        destination
                        / f"output-samples-{dtype}-{layer}-{batch}-{int(tail)}.pt",
                    )
                    del cpu_outputs
                    if dtype == "mixed":
                        scale_checks = {}
                        for backend in ["fa3", "fa2"]:
                            original_output = tuple(x.clone() for x in outputs[backend])
                            os.environ["FLASHINFER_FUSE_V_SCALE"] = "0"
                            operations[backend]()
                            scale_checks[backend] = all(
                                torch.equal(x, y)
                                for x, y in zip(original_output, outputs[backend])
                            )
                            os.environ["FLASHINFER_FUSE_V_SCALE"] = "1"
                            operations[backend]()
                        checks["exact_scale_fusion"] = scale_checks
                    numerical.append(checks)
                    if not tail:
                        for block, order in enumerate(config["component_orders"]):
                            for backend in order:
                                rows = measure(
                                    operations[backend],
                                    plans[backend],
                                    config["warmup"],
                                    config["repetitions"],
                                )
                                results.append(
                                    dict(
                                        dtype=dtype,
                                        layer=layer,
                                        batch=batch,
                                        block=block,
                                        backend=backend,
                                        rows=rows,
                                    )
                                )
                        # Profile separate calls, after all unprofiled measurements.
                        for backend in ["fa3", "fa2"]:
                            with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                            ) as prof:
                                operations[backend]()
                                torch.cuda.synchronize()
                            prof.export_chrome_trace(
                                str(
                                    destination
                                    / f"trace-{dtype}-{layer}-{batch}-{backend}.json"
                                )
                            )
                    save(destination / "numerical.json", numerical)
                    save(destination / "timings.json", results)
                    save(destination / "compilation.json", compilation)
                    save(destination / "dispatch.json", dispatch)
                    del operations, plans, outputs, keep, q, k, v
                    torch.cuda.empty_cache()
    ratios = []
    for block in range(3):
        sums = {
            backend: sum(
                statistics.median(x["inclusive_ms"] for x in row["rows"])
                for row in results
                if row["dtype"] == "mixed"
                and row["batch"] == 8
                and row["block"] == block
                and row["backend"] == backend
            )
            for backend in ["fa2", "fa3"]
        }
        ratios.append(sums["fa3"] / sums["fa2"])
    numerical_passed = all(
        row[kind]["passed"] for row in numerical for kind in ["output", "lse"]
    )
    numerical_passed = numerical_passed and all(
        all(row.get("exact_scale_fusion", {}).values()) for row in numerical
    )
    passed = numerical_passed and min(ratios) > 1 and statistics.median(ratios) >= 1.01
    decision = dict(
        passed=passed,
        numerical_passed=numerical_passed,
        ratios=ratios,
        median=statistics.median(ratios),
        baseline="FA3",
        candidate="FA2",
        primary="sum of three layer median inclusive prefill times, replicated B8, length8192",
        batch_replication="Each B8 fixture repeats the captured B1 request; distinct page storage.",
    )
    save(destination / "decision.json", decision)
    return decision


def expansion_diagnostic(fixtures, destination, config):
    """Count a transient value cast, native attention, and rounded output scale."""
    destination = Path(destination)
    rows = []
    for layer in config["layers"]:
        path = Path(fixtures) / "mixed" / f"layer-{layer}.pt"
        meta = json.loads(path.with_suffix(".json").read_text())
        data = torch.load(path, weights_only=True, map_location="cpu")
        for batch in [1, 8]:
            q, k, v, metadata = fixture(data, batch)
            scratch = torch.empty(v.shape, dtype=torch.float16, device="cuda")
            scratch.copy_(v)
            mixed, mplan, mout, mlse, _ = build(q, k, v, metadata, "fa3")
            native, nplan, nout, nlse, seconds = build(q, k, scratch, metadata, "fa3")

            def baseline(
                mixed=mixed, q=q, k=k, v=v, mout=mout, mlse=mlse, scale=meta["scale"]
            ):
                return mixed.run(
                    q,
                    (k, v),
                    v_scale=scale,
                    out=mout,
                    lse=mlse,
                    return_lse=True,
                )

            def expanded(
                scratch=scratch,
                v=v,
                native=native,
                q=q,
                k=k,
                nout=nout,
                nlse=nlse,
                scale=meta["scale"],
            ):
                scratch.copy_(v)
                native.run(q, (k, scratch), out=nout, lse=nlse, return_lse=True)
                nout.mul_(scale)

            baseline()
            expanded()
            numerical = dict(
                output=compare_close(nout, mout), lse=compare_close(nlse, mlse)
            )
            for block, order in enumerate(config["component_orders"]):
                for choice in order:
                    operation, plan = (
                        (baseline, mplan) if choice == "fa3" else (expanded, nplan)
                    )
                    measurements = measure(
                        operation, plan, config["warmup"], config["repetitions"]
                    )
                    rows.append(
                        dict(
                            layer=layer,
                            batch=batch,
                            block=block,
                            operation=(
                                "mixed-fa3"
                                if choice == "fa3"
                                else "expand-native-fa3-scale"
                            ),
                            rows=measurements,
                            numerical=numerical,
                            scratch_bytes=scratch.numel() * scratch.element_size(),
                            first_native_plan_seconds=seconds,
                            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            device_used_bytes=torch.cuda.device_memory_used(),
                        )
                    )
                    save(destination / "expansion.json", rows)
            del q, k, v, scratch, mixed, native, baseline, expanded
            torch.cuda.empty_cache()
    return rows
