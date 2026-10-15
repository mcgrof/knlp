# SPDX-License-Identifier: MIT
"""Importable vLLM callbacks; observe buffers and preserve original logits."""

import json
from pathlib import Path
import torch


def ensure_schedule_observer(model, path):
    from . import schedule

    if not schedule._installed:
        schedule.install(path, external_labels=True)
    elif schedule._sink != Path(path):
        raise RuntimeError("scheduler observer points at a different experiment")


def require_finite(model):
    if hasattr(model, "_study_finite_original"):
        raise RuntimeError("finite-logit check already installed")
    original = model.compute_logits
    model._study_finite_original = original

    def checked(*args, **kwargs):
        logits = original(*args, **kwargs)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite native generation logits")
        return logits

    model.compute_logits = checked


def ensure_no_teacher(model):
    if hasattr(model, "_study_original_compute_logits"):
        raise RuntimeError("teacher forcing remains active during generation")
    return True


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def tensor_inventory(value):
    if isinstance(value, torch.Tensor):
        return [
            {
                "shape": list(value.shape),
                "strides": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "element_size": value.element_size(),
                "tensor_pointer": value.data_ptr(),
                "contiguous": value.is_contiguous(),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "logical_bytes": value.numel() * value.element_size(),
                "storage_pointer": value.untyped_storage().data_ptr(),
                "storage_bytes": value.untyped_storage().nbytes(),
            }
        ]
    if isinstance(value, (tuple, list)):
        return [x for part in value for x in tensor_inventory(part)]
    return []


def observe_wrappers(model, trace_dir, expected_types=None):
    import flashinfer

    if hasattr(model, "_study_wrapper_originals"):
        raise RuntimeError("wrapper observer already installed")
    model._study_wrapper_originals = []
    read_seen = set()
    read_count = [0]
    backing = {
        t["storage_pointer"]: name
        for name, module in model.named_modules()
        if hasattr(module, "_k_scale") and hasattr(module, "kv_cache")
        for t in tensor_inventory(module.kv_cache)
    }
    for kind in (
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
    ):
        cls = getattr(flashinfer, kind)
        original = cls.run
        seen = [0]

        def observed(self, *args, _original=original, _kind=kind, _seen=seen, **kwargs):
            backend = str(getattr(self, "_backend", None))
            cache = args[1] if len(args) > 1 else kwargs.get("paged_kv_cache")
            values = tensor_inventory(cache)
            if expected_types is not None:
                if {t["dtype"] for t in values} != expected_types:
                    raise RuntimeError(
                        "attention read dtype differs from selected mode"
                    )
                if any(t["storage_pointer"] not in backing for t in values):
                    raise RuntimeError("attention read is outside model cache storage")
                layer = backing[values[0]["storage_pointer"]]
                if any(backing[t["storage_pointer"]] != layer for t in values):
                    raise RuntimeError("attention read mixes different layer storages")
                key = (_kind, layer)
                if key not in read_seen:
                    record = {"wrapper": _kind, "layer": layer, "cache": values}
                    with (Path(trace_dir) / "kernel-read-events.jsonl").open(
                        "a"
                    ) as stream:
                        stream.write(json.dumps(record) + "\n")
                    if read_count[0] < 3:
                        save(
                            Path(trace_dir) / f"kernel-read-{read_count[0]}.json",
                            values,
                        )
                    read_count[0] += 1
                    read_seen.add(key)
            if _seen[0] < 4:
                record = {
                    "wrapper": _kind,
                    "backend": backend,
                    "use_tensor_cores": getattr(
                        self,
                        "use_tensor_cores",
                        getattr(self, "_use_tensor_cores", None),
                    ),
                    "query": tensor_inventory(args[0]) if args else [],
                    "cache": values,
                    "cached_module": repr(getattr(self, "_cached_module", None))[:4000],
                }
                path = Path(trace_dir) / "wrapper-dispatch.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                _seen[0] += 1
            with torch.profiler.record_function("native:" + _kind + ":" + backend):
                return _original(self, *args, **kwargs)

        model._study_wrapper_originals.append((cls, original))
        cls.run = observed


def restore_wrappers(model):
    for cls, original in model._study_wrapper_originals:
        cls.run = original
    del model._study_wrapper_originals


def native_memory_state(worker):
    runner = worker.model_runner
    seen = set()

    def collect(value, depth=0):
        if id(value) in seen or depth > 5:
            return []
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            return tensor_inventory(value)
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (tuple, list)):
            children = value
        elif hasattr(value, "__dict__"):
            children = vars(value).values()
        else:
            return []
        return [item for child in children for item in collect(child, depth + 1)]

    batch = getattr(runner, "input_batch", runner)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    metadata = {}
    for name in (
        "block_table",
        "block_tables",
        "req_states",
        "slot_mapping",
        "slot_mapping_cpu",
        "slot_mapping_gpu",
        "seq_lens",
        "seq_lens_cpu",
        "seq_lens_gpu",
    ):
        if hasattr(batch, name):
            metadata[name] = collect(getattr(batch, name))
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated_since_engine_start": torch.cuda.max_memory_allocated(),
        "cuda_device_used_bytes": total_bytes - free_bytes,
        "cuda_device_total_bytes": total_bytes,
        "metadata_tensors": metadata,
    }


def configure(model, scales, arm, trace_dir, packed_expectation=None):
    selected = []
    packed = []
    for name, module in model.named_modules():
        if hasattr(module, "qweight"):
            if module.qweight.dtype != torch.int32:
                raise RuntimeError("native AWQ weight storage changed")
            packed.append(
                {
                    "name": name,
                    "method": type(getattr(module, "quant_method", None)).__name__,
                    "kernel": type(
                        getattr(getattr(module, "quant_method", None), "kernel", None)
                    ).__name__,
                    "weight_bits": getattr(
                        getattr(
                            getattr(module, "quant_method", None), "quant_config", None
                        ),
                        "weight_bits",
                        4,
                    ),
                    "buffers": tensor_inventory(
                        (module.qweight, module.qzeros, module.scales)
                    ),
                }
            )
        if hasattr(module, "_k_scale") and hasattr(module, "kv_cache"):
            index = int(name.split("layers.")[1].split(".")[0])
            k = scales[f"{index}.k"] if arm == "k8_v8" else 1.0
            v = scales[f"{index}.v"] if arm in ("k16_v8", "k8_v8") else 1.0
            module._k_scale.fill_(k)
            module._v_scale.fill_(v)
            module._k_scale_float, module._v_scale_float = k, v
            module._q_scale.fill_(1)
            module._q_scale_float = 1.0
            module.calculate_kv_scales = False
            if hasattr(module.impl, "bmm1_scale"):
                module.impl.bmm1_scale = None
                module.impl.bmm2_scale = None
            selected.append(
                {
                    "name": name,
                    "k_scale": k,
                    "v_scale": v,
                    "impl": type(module.impl).__name__,
                    "cache": tensor_inventory(module.kv_cache),
                    "scale_buffers": tensor_inventory(
                        (module._q_scale, module._k_scale, module._v_scale)
                    ),
                }
            )
    if len(selected) != 36 or len(packed) != 144:
        raise RuntimeError("native packed attention inventory is incomplete")
    expected = packed_expectation or {
        "method": "AWQLinearMethod",
        "kernel": "NoneType",
        "weight_bits": 4,
        "dispatch_marker": "gemm_forward_4bit_cuda",
    }
    for module in packed:
        if any(
            module[key] != expected[key] for key in ("method", "kernel", "weight_bits")
        ):
            raise RuntimeError(
                "native packed AWQ implementation differs from frozen expectation"
            )
    model._study_packed_dispatch_marker = expected["dispatch_marker"]
    save(
        Path(trace_dir) / "installed.json",
        {
            "layers": selected,
            "packed": packed,
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        },
    )
    observed = {t["dtype"] for layer in selected for t in layer["cache"]}
    expected = (
        {"torch.float16"}
        if arm == "k16_v16"
        else (
            {"torch.float16", "torch.float8_e4m3fn"}
            if arm == "k16_v8"
            else {"torch.float8_e4m3fn"}
        )
    )
    backing_ok = observed == expected or (
        arm == "k8_v8" and observed == {"torch.uint8"}
    )
    if not backing_ok:
        raise RuntimeError(
            f"physical cache dtype gate failed: {observed} != {expected}"
        )
    if arm == "k16_v8":
        for layer in selected:
            key, value = layer["cache"]
            if key["storage_pointer"] != value["storage_pointer"]:
                raise RuntimeError("mixed cache has copied backing allocations")
            a = key["tensor_pointer"]
            b = value["tensor_pointer"]
            if not (
                key["contiguous"]
                and value["contiguous"]
                and a + key["logical_bytes"] == b
            ):
                raise RuntimeError(
                    "mixed K/V regions are not contiguous disjoint views"
                )
    return {
        "layers": len(selected),
        "packed_modules": len(packed),
        "cache_dtypes": sorted(expected),
    }


def install(model, scales, arm, trace_dir, packed_expectation=None):
    receipt = configure(model, scales, arm, trace_dir, packed_expectation)
    expected = set(receipt.pop("cache_dtypes"))
    model._study_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    )
    model._study_profiler.__enter__()
    observe_wrappers(model, trace_dir, expected_types=expected)
    return receipt


def finish_profile(model, trace_dir):
    model._study_profiler.__exit__(None, None, None)
    restore_wrappers(model)
    events = [item.key for item in model._study_profiler.key_averages()]
    save(Path(trace_dir) / "dispatch.json", events)
    save(
        Path(trace_dir) / "jit-module-operators.json",
        [name for name in events if name.startswith("flashinfer::")],
    )
    matched = any(
        model._study_packed_dispatch_marker.lower() in name.lower() for name in events
    )
    if not matched:
        raise RuntimeError("native model-specific packed AWQ kernel missing")
    model._study_profiler = None
    return {"packed_awq_dispatch": matched}


def stage_teacher(model, study):
    if hasattr(model, "_study_original_compute_logits"):
        raise RuntimeError("native teacher is already active")
    original = model.compute_logits
    model._study_original_compute_logits = original
    step = [0]

    def compute_logits(*args, **kwargs):
        logits = original(*args, **kwargs)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("teacher requires exactly one native request")
        index = step[0]
        if index >= len(study["teacher"]):
            raise RuntimeError("native teacher sequence exhausted")
        values = logits[0].float()
        if not torch.isfinite(values).all():
            raise RuntimeError("nonfinite native teacher logits")
        target = study["teacher"][index]
        row = {
            "sample_id": study["sample_id"],
            "step": index,
            "target": target,
            "nll": -float(values.log_softmax(-1)[target]),
            "top1": int(values.argmax()),
            "provenance": study["provenance"],
        }
        with open(study["path"], "a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        logits.fill_(-torch.inf)
        logits[0, target] = 0
        step[0] += 1
        return logits

    model.compute_logits = compute_logits


def clear_teacher(model):
    model.compute_logits = model._study_original_compute_logits
    del model._study_original_compute_logits


def profile_start(model, trace_dir):
    model._study_profile_dir = str(trace_dir)
    model._study_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    )
    model._study_profiler.__enter__()
    observe_wrappers(model, trace_dir)


def profile_end(model):
    import gzip

    model._study_profiler.__exit__(None, None, None)
    restore_wrappers(model)
    directory = Path(model._study_profile_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trace.json"
    model._study_profiler.export_chrome_trace(str(path))
    raw = path.read_bytes()
    (directory / "trace.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    path.unlink()
    data = json.loads(raw)
    kernels = [
        {"name": e["name"], "duration_us": e["dur"], "args": e.get("args", {})}
        for e in data["traceEvents"]
        if e.get("cat") == "kernel" and "dur" in e
    ]
    save(directory / "gpu-kernels.json", kernels)
    model._study_profiler = None


def reset_warm_peak(worker):
    torch.cuda.reset_peak_memory_stats()
    return native_memory_state(worker)
