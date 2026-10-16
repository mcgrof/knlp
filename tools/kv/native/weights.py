# SPDX-License-Identifier: MIT
"""Verify unpacked FP16 model weights independently of cache precision."""

import torch

from .instrument import tensor_inventory


def inspect_fp16_weights(model, expected_modules=144, expected_embeddings=2):
    """Reject quantized methods, packed storage, and unexpected compute weights."""
    linears = []
    embeddings = []
    parameters = []
    seen = set()
    for name, module in model.named_modules():
        if hasattr(module, "qweight") or hasattr(module, "qzeros"):
            raise RuntimeError("FP16 execution contains packed weights")
        method = getattr(module, "quant_method", None)
        if method is None:
            continue
        if type(method).__name__ == "UnquantizedEmbeddingMethod":
            embeddings.append(name)
            if module.weight.dtype != torch.float16:
                raise RuntimeError("FP16 embedding weight has another dtype")
            continue
        if type(method).__name__ != "UnquantizedLinearMethod":
            raise RuntimeError("FP16 execution contains a quantized method")
        if not isinstance(getattr(module, "weight", None), torch.Tensor):
            raise RuntimeError("FP16 linear weight is missing")
        if module.weight.dtype != torch.float16:
            raise RuntimeError("FP16 linear weight has another dtype")
        linears.append(
            dict(
                name=name,
                method=type(method).__name__,
                weight=tensor_inventory(module.weight),
            )
        )
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float16:
            raise RuntimeError("FP16 parameter has another dtype: " + name)
        storage = parameter.untyped_storage()
        key = (str(parameter.device), storage.data_ptr())
        unique = key not in seen
        seen.add(key)
        parameters.append(
            dict(
                name=name,
                dtype=str(parameter.dtype),
                shape=list(parameter.shape),
                unique_storage_bytes=storage.nbytes() if unique else 0,
            )
        )
    if (
        len(linears) != expected_modules
        or len(embeddings) != expected_embeddings
        or not parameters
    ):
        raise RuntimeError("FP16 weight inventory is incomplete")
    return dict(
        format="fp16",
        linear_modules=linears,
        embedding_modules=embeddings,
        parameters=parameters,
        unique_parameter_bytes=sum(x["unique_storage_bytes"] for x in parameters),
        dispatch_marker="aten::mm",
    )


def fp16_dispatch_from_trace(trace, minimum_calls=144):
    """Join dense matrix calls to actual CUDA work without naming kernels."""
    events = trace["traceEvents"]
    calls = [
        event
        for event in events
        if event.get("cat") == "cpu_op" and event.get("name") == "aten::mm"
    ]
    if len(calls) < minimum_calls:
        raise RuntimeError("FP16 profile omits expected dense matrix calls")
    identifiers = {event["args"]["External id"] for event in calls}
    if len(identifiers) != len(calls):
        raise RuntimeError("FP16 matrix call identifiers are not unique")
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("args", {}).get("External id") in identifiers
        and event.get("dur", 0) > 0
    ]
    covered = {event["args"]["External id"] for event in kernels}
    if covered != identifiers:
        raise RuntimeError("FP16 matrix calls do not all join CUDA kernels")
    names = sorted({event["name"] for event in kernels})
    if any("machete" in name.lower() or "4bit" in name.lower() for name in names):
        raise RuntimeError("FP16 matrix trace contains packed weight kernels")
    devices = sorted({event["args"]["device"] for event in kernels})
    if len(devices) != 1:
        raise RuntimeError("FP16 matrix trace requires one CUDA device")
    return dict(
        cpu_mm_calls=len(calls),
        covered_cpu_mm_calls=len(covered),
        cuda_kernels=len(kernels),
        kernel_names=names,
        devices=devices,
        matching="CPU aten::mm and positive-duration CUDA kernels by External id",
        passed=True,
    )
