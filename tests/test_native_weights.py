# SPDX-License-Identifier: MIT
"""Check weight provenance rejects incompatible loaders and alias inflation."""

import pytest
import torch

from tools.kv.native.weights import inspect_fp16_weights


class UnquantizedLinearMethod:
    pass


class AWQLinearMethod:
    pass


def model():
    value = torch.nn.Sequential(
        torch.nn.Linear(4, 8, bias=False, dtype=torch.float16),
        torch.nn.Linear(8, 4, bias=False, dtype=torch.float16),
    )
    for layer in value:
        layer.quant_method = UnquantizedLinearMethod()
    return value


def test_fp16_storage_and_shared_parameter_accounting():
    value = model()
    value.alias = value[0]
    receipt = inspect_fp16_weights(value, 2, 0)
    assert receipt["format"] == "fp16"
    assert receipt["unique_parameter_bytes"] == 128
    assert len(receipt["linear_modules"]) == 2


@pytest.mark.parametrize("change", ["packed", "method", "dtype", "count"])
def test_reject_incompatible_weights(change):
    value = model()
    if change == "packed":
        value[0].qweight = torch.zeros(1, dtype=torch.int32)
    elif change == "method":
        value[0].quant_method = AWQLinearMethod()
    elif change == "dtype":
        value[0].weight = torch.nn.Parameter(value[0].weight.float())
    with pytest.raises(RuntimeError):
        inspect_fp16_weights(value, 3 if change == "count" else 2, 0)


def trace(calls=144):
    events = []
    for identifier in range(calls):
        events.append(
            dict(cat="cpu_op", name="aten::mm", args={"External id": identifier})
        )
        events.append(
            dict(
                cat="kernel",
                name="nvjet_hsh_384x8_64x4_2x1_v_bz_TNT",
                dur=2.5,
                args={"External id": identifier, "device": 0},
            )
        )
    return {"traceEvents": events}


def test_dense_cuda_join_accepts_opaque_kernel_names():
    from tools.kv.native.weights import fp16_dispatch_from_trace

    result = fp16_dispatch_from_trace(trace())
    assert result["covered_cpu_mm_calls"] == 144
    assert result["cuda_kernels"] == 144


@pytest.mark.parametrize(
    "change", ["cpu_only", "unrelated", "packed", "missing", "duplicate"]
)
def test_dense_cuda_join_rejects_incomplete_or_wrong_dispatch(change):
    from tools.kv.native.weights import fp16_dispatch_from_trace

    data = trace()
    if change == "cpu_only":
        data["traceEvents"] = data["traceEvents"][::2]
    elif change == "unrelated":
        data["traceEvents"][1]["args"]["External id"] = 5000
    elif change == "packed":
        data["traceEvents"][1]["name"] = "machete_4bit_kernel"
    elif change == "missing":
        data["traceEvents"] = data["traceEvents"][:-2]
    elif change == "duplicate":
        data["traceEvents"][2]["args"]["External id"] = 0
    with pytest.raises(RuntimeError):
        fp16_dispatch_from_trace(data)


def test_bf16_storage_rejects_fp16_and_counts_bias():
    from tools.kv.native.weights import inspect_bf16_weights

    value = model().to(torch.bfloat16)
    value[0].bias = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
    receipt = inspect_bf16_weights(value, 2, 0)
    assert receipt["format"] == "bf16"
    assert receipt["unique_parameter_bytes"] == 144
    with pytest.raises(RuntimeError):
        inspect_fp16_weights(value, 2, 0)
    with pytest.raises(RuntimeError):
        inspect_bf16_weights(value.to(torch.float16), 2, 0)


def test_bf16_dispatch_joins_biased_dense_projections():
    from tools.kv.native.weights import bf16_dispatch_from_trace

    data = trace(112)
    for event in data["traceEvents"][::8]:
        event["name"] = "aten::addmm"
    assert bf16_dispatch_from_trace(data)["covered_cpu_mm_calls"] == 112
    data["traceEvents"][1]["args"]["External id"] = 9999
    with pytest.raises(RuntimeError):
        bf16_dispatch_from_trace(data)
