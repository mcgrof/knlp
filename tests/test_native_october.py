"""Check actual launch guards without importing a GPU runtime."""

import copy
import json

import pytest

from tools.kv.native.october import (
    VARIANTS,
    check_protocol,
    check_qualification,
    check_timing_protocol,
    check_scales,
    digest,
)
from tools.kv.native.completion import request_metrics


def test_scored_serving_rejects_native_calibration_placeholder():
    value = {
        "status": "native_only_calibration_placeholder",
        "scales": {"0.k": 1.0, "0.v": 1.0},
    }
    with pytest.raises(ValueError, match="native-only placeholder"):
        check_scales(value, {"layers": 1})
    value.pop("status")
    with pytest.raises(ValueError, match="calibrated scales"):
        check_scales(value, {"layers": 1})
    value["method"] = "max_abs_div_448_float32_once"
    assert check_scales(value, {"layers": 1}) is value["scales"]
    value["scales"]["0.v"] = float("nan")
    with pytest.raises(ValueError, match="inventory"):
        check_scales(value, {"layers": 1})


@pytest.mark.parametrize("flag", [None, True, 0])
def test_timing_requires_explicit_statistics(flag):
    value = protocol("native")
    if flag is not None:
        value["engine"]["disable_log_stats"] = flag
    with pytest.raises(ValueError, match="explicit disable_log_stats=False"):
        check_timing_protocol(value)
    value["engine"]["disable_log_stats"] = False
    check_timing_protocol(value)


def test_request_clocks_support_slots_and_reject_disabled_metrics():
    class Metrics:
        __slots__ = (
            "arrival_time",
            "first_token_ts",
            "last_token_ts",
            "first_token_latency",
        )

    metrics = Metrics()
    for i, name in enumerate(Metrics.__slots__):
        setattr(metrics, name, float(i))
    assert request_metrics(metrics) == {
        name: float(i) for i, name in enumerate(Metrics.__slots__)
    }
    with pytest.raises(RuntimeError, match="disable_log_stats=False"):
        request_metrics(None)


def protocol(variant):
    return {
        "backend_sources": dict(VARIANTS[variant].sources),
        "engine": {
            "enforce_eager": False,
            "enable_chunked_prefill": False,
            "tensor_parallel_size": 1,
            "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        },
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_variant_and_capture(variant):
    env = {
        **VARIANTS[variant].environment({}),
        "VLLM_COMPACT_PREFILL": str(int(variant == "compact")),
    }
    assert check_protocol(protocol(variant), variant, env) is VARIANTS[variant]
    env["VLLM_COMPACT_PREFILL"] = str(int(variant != "compact"))
    with pytest.raises(ValueError, match="compact flag"):
        check_protocol(protocol(variant), variant, env)


@pytest.mark.parametrize("change", ["source", "eager", "chunking", "capture"])
def test_recipe_drift_is_rejected(change):
    value = copy.deepcopy(protocol("compact"))
    if change == "source":
        value["backend_sources"]["vllm"] = VARIANTS["mixed"].sources["vllm"]
    elif change == "eager":
        value["engine"]["enforce_eager"] = True
    elif change == "chunking":
        value["engine"]["enable_chunked_prefill"] = True
    else:
        value["engine"]["compilation_config"]["cudagraph_mode"] = "NONE"
    env = {**VARIANTS["compact"].environment({}), "VLLM_COMPACT_PREFILL": "1"}
    with pytest.raises(ValueError):
        check_protocol(value, "compact", env)


def test_timing_requires_exact_complete_quality_and_workload(tmp_path):
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol("compact")))
    gate = tmp_path / "quality.json"
    value = {
        "passed": True,
        "protocol_sha256": [digest(source)],
        "production": True,
        "nll": True,
        "retrieval": True,
        "gsm8k": True,
        "workloads": [{"input_tokens": 8192, "batch": 8}],
    }
    gate.write_text(json.dumps(value))
    check_qualification(gate, source, length=8192, batch=8)
    with pytest.raises(ValueError, match="workload"):
        check_qualification(gate, source, length=8192, batch=32)
    value["gsm8k"] = False
    gate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="incomplete"):
        check_qualification(gate, source, length=8192, batch=8)
    value["gsm8k"] = True
    gate.write_text(json.dumps(value))
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="exact protocol"):
        check_qualification(gate, source, length=8192, batch=8)
