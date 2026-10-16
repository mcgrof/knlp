"""Check actual launch guards without importing a GPU runtime."""

import copy
import json

import pytest

from tools.kv.native.october import (
    VARIANTS,
    check_protocol,
    check_qualification,
    digest,
)


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
