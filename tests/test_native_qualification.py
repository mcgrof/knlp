# SPDX-License-Identifier: MIT
"""Check that numerical failures retain their physical operands and bounds."""

import json
from types import SimpleNamespace

import pytest
import torch

from tools.kv.native.qualification import record_attention_check


@pytest.mark.parametrize("kind", ["pass", "error", "nan"])
def test_capture_before_raise(tmp_path, kind):
    raw = torch.arange(64, dtype=torch.uint8)
    cache = (raw[:32].view(torch.float16), raw[32:].view(torch.float8_e4m3fn))
    result = torch.zeros(1, 2, 3)
    if kind == "error":
        result[0, 1, 2] = 0.1
    elif kind == "nan":
        result[0, 1, 2] = float("nan")
    kwargs = {
        "wrapper": SimpleNamespace(_backend="test", _use_tensor_cores=True),
        "result": result,
        "reference": torch.zeros_like(result),
        "tensors": {
            "query": torch.ones_like(result),
            "page_indices": torch.tensor([3, 0, 2]),
        },
        "cache": cache,
        "configuration": {"head_dim": 64, "v_scale": 1 / 64},
    }
    if kind == "pass":
        record_attention_check(tmp_path, "attention", **kwargs)
    else:
        with pytest.raises(RuntimeError, match="retained"):
            record_attention_check(tmp_path, "attention", **kwargs)
    receipt = json.loads((tmp_path / "attention/check.json").read_text())
    artifact = torch.load(tmp_path / "attention/tensors.pt", weights_only=True)
    assert receipt["passed"] == (kind == "pass")
    assert receipt["maximum_tolerance"] == 0.005
    assert receipt["mean_tolerance"] == 0.001
    assert receipt["offending_index"] == ([0, 0, 0] if kind == "pass" else [0, 1, 2])
    assert torch.equal(artifact["physical_cache_storage_0"], raw)
    assert "physical_cache_storage_1" not in artifact
    assert artifact["page_indices"].tolist() == [3, 0, 2]
    assert receipt["backend"] == "test" and receipt["use_tensor_cores"]
