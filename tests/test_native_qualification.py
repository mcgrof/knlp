# SPDX-License-Identifier: MIT
"""Check that numerical failures retain their physical operands and bounds."""

import json
import hashlib
from types import SimpleNamespace

import pytest
import torch

from tools.kv.native.qualification import (
    record_attention_check,
    snapshot_attention_libraries,
)


@pytest.mark.parametrize("kind", ["pass", "error", "nan"])
def test_capture_before_raise(tmp_path, kind, monkeypatch):
    binary = tmp_path / "current.so"
    binary.write_bytes(b"recorded binary")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "tools.kv.native.qualification.loaded_attention_libraries",
        lambda: [{"path": str(binary), "sha256": digest}],
    )
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
    retained = receipt["retained_flashinfer_libraries"]
    assert retained == [{"sha256": digest, "file": digest + ".so"}]
    binary.write_bytes(b"later compilation")
    assert (
        tmp_path / "attention/libraries" / retained[0]["file"]
    ).read_bytes() == b"recorded binary"


def test_snapshot_preserves_versions_of_reused_cache_path(tmp_path):
    binary = tmp_path / "current.so"
    output = tmp_path / "retained"
    versions = []
    for data in [b"first binary", b"later binary"]:
        binary.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        rows = snapshot_attention_libraries(
            [{"path": str(binary), "sha256": digest}], output
        )
        versions.append((rows[0]["file"], data))
    assert all((output / filename).read_bytes() == data for filename, data in versions)


@pytest.mark.parametrize(
    "fault", ["source_changed", "retained_corrupt", "invalid_hash"]
)
def test_snapshot_rejects_wrong_binary_bytes(tmp_path, fault):
    binary = tmp_path / "current.so"
    binary.write_bytes(b"original")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    rows = [{"path": str(binary), "sha256": digest}]
    output = tmp_path / "retained"
    snapshot_attention_libraries(rows, output)
    if fault == "source_changed":
        binary.write_bytes(b"new binary at same path")
    elif fault == "retained_corrupt":
        (output / (digest + ".so")).write_bytes(b"corrupt")
    else:
        rows[0]["sha256"] = "../outside"
    with pytest.raises((RuntimeError, ValueError)):
        snapshot_attention_libraries(rows, output)


def test_loaded_binary_in_isolated_jit_root(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace

    from tools.kv.native.qualification import loaded_attention_libraries

    root = tmp_path / "isolated/cached_ops"
    root.mkdir(parents=True)
    binary = root / "batch_prefill.so"
    binary.write_bytes(b"known-test-binary")
    maps = tmp_path / "maps"
    maps.write_text("0000-ffff r-xp 0000 00:00 0 " + str(binary) + "\n")
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.jit.env",
        SimpleNamespace(FLASHINFER_JIT_DIR=root, FLASHINFER_AOT_DIR=tmp_path / "empty"),
    )
    assert loaded_attention_libraries(maps) == [
        {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    ]
