# SPDX-License-Identifier: MIT
"""Reject altered checkpoint metadata as well as altered weight bytes."""

import hashlib
import json

import pytest
from tools.kv.native.october_setup import recipe_pins
from tools.kv.decision.production import (
    MODEL,
    REVISION,
    SOURCES,
    checkpoint,
    fp32,
    transferred_scales,
)


def test_worker_requires_explicit_installer_revision_and_keeps_other_recipes():
    default = recipe_pins("decision-http")
    assert default["vllm"][1] == "cca5181dd50b24ff58f114e3ee7d200293c9fbf6"
    assert SOURCES["vllm"] == "e5c8d87fd6bbf9cdedf266d160890599d938abb0"
    assert default["flashinfer"][1] == SOURCES["flashinfer"]
    assert (
        recipe_pins("current")["vllm"][1] == "276305b29e0d11f95ada503ba76323ab24992e59"
    )
    assert (
        recipe_pins("recent-transfer")["vllm"][1]
        == "a74386bbdf700e3cfd90a7bbfc9f37026afa652e"
    )


def fixture(tmp_path, monkeypatch):
    import huggingface_hub

    model = tmp_path / "model"
    model.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config = {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "model_type": "qwen3",
        "rms_norm_eps": 1e-6,
    }
    (model / "config.json").write_text(json.dumps(config))
    (model / "generation_config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test-only checkpoint digest payload")
    (inputs / "official-weights.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "files": {
                    "model.safetensors": {
                        "sha256": hashlib.sha256(
                            (model / "model.safetensors").read_bytes()
                        ).hexdigest()
                    }
                },
            }
        )
    )
    files = {}
    for path in model.glob("*.json"):
        raw = path.read_bytes()
        files[path.name] = {
            "algorithm": "git_blob_sha1",
            "bytes": len(raw),
            "digest": hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest(),
        }
    (inputs / "official-model-files.json").write_text(
        json.dumps({"model": MODEL, "revision": REVISION, "files": files})
    )

    def local(repo_id, revision, local_files_only):
        assert repo_id == MODEL and revision == REVISION and local_files_only
        return str(model)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", local)
    return inputs, model


def test_exact_checkpoint_and_metadata_identities(tmp_path, monkeypatch):
    inputs, model = fixture(tmp_path, monkeypatch)
    path, identity = checkpoint(inputs)
    assert path == model and len(identity["weights_sha256"]) == 1
    assert len(identity["configuration_and_tokenizer_json_sha256"]) == 3


@pytest.mark.parametrize(
    "bad", ["weights", "nongemetry_config", "missing_config", "algorithm"]
)
def test_unfrozen_checkpoint_bytes_rejected(tmp_path, monkeypatch, bad):
    inputs, model = fixture(tmp_path, monkeypatch)
    metadata = inputs / "official-model-files.json"
    if bad == "weights":
        (model / "model.safetensors").write_bytes(b"altered")
    elif bad == "nongemetry_config":
        path = model / "config.json"
        value = json.loads(path.read_text())
        value["rms_norm_eps"] = 1e-4
        path.write_text(json.dumps(value))
    else:
        value = json.loads(metadata.read_text())
        if bad == "missing_config":
            del value["files"]["config.json"]
        else:
            value["files"]["config.json"].update(algorithm="unchecked", digest=None)
        metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        checkpoint(inputs)


def test_scale_transfer_only_rounds_to_declared_fp32_wire_type():
    row = {
        "model": MODEL,
        "revision": REVISION,
        "scales": [{"K": 0.1 + i / 100, "V": 0.03 + i / 100} for i in range(36)],
    }
    result = transferred_scales(row)
    assert (
        len(result) == 72
        and result["0.k"] == fp32(0.1)
        and result["35.v"] == fp32(0.38)
    )
    assert result["0.k"] != 0.1
    for value in [0, -1, float("nan"), float("inf"), 1e-99]:
        with pytest.raises(ValueError):
            fp32(value)
