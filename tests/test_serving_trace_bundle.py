import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.kv.native.serving_trace_bundle import read_json, restore_bundle


@pytest.fixture
def synthetic_bundle(tmp_path):
    # Synthetic tokens and arrivals; no captured experiment data in source.
    samples = [
        {"kind": "nll", "length": 8192, "token_ids": [i + 1] * 8192} for i in range(32)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}) + "\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    requests = [
        {"id": "request-1", "offset_ns": 0, "prompt_index": 3, "max_tokens": 4},
        {"id": "request-2", "offset_ns": 10000, "prompt_index": 7, "max_tokens": 8},
    ]
    expected = {
        "manifest_sha256": manifest_sha,
        "requests": [
            {
                "id": "request-1",
                "offset_ns": 0,
                "prompt_token_ids": [4] * 8192,
                "max_tokens": 4,
            },
            {
                "id": "request-2",
                "offset_ns": 10000,
                "prompt_token_ids": [8] * 8192,
                "max_tokens": 8,
            },
        ],
    }
    raw = (json.dumps(expected, separators=(",", ":")) + "\n").encode()
    catalog = {
        "schema": 1,
        "input_tokens": 8192,
        "manifest_sha256": manifest_sha,
        "traces": {
            "traces/idle.json": {
                "metadata": {"manifest_sha256": manifest_sha},
                "field_order": ["manifest_sha256", "requests"],
                "requests": requests,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
    }
    return catalog, manifest


def test_all_measured_traces_restore_exactly():
    root = os.environ.get("KNLP_KV_FIXTURES")
    if not root:
        pytest.skip("set KNLP_KV_FIXTURES for archived-trace integration")
    catalog_path = Path(root) / "serving/fp16-8k/catalog.json.gz"
    manifest = Path(root) / "current/fp16/qwen.json.gz"
    catalog = read_json(catalog_path)
    expected = read_json(catalog_path.parent / "manifest.json")
    assert (
        hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        == expected["catalog_sha256"]
    )
    assert restore_bundle(catalog, manifest) == expected["restored_files"]


def test_restored_file_preserves_ids_arrivals_and_tokens(tmp_path, synthetic_bundle):
    catalog, manifest = synthetic_bundle
    name = "traces/idle.json"
    catalog["traces"] = {name: catalog["traces"][name]}
    out = tmp_path / "restored"
    restore_bundle(catalog, manifest, out)
    trace = json.loads((out / name).read_bytes())
    compact = catalog["traces"][name]["requests"]
    assert [r["id"] for r in trace["requests"]] == [r["id"] for r in compact]
    assert [r["offset_ns"] for r in trace["requests"]] == [
        r["offset_ns"] for r in compact
    ]
    assert all(len(r["prompt_token_ids"]) == 8192 for r in trace["requests"])
    with pytest.raises(FileExistsError):
        restore_bundle(catalog, manifest, out)


def test_changed_inputs_are_rejected(tmp_path, synthetic_bundle):
    catalog, manifest = synthetic_bundle
    changed = copy.deepcopy(catalog)
    changed["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt manifest"):
        restore_bundle(changed, manifest)
    changed = copy.deepcopy(catalog)
    next(iter(changed["traces"].values()))["requests"][0]["offset_ns"] += 1
    with pytest.raises(ValueError, match="measured bytes"):
        restore_bundle(changed, manifest)
    changed = copy.deepcopy(catalog)
    changed["traces"] = {"../escape.json": next(iter(changed["traces"].values()))}
    with pytest.raises(ValueError, match="relative paths"):
        restore_bundle(changed, manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()
