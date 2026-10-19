import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.kv.native.serving_trace_bundle import read_json, restore_bundle


FIXTURES = Path(__file__).resolve().parents[1] / "tools/kv/native/fixtures"
CATALOG = FIXTURES / "serving/fp16-8k/catalog.json.gz"
MANIFEST = FIXTURES / "current/fp16/qwen.json.gz"


def test_all_measured_traces_restore_exactly():
    catalog = read_json(CATALOG)
    expected = read_json(CATALOG.parent / "manifest.json")
    assert (
        hashlib.sha256(CATALOG.read_bytes()).hexdigest() == expected["catalog_sha256"]
    )
    assert restore_bundle(catalog, MANIFEST) == expected["restored_files"]


def test_restored_file_preserves_ids_arrivals_and_tokens(tmp_path):
    catalog = read_json(CATALOG)
    name = "traces/idle.json"
    catalog["traces"] = {name: catalog["traces"][name]}
    out = tmp_path / "restored"
    restore_bundle(catalog, MANIFEST, out)
    trace = json.loads((out / name).read_bytes())
    compact = catalog["traces"][name]["requests"]
    assert [r["id"] for r in trace["requests"]] == [r["id"] for r in compact]
    assert [r["offset_ns"] for r in trace["requests"]] == [
        r["offset_ns"] for r in compact
    ]
    assert all(len(r["prompt_token_ids"]) == 8192 for r in trace["requests"])
    with pytest.raises(FileExistsError):
        restore_bundle(catalog, MANIFEST, out)


def test_changed_inputs_are_rejected(tmp_path):
    catalog = read_json(CATALOG)
    changed = copy.deepcopy(catalog)
    changed["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt manifest"):
        restore_bundle(changed, MANIFEST)
    changed = copy.deepcopy(catalog)
    next(iter(changed["traces"].values()))["requests"][0]["offset_ns"] += 1
    with pytest.raises(ValueError, match="measured bytes"):
        restore_bundle(changed, MANIFEST)
    changed = copy.deepcopy(catalog)
    changed["traces"] = {"../escape.json": next(iter(changed["traces"].values()))}
    with pytest.raises(ValueError, match="relative paths"):
        restore_bundle(changed, MANIFEST, tmp_path / "out")
    assert not (tmp_path / "out").exists()
