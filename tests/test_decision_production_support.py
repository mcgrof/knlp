# SPDX-License-Identifier: MIT
"""A support failure cannot erase a quality result or free its alpha slot."""

import json

import pytest

from tools.kv.decision.production import sha
from tools.kv.decision.production_analysis import build_identity, support_gap


def failed_worker(tmp_path, phase="numerical"):
    base = tmp_path / "SF"
    (base / "server").mkdir(parents=True)
    protocol = {"source_sha": "source"}
    records = {
        "server/launch.json": {
            "arm": "SF",
            "source_sha": "source",
            "protocol_sha256": "protocol",
            "allocation": "capacity",
        },
        "server.log": {"diagnostic": "retained unsupported operation"},
        "server-exit.json": {"error": "collector failed"},
    }
    if phase == "numerical":
        records["numerical-command.json"] = {"error": "oracle failed"}
    for name, value in records.items():
        (base / name).write_text(json.dumps(value))
    declaration = {
        "arm": "SF",
        "source_sha": "source",
        "protocol_sha256": "protocol",
        "phase": phase,
        "classification": "unsupported_interface",
        "reason": "Raw log identifies an unsupported operation before quality.",
        "evidence_sha256": {name: sha(base / name) for name in records},
    }
    return base, declaration, protocol


@pytest.mark.parametrize("phase", ["startup", "numerical"])
def test_pre_quality_support_gap_keeps_unfilled_slots(tmp_path, phase):
    _, declaration, protocol = failed_worker(tmp_path, phase)
    result = support_gap(tmp_path, declaration, protocol, "protocol")
    assert not result["quality_evaluated"] and not result["alpha_reallocated"]


@pytest.mark.parametrize(
    "defect",
    [
        "quality_started",
        "completed",
        "changed_raw",
        "missing_raw",
        "other_arm",
        "qualified",
        "wrong_source",
        "no_failure",
        "wrong_allocation",
    ],
)
def test_support_gap_rejects_relabelling_or_missing_evidence(tmp_path, defect):
    base, declaration, protocol = failed_worker(tmp_path)
    if defect == "quality_started":
        (base / "quality").mkdir()
    elif defect == "completed":
        (base / "DONE.json").write_text("{}")
    elif defect == "changed_raw":
        (base / "server.log").write_text("changed")
    elif defect == "missing_raw":
        declaration["evidence_sha256"].pop("server.log")
    elif defect == "other_arm":
        declaration["arm"] = "AF"
    elif defect == "qualified":
        (base / "numerical").mkdir()
        (base / "numerical/QUALIFIED.json").write_text("{}")
    elif defect == "wrong_source":
        declaration["source_sha"] = "changed"
    else:
        name = "server-exit.json" if defect == "no_failure" else "server/launch.json"
        value = json.loads((base / name).read_text())
        value["error" if defect == "no_failure" else "allocation"] = None
        (base / name).write_text(json.dumps(value))
        declaration["evidence_sha256"][name] = sha(base / name)
    with pytest.raises(ValueError):
        support_gap(tmp_path, declaration, protocol, "protocol")


@pytest.mark.parametrize(
    "defect", [None, "boundary", "bytes", "counts", "empty", "digest"]
)
def test_kernel_inventory_is_bound_to_its_exact_stage(tmp_path, defect):
    value = {
        "loaded_shared_objects_sha256": {"worker.so": "a" * 64},
        "generated_cache_files_sha256": {"kernel.cubin": "b" * 64},
    }
    if defect == "empty":
        value["generated_cache_files_sha256"] = {}
    elif defect == "digest":
        value["loaded_shared_objects_sha256"]["worker.so"] = "not-a-hash"
    path = tmp_path / "build-identity-quality.json"
    path.write_text(json.dumps(value))
    receipt = {
        "results": [
            {
                "file": path.name,
                "sha256": sha(path),
                "loaded_shared_objects": 1,
                "generated_files": 1,
            }
        ]
    }
    if defect == "boundary":
        receipt["results"][0]["file"] = "build-identity-numerical.json"
    elif defect == "counts":
        receipt["results"][0]["generated_files"] = 2
    elif defect == "bytes":
        path.write_text("{}")
    if defect is None:
        assert build_identity(tmp_path, receipt, "quality") == value
    else:
        with pytest.raises(ValueError):
            build_identity(tmp_path, receipt, "quality")
