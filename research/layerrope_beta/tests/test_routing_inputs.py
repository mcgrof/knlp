# SPDX-License-Identifier: MIT
"""Check replay provenance without training or reading a final holdout."""

import copy

import numpy as np
import pytest
import torch

from research.layerrope_beta.time_budget.routing_inputs import (
    jobs,
    manifest,
    training_data,
)
from research.layerrope_beta.time_budget.run import job
from research.layerrope_beta.time_budget.protocol import freeze
from research.layerrope_beta.time_budget.state import json_hash
from research.layerrope_beta.time_budget.train import source_identity
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.train import state_hash
from research.tinystories_depth.data import DATA_REV, TOKEN_REV


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "training"
    data.mkdir()
    np.save(data / "train.npy", np.arange(97, dtype=np.uint16))
    np.save(data / "validation.npy", np.arange(32769, dtype=np.uint16))
    write(
        data / "TRAINING.json",
        dict(files={n: sha(data / n) for n in ("train.npy", "validation.npy")}),
    )
    results = []
    for row in ("M", "ML"):
        root = tmp_path / row
        root.mkdir()
        identity = dict(
            job=job(row, 81234, 0.001, "qualification"),
            source=source_identity(),
            protocol_sha256=json_hash(freeze()),
            data_sha256=sha(data / "TRAINING.json"),
        )
        points = []
        for target in (0.0, 200.0, 787.0):
            path = root / (str(target) + ".pt")
            model = dict(weight=torch.eye(4))
            torch.save(
                dict(
                    identity=identity,
                    model=model,
                    step=int(target),
                    training_seconds=target,
                ),
                path,
            )
            points.append(
                dict(
                    target_seconds=target,
                    timing_valid=True,
                    step=int(target),
                    training_seconds=target,
                    validation=dict(ce=1.5, window_ce=[1.5] * 128),
                    checkpoint=dict(
                        path=str(path),
                        sha256=sha(path),
                        model_sha256=state_hash(model),
                        step=int(target),
                        training_seconds=target,
                    ),
                )
            )
        result = root / "RESULT.json"
        write(
            result,
            dict(status="complete", identity=identity, history=dict(landmarks=points)),
        )
        results.append(result)
    return data, results


def test_manifest_preserves_original_losses_and_binds_every_payload(inputs, tmp_path):
    data, results = inputs
    card = manifest(data, results, tmp_path / "AUDIT-INPUTS.json")
    assert len(card["checkpoints"]) == 6
    assert card["validation"]["sha256"] == sha(data / "validation.npy")
    assert all(r["validation"]["ce"] == 1.5 for r in card["checkpoints"])
    assert all(r["sha256"] == sha(r["path"]) for r in card["checkpoints"])
    # A copied result can resolve adjacent files after its original absolute
    # checkpoint location is no longer mounted.
    value = read(results[0])
    for p in value["history"]["landmarks"]:
        p["checkpoint"]["path"] = "/unmounted/" + p["checkpoint"]["path"].split("/")[-1]
    write(results[0], value)
    relocated = manifest(data, results, tmp_path / "relocated.json")
    assert len(relocated["checkpoints"]) == 6


@pytest.mark.parametrize(
    "mutation", ["clock", "source", "data", "validation", "duplicate"]
)
def test_manifest_rejects_unreplayable_evidence(inputs, tmp_path, mutation):
    data, results = inputs
    r = read(results[0])
    point = r["history"]["landmarks"][0]
    if mutation == "clock":
        p = point["checkpoint"]
        state = torch.load(p["path"], weights_only=False)
        state["step"] += 1
        torch.save(state, p["path"])
        p["sha256"] = sha(p["path"])
    elif mutation == "source":
        r["identity"]["source"]["research/layerrope_beta/model.py"] = "0" * 64
    elif mutation == "data":
        with (data / "validation.npy").open("ab") as f:
            f.write(b"changed")
    elif mutation == "validation":
        point["validation"]["window_ce"].pop()
    elif mutation == "duplicate":
        r["history"]["landmarks"].append(copy.deepcopy(point))
    write(results[0], r)
    with pytest.raises(ValueError):
        manifest(data, results, tmp_path / "rejected.json")
    assert not (tmp_path / "rejected.json").exists()


def test_jobs_are_explicit_qualification_recipes(tmp_path):
    root = tmp_path / "jobs"
    jobs(root, 81234)
    assert read(root / "M.json") == job("M", 81234, 0.001, "qualification")
    assert read(root / "ML.json")["stage"] == "qualification"
    with pytest.raises(FileExistsError):
        jobs(root, 81234)


def test_public_data_adapter_rejects_changed_arrays_before_copy(tmp_path):
    src, dest = tmp_path / "public", tmp_path / "training"
    src.mkdir()
    np.save(src / "train.npy", np.arange(8, dtype=np.uint16))
    write(
        src / "DATA.json",
        dict(
            dataset_revision=DATA_REV,
            tokenizer_revision=TOKEN_REV,
            files={"train.npy": "0" * 64},
        ),
    )
    with pytest.raises(ValueError, match="digest"):
        training_data(src, dest)
    assert not dest.exists()
