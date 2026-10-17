# SPDX-License-Identifier: MIT
"""Exercise execution barriers and inferential failure cases on CPU artifacts."""

import json
from pathlib import Path
import pytest
from research.layerrope_beta.time_budget.protocol import freeze
from research.layerrope_beta.time_budget.run import (
    prepare,
    load_card,
    resource_gate,
    Supervisor,
    final_training_barrier,
    job,
    seeds_in,
    worker,
)
from research.mo_variants.stage_c_common import write, sha


def fixture_card(root):
    data = root / "data"
    (data / "training").mkdir(parents=True)
    (data / "sealed").mkdir()
    for p in (data / "training/TRAINING.json", data / "sealed/SEALED.json"):
        write(p, {})
    write(
        data / "PREPARED.json",
        dict(
            training_manifest_sha256=sha(data / "training/TRAINING.json"),
            sealed_manifest_sha256=sha(data / "sealed/SEALED.json"),
        ),
    )
    prior = root / "PRIOR.json"
    write(prior, dict(jobs=[dict(seed=77311)], pilot_seeds=[77301]))
    audit = root / "AUDIT.json"
    write(audit, [dict(path=str(prior), sha256=sha(prior))])
    card = root / "CARD.json"
    prepare(data, audit, card)
    return card


def test_frozen_card_rejects_changed_data_and_prior_seed_use(tmp_path):
    card = fixture_card(tmp_path)
    assert load_card(card)["seed_audit"]["collisions"] == []
    write(tmp_path / "data/training/TRAINING.json", dict(changed=True))
    with pytest.raises(ValueError, match="Data capability changed"):
        load_card(card)
    prior = tmp_path / "PRIOR.json"
    write(prior, dict(seed=77611))
    audit = tmp_path / "AUDIT.json"
    write(audit, [dict(path=str(prior), sha256=sha(prior))])
    with pytest.raises(ValueError, match="prior use"):
        prepare(tmp_path / "data", audit, tmp_path / "OTHER.json")


def test_resource_decision_requires_matching_explicit_action_and_caps(tmp_path):
    card = fixture_card(tmp_path)
    decision = tmp_path / "RESOURCE.json"
    value = dict(
        authorized=False,
        action="qualify",
        card_sha256=sha(card),
        gpu_count=2,
        max_seconds=3600,
        max_aggregate_gpu_hours=2,
        max_output_bytes=50_000_000_000,
    )
    write(decision, value)
    with pytest.raises(ValueError, match="explicit resource decision"):
        resource_gate(card, decision, "qualify")
    value["authorized"] = True
    write(decision, value)
    assert resource_gate(card, decision, "qualify")[1]["authorized"]
    with pytest.raises(ValueError, match="explicit resource decision"):
        resource_gate(card, decision, "run")
    value["max_seconds"] = 0
    write(decision, value)
    with pytest.raises(ValueError, match="finite process limit"):
        resource_gate(card, decision, "qualify")


def test_supervisor_invokes_real_module_and_retains_failed_attempt(
    tmp_path, monkeypatch
):
    card = fixture_card(tmp_path)
    calls = []

    class Process:
        returncode = 1

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))

        def poll(self):
            return self.returncode

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setattr("subprocess.Popen", Process)
    s = Supervisor(
        card,
        dict(max_seconds=30, max_aggregate_gpu_hours=1, max_output_bytes=10**8),
        tmp_path / "run",
    )
    with pytest.raises(RuntimeError, match="Worker failed"):
        s.batch([dict(kind="train"), dict(kind="train")])
    assert [slot for _, slot in calls] == ["3", "7"]
    assert all(cmd[2] == "research.layerrope_beta.time_budget.run" for cmd, _ in calls)
    assert all(
        x["status"] == "failed"
        for x in json.loads((s.root / "OPERATIONS.json").read_text())
    )


def test_seed_audit_includes_scalar_pilot_and_probe_fields():
    assert set(
        seeds_in(dict(pilot_seed=77601, probe_seed=77690, final_seeds=[77611]))
    ) == {77601, 77690, 77611}


def final_receipts(root, card):
    selection = dict(selected_lrs={row: 0.001 for row in card["protocol"]["rows"]})
    receipts = []
    for row in card["protocol"]["rows"]:
        for seed in card["protocol"]["final_seeds"]:
            path = root / f"{row}-{seed}.json"
            frames = [
                dict(
                    target_seconds=t,
                    training_seconds=t,
                    preceding_training_seconds=t - 0.1,
                    last_update_seconds=0.1,
                    timing_valid=True,
                    step=int(t),
                    validation=dict(ce=2.0),
                )
                for t in card["protocol"]["landmarks"]
            ]
            write(
                path,
                dict(
                    status="complete",
                    identity=dict(
                        job=job(row, seed, 0.001, "final"),
                        source=card["source"],
                        protocol_sha256=card["protocol_sha256"],
                        data_sha256=card["training_sha256"],
                    ),
                    history=dict(landmarks=frames),
                    training_seconds=787.0,
                    completed_steps=787,
                    cursor=dict(cursor=787 * 8),
                    common_initial_weights={"matrix": str(seed)},
                ),
            )
            receipts.append(dict(path=str(path), sha256=sha(path)))
    return selection, receipts


@pytest.mark.parametrize("mismatch", ["source", "lr", "clock", "initial_weights"])
def test_final_barrier_rejects_other_sources_recipes_clocks_and_unpaired_weights(
    tmp_path, mismatch
):
    card_path = fixture_card(tmp_path)
    card = load_card(card_path)
    selection, receipts = final_receipts(tmp_path, card)
    assert len(final_training_barrier(card, selection, receipts)) == 32
    path = Path(receipts[0]["path"])
    value = json.loads(path.read_text())
    if mismatch == "source":
        value["identity"]["source"] = {}
    elif mismatch == "lr":
        value["identity"]["job"]["lr"] = 0.03
    elif mismatch == "clock":
        value["history"]["landmarks"][-1]["training_seconds"] += 50
    else:
        value["common_initial_weights"] = {"matrix": "different"}
    write(path, value)
    receipts[0]["sha256"] = sha(path)
    with pytest.raises(ValueError):
        final_training_barrier(card, selection, receipts)


def test_incomplete_matrix_cannot_open_final_holdout(tmp_path, monkeypatch):
    card_path = fixture_card(tmp_path)
    card = load_card(card_path)
    selection, receipts = final_receipts(tmp_path, card)
    selection["card_sha256"] = sha(card_path)
    path = tmp_path / "selection.json"
    write(path, selection)

    def deny(*args, **kwargs):
        raise AssertionError("Final data was accessed before the matrix barrier")

    monkeypatch.setattr("numpy.load", deny)
    with pytest.raises(ValueError, match="Final matrix barrier"):
        worker(
            card_path,
            dict(
                kind="evaluate",
                selection=dict(path=str(path), sha256=sha(path)),
                completed_final_training=receipts[:-1],
                result=receipts[0]["path"],
            ),
            tmp_path / "out",
        )
