# SPDX-License-Identifier: MIT
"""Exercise frozen capabilities, final-data admission and single-policy floors."""

from pathlib import Path
import pytest
from research.layerrope_beta.qknorm.analysis import floor, summary
from research.layerrope_beta.qknorm.protocol import freeze, menu
from research.layerrope_beta.qknorm.run import (
    prepare,
    load_card,
    resource_gate,
    jobs,
    name,
    training_barrier,
    worker,
)
from research.mo_variants.stage_c_common import sha, write, read


def fixture_card(root):
    data = root / "data"
    for directory in ("training", "sealed"):
        (data / directory).mkdir(parents=True)
    for path in (data / "training/TRAINING.json", data / "sealed/SEALED.json"):
        write(path, {})
    write(
        data / "PREPARED.json",
        dict(
            training_manifest_sha256=sha(data / "training/TRAINING.json"),
            sealed_manifest_sha256=sha(data / "sealed/SEALED.json"),
        ),
    )
    previous = root / "PREVIOUS.json"
    write(previous, dict(final_seeds=[77611, 77612, 77613], probe_seed=77690))
    audit = root / "AUDIT.json"
    write(audit, [dict(path=str(previous), sha256=sha(previous))])
    path = root / "CARD.json"
    prepare(data, audit, path)
    return path


def test_frozen_seeds_data_and_protocol(tmp_path):
    path = fixture_card(tmp_path)
    card = load_card(path)
    assert card["seed_audit"]["collisions"] == []
    assert len(jobs(card)) == 24
    assert len({name(j) for j in jobs(card)}) == 24
    write(tmp_path / "data/training/TRAINING.json", dict(changed=True))
    with pytest.raises(ValueError, match="capability changed"):
        load_card(path)
    previous = tmp_path / "PREVIOUS.json"
    write(previous, dict(final_seeds=[77811]))
    audit = tmp_path / "AUDIT.json"
    write(audit, [dict(path=str(previous), sha256=sha(previous))])
    with pytest.raises(ValueError, match="seed audit failed"):
        prepare(tmp_path / "data", audit, tmp_path / "OTHER.json")


def test_explicit_resource_admission(tmp_path):
    card = fixture_card(tmp_path)
    resource = tmp_path / "RESOURCE.json"
    value = dict(
        authorized=False,
        action="qualify",
        card_sha256=sha(card),
        gpu_count=1,
        max_aggregate_gpu_hours=12,
        max_seconds=3600,
        max_total_usd=60,
        max_output_bytes=90_000_000_000,
    )
    write(resource, value)
    with pytest.raises(ValueError, match="explicit resource"):
        resource_gate(card, resource, "qualify")
    value["authorized"] = True
    write(resource, value)
    assert resource_gate(card, resource, "qualify")[1]["gpu_count"] == 1
    value["gpu_count"] = 2
    write(resource, value)
    with pytest.raises(ValueError, match="One GPU"):
        resource_gate(card, resource, "qualify")


def test_incomplete_or_unpaired_runs_never_open_final(tmp_path, monkeypatch):
    card = fixture_card(tmp_path)

    def denied(*args, **kwargs):
        raise AssertionError("Final tokens opened")

    monkeypatch.setattr("numpy.load", denied)
    with pytest.raises(FileNotFoundError):
        worker(
            card,
            dict(
                kind="evaluate",
                run_root=str(tmp_path / "run"),
                run_name="R-off-seed77811",
            ),
        )


def test_bytes_and_all_seed_single_codec_floor():
    expected = [16384, 9984, 9472, 9984, 9472, 3584, 3072, 3072, 2560, 3072]
    assert [c["storage"]["ideal_total_bytes"] for c in menu()] == expected
    cells = [
        dict(
            policy=codec,
            seed=s,
            ce=2.0 if (s + codec) % 2 else 2.2,
            ideal_total_bytes=3072,
        )
        for codec in (0, 1)
        for s in (1, 2, 3)
    ]
    assert floor(cells, [1, 2, 3], {s: 2.1 for s in (1, 2, 3)})["minimum_bytes"] is None
    cells.extend(
        dict(policy=2, seed=s, ce=2.05, ideal_total_bytes=3584) for s in (1, 2, 3)
    )
    assert floor(cells, [1, 2, 3], {s: 2.1 for s in (1, 2, 3)})["minimum_bytes"] == 3584
    assert summary([-0.02, 0.01, -0.03])["values"] == [-0.02, 0.01, -0.03]


def test_barrier_checks_pairing_and_equal_token_coverage(tmp_path):
    path = fixture_card(tmp_path)
    card = load_card(path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"deterministic barrier fixture")
    for job in jobs(card):
        write(
            tmp_path / "run/training" / name(job) / "RESULT.json",
            dict(
                status="complete",
                identity=dict(
                    job=job,
                    protocol_sha256=card["protocol_sha256"],
                    data_sha256=card["training_sha256"],
                    source=card["source"],
                ),
                completed_steps=1001,
                cursor=dict(cursor=1001 * 8),
                common_initial_weights={"shared": str(job["seed"])},
                initial_parameter_hashes={
                    "shared": str(job["seed"]),
                    **({"qknorm_gain": "constant"} if job["qknorm"] else {}),
                },
                history=dict(
                    landmarks=[
                        dict(
                            target_seconds=787.0,
                            timing_valid=True,
                            checkpoint=dict(
                                path=str(checkpoint), sha256=sha(checkpoint)
                            ),
                        )
                    ],
                    update_landmarks=[
                        dict(
                            step=1000,
                            checkpoint=dict(
                                path=str(checkpoint), sha256=sha(checkpoint)
                            ),
                        )
                    ],
                ),
            ),
        )
    assert len(training_barrier(card, tmp_path / "run")) == 24
    result = tmp_path / "run/training/R-on-seed77811/RESULT.json"
    value = read(result)
    value["initial_parameter_hashes"]["shared"] = "different"
    write(result, value)
    with pytest.raises(ValueError, match="initial weights"):
        training_barrier(card, tmp_path / "run")
    value["initial_parameter_hashes"]["shared"] = "77811"
    value["history"]["update_landmarks"] = []
    write(result, value)
    with pytest.raises(ValueError, match="token endpoint coverage"):
        training_barrier(card, tmp_path / "run")
