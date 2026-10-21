# SPDX-License-Identifier: MIT
"""Missing/late telemetry or a resource overrun cannot qualify serving."""

import copy
import json

import pytest
from decision_storage_fixture import memory_fixture

from tools.kv.decision.production_resources import POLICY, audit_window, placement
from tools.kv.decision.production import sha
from tools.kv.decision.production_analysis import resources


def samples():
    return [
        {
            "monotonic_ns": i * 100_000_000,
            "uuid": "GPU-fixed",
            "name": POLICY["device_name"],
            "driver_total_bytes": 80 * 1024**3,
            "driver_used_bytes": 65 * 1024**3,
        }
        for i in range(11)
    ]


def test_sampled_peak_includes_bracketing_observations():
    rows = samples()
    rows[1]["driver_used_bytes"] = 75 * 1024**3
    result = audit_window(rows, 150_000_000, 850_000_000)
    assert result["eligible"]
    assert result["samples"] == 9
    assert result["maximum_observed_driver_used_bytes"] == 75 * 1024**3


@pytest.mark.parametrize("edge", ["before", "after"])
def test_missing_interval_edge_is_not_silent(edge):
    rows = samples()[2:] if edge == "before" else samples()[:-2]
    with pytest.raises(ValueError, match="bracket"):
        audit_window(rows, 150_000_000, 850_000_000)


def test_sampling_gap_disqualifies_even_without_observed_memory_overrun():
    rows = samples()[:2] + samples()[8:]
    result = audit_window(rows, 50_000_000, 950_000_000)
    assert not result["sampling_coverage_passed"]
    assert result["envelope_passed_at_samples"]
    assert not result["eligible"]


def test_numerical_exemption_is_explicit_and_does_not_change_observed_peak():
    rows = samples()
    rows[5]["driver_used_bytes"] = 77 * 1024**3
    quality = audit_window(rows, 50_000_000, 950_000_000)
    numerical = audit_window(rows, 50_000_000, 950_000_000, enforce=False)
    assert not quality["eligible"]
    assert numerical["eligible"] and not numerical["enforced"]
    assert not numerical["envelope_passed_at_samples"]
    assert (
        quality["maximum_observed_driver_used_bytes"]
        == numerical["maximum_observed_driver_used_bytes"]
    )


@pytest.mark.parametrize("defect", ["clock", "device", "negative", "hardware"])
def test_bad_telemetry_is_rejected(defect):
    rows = copy.deepcopy(samples())
    if defect == "clock":
        rows[5]["monotonic_ns"] = rows[4]["monotonic_ns"]
    elif defect == "device":
        rows[5]["uuid"] = "GPU-other"
    elif defect == "negative":
        rows[5]["driver_used_bytes"] = -1
    else:
        rows[5]["name"] = "another GPU"
    with pytest.raises(ValueError):
        audit_window(rows, 50_000_000, 950_000_000)


def test_cpu_resources_are_disjoint_and_respect_effective_quota():
    cpu = placement(range(20, 36), 16)
    assert cpu == {
        "server": list(range(20, 28)),
        "client": [28, 29],
        "control": [30, 31],
    }
    with pytest.raises(ValueError):
        placement(range(32), 8)
    with pytest.raises(ValueError):
        placement(range(8))


def retained_resource_fixture(root):
    def save(name, row):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))

    cpu = placement(range(16), 16)
    identity = {
        "arm": "AF",
        "protocol_sha256": "protocol",
        "source_sha": "source",
        "allocation": "capacity",
        "cpu_affinity": cpu["server"],
    }
    save("server/worker-identity.json", identity)
    save(
        "DONE.json",
        {
            **identity,
            "phase": "numerical_quality",
            "worker_identity_sha256": sha(root / "server/worker-identity.json"),
        },
    )
    save(
        "cpu-contract.json",
        {
            "allowed_cpus": list(range(16)),
            "cpu_quota": 16,
            "placement": cpu,
        },
    )
    save("server-exit.json", {"returncode": -15})
    for path in ["numerical/QUALIFIED.json", "quality/START.json", "quality/DONE.json"]:
        save(path, {"client_cpu_affinity": cpu["client"]})
    rows = samples()
    (root / "driver-samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    save("startup-resources.json", audit_window(rows, 50_000_000, 150_000_000))
    for name, enforce, start, end in [
        ("numerical", False, 250_000_000, 450_000_000),
        ("quality", True, 550_000_000, 850_000_000),
    ]:
        save(
            name + "-command.json",
            {
                "started_ns": start,
                "ended_ns": end,
                "returncode": 0,
                "resources": audit_window(rows, start, end, enforce=enforce),
            },
        )
    geometry, storage = memory_fixture()
    save("server/initial-storage.json", storage)
    return {
        "source_sha": "source",
        "resource_policy": POLICY,
        "geometry": geometry,
        "persistent_budget_bytes": 4096,
    }


def test_resource_reader_replays_raw_samples_and_actual_cpu_placement(tmp_path):
    protocol = retained_resource_fixture(tmp_path)
    result = resources(tmp_path, protocol, "protocol", "AF")
    assert result["quality"]["eligible"]
    assert result["numerical"]["enforced"] is False


def test_equal_slot_probe_is_bound_to_actual_block_count_without_quality_collectors(
    tmp_path,
):
    protocol = retained_resource_fixture(tmp_path)
    identity_path = tmp_path / "server/worker-identity.json"
    identity = json.loads(identity_path.read_text())
    identity["allocation"] = "equal_slots"
    identity_path.write_text(json.dumps(identity))
    done_path = tmp_path / "DONE.json"
    done = json.loads(done_path.read_text())
    done.update(phase="equal_slots", worker_identity_sha256=sha(identity_path))
    done_path.write_text(json.dumps(done))
    protocol["equal_slot_blocks"] = 2
    result = resources(tmp_path, protocol, "protocol", "AF", phase="equal_slots")
    assert result["storage"]["physical_blocks_per_layer"] == 2
    assert "quality" not in result and "numerical" not in result
    protocol["equal_slot_blocks"] = 3
    with pytest.raises(ValueError):
        resources(tmp_path, protocol, "protocol", "AF", phase="equal_slots")


@pytest.mark.parametrize(
    "defect", ["sample", "exemption", "affinity", "worker", "exit"]
)
def test_resource_reader_rejects_inconsistent_retained_evidence(tmp_path, defect):
    protocol = retained_resource_fixture(tmp_path)
    if defect == "sample":
        path = tmp_path / "driver-samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[7]["driver_used_bytes"] = 77 * 1024**3
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    else:
        file = {
            "exemption": "quality-command.json",
            "affinity": "quality/START.json",
            "worker": "server/worker-identity.json",
            "exit": "server-exit.json",
        }[defect]
        path = tmp_path / file
        row = json.loads(path.read_text())
        if defect == "exemption":
            row["resources"]["enforced"] = False
        elif defect == "affinity":
            row["client_cpu_affinity"] = list(range(8))
        elif defect == "worker":
            row["arm"] = "SF"
        else:
            row["returncode"] = 1
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        resources(tmp_path, protocol, "protocol", "AF")
