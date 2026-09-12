"""X-Plane shadow-transfer trace analysis tests."""

import json
import os
from pathlib import Path

import pytest

from rl.flight.analyze_shadow import analyze_trace
from rl.flight.contracts import FlightContract, TelemetryFrame


def load_contract() -> FlightContract:
    xplane_root = os.environ.get("XPLANE_UFO_ROOT")
    if not xplane_root:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    return FlightContract.from_json(
        Path(xplane_root) / "schemas" / "ufo-wrench-v1.json"
    )


def write_trace(path: Path, contract: FlightContract) -> None:
    rows = []
    velocity = (2.0, -1.0, 0.5)
    action = (100.0, 200.0, -10_000.0, 10.0, 20.0, 30.0)
    for sequence in range(3):
        elapsed = sequence * 0.02
        frame = TelemetryFrame.create(
            contract,
            episode_id="test",
            sequence=sequence,
            monotonic_ns=1_000_000_000 + sequence * 20_000_000,
            dt_s=0.02,
            observation=(
                velocity[0] * elapsed,
                velocity[1] * elapsed,
                -100.0 + velocity[2] * elapsed,
                *velocity,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            goal=(0.0, 0.0, 0.0, 0.0),
            requested_action=action,
            applied_action=action,
            simulator_total_wrench=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            aerodynamic_wrench=(-1.0, -2.0, -3.0, -4.0, -5.0, -6.0),
            vehicle_mass_kg=1600.0 + sequence,
            vehicle_inertia_kg_m2=(100.0 + sequence, 200.0, 300.0),
        )
        rows.append(
            {
                "schema_version": 1,
                "kind": "shadow_action",
                "checkpoint_sha256": "checkpoint",
                "model_sha256": "model",
                "received_monotonic_ns": frame.monotonic_ns + 10_000,
                "inference_ns": 20_000 + sequence,
                "telemetry": json.loads(frame.to_wire()),
                "proposed_action": action,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_analyze_consistent_shadow_trace(tmp_path):
    contract = load_contract()
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, contract)
    report = analyze_trace(trace, contract)
    assert report["frames"] == 3
    assert report["episodes"] == 1
    assert report["sequence_gaps"] == 0
    assert report["monotonic_errors"] == 0
    assert report["monotonic_elapsed_s"] == pytest.approx(0.04)
    assert report["position_velocity_consistency_rmse_mps"] < 1e-12
    assert report["quaternion_maximum_norm_error"] == 0.0
    assert report["action_normalized_delta"]["maximum"] == 0.0
    assert report["applied_action_frames"] == 3
    assert report["requested_action_frames"] == 3
    assert report["requested_action_ranges"]["force_body_x"] == {
        "minimum": 100.0,
        "maximum": 100.0,
    }
    assert report["applied_action_ranges"]["force_body_z"] == {
        "minimum": -10_000.0,
        "maximum": -10_000.0,
    }
    assert report["simulator_total_wrench_frames"] == 3
    assert report["simulator_total_wrench_ranges"]["moment_body_z"] == {
        "minimum": 6.0,
        "maximum": 6.0,
    }
    assert report["aerodynamic_wrench_frames"] == 3
    assert report["aerodynamic_wrench_ranges"]["force_body_x"] == {
        "minimum": -1.0,
        "maximum": -1.0,
    }
    assert report["vehicle_mass_frames"] == 3
    assert report["vehicle_mass_kg"]["median"] == 1601.0
    assert report["vehicle_inertia_frames"] == 3
    assert report["vehicle_inertia_kg_m2"]["x"]["median"] == 101.0


def test_analyze_rejects_non_shadow_record(tmp_path):
    contract = load_contract()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"kind": "control"}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        analyze_trace(trace, contract)
