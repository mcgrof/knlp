"""UFO response-identification tests."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from rl.flight.analyze_ufo_response import analyze_response
from rl.flight.contracts import FlightContract, TelemetryFrame


def load_contract() -> FlightContract:
    xplane_root = os.environ.get("XPLANE_UFO_ROOT")
    if not xplane_root:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    return FlightContract.from_json(
        Path(xplane_root) / "schemas" / "ufo-wrench-v1.json"
    )


def state_at(time_s: float) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.asarray(
        (
            10.0 + 2.0 * math.sin(0.7 * time_s),
            -3.0 + 1.5 * math.cos(1.1 * time_s),
            0.7 * math.sin(0.9 * time_s),
        )
    )
    omega = np.asarray(
        (
            0.20 * math.sin(0.8 * time_s),
            0.16 * math.cos(0.6 * time_s),
            0.12 * math.sin(1.0 * time_s),
        )
    )
    return velocity, omega


def write_response_trace(path: Path, contract: FlightContract) -> None:
    dt_s = 0.01
    mass_kg = 1_000.0
    inertia = np.asarray((300.0, 450.0, 600.0))
    drag = np.asarray((12.0, 24.0, 36.0))
    angular_drag = np.asarray((40.0, 50.0, 60.0))
    position = np.asarray((0.0, 0.0, -100.0))
    previous_velocity, previous_omega = state_at(0.0)
    rows = []
    for sequence in range(2_001):
        time_s = sequence * dt_s
        velocity, omega = state_at(time_s)
        if sequence:
            acceleration = (velocity - previous_velocity) / dt_s
            angular_acceleration = (omega - previous_omega) / dt_s
            position += velocity * dt_s
        else:
            next_velocity, next_omega = state_at(dt_s)
            acceleration = (next_velocity - velocity) / dt_s
            angular_acceleration = (next_omega - omega) / dt_s

        aerodynamic_force = -drag * velocity
        total_force = mass_kg * (acceleration - np.asarray((0.0, 0.0, 9.80665)))
        total_moment = inertia * angular_acceleration + np.cross(omega, inertia * omega)
        aerodynamic = np.concatenate((aerodynamic_force, -angular_drag * omega))
        total = np.concatenate((total_force, total_moment))
        applied = total - aerodynamic
        frame = TelemetryFrame.create(
            contract,
            episode_id="synthetic-response",
            sequence=sequence,
            monotonic_ns=1_000_000_000 + sequence * 10_000_000,
            dt_s=dt_s,
            observation=(
                *position,
                *velocity,
                1.0,
                0.0,
                0.0,
                0.0,
                *omega,
            ),
            goal=(0.0, 0.0, 0.0, 0.0),
            applied_action=applied,
            simulator_total_wrench=total,
            aerodynamic_wrench=aerodynamic,
            vehicle_mass_kg=mass_kg,
        )
        rows.append(
            {
                "kind": "shadow_action",
                "telemetry": json.loads(frame.to_wire()),
            }
        )
        previous_velocity = velocity
        previous_omega = omega
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_identifies_synthetic_mass_inertia_and_drag(tmp_path):
    contract = load_contract()
    trace = tmp_path / "response.jsonl"
    write_response_trace(trace, contract)

    report = analyze_response(trace, contract)

    assert report["frames"] == 2_001
    assert report["airborne_frames"] == 2_001
    assert report["response_windows"] >= 100
    assert report["total_force_consistency"]["gate_passed"]
    assert report["inertia"]["inertia_kg_m2"] == pytest.approx(
        (300.0, 450.0, 600.0), rel=2e-4
    )
    assert report["inertia"]["gate_passed"]
    assert [
        axis["coefficient_n_per_mps"] for axis in report["aerodynamic_linear_drag_ned"]
    ] == pytest.approx((12.0, 24.0, 36.0))
    assert report["linear_drag_gate_passed"]
    assert [
        axis["coefficient_nm_per_radps"]
        for axis in report["aerodynamic_angular_drag_body"]
    ] == pytest.approx((40.0, 50.0, 60.0))
    assert report["angular_drag_gate_passed"]
    assert report["mass_and_inertia_gate_passed"]
    assert report["full_standalone_model_gate_passed"]


def test_rejects_trace_without_response_diagnostics(tmp_path):
    contract = load_contract()
    frame = TelemetryFrame.create(
        contract,
        episode_id="missing",
        sequence=0,
        monotonic_ns=1,
        dt_s=0.01,
        observation=(0.0,) * 6 + (1.0, 0.0, 0.0, 0.0) + (0.0,) * 3,
        goal=(0.0,) * 4,
    )
    trace = tmp_path / "missing.jsonl"
    trace.write_text(
        json.dumps(
            {
                "kind": "shadow_action",
                "telemetry": json.loads(frame.to_wire()),
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="response diagnostics are missing"):
        analyze_response(trace, contract)
