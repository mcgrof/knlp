"""Standalone-model matching tests for X-Plane UFO traces."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rl.flight.analyze_ufo_model_match import analyze_model_match
from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.ufo_reference import UfoReferenceParameters


def load_contract() -> FlightContract:
    xplane_root = os.environ.get("XPLANE_UFO_ROOT")
    if not xplane_root:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    return FlightContract.from_json(
        Path(xplane_root) / "schemas" / "ufo-wrench-v1.json"
    )


def write_trace(
    path: Path,
    contract: FlightContract,
    *,
    angular_scale: float = 1.0,
) -> None:
    parameters = UfoReferenceParameters()
    dt_s = 0.01
    inertia = np.asarray(parameters.inertia_kg_m2)
    linear_drag = np.asarray(parameters.linear_drag_n_per_mps)
    angular_drag = np.asarray(parameters.angular_drag_nm_per_radps)
    position = np.asarray((0.0, 0.0, -100.0))
    velocity = np.asarray((2.0, -1.0, 0.5))
    omega = np.asarray((0.1, -0.07, 0.05))
    rows = []
    for sequence in range(2_001):
        time_s = sequence * dt_s
        requested = np.asarray(
            (
                1500.0 * np.sin(0.71 * time_s),
                1100.0 * np.cos(0.53 * time_s),
                -12000.0 + 900.0 * np.sin(0.37 * time_s),
                700.0 * np.sin(0.83 * time_s),
                900.0 * np.cos(0.61 * time_s),
                500.0 * np.sin(1.07 * time_s),
            )
        )
        acceleration = requested[:3] / parameters.mass_kg
        acceleration -= linear_drag * velocity / parameters.mass_kg
        acceleration += np.asarray((0.0, 0.0, parameters.gravity_mps2))
        angular_acceleration = (
            requested[3:]
            - np.cross(omega, inertia * omega)
            - angular_drag * omega
        ) / inertia
        if sequence:
            velocity += acceleration * dt_s
            omega += angular_scale * angular_acceleration * dt_s
            position += velocity * dt_s
        total = np.zeros(6)
        aerodynamic = np.zeros(6)
        applied = np.zeros(6)
        frame = TelemetryFrame.create(
            contract,
            episode_id="synthetic-model-match",
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
            requested_action=requested,
            applied_action=applied,
            simulator_total_wrench=total,
            aerodynamic_wrench=aerodynamic,
            vehicle_mass_kg=918.0,
            vehicle_inertia_kg_m2=(1.0, 2.0, 3.0),
        )
        rows.append({"kind": "shadow_action", "telemetry": json.loads(frame.to_wire())})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_accepts_target_model_and_ignores_reported_live_inertia(tmp_path):
    contract = load_contract()
    trace = tmp_path / "matched.jsonl"
    write_trace(trace, contract)

    report = analyze_model_match(trace, contract)

    assert report["frames"] == 2_001
    assert report["linear_response"]["gate_passed"]
    assert report["angular_response"]["gate_passed"]
    assert report["model_match_gate_passed"]


def test_rejects_rotational_mismatch(tmp_path):
    contract = load_contract()
    trace = tmp_path / "mismatched.jsonl"
    write_trace(trace, contract, angular_scale=0.6)

    report = analyze_model_match(trace, contract)

    assert report["linear_response"]["gate_passed"]
    assert not report["angular_response"]["gate_passed"]
    assert not report["model_match_gate_passed"]


def test_rejects_trace_without_logical_wrench(tmp_path):
    contract = load_contract()
    trace = tmp_path / "missing.jsonl"
    write_trace(trace, contract)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    for row in rows:
        del row["telemetry"]["requested_action"]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="requested_action is missing"):
        analyze_model_match(trace, contract)
