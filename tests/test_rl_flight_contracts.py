"""Tests for the reusable, versioned flight-control contract."""

import json

import pytest

from rl.flight.contracts import (
    ControlCommand,
    EnemyPose,
    EnemyShot,
    FlightContract,
    TelemetryFrame,
    VectorSpec,
)


def vector(fields, low=-1.0, high=1.0):
    width = len(fields)
    return VectorSpec(
        fields=tuple(fields),
        units=("normalized",) * width,
        low=(low,) * width,
        high=(high,) * width,
    )


@pytest.fixture
def contract():
    return FlightContract(
        name="test-flight",
        revision=1,
        nominal_dt_s=0.02,
        observation=vector(("roll", "pitch")),
        goal=vector(("target_roll",)),
        action=vector(("roll_moment", "pitch_moment")),
    )


def test_contract_digest_changes_with_semantics(contract):
    same = FlightContract(
        name=contract.name,
        revision=contract.revision,
        nominal_dt_s=contract.nominal_dt_s,
        observation=contract.observation,
        goal=contract.goal,
        action=contract.action,
    )
    changed = FlightContract(
        name=contract.name,
        revision=2,
        nominal_dt_s=contract.nominal_dt_s,
        observation=contract.observation,
        goal=contract.goal,
        action=contract.action,
    )
    assert same.digest == contract.digest
    assert changed.digest != contract.digest
    assert json.loads(contract.canonical_json())["name"] == "test-flight"


def test_contract_loads_strict_json_without_changing_its_digest(tmp_path, contract):
    path = tmp_path / "contract.json"
    path.write_text(contract.canonical_json())
    loaded = FlightContract.from_json(path)
    assert loaded == contract
    assert loaded.digest == contract.digest

    payload = json.loads(contract.canonical_json())
    payload["observation"]["unknown"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unexpected"):
        FlightContract.from_json(path)


def test_telemetry_and_control_round_trip(contract):
    frame = TelemetryFrame.create(
        contract,
        episode_id="episode-7",
        sequence=4,
        monotonic_ns=1_000,
        dt_s=0.02,
        observation=(0.2, -0.1),
        goal=(0.0,),
        requested_action=(-0.2, 0.1),
        applied_action=(-3.0, 4.0),
        simulator_total_wrench=(-3.0, 4.0),
        aerodynamic_wrench=(-2.5, 3.5),
        vehicle_mass_kg=1200.0,
        vehicle_inertia_kg_m2=(100.0, 200.0, 300.0),
    )
    command = ControlCommand.create(
        contract,
        episode_id=frame.episode_id,
        source_sequence=frame.sequence,
        issued_monotonic_ns=1_100,
        valid_until_monotonic_ns=21_100_000,
        action=(-0.2, 0.1),
    )
    assert TelemetryFrame.from_wire(frame.to_wire(), contract) == frame
    decoded = ControlCommand.from_wire(command.to_wire(), contract)
    decoded.validate(contract, frame=frame, now_monotonic_ns=1_200)
    assert decoded == command


def test_control_round_trip_with_enemy_poses(contract):
    enemies = (
        EnemyPose.create(
            slot=0,
            position_ned_m=(100.0, -20.0, -150.0),
            quaternion_body_to_ned=(1.0, 0.0, 0.0, 0.0),
        ),
        EnemyPose.create(
            slot=2,
            position_ned_m=(200.0, 40.0, -170.0),
            quaternion_body_to_ned=(1.0, 0.0, 0.0, 0.0),
            shield_requested=True,
        ),
    )
    command = ControlCommand.create(
        contract,
        episode_id="swarm",
        source_sequence=1,
        issued_monotonic_ns=100,
        valid_until_monotonic_ns=200,
        action=(0.0, 0.0),
        enemies=enemies,
        enemy_shots=(
            EnemyShot.create(
                slot=2,
                aim_position_ned_m=(0.0, 0.0, -100.0),
            ),
        ),
    )
    assert ControlCommand.from_wire(command.to_wire(), contract) == command


def test_enemy_pose_accepts_legacy_wire_without_shield_request():
    pose = EnemyPose.from_mapping(
        {
            "slot": 1,
            "position_ned_m": (0.0, 0.0, -100.0),
            "quaternion_body_to_ned": (1.0, 0.0, 0.0, 0.0),
        }
    )
    assert not pose.shield_requested


def test_control_rejects_enemy_shot_without_matching_pose(contract):
    with pytest.raises(ValueError, match="matching enemy pose"):
        ControlCommand.create(
            contract,
            episode_id="swarm",
            source_sequence=1,
            issued_monotonic_ns=100,
            valid_until_monotonic_ns=200,
            action=(0.0, 0.0),
            enemy_shots=(
                EnemyShot.create(
                    slot=0,
                    aim_position_ned_m=(0.0, 0.0, -100.0),
                ),
            ),
        )


def test_control_rejects_unordered_enemy_slots(contract):
    enemies = tuple(
        EnemyPose.create(
            slot=slot,
            position_ned_m=(0.0, 0.0, -100.0),
            quaternion_body_to_ned=(1.0, 0.0, 0.0, 0.0),
        )
        for slot in (1, 0)
    )
    with pytest.raises(ValueError, match="unique and increasing"):
        ControlCommand.create(
            contract,
            episode_id="swarm",
            source_sequence=1,
            issued_monotonic_ns=100,
            valid_until_monotonic_ns=200,
            action=(0.0, 0.0),
            enemies=enemies,
        )


def test_telemetry_accepts_legacy_frames_without_vehicle_diagnostics(contract):
    frame = TelemetryFrame.create(
        contract,
        episode_id="legacy",
        sequence=1,
        monotonic_ns=1_000,
        dt_s=0.02,
        observation=(0.0, 0.0),
        goal=(0.0,),
    )
    payload = json.loads(frame.to_wire())
    del payload["requested_action"]
    del payload["applied_action"]
    del payload["simulator_total_wrench"]
    del payload["aerodynamic_wrench"]
    del payload["vehicle_mass_kg"]
    del payload["vehicle_inertia_kg_m2"]
    decoded = TelemetryFrame.from_wire(json.dumps(payload).encode(), contract)
    assert decoded.requested_action is None
    assert decoded.applied_action is None
    assert decoded.simulator_total_wrench is None
    assert decoded.aerodynamic_wrench is None
    assert decoded.vehicle_mass_kg is None
    assert decoded.vehicle_inertia_kg_m2 is None


def test_contract_rejects_mismatches_and_stale_control(contract):
    frame = TelemetryFrame.create(
        contract,
        episode_id="episode-7",
        sequence=4,
        monotonic_ns=1_000,
        dt_s=0.02,
        observation=(0.2, -0.1),
        goal=(0.0,),
    )
    with pytest.raises(ValueError, match="width"):
        TelemetryFrame.create(
            contract,
            episode_id="episode-7",
            sequence=5,
            monotonic_ns=2_000,
            dt_s=0.02,
            observation=(0.2,),
            goal=(0.0,),
        )
    with pytest.raises(ValueError, match="outside"):
        ControlCommand.create(
            contract,
            episode_id="episode-7",
            source_sequence=4,
            issued_monotonic_ns=1_100,
            valid_until_monotonic_ns=2_000,
            action=(2.0, 0.0),
        )
    with pytest.raises(ValueError, match="vehicle_mass_kg"):
        TelemetryFrame.create(
            contract,
            episode_id="episode-7",
            sequence=5,
            monotonic_ns=2_000,
            dt_s=0.02,
            observation=(0.2, -0.1),
            goal=(0.0,),
            vehicle_mass_kg=0.0,
        )
    with pytest.raises(ValueError, match="requested_action"):
        TelemetryFrame.create(
            contract,
            episode_id="episode-7",
            sequence=5,
            monotonic_ns=2_000,
            dt_s=0.02,
            observation=(0.2, -0.1),
            goal=(0.0,),
            requested_action=(2.0, 0.0),
        )
    with pytest.raises(ValueError, match="vehicle_inertia_kg_m2"):
        TelemetryFrame.create(
            contract,
            episode_id="episode-7",
            sequence=5,
            monotonic_ns=2_000,
            dt_s=0.02,
            observation=(0.2, -0.1),
            goal=(0.0,),
            vehicle_inertia_kg_m2=(1.0, -2.0, 3.0),
        )
    with pytest.raises(ValueError, match="simulator_total_wrench"):
        TelemetryFrame.create(
            contract,
            episode_id="episode-7",
            sequence=5,
            monotonic_ns=2_000,
            dt_s=0.02,
            observation=(0.2, -0.1),
            goal=(0.0,),
            simulator_total_wrench=(1.0,),
        )
    command = ControlCommand.create(
        contract,
        episode_id="episode-7",
        source_sequence=4,
        issued_monotonic_ns=1_100,
        valid_until_monotonic_ns=2_000,
        action=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="expired"):
        command.validate(contract, frame=frame, now_monotonic_ns=2_001)


def test_wire_messages_fail_closed(contract):
    frame = TelemetryFrame.create(
        contract,
        episode_id="episode-7",
        sequence=4,
        monotonic_ns=1_000,
        dt_s=0.02,
        observation=(0.2, -0.1),
        goal=(0.0,),
    )
    payload = json.loads(frame.to_wire())
    payload["contract_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        TelemetryFrame.from_wire(json.dumps(payload).encode(), contract)
    payload = json.loads(frame.to_wire())
    payload["surprise"] = True
    with pytest.raises(ValueError, match="unexpected"):
        TelemetryFrame.from_wire(json.dumps(payload).encode(), contract)
