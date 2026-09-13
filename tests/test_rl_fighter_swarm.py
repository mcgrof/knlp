"""Tests for learned fixed-wing formation flight."""

import os
from pathlib import Path

import numpy as np
import pytest

from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.fighter_swarm import DEFAULT_CONTRACT, RlF14Swarm
from rl.flight.geometry import quaternion_from_euler


def _actor(path: Path, contract: FlightContract) -> None:
    hidden = 4
    input_width = contract.observation.width + contract.goal.width
    np.savez(
        path,
        format_version=np.asarray(1),
        contract_hash=np.asarray(contract.digest),
        checkpoint_sha256=np.asarray("test-checkpoint"),
        observation_mean=np.zeros(input_width, dtype=np.float32),
        observation_scale=np.ones(input_width, dtype=np.float32),
        action_mid=np.asarray((0.5, 0.0, 0.0, 0.0), dtype=np.float32),
        action_scale=np.asarray((0.5, 1.0, 1.0, 1.0), dtype=np.float32),
        layer_0_weight=np.zeros((hidden, input_width), dtype=np.float32),
        layer_0_bias=np.zeros(hidden, dtype=np.float32),
        layer_1_weight=np.zeros((hidden, hidden), dtype=np.float32),
        layer_1_bias=np.zeros(hidden, dtype=np.float32),
        layer_2_weight=np.zeros((contract.action.width, hidden), dtype=np.float32),
        layer_2_bias=np.zeros(contract.action.width, dtype=np.float32),
    )


def _player_frame(contract: FlightContract, position=(0.0, 0.0, -1500.0)):
    return TelemetryFrame.create(
        contract,
        episode_id="f14-formation-test",
        sequence=1,
        monotonic_ns=1_000,
        dt_s=0.02,
        observation=(
            *position,
            180.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        goal=(0.0, 0.0, 0.0, 0.0),
    )


def test_f14_swarm_uses_native_aircraft_and_follows_relocation(tmp_path):
    root_value = os.environ.get("XPLANE_UFO_ROOT")
    if not root_value:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    output_contract = FlightContract.from_json(
        Path(root_value) / "schemas/ufo-wrench-v1.json"
    )
    fighter_contract = FlightContract.from_json(DEFAULT_CONTRACT)
    model = tmp_path / "fighter.npz"
    _actor(model, fighter_contract)
    swarm = RlF14Swarm(output_contract, model, 4, DEFAULT_CONTRACT)

    frame = _player_frame(output_contract)
    poses = swarm.update(frame)
    assert [pose.slot for pose in poses] == list(range(4))
    assert len({pose.position_ned_m for pose in poses}) == 4
    assert all(pose.native_visual for pose in poses)
    assert all(not pose.shield_requested for pose in poses)
    assert swarm.shots == ()
    assert swarm.control_frame(frame) is frame
    assert swarm.checkpoint_sha256 == "test-checkpoint"
    assert swarm.position_gain_per_s == 0.30
    assert swarm.maximum_formation_speed_mps == 780.0

    displacement = np.asarray((8_000.0, 1_200.0, -600.0))
    blink_observation = np.asarray(frame.observation, dtype=np.float64)
    blink_observation[:3] += displacement
    blink = TelemetryFrame.create(
        output_contract,
        episode_id=frame.episode_id,
        sequence=2,
        monotonic_ns=21_000_000,
        dt_s=0.02,
        observation=blink_observation,
        goal=frame.goal,
    )
    after = swarm.update(blink)
    for before, moved in zip(poses, after, strict=True):
        actual = np.asarray(moved.position_ned_m) - np.asarray(
            before.position_ned_m
        )
        # Followers continue their normal flight during the relocated frame.
        assert actual == pytest.approx(displacement, abs=5.0)


def test_legacy_actor_keeps_its_original_formation_envelope(tmp_path):
    root_value = os.environ.get("XPLANE_UFO_ROOT")
    if not root_value:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    output_contract = FlightContract.from_json(
        Path(root_value) / "schemas/ufo-wrench-v1.json"
    )
    legacy_contract_path = (
        Path(__file__).parents[1] / "rl/contracts/fighter-controls-v1.json"
    )
    legacy_contract = FlightContract.from_json(legacy_contract_path)
    model = tmp_path / "legacy-fighter.npz"
    _actor(model, legacy_contract)
    swarm = RlF14Swarm(
        output_contract,
        model,
        2,
        legacy_contract_path,
    )
    assert swarm.position_gain_per_s == 0.12
    assert swarm.maximum_position_correction_mps == 90.0
    assert swarm.maximum_formation_speed_mps == 280.0


def test_f14_formation_goal_includes_turning_slot_velocity(tmp_path):
    root_value = os.environ.get("XPLANE_UFO_ROOT")
    if not root_value:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    output_contract = FlightContract.from_json(
        Path(root_value) / "schemas/ufo-wrench-v1.json"
    )
    fighter_contract = FlightContract.from_json(DEFAULT_CONTRACT)
    model = tmp_path / "fighter.npz"
    _actor(model, fighter_contract)
    swarm = RlF14Swarm(output_contract, model, 2, DEFAULT_CONTRACT)
    player = np.asarray(_player_frame(output_contract).observation)
    swarm._reset(player, "turn-test")
    fighter = swarm.fighters[0].dynamics.state_vector()

    straight = swarm._goal(0, fighter, player, 0.0)
    turning = swarm._goal(0, fighter, player, 0.05)

    assert turning[0] != pytest.approx(straight[0])
    assert turning[2] != pytest.approx(straight[2])
    assert abs(turning[2]) <= 0.08


def test_state_adapter_measures_leader_turn_rate(tmp_path):
    root_value = os.environ.get("XPLANE_UFO_ROOT")
    if not root_value:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    output_contract = FlightContract.from_json(
        Path(root_value) / "schemas/ufo-wrench-v1.json"
    )
    fighter_contract = FlightContract.from_json(DEFAULT_CONTRACT)
    model = tmp_path / "fighter.npz"
    _actor(model, fighter_contract)
    swarm = RlF14Swarm(output_contract, model, 2, DEFAULT_CONTRACT)
    player = np.asarray(_player_frame(output_contract).observation)
    swarm.update_state(
        player,
        episode_id="state-test",
        sequence=0,
        monotonic_ns=1,
        dt_s=0.02,
    )
    player[6:10] = quaternion_from_euler(0.0, 0.0, 0.002)
    swarm.update_state(
        player,
        episode_id="state-test",
        sequence=1,
        monotonic_ns=20_000_001,
        dt_s=0.02,
    )
    assert swarm.previous_player_heading_rad == pytest.approx(0.002)
