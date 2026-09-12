"""Tests for learned fixed-wing formation flight."""

import os
from pathlib import Path

import numpy as np
import pytest

from rl.envs.fighter_env import DEFAULT_CONTRACT
from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.fighter_swarm import RlF14Swarm


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
