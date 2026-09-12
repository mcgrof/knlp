"""Tests for actor-driven multiplayer UFO motion."""

import os
from pathlib import Path

import numpy as np
import pytest

from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.ufo_swarm import RlUfoSwarm


def test_swarm_emits_distinct_normalized_enemy_poses(tmp_path):
    root_value = os.environ.get("XPLANE_UFO_ROOT")
    if not root_value:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    root = Path(root_value)
    library = root / "build/libxplane_ufo_dynamics.so"
    if not library.is_file():
        pytest.skip("xplane-ufo dynamics library is not built")
    contract = FlightContract.from_json(root / "schemas/ufo-wrench-v1.json")
    model = tmp_path / "actor.npz"
    hidden = 4
    input_width = contract.observation.width + contract.goal.width
    np.savez(
        model,
        format_version=np.asarray(1),
        contract_hash=np.asarray(contract.digest),
        checkpoint_sha256=np.asarray("test-checkpoint"),
        observation_mean=np.zeros(input_width, dtype=np.float32),
        observation_scale=np.ones(input_width, dtype=np.float32),
        action_mid=np.zeros(contract.action.width, dtype=np.float32),
        action_scale=np.ones(contract.action.width, dtype=np.float32),
        layer_0_weight=np.zeros((hidden, input_width), dtype=np.float32),
        layer_0_bias=np.zeros(hidden, dtype=np.float32),
        layer_1_weight=np.zeros((hidden, hidden), dtype=np.float32),
        layer_1_bias=np.zeros(hidden, dtype=np.float32),
        layer_2_weight=np.zeros((contract.action.width, hidden), dtype=np.float32),
        layer_2_bias=np.zeros(contract.action.width, dtype=np.float32),
    )
    frame = TelemetryFrame.create(
        contract,
        episode_id="swarm-test",
        sequence=1,
        monotonic_ns=1_000,
        dt_s=0.02,
        observation=(
            0.0,
            0.0,
            -100.0,
            0.0,
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
    swarm = RlUfoSwarm(contract, model, library, 5)
    poses = swarm.update(frame)
    assert [pose.slot for pose in poses] == list(range(5))
    assert len({tuple(pose.position_ned_m) for pose in poses}) == 5
    for pose in poses:
        assert np.linalg.norm(pose.quaternion_body_to_ned) == pytest.approx(1.0)
    assert [shot.slot for shot in swarm.shots] == [0]
    assert swarm.shots[0].aim_position_ned_m == pytest.approx(
        frame.observation[:3]
    )
    controlled = swarm.control_frame(frame)
    assert swarm.enemies[0].goal[0] == 75.0
    assert controlled.goal[0] == 75.0
    assert controlled.goal[1] == 40.0
    assert controlled.goal[3] == 1.2

    enemy_state = swarm.enemies[0].world_state()
    swarm.enemies[0].goal_until_s = 0.0
    close_observation = enemy_state.copy()
    close_observation[:3] += (100.0, 0.0, 0.0)
    close_frame = TelemetryFrame.create(
        contract,
        episode_id="swarm-test",
        sequence=2,
        monotonic_ns=21_000_000,
        dt_s=0.02,
        observation=close_observation,
        goal=(0.0, 0.0, 0.0, 0.0),
    )
    close_poses = swarm.update(close_frame)
    assert close_poses[0].shield_requested
    assert swarm.enemies[0].shield_used


def test_formation_tracks_player_without_firing(tmp_path):
    root_value = os.environ.get("XPLANE_UFO_ROOT")
    if not root_value:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    root = Path(root_value)
    library = root / "build/libxplane_ufo_dynamics.so"
    if not library.is_file():
        pytest.skip("xplane-ufo dynamics library is not built")
    contract = FlightContract.from_json(root / "schemas/ufo-wrench-v1.json")
    model = tmp_path / "actor.npz"
    hidden = 4
    input_width = contract.observation.width + contract.goal.width
    np.savez(
        model,
        format_version=np.asarray(1),
        contract_hash=np.asarray(contract.digest),
        checkpoint_sha256=np.asarray("test-checkpoint"),
        observation_mean=np.zeros(input_width, dtype=np.float32),
        observation_scale=np.ones(input_width, dtype=np.float32),
        action_mid=np.zeros(contract.action.width, dtype=np.float32),
        action_scale=np.ones(contract.action.width, dtype=np.float32),
        layer_0_weight=np.zeros((hidden, input_width), dtype=np.float32),
        layer_0_bias=np.zeros(hidden, dtype=np.float32),
        layer_1_weight=np.zeros((hidden, hidden), dtype=np.float32),
        layer_1_bias=np.zeros(hidden, dtype=np.float32),
        layer_2_weight=np.zeros((contract.action.width, hidden), dtype=np.float32),
        layer_2_bias=np.zeros(contract.action.width, dtype=np.float32),
    )
    frame = TelemetryFrame.create(
        contract,
        episode_id="formation-test",
        sequence=1,
        monotonic_ns=1_000,
        dt_s=0.02,
        observation=(
            0.0,
            0.0,
            -300.0,
            40.0,
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
    swarm = RlUfoSwarm(contract, model, library, 4, behavior="formation")
    poses = swarm.update(frame)
    assert len(poses) == 4
    assert swarm.shots == ()
    assert all(not pose.shield_requested for pose in poses)
    assert [enemy.goal[0] for enemy in swarm.enemies] == pytest.approx([40.0] * 4)
    assert [pose.position_ned_m[0] for pose in poses] == pytest.approx(
        [-90.0, -90.0, -180.0, -180.0], abs=0.1
    )

    blink_displacement = np.asarray((8000.0, 1200.0, -600.0))
    blink_observation = np.asarray(frame.observation, dtype=np.float64)
    blink_observation[:3] += blink_displacement
    blink_frame = TelemetryFrame.create(
        contract,
        episode_id="formation-test",
        sequence=2,
        monotonic_ns=21_000_000,
        dt_s=0.02,
        observation=blink_observation,
        goal=(0.0, 0.0, 0.0, 0.0),
    )
    blink_poses = swarm.update(blink_frame)
    for before, after in zip(poses, blink_poses, strict=True):
        actual = np.asarray(after.position_ned_m) - np.asarray(
            before.position_ned_m
        )
        assert actual == pytest.approx(blink_displacement, abs=0.1)
