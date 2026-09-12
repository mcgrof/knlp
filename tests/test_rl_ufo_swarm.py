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
    poses = RlUfoSwarm(contract, model, library, 5).update(frame)
    assert [pose.slot for pose in poses] == list(range(5))
    assert len({tuple(pose.position_ned_m) for pose in poses}) == 5
    for pose in poses:
        assert np.linalg.norm(pose.quaternion_body_to_ned) == pytest.approx(1.0)
