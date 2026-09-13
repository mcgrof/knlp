"""Fixed-wing fighter surrogate and reference-controller gates."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rl.controls.fighter import FighterReferenceController  # noqa: E402
from rl.envs import make_env  # noqa: E402
from rl.envs.fighter_env import FighterEnv, LIVE_CONTRACT  # noqa: E402


def rollout(env, controller, limit=3000):
    last = None
    for _ in range(limit):
        last = env.step(controller(env))
        if last[2] or last[3]:
            break
    assert last is not None
    return last


def test_fighter_env_passes_gymnasium_check():
    from gymnasium.utils.env_checker import check_env

    env = FighterEnv(max_seconds=1.0)
    check_env(env, skip_render_check=True)
    env.close()


def test_fixed_seed_trajectory_is_reproducible():
    first = FighterEnv(max_seconds=1.0)
    second = FighterEnv(max_seconds=1.0)
    try:
        a, _ = first.reset(seed=17)
        b, _ = second.reset(seed=17)
        assert np.array_equal(a, b)
        action = np.asarray((0.4, 0.2, -0.1, 0.05))
        for _ in range(25):
            a = first.step(action)[0]
            b = second.step(action)[0]
            assert np.array_equal(a, b)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    "goal",
    (
        (180.0, 0.0, 0.0),
        (240.0, 20.0, 0.0),
        (150.0, -15.0, 0.06),
        (220.0, 10.0, -0.06),
    ),
)
def test_reference_tracks_formation_envelope(goal):
    env = FighterEnv(
        goal=goal,
        max_seconds=30.0,
        random_start=False,
    )
    try:
        env.reset(seed=0)
        last = rollout(env, FighterReferenceController())
        error = np.abs(env.tracking() - np.asarray(goal))
        assert last[3] and not last[2]
        assert error[0] < 6.0
        assert error[1] < 4.0
        assert error[2] < 0.02
    finally:
        env.close()


def test_registry_builds_maneuver_environment():
    env = make_env("f14:formation", max_seconds=0.2)
    try:
        observation, _ = env.reset(seed=23)
        assert observation.shape == (13,)
        assert env.action_space.shape == (4,)
        assert np.all(env.goal >= np.asarray((120.0, -30.0, -0.08)))
        assert np.all(env.goal <= np.asarray((280.0, 30.0, 0.08)))
    finally:
        env.close()


def test_live_contract_is_yaw_invariant_and_runs_at_ten_hz():
    env = FighterEnv(
        contract_path=LIVE_CONTRACT,
        max_seconds=0.2,
    )
    try:
        observation, _ = env.reset(seed=23)
        assert observation.shape == (10,)
        assert env.contract.revision == 2
        assert env.dt_s == 0.1
    finally:
        env.close()
