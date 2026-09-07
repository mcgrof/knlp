"""Standalone UFO environment and deterministic reference-controller gates."""

import os
import time

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rl.controls.ufo import VelocityTargetController, hover_wrench, zero_wrench
from rl.envs import make_env
from rl.envs.ufo_env import UfoEnv

UFO_ROOT = os.environ.get("XPLANE_UFO_ROOT")
pytestmark = pytest.mark.skipif(not UFO_ROOT, reason="XPLANE_UFO_ROOT is not set")


def rollout(env, controller, limit=2000):
    last = None
    for _ in range(limit):
        action = controller(env)
        last = env.step(action)
        if last[2] or last[3]:
            break
    assert last is not None
    return last


def test_ufo_env_passes_gymnasium_check():
    from gymnasium.utils.env_checker import check_env

    env = UfoEnv(root=UFO_ROOT, max_seconds=1.0)
    check_env(env, skip_render_check=True)
    env.close()


def test_fixed_seed_reset_and_trajectory_are_identical():
    first = UfoEnv(root=UFO_ROOT, max_seconds=1.0)
    second = UfoEnv(root=UFO_ROOT, max_seconds=1.0)
    try:
        a, _ = first.reset(seed=17)
        b, _ = second.reset(seed=17)
        assert np.array_equal(a, b)
        for _ in range(25):
            action = np.asarray([100.0, -20.0, -500.0, 3.0, -2.0, 1.0])
            a = first.step(action)[0]
            b = second.step(action)[0]
            assert np.array_equal(a, b)
        first.reset(seed=17)
        assert np.array_equal(
            first.dynamics.state_vector(), second.reset(seed=17)[1]["state"]
        )
    finally:
        first.close()
        second.close()


def test_hover_holds_altitude_while_zero_wrench_hits_ground():
    hover = make_env("ufo:hover", root=UFO_ROOT, max_seconds=6.0, random_start=False)
    falling = make_env("ufo:hover", root=UFO_ROOT, max_seconds=6.0, random_start=False)
    try:
        hover.reset(seed=0)
        falling.reset(seed=0)
        hover_last = rollout(hover, hover_wrench)
        fall_last = rollout(falling, zero_wrench)
        assert hover_last[3] and not hover_last[2]
        assert abs(hover.dynamics.state.position_ned_m[2] + 100.0) < 1e-6
        assert fall_last[2] and fall_last[4]["episode_stats"]["ground_contact"] == 1.0
    finally:
        hover.close()
        falling.close()


def test_velocity_controller_tracks_forward_goal():
    env = make_env("ufo:forward", root=UFO_ROOT, max_seconds=8.0, random_start=False)
    controller = VelocityTargetController()
    try:
        env.reset(seed=0)
        last = rollout(env, controller)
        velocity = env.body_velocity()
        assert last[3] and not last[2]
        assert velocity[0] == pytest.approx(10.0, abs=0.15)
        assert abs(velocity[1]) < 0.05 and abs(velocity[2]) < 0.05
        assert last[4]["episode_stats"]["success"] == 1.0
    finally:
        env.close()


def test_out_of_contract_action_fails_closed():
    env = UfoEnv(root=UFO_ROOT)
    try:
        env.reset(seed=0)
        action = env.action_space.high.astype(np.float64)
        action[0] += 1.0
        with pytest.raises(ValueError, match="outside"):
            env.step(action)
    finally:
        env.close()


def test_standalone_step_rate_is_above_real_time():
    env = UfoEnv(root=UFO_ROOT, max_seconds=30.0, random_start=False)
    try:
        env.reset(seed=0)
        start = time.perf_counter()
        steps = 1000
        for _ in range(steps):
            env.step(hover_wrench(env))
        rate = steps / (time.perf_counter() - start)
        assert rate > 1000.0
    finally:
        env.close()
