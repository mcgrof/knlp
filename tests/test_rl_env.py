"""Contract v0 over the simulator backend, and the bridge client parser."""

import json

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rl.envs import make_env, parse_env_id
from rl.envs.etr_env import (
    ACTION_TABLE,
    DEFAULT_REWARD,
    OBS_DIM,
    build_actions,
    encode_observation,
)
from rl.envs.etr_sim import EtrSim


def test_parse_env_id():
    assert parse_env_id("sim") == ("sim", "sim_trees")
    assert parse_env_id("etr") == ("etr", "bunny_hill")
    assert parse_env_id("etr:twisty_slope") == ("etr", "twisty_slope")


def test_sim_backend_is_deterministic():
    a, b = EtrSim(), EtrSim()
    ra, rb = a.reset(seed=3, course="sim_trees"), b.reset(seed=3, course="sim_trees")
    assert json.dumps(ra, sort_keys=True) == json.dumps(rb, sort_keys=True)
    for i in range(50):
        turn = [-1, 0, 1][i % 3]
        sa = a.step(turn=turn, paddle=True, ticks=4)
        sb = b.step(turn=turn, paddle=True, ticks=4)
        assert json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)
    assert sa["tick"] == 200
    assert sa["progress"] > ra["progress"]


def test_encode_observation_shape_and_ranges():
    sim = EtrSim()
    raw = sim.reset(seed=1, course="sim_trees")
    obs = encode_observation(raw)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert 0.0 <= obs[0] <= 1.2  # progress fraction
    assert -1.0 <= obs[1] <= 1.0  # lateral


def test_env_passes_gymnasium_check():
    from gymnasium.utils.env_checker import check_env

    env = make_env("sim:sim_trees")
    check_env(env, skip_render_check=True)
    env.close()


def test_env_episode_terminates_and_reports_terms():
    env = make_env("sim:sim_flat", max_seconds=60.0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    total = 0.0
    done = False
    steps = 0
    paddle_straight = [i for i, (s, p, b) in enumerate(ACTION_TABLE) if s == 0.0 and p][
        0
    ]
    while not done:
        obs, r, term, trunc, info = env.step(paddle_straight)
        assert set(info["reward_terms"]) == set(DEFAULT_REWARD)
        total += r
        steps += 1
        done = term or trunc
        assert steps < 5000
    assert term, "a straight paddling policy should finish the flat course"
    stats = info["episode_stats"]
    assert stats["finish"] == 1.0
    assert stats["r_finish"] > 0 and stats["r_progress"] > 0
    assert abs(sum(v for k, v in stats.items() if k.startswith("r_")) - total) < 1e-6


def test_env_truncates_when_stuck():
    env = make_env("sim:sim_flat", max_seconds=60.0, stuck_seconds=1.0)
    env.reset(seed=0)
    brake_only = [i for i, (s, p, b) in enumerate(ACTION_TABLE) if s == 0.0 and b][0]
    # the simulator never drops below its minimum speed, so we force a
    # no-progress condition by monkeypatching the maximum progress high
    env._max_progress = 1e9
    for _ in range(200):
        _, r, term, trunc, info = env.step(brake_only)
        if trunc:
            break
    assert trunc and not term
    assert info["reward_terms"]["stuck"] < 0


def test_progress_reward_cannot_be_farmed_by_backtracking():
    env = make_env("sim:sim_flat")
    env.reset(seed=0)
    paddle_straight = [i for i, (s, p, b) in enumerate(ACTION_TABLE) if s == 0.0 and p][
        0
    ]
    _, _, _, _, info = env.step(paddle_straight)
    gained = info["reward_terms"]["progress"]
    assert gained > 0
    # pretend the racer slid back: progress below the maximum yields nothing
    env._raw = dict(env._raw, progress=env._raw["progress"] - 5.0)
    env.backend.z += 5.0
    _, _, _, _, info = env.step(paddle_straight)
    assert info["reward_terms"]["progress"] == 0.0


def test_v1_action_set_adds_jump_and_the_reset_key():
    table, names = build_actions("v1")
    assert len(table) == 13
    assert names[-1] == "recover" and table[-1] is None
    assert "straight+jump" in names
    # v0 is untouched, so checkpoints trained against it still load
    assert len(build_actions("v0")[0]) == 9


def test_recover_costs_time_and_moves_backwards():
    env = make_env("sim:sim_trees", action_set="v1")
    env.reset(seed=0)
    paddle = env.action_names.index("straight+paddle")
    for _ in range(60):
        _, _, _, _, info = env.step(paddle)
    before, t0 = info["progress"], info["time"]
    _, r, _, _, info = env.step(env.action_names.index("recover"))
    assert info["progress"] < before
    assert info["time"] > t0
    assert info["reward_terms"]["recover"] < 0
    assert info["reward_terms"]["progress"] == 0.0  # cannot farm progress by resetting
