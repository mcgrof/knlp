"""Integrity checks against the real game bridge (needs the patched etr).

Skipped unless a binary is found (``ETR_RL_BIN`` or the default build
location); on the training host these are the Gate 0 checks from the
lane plan: determinism across runs and across in-process resets, a
faster-than-real-time step rate, and the Gymnasium contract over the
game itself.
"""

import json
import time

import pytest

pytest.importorskip("gymnasium")

from rl.envs.etr_bridge import BridgeError, EtrBridge, find_binary

try:
    BINARY = find_binary()
except BridgeError:
    BINARY = None

pytestmark = pytest.mark.skipif(BINARY is None, reason="no etr --rl binary available")


def _script(bridge, seed=7, n=60):
    out = [bridge.reset(seed=seed, course="bunny_hill")]
    for i in range(n):
        out.append(
            bridge.step(
                turn=[-1, 0, 1][(i // 10) % 3],
                paddle=(i // 5) % 2,
                brake=(i // 25) % 2,
                ticks=4,
            )
        )
    return json.dumps(out, sort_keys=True)


def test_info_lists_default_courses():
    with EtrBridge(BINARY) as b:
        info = b.info()
    dirs = {c["dir"] for c in info["courses"] if c["group"] == "default"}
    assert "bunny_hill" in dirs and len(dirs) >= 10


def test_deterministic_across_processes_and_resets():
    with EtrBridge(BINARY) as a, EtrBridge(BINARY) as b:
        first = _script(a)
        assert first == _script(b)
        assert first == _script(a)  # second reset in the same process


def test_reset_restores_items_and_counters():
    with EtrBridge(BINARY) as b:
        r0 = b.reset(seed=1, course="bunny_hill")
        for _ in range(300):
            s = b.step(paddle=True, ticks=4)
        assert s["tick"] == 1200 and s["progress"] > r0["progress"]
        r1 = b.reset(seed=1, course="bunny_hill")
        assert r1["tick"] == 0 and r1["herring"] == 0 and r1["collisions"] == 0
        assert r1["items_left"] == r0["items_left"]
        assert r1["pos"] == r0["pos"]


def test_step_rate_is_far_above_real_time():
    with EtrBridge(BINARY) as b:
        b.reset(seed=1, course="bunny_hill")
        t0 = time.time()
        ticks = 0
        for _ in range(500):
            s = b.step(paddle=True, ticks=4)
            ticks += 4
            if s["done"]:
                break
        rate = ticks / (time.time() - t0)
    assert rate > 60 * 20, f"only {rate:.0f} ticks/s"


def test_gym_contract_over_the_game():
    from gymnasium.utils.env_checker import check_env

    from rl.envs import make_env

    env = make_env("etr:bunny_hill", binary=BINARY)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_controls_finish_bunny_hill():
    from rl.controls.policies import POLICIES, run_episodes

    stats, steps, wall = run_episodes(
        "etr:bunny_hill", POLICIES["centerline"], 1, 0, binary=BINARY
    )
    assert stats[0]["finish"] == 1.0, stats
    assert stats[0]["progress"] >= 470.0
