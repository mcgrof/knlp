"""Reference policies that do not learn, and a runner to score them.

    python -m rl.controls.policies --env sim:sim_trees --policy centerline --episodes 5
    python -m rl.controls.policies --env etr:bunny_hill --policy all --episodes 3

``noop`` coasts straight, ``random`` samples macro-actions uniformly,
``centerline`` is a proportional steering controller on the lateral
offset that always paddles. They bound what "learned" has to beat and
double as an environment integrity check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable

import numpy as np

from rl.envs import make_env
from rl.envs.etr_env import ACTION_NAMES, ACTION_TABLE

Policy = Callable[[np.ndarray, np.random.Generator], int]


def _action_index(steer: float, mode: str) -> int:
    for i, (s, paddle, brake) in enumerate(ACTION_TABLE):
        if (
            s == steer
            and ((mode == "paddle") == paddle)
            and ((mode == "brake") == brake)
        ):
            return i
    raise ValueError((steer, mode))


A_STRAIGHT_COAST = _action_index(0.0, "coast")
A_STRAIGHT_PADDLE = _action_index(0.0, "paddle")
A_LEFT_PADDLE = _action_index(-1.0, "paddle")
A_RIGHT_PADDLE = _action_index(1.0, "paddle")


def noop(obs: np.ndarray, rng: np.random.Generator) -> int:
    return A_STRAIGHT_COAST


def random_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(ACTION_TABLE)))


def centerline(obs: np.ndarray, rng: np.random.Generator, gain: float = 2.0) -> int:
    lateral = float(obs[1])  # -1 .. 1 across the playable width
    heading = float(obs[2])  # -1 .. 1 (fraction of pi)
    command = -gain * lateral - 1.5 * heading
    if command > 0.15:
        return A_RIGHT_PADDLE
    if command < -0.15:
        return A_LEFT_PADDLE
    return A_STRAIGHT_PADDLE


POLICIES: dict[str, Policy] = {
    "noop": noop,
    "random": random_policy,
    "centerline": centerline,
}


def run_episodes(env_id: str, policy: Policy, episodes: int, seed: int, **env_kwargs):
    env = make_env(env_id, **env_kwargs)
    rng = np.random.default_rng(seed)
    stats = []
    t0 = time.time()
    steps = 0
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ret = 0.0
            while not done:
                obs, r, term, trunc, info = env.step(policy(obs, rng))
                ret += r
                steps += 1
                done = term or trunc
            s = dict(info["episode_stats"])
            s["return"] = ret
            stats.append(s)
    finally:
        env.close()
    wall = time.time() - t0
    return stats, steps, wall


def summarize(name: str, stats: list[dict], steps: int, wall: float) -> dict:
    keys = ("return", "finish", "time", "progress", "herring", "collisions")
    out = {"policy": name, "episodes": len(stats), "agent_steps": steps, "wall_s": wall}
    for k in keys:
        vals = [s[k] for s in stats]
        out[k] = statistics.mean(vals)
        out[k + "_sd"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    out["steps_per_s"] = steps / wall if wall > 0 else 0.0
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="sim:sim_trees")
    p.add_argument("--policy", default="all", choices=["all", *POLICIES])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="write the summary rows here")
    args = p.parse_args(argv)
    names = list(POLICIES) if args.policy == "all" else [args.policy]
    rows = []
    for name in names:
        stats, steps, wall = run_episodes(
            args.env, POLICIES[name], args.episodes, args.seed
        )
        row = summarize(name, stats, steps, wall)
        rows.append(row)
        print(
            f"{name:10s} return {row['return']:8.2f}  finish {row['finish']:.2f}  "
            f"time {row['time']:7.2f}s  progress {row['progress']:7.1f}m  "
            f"herring {row['herring']:.1f}  collisions {row['collisions']:.1f}  "
            f"({row['steps_per_s']:.0f} agent steps/s)"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"env": args.env, "actions": ACTION_NAMES, "rows": rows}, f, indent=2
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
