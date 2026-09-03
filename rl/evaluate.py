"""Deterministic evaluation helpers shared by the tools in this tree.

A policy here is any callable ``policy(obs, rng) -> action``; these
helpers roll it through an environment, collect transitions, and score
it over a fixed list of seeds.
"""

from __future__ import annotations

import time

import numpy as np

from rl.envs.etr_env import ACTION_TABLE

N_ACTIONS = len(ACTION_TABLE)


def rollout(env, policy, seed: int, epsilon: float = 0.0, rng=None):
    """One episode; returns transitions (S, A, R, S2, D) and the episode stats."""
    rng = rng or np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    S, A, R, S2, D = [], [], [], [], []
    done = False
    ret = 0.0
    while not done:
        a = (
            int(rng.integers(N_ACTIONS))
            if rng.random() < epsilon
            else int(policy(obs, rng))
        )
        nobs, r, term, trunc, info = env.step(a)
        S.append(obs)
        A.append(a)
        R.append(r)
        S2.append(nobs)
        D.append(float(term))
        ret += r
        obs = nobs
        done = term or trunc
    stats = dict(info["episode_stats"])
    stats["return"] = ret
    return (
        np.array(S),
        np.array(A),
        np.array(R, dtype=np.float32),
        np.array(S2),
        np.array(D, dtype=np.float32),
    ), stats


def evaluate(env, policy, seeds):
    stats = [rollout(env, policy, s)[1] for s in seeds]
    times = [s["time"] for s in stats if s["finish"]]
    return {
        "finish_rate": float(np.mean([s["finish"] for s in stats])),
        "mean_time": float(np.mean(times)) if times else float("nan"),
        "mean_return": float(np.mean([s["return"] for s in stats])),
        "mean_herring": float(np.mean([s["herring"] for s in stats])),
        "mean_collisions": float(np.mean([s["collisions"] for s in stats])),
    }


def decision_cost_us(policy, obs, n: int = 2000) -> float:
    """Microseconds per decision at batch size one."""
    rng = np.random.default_rng(0)
    policy(obs, rng)
    t0 = time.perf_counter()
    for _ in range(n):
        policy(obs, rng)
    return (time.perf_counter() - t0) / n * 1e6
