"""A minimal synchronous vector environment with same-step autoreset.

When a sub-environment ends, it is reset immediately and the returned
observation is the first one of the new episode; the terminal
observation is not needed by PPO with bootstrapping masked by
``done``. Episode statistics (return, length, and whatever the env
puts in ``info["episode_stats"]``) are surfaced in ``infos["episodes"]``.
This keeps the trainer independent of Gymnasium's autoreset mode
changes across versions.
"""

from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import numpy as np


class SyncVec:
    def __init__(self, factories: Sequence[Callable[[], gym.Env]]):
        self.envs = [f() for f in factories]
        self.n = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self._ret = np.zeros(self.n, dtype=np.float64)
        self._len = np.zeros(self.n, dtype=np.int64)
        self._seeds: list[int] = [0] * self.n

    def reset(self, seed: int = 0):
        obs = []
        for i, env in enumerate(self.envs):
            self._seeds[i] = seed + i
            o, _ = env.reset(seed=seed + i)
            obs.append(o)
        self._ret[:] = 0.0
        self._len[:] = 0
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        obs, rews, dones, episodes = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(int(actions[i]))
            self._ret[i] += r
            self._len[i] += 1
            done = term or trunc
            if done:
                ep = {"r": float(self._ret[i]), "l": int(self._len[i]), "env": i}
                ep.update(info.get("episode_stats", {}))
                episodes.append(ep)
                self._ret[i] = 0.0
                self._len[i] = 0
                o, _ = env.reset()
            obs.append(o)
            rews.append(r)
            dones.append(done)
        return (
            np.stack(obs),
            np.asarray(rews, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            {"episodes": episodes},
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
