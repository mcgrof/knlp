"""A minimal synchronous vector environment with same-step autoreset.

When a sub-environment ends, it is reset immediately and the returned
observation is the first one of the new episode.  Termination and truncation
remain distinct, and the final observation is retained so a trainer can
bootstrap a time limit without treating it as a terminal state.  Episode
statistics (return, length, and whatever the environment puts in
``info["episode_stats"]``) are surfaced in ``infos["episodes"]``.  This keeps
the trainer independent of Gymnasium's autoreset mode changes across versions.
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
        obs, rews, terms, truncs, episodes = [], [], [], [], []
        final_observations: list[np.ndarray | None] = []
        final_infos: list[dict | None] = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(action)
            else:
                action = np.asarray(action, dtype=env.action_space.dtype)
            o, r, term, trunc, info = env.step(action)
            self._ret[i] += r
            self._len[i] += 1
            done = term or trunc
            if done:
                ep = {"r": float(self._ret[i]), "l": int(self._len[i]), "env": i}
                ep.update(info.get("episode_stats", {}))
                episodes.append(ep)
                self._ret[i] = 0.0
                self._len[i] = 0
                final_observations.append(np.asarray(o).copy())
                final_infos.append(info)
                o, _ = env.reset()
            else:
                final_observations.append(None)
                final_infos.append(None)
            obs.append(o)
            rews.append(r)
            terms.append(term)
            truncs.append(trunc)
        terminations = np.asarray(terms, dtype=np.bool_)
        truncations = np.asarray(truncs, dtype=np.bool_)
        return (
            np.stack(obs),
            np.asarray(rews, dtype=np.float32),
            terminations | truncations,
            {
                "episodes": episodes,
                "terminations": terminations,
                "truncations": truncations,
                "final_observations": tuple(final_observations),
                "final_infos": tuple(final_infos),
            },
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
