"""Environment registry: ``make_env("etr:bunny_hill")`` style ids.

An id is ``<backend>:<course>``. Backends: ``etr`` (the headless game
bridge, needs the patched binary) and ``sim`` (the pure-Python stand-in).
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym

from rl.envs.etr_env import EtrEnv

DEFAULT_COURSE = {"etr": "bunny_hill", "sim": "sim_trees"}


def parse_env_id(env_id: str) -> tuple[str, str]:
    backend, _, course = env_id.partition(":")
    backend = backend or "sim"
    course = course or DEFAULT_COURSE[backend]
    return backend, course


def make_env(env_id: str, **kwargs) -> gym.Env:
    backend, course = parse_env_id(env_id)
    return EtrEnv(backend=backend, course=course, **kwargs)


def env_factory(env_id: str, **kwargs) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        return make_env(env_id, **kwargs)

    return thunk


__all__ = ["EtrEnv", "make_env", "env_factory", "parse_env_id"]
