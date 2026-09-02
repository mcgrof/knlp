"""Gymnasium environment over the Extreme Tux Racer bridge (contract v0).

The contract is frozen here on purpose: observation layout, the
nine-way macro-action table, the frame skip and every reward term are
named constants so a run can be reproduced and so a change is a
visible diff rather than a tuning drift. Reward components are
reported separately in ``info["reward_terms"]`` so reward hacking is
visible in the logs.

Observation (float32, ``OBS_DIM``):
    progress fraction, lateral offset from the course centre,
    heading error, forward/lateral/vertical velocity, speed,
    terrain normal tilt (x, z), airborne/braking/paddling flags,
    surface friction, 20 forward terrain heights, 20 forward
    frictions, the 4 nearest trees (dx, dz, present) and the 4
    nearest items (dx, dz, present), all in course coordinates.

Action (``Discrete(9)``): steer {left, straight, right} x
    {coast, paddle, brake}, held for ``FRAME_SKIP`` physics ticks at
    60 Hz (a 15 Hz decision rate). Jumping is not exposed in v0.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.envs.etr_bridge import DEFAULT_DT, NEAREST, PROBE_AHEAD, PROBE_SIDE, EtrBridge
from rl.envs.etr_sim import EtrSim

FRAME_SKIP = 4
TICK_DT = DEFAULT_DT
N_PROBES = len(PROBE_AHEAD) * len(PROBE_SIDE)

# steer, paddle, brake for each macro action
ACTION_TABLE = tuple(
    (steer, mode == "paddle", mode == "brake")
    for steer in (-1.0, 0.0, 1.0)
    for mode in ("coast", "paddle", "brake")
)
ACTION_NAMES = tuple(
    f"{s}+{m}"
    for s in ("left", "straight", "right")
    for m in ("coast", "paddle", "brake")
)

OBS_DIM = (
    1 + 1 + 1 + 3 + 1 + 2 + 3 + 1 + N_PROBES + N_PROBES + 3 * NEAREST + 3 * NEAREST
)

# Normalisation constants (fixed, not learned).
OBS_BOUND = 10.0  # every feature is scaled to sit well inside this box
SPEED_SCALE = 30.0
PROBE_H_SCALE = 5.0
NEAR_SCALE = 25.0

DEFAULT_REWARD = {
    "progress": 1.0,  # per metre of new maximum progress along the course
    "herring": 2.0,  # per item collected
    "time": 0.2,  # per agent step (1/15 s), subtracted: 3 reward per race second
    "collision": 2.0,  # per tree contact event, subtracted
    "finish": 50.0,  # once, on crossing the finish line
    "stuck": 5.0,  # once, subtracted when the episode is truncated for no progress
}

BACKENDS = {"etr": EtrBridge, "sim": EtrSim}


def encode_observation(raw: dict) -> np.ndarray:
    """Map the bridge's raw dictionary to the fixed observation vector."""
    size = raw["size"]
    play = raw["play"]
    x, y, z = raw["pos"]
    vx, vy, vz = raw["vel"]
    width = float(play[0]) if play[0] else 1.0
    length = float(play[1]) if play[1] else 1.0
    centre = float(size[0]) / 2.0
    half = max(width / 2.0, 1e-6)
    heading = math.atan2(vx, -vz) / math.pi if (abs(vx) + abs(vz)) > 1e-9 else 0.0
    nml = raw["nml"]
    out = [
        min(max(-z / length, 0.0), 1.2),
        (x - centre) / half,
        heading,
        -vz / SPEED_SCALE,
        vx / SPEED_SCALE,
        vy / SPEED_SCALE,
        raw["speed"] / SPEED_SCALE,
        nml[0],
        nml[2],
        float(raw["airborne"]),
        float(raw["braking"]),
        float(raw["paddling"]),
        raw["friction"],
    ]
    probe_h = list(raw["probe_h"])[:N_PROBES]
    probe_h += [0.0] * (N_PROBES - len(probe_h))
    out += [min(max(h / PROBE_H_SCALE, -3.0), 3.0) for h in probe_h]
    probe_f = list(raw["probe_f"])[:N_PROBES]
    probe_f += [0.0] * (N_PROBES - len(probe_f))
    out += probe_f
    for k in range(NEAREST):
        if k < len(raw["trees"]):
            dx, dz = raw["trees"][k][0], raw["trees"][k][1]
            out += [dx / NEAR_SCALE, dz / NEAR_SCALE, 1.0]
        else:
            out += [0.0, 0.0, 0.0]
    for k in range(NEAREST):
        if k < len(raw["items"]):
            dx, dz = raw["items"][k][0], raw["items"][k][1]
            out += [dx / NEAR_SCALE, dz / NEAR_SCALE, 1.0]
        else:
            out += [0.0, 0.0, 0.0]
    arr = np.clip(np.asarray(out, dtype=np.float32), -OBS_BOUND, OBS_BOUND)
    assert arr.shape == (OBS_DIM,), arr.shape
    return arr


class EtrEnv(gym.Env):
    """Contract v0 over either the game bridge or the simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: str = "etr",
        course: str = "bunny_hill",
        group: str = "default",
        wind: int = 0,
        frame_skip: int = FRAME_SKIP,
        max_seconds: float = 120.0,
        stuck_seconds: float = 6.0,
        reward_weights: Optional[dict] = None,
        binary: Optional[str] = None,
        stderr_path: Optional[str] = None,
    ):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
        self.backend_name = backend
        self.course = course
        self.group = group if backend == "etr" else "sim"
        self.wind = wind
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_seconds / (TICK_DT * self.frame_skip))
        self.stuck_steps = int(stuck_seconds / (TICK_DT * self.frame_skip))
        self.weights = dict(DEFAULT_REWARD)
        if reward_weights:
            self.weights.update(reward_weights)
        self.observation_space = spaces.Box(
            -OBS_BOUND, OBS_BOUND, (OBS_DIM,), np.float32
        )
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        kwargs: dict[str, Any] = {"binary": binary}
        if backend == "etr":
            kwargs["stderr_path"] = stderr_path
        self.backend = BACKENDS[backend](**kwargs)
        self._raw: Optional[dict] = None
        self._max_progress = 0.0
        self._steps = 0
        self._last_gain_step = 0
        self._episode_seed = 0
        self._terms_total: dict[str, float] = {}

    # ---- gym API ----------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        course = (options or {}).get("course", self.course)
        raw = self.backend.reset(
            seed=self._episode_seed,
            course=course,
            group=self.group,
            wind=self.wind,
            dt=TICK_DT,
        )
        self._raw = raw
        self._max_progress = float(raw["progress"])
        self._steps = 0
        self._last_gain_step = 0
        self._terms_total = {k: 0.0 for k in self.weights}
        return encode_observation(raw), self._info(raw, {})

    def step(self, action: int):
        assert self._raw is not None, "reset first"
        steer, paddle, brake = ACTION_TABLE[int(action)]
        prev = self._raw
        raw = self.backend.step(
            turn=steer, brake=brake, paddle=paddle, jump=False, ticks=self.frame_skip
        )
        self._raw = raw
        self._steps += 1

        terms: dict[str, float] = {}
        gain = max(0.0, float(raw["progress"]) - self._max_progress)
        if gain > 1e-6:
            self._last_gain_step = self._steps
        self._max_progress = max(self._max_progress, float(raw["progress"]))
        terms["progress"] = self.weights["progress"] * gain
        terms["herring"] = self.weights["herring"] * (raw["herring"] - prev["herring"])
        terms["time"] = -self.weights["time"]
        terms["collision"] = -self.weights["collision"] * (
            raw["collisions"] - prev["collisions"]
        )
        terminated = bool(raw["done"])
        terms["finish"] = self.weights["finish"] if terminated else 0.0
        truncated = False
        terms["stuck"] = 0.0
        if not terminated:
            if self._steps >= self.max_steps:
                truncated = True
            elif self._steps - self._last_gain_step >= self.stuck_steps:
                truncated = True
                terms["stuck"] = -self.weights["stuck"]
        for k, v in terms.items():
            self._terms_total[k] += v
        reward = float(sum(terms.values()))
        info = self._info(raw, terms)
        if terminated or truncated:
            info["episode_stats"] = {
                "finish": float(terminated),
                "time": float(raw["time"]),
                "progress": float(self._max_progress),
                "herring": float(raw["herring"]),
                "collisions": float(raw["collisions"]),
                "steps": float(self._steps),
                "seed": float(self._episode_seed),
                **{f"r_{k}": v for k, v in self._terms_total.items()},
            }
        return encode_observation(raw), reward, terminated, truncated, info

    def _info(self, raw: dict, terms: dict) -> dict:
        return {
            "reward_terms": terms,
            "progress": float(raw["progress"]),
            "speed": float(raw["speed"]),
            "herring": int(raw["herring"]),
            "collisions": int(raw["collisions"]),
            "time": float(raw["time"]),
            "tick": int(raw["tick"]),
        }

    def raw(self) -> Optional[dict]:
        return self._raw

    def close(self) -> None:
        self.backend.close()
