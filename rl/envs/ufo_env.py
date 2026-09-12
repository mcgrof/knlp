"""Standalone Gymnasium environment backed by xplane-ufo's C dynamics."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.flight.contracts import FlightContract
from rl.flight.ufo_dynamics import UfoDynamics
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler

OBSERVATION_FIELDS = (
    "position_ned_x",
    "position_ned_y",
    "position_ned_z",
    "velocity_ned_x",
    "velocity_ned_y",
    "velocity_ned_z",
    "quaternion_body_to_ned_w",
    "quaternion_body_to_ned_x",
    "quaternion_body_to_ned_y",
    "quaternion_body_to_ned_z",
    "angular_velocity_body_x",
    "angular_velocity_body_y",
    "angular_velocity_body_z",
)
GOAL_FIELDS = (
    "velocity_body_x",
    "velocity_body_y",
    "velocity_body_z",
    "yaw_rate_body",
)
ACTION_FIELDS = (
    "force_body_x",
    "force_body_y",
    "force_body_z",
    "moment_body_x",
    "moment_body_y",
    "moment_body_z",
)

MANEUVER_PROFILES = {
    "certified": {
        "low": (-20.0, -30.0, -12.0, -1.0),
        "high": (55.0, 30.0, 12.0, 1.0),
        "reference_max_acceleration_mps2": 8.0,
    },
    "showcase": {
        "low": (-20.0, -50.0, -24.0, -1.4),
        "high": (90.0, 50.0, 24.0, 1.4),
        "reference_max_acceleration_mps2": 24.0,
    },
}


def default_ufo_root() -> Path:
    return Path(os.environ.get("XPLANE_UFO_ROOT", "~/devel/xplane-ufo")).expanduser()


class UfoEnv(gym.Env):
    """Free-flight velocity-tracking task over the common UFO dynamics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        contract_path: str | Path | None = None,
        library_path: str | Path | None = None,
        goal: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        goal_mode: str = "fixed",
        maneuver_profile: str = "certified",
        goal_hold_seconds: float = 4.0,
        max_seconds: float = 20.0,
        random_start: bool = True,
    ):
        super().__init__()
        root_path = Path(root).expanduser() if root else default_ufo_root()
        contract_path = contract_path or root_path / "schemas/ufo-wrench-v1.json"
        library_path = library_path or root_path / "build/libxplane_ufo_dynamics.so"
        self.contract = FlightContract.from_json(contract_path)
        self._check_concrete_contract()
        self.dynamics = UfoDynamics(library_path)
        self.dt_s = self.contract.nominal_dt_s
        if not math.isfinite(max_seconds) or max_seconds <= self.dt_s:
            raise ValueError("max_seconds must exceed one environment step")
        self.max_steps = int(math.ceil(max_seconds / self.dt_s))
        self.random_start = bool(random_start)
        if goal_mode not in {"fixed", "maneuver"}:
            raise ValueError(f"unknown UFO goal mode {goal_mode!r}")
        if maneuver_profile not in MANEUVER_PROFILES:
            raise ValueError(f"unknown UFO maneuver profile {maneuver_profile!r}")
        if not math.isfinite(goal_hold_seconds) or goal_hold_seconds <= self.dt_s:
            raise ValueError("goal hold time must exceed one environment step")
        self.goal_mode = goal_mode
        self.maneuver_profile = maneuver_profile
        self.reference_max_acceleration_mps2 = float(
            MANEUVER_PROFILES[maneuver_profile]["reference_max_acceleration_mps2"]
        )
        self.goal_hold_steps = int(math.ceil(goal_hold_seconds / self.dt_s))
        self.default_goal = self.contract.goal.validate(goal, "goal")
        self.goal = np.asarray(self.default_goal, dtype=np.float64)
        self._state_low = np.asarray(self.contract.observation.low, dtype=np.float64)
        self._state_high = np.asarray(self.contract.observation.high, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.asarray(
                (*self.contract.observation.low, *self.contract.goal.low),
                dtype=np.float32,
            ),
            high=np.asarray(
                (*self.contract.observation.high, *self.contract.goal.high),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray(self.contract.action.low, dtype=np.float32),
            high=np.asarray(self.contract.action.high, dtype=np.float32),
            dtype=np.float32,
        )
        self._steps = 0
        self._return = 0.0

    def _check_concrete_contract(self) -> None:
        expected = {
            "observation": OBSERVATION_FIELDS,
            "goal": GOAL_FIELDS,
            "action": ACTION_FIELDS,
        }
        for name, fields in expected.items():
            if getattr(self.contract, name).fields != fields:
                raise ValueError(f"xplane-ufo {name} ABI does not match the contract")

    def _observation(
        self, state: np.ndarray | None = None, *, clip_terminal: bool = False
    ) -> np.ndarray:
        state = self.dynamics.state_vector() if state is None else state
        if clip_terminal:
            state = np.nan_to_num(
                state,
                nan=0.0,
                posinf=self._state_high,
                neginf=self._state_low,
            )
            state = np.clip(state, self._state_low, self._state_high)
        else:
            self.contract.observation.validate(state, "observation")
        return np.asarray((*state, *self.goal), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.dynamics.reset()
        self._steps = 0
        self._return = 0.0
        state = self.dynamics.state
        state.position_ned_m[2] = -100.0
        if options.get("randomize", self.random_start):
            for axis in range(2):
                state.position_ned_m[axis] = self.np_random.uniform(-5.0, 5.0)
            state.position_ned_m[2] += self.np_random.uniform(-2.0, 2.0)
            for axis in range(3):
                state.velocity_ned_mps[axis] = self.np_random.uniform(-1.0, 1.0)
                state.angular_velocity_body_radps[axis] = self.np_random.uniform(
                    -0.05, 0.05
                )
            attitude = quaternion_from_euler(
                *self.np_random.uniform(-math.radians(5.0), math.radians(5.0), 3)
            )
            for index, value in enumerate(attitude):
                state.quaternion_body_to_ned[index] = value
        self.goal = np.asarray(
            self.contract.goal.validate(options.get("goal", self.default_goal), "goal"),
            dtype=np.float64,
        )
        if self.goal_mode == "maneuver" and "goal" not in options:
            self.goal = self._sample_maneuver_goal()
        observation = self._observation()
        return observation, self._info({})

    def step(self, action):
        values = np.asarray(action, dtype=np.float64)
        self.contract.action.validate(values, "action")
        applied_goal = self.goal.copy()
        state = self.dynamics.step(values, self.dt_s)
        self._steps += 1
        state_is_finite = bool(np.isfinite(state).all())
        out_of_envelope = bool(
            not state_is_finite
            or np.any(state < self._state_low)
            or np.any(state > self._state_high)
        )
        terms = (
            self._reward_terms(state, values)
            if state_is_finite
            else {"alive": 0.0, "nonfinite_state": -100.0}
        )
        reward = float(sum(terms.values()))
        ground_contact = bool(state_is_finite and state[2] >= 0.0)
        if ground_contact:
            reward -= 25.0
            terms["ground"] = -25.0
        if out_of_envelope:
            reward -= 25.0
            terms["out_of_envelope"] = -25.0
        self._return += reward
        terminated = bool(ground_contact or out_of_envelope)
        truncated = bool(not terminated and self._steps >= self.max_steps)
        info = self._info(terms, applied_goal=applied_goal)
        if terminated or truncated:
            body_velocity = (
                self.body_velocity(state) if state_is_finite else np.full(3, np.inf)
            )
            info["episode_stats"] = {
                "return": self._return,
                "time": self._steps * self.dt_s,
                "success": float(
                    np.linalg.norm(body_velocity - applied_goal[:3]) < 1.0
                    and abs(state[12] - applied_goal[3]) < 0.1
                    and not ground_contact
                    and not out_of_envelope
                ),
                "ground_contact": float(ground_contact),
                "out_of_envelope": float(out_of_envelope),
            }
        if (
            not terminated
            and not truncated
            and self.goal_mode == "maneuver"
            and self._steps % self.goal_hold_steps == 0
        ):
            self.goal = self._sample_maneuver_goal()
            info["goal"] = self.goal.copy()
        observation = self._observation(state, clip_terminal=out_of_envelope)
        return observation, reward, terminated, truncated, info

    def _sample_maneuver_goal(self) -> np.ndarray:
        """Sample a broad but controllable body-velocity command."""

        profile = MANEUVER_PROFILES[self.maneuver_profile]
        low = np.asarray(profile["low"], dtype=np.float64)
        high = np.asarray(profile["high"], dtype=np.float64)
        position_ned_z = float(self.dynamics.state.position_ned_m[2])
        hold_seconds = self.goal_hold_steps * self.dt_s
        maximum_descent = (-30.0 - position_ned_z) / hold_seconds
        high[2] = max(low[2], min(high[2], maximum_descent))
        sampled = self.np_random.uniform(low, high)
        return np.asarray(
            self.contract.goal.validate(sampled, "maneuver goal"),
            dtype=np.float64,
        )

    def body_velocity(self, state: np.ndarray | None = None) -> np.ndarray:
        state = self.dynamics.state_vector() if state is None else state
        rotation = quaternion_body_to_ned(state[6:10])
        return rotation.T @ state[3:6]

    def _reward_terms(self, state: np.ndarray, action: np.ndarray) -> dict[str, float]:
        body_velocity = self.body_velocity(state)
        velocity_error = (body_velocity - self.goal[:3]) / np.asarray(
            [20.0, 20.0, 10.0]
        )
        yaw_error = (state[12] - self.goal[3]) / 1.0
        rotation = quaternion_body_to_ned(state[6:10])
        tilt = 1.0 - float(np.clip(rotation[2, 2], -1.0, 1.0))
        action_scale = np.maximum(np.abs(self.action_space.low), self.action_space.high)
        normalized_action = np.asarray(action, dtype=np.float64) / action_scale
        return {
            "alive": 1.0,
            "velocity": -float(np.dot(velocity_error, velocity_error)),
            "yaw_rate": -float(yaw_error * yaw_error),
            "tilt": -2.0 * tilt,
            "effort": -0.01
            * float(np.dot(normalized_action, normalized_action) / len(action)),
        }

    def _info(
        self,
        reward_terms: dict[str, float],
        *,
        applied_goal: np.ndarray | None = None,
    ) -> dict:
        info = {
            "contract_hash": self.contract.digest,
            "step": self._steps,
            "dt_s": self.dt_s,
            "goal": self.goal.copy(),
            "state": self.dynamics.state_vector(),
            "reward_terms": reward_terms,
        }
        if applied_goal is not None:
            info["applied_goal"] = applied_goal.copy()
        return info
