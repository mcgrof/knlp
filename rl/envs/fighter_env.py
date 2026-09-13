"""Headless fixed-wing environment for fighter motor-control policies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.flight.contracts import FlightContract
from rl.flight.fighter_dynamics import (
    GRAVITY_MPS2,
    FighterDynamics,
    fighter_envelope,
)
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler

CONTRACT_DIRECTORY = Path(__file__).resolve().parents[1] / "contracts"
DEFAULT_CONTRACT = CONTRACT_DIRECTORY / "fighter-controls-v1.json"
LIVE_CONTRACT = CONTRACT_DIRECTORY / "fighter-controls-v2.json"
FORMATION_CONTRACT = CONTRACT_DIRECTORY / "fighter-controls-v3.json"
LEGACY_MANEUVER_LOW = np.asarray((120.0, -30.0, -0.08))
LEGACY_MANEUVER_HIGH = np.asarray((280.0, 30.0, 0.08))
FAST_MANEUVER_LOW = np.asarray((120.0, -30.0, -0.08))
FAST_MANEUVER_HIGH = np.asarray((760.0, 30.0, 0.08))


class FighterEnv(gym.Env):
    """Track airspeed, climb rate, and turn rate in coordinated flight."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        contract_path: str | Path = DEFAULT_CONTRACT,
        goal: Sequence[float] = (180.0, 0.0, 0.0),
        goal_mode: str = "fixed",
        goal_hold_seconds: float = 6.0,
        max_seconds: float = 30.0,
        random_start: bool = True,
    ):
        super().__init__()
        self.contract = FlightContract.from_json(contract_path)
        self.dynamics = FighterDynamics(fighter_envelope(self.contract.revision))
        if self.contract.revision == 3:
            self.maneuver_low = FAST_MANEUVER_LOW
            self.maneuver_high = FAST_MANEUVER_HIGH
        else:
            self.maneuver_low = LEGACY_MANEUVER_LOW
            self.maneuver_high = LEGACY_MANEUVER_HIGH
        self.dt_s = self.contract.nominal_dt_s
        if goal_mode not in {"fixed", "maneuver"}:
            raise ValueError(f"unknown fighter goal mode {goal_mode!r}")
        if max_seconds <= self.dt_s or not math.isfinite(max_seconds):
            raise ValueError("max_seconds must exceed one environment step")
        if goal_hold_seconds <= self.dt_s or not math.isfinite(goal_hold_seconds):
            raise ValueError("goal hold time must exceed one environment step")
        self.goal_mode = goal_mode
        self.goal_hold_steps = int(math.ceil(goal_hold_seconds / self.dt_s))
        self.max_steps = int(math.ceil(max_seconds / self.dt_s))
        self.random_start = bool(random_start)
        self.default_goal = np.asarray(
            self.contract.goal.validate(goal, "goal"), dtype=np.float64
        )
        self.goal = self.default_goal.copy()
        velocity_bound = max(
            500.0, self.dynamics.envelope.maximum_airspeed_mps + 10.0
        )
        self._state_low = np.asarray(
            (
                -1_000_000.0,
                -1_000_000.0,
                -100_000.0,
                -velocity_bound,
                -velocity_bound,
                -velocity_bound,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -4.0,
                -4.0,
                -4.0,
            )
        )
        self._state_high = -self._state_low
        self._state_high[2] = 10_000.0
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

    def _sample_goal(self) -> np.ndarray:
        goal = self.np_random.uniform(self.maneuver_low, self.maneuver_high)
        if self.contract.revision == 3:
            maximum_turn = min(
                self.maneuver_high[2],
                GRAVITY_MPS2
                * math.tan(
                    self.dynamics.envelope.maximum_commanded_roll_rad
                )
                / goal[0],
            )
            goal[2] = self.np_random.uniform(-maximum_turn, maximum_turn)
        return goal

    def tracking(self, state: np.ndarray | None = None) -> np.ndarray:
        values = self.dynamics.state_vector() if state is None else state
        return np.asarray(
            (
                float(np.linalg.norm(values[3:6])),
                -float(values[5]),
                float(values[12]),
            )
        )

    def _observation(self, state: np.ndarray | None = None) -> np.ndarray:
        values = self.dynamics.state_vector() if state is None else state
        if self.contract.revision in {1, 3}:
            motor_state = values[3:]
        elif self.contract.revision == 2:
            rotation = quaternion_body_to_ned(values[6:10])
            roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
            motor_state = np.asarray(
                (
                    float(np.linalg.norm(values[3:6])),
                    -float(values[5]),
                    roll,
                    pitch,
                    *values[10:13],
                )
            )
        else:
            raise ValueError(
                "unsupported fighter contract revision "
                f"{self.contract.revision}"
            )
        self.contract.observation.validate(motor_state, "observation")
        return np.asarray((*motor_state, *self.goal), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.dynamics.reset()
        self._steps = 0
        self._return = 0.0
        state = self.dynamics.state
        if options.get("randomize", self.random_start):
            state[:2] = self.np_random.uniform(-20.0, 20.0, 2)
            state[2] = self.np_random.uniform(-1800.0, -1200.0)
            maximum_start_speed = (
                780.0 if self.contract.revision == 3 else 330.0
            )
            speed = self.np_random.uniform(110.0, maximum_start_speed)
            attitude = quaternion_from_euler(
                self.np_random.uniform(
                    -math.radians(10.0), math.radians(10.0)
                ),
                self.np_random.uniform(
                    -math.radians(10.0), math.radians(10.0)
                ),
                self.np_random.uniform(-math.pi, math.pi),
            )
            state[6:10] = attitude
            state[3:6] = quaternion_body_to_ned(attitude)[:, 0] * speed
            state[10:13] = self.np_random.uniform(-0.08, 0.08, 3)
        self.goal = np.asarray(
            self.contract.goal.validate(
                options.get(
                    "goal",
                    self._sample_goal()
                    if self.goal_mode == "maneuver"
                    else self.default_goal,
                ),
                "goal",
            )
        )
        return self._observation(), self._info({})

    def step(self, action):
        controls = np.asarray(action, dtype=np.float64)
        self.contract.action.validate(controls, "action")
        applied_goal = self.goal.copy()
        state = self.dynamics.step(controls, self.dt_s)
        self._steps += 1
        out_of_envelope = bool(
            not np.isfinite(state).all()
            or np.any(state < self._state_low)
            or np.any(state > self._state_high)
        )
        ground_contact = bool(np.isfinite(state).all() and state[2] >= 0.0)
        speed_error_scale = 80.0 if self.contract.revision == 3 else 35.0
        tracking_error = (self.tracking(state) - applied_goal) / np.asarray(
            (speed_error_scale, 15.0, 0.04)
        )
        centered_action = controls.copy()
        centered_action[0] -= (
            self.tracking(state)[0]
            / self.dynamics.envelope.thrust_speed_mps
        ) ** 2
        terms = {
            "alive": 1.0,
            "tracking": -float(np.dot(tracking_error, tracking_error)),
            "effort": -0.01 * float(np.dot(centered_action, centered_action)),
        }
        if ground_contact:
            terms["ground"] = -100.0
        if out_of_envelope:
            terms["out_of_envelope"] = -100.0
        reward = float(sum(terms.values()))
        self._return += reward
        terminated = ground_contact or out_of_envelope
        truncated = not terminated and self._steps >= self.max_steps
        info = self._info(terms, applied_goal=applied_goal)
        if terminated or truncated:
            error = np.abs(self.tracking(state) - applied_goal)
            info["episode_stats"] = {
                "return": self._return,
                "time": self._steps * self.dt_s,
                "success": float(
                    not terminated
                    and error[0] < 5.0
                    and error[1] < 3.0
                    and error[2] < 0.015
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
            self.goal = self._sample_goal()
            info["goal"] = self.goal.copy()
        return self._observation(state), reward, terminated, truncated, info

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
            "tracking": self.tracking(),
            "reward_terms": reward_terms,
        }
        if applied_goal is not None:
            info["applied_goal"] = applied_goal.copy()
        return info
