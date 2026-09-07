"""Standalone Gymnasium environment backed by xplane-ufo's C dynamics."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.flight.contracts import FlightContract
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
DYNAMICS_ABI_VERSION = 1


class UfoParameters(ctypes.Structure):
    _fields_ = [
        ("mass_kg", ctypes.c_double),
        ("inertia_kg_m2", ctypes.c_double * 3),
        ("linear_drag_n_per_mps", ctypes.c_double * 3),
        ("angular_drag_nm_per_radps", ctypes.c_double * 3),
        ("gravity_mps2", ctypes.c_double),
    ]


class UfoState(ctypes.Structure):
    _fields_ = [
        ("position_ned_m", ctypes.c_double * 3),
        ("velocity_ned_mps", ctypes.c_double * 3),
        ("quaternion_body_to_ned", ctypes.c_double * 4),
        ("angular_velocity_body_radps", ctypes.c_double * 3),
    ]


class UfoWrench(ctypes.Structure):
    _fields_ = [
        ("force_body_n", ctypes.c_double * 3),
        ("moment_body_nm", ctypes.c_double * 3),
    ]


class UfoDynamics:
    """Typed access to the common C integrator."""

    def __init__(self, library_path: str | Path):
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"xplane-ufo dynamics library does not exist: {path}")
        self.library_path = path
        self._library = ctypes.CDLL(str(path))
        self._library.xplane_ufo_dynamics_abi_version.argtypes = []
        self._library.xplane_ufo_dynamics_abi_version.restype = ctypes.c_uint
        actual_abi = self._library.xplane_ufo_dynamics_abi_version()
        if actual_abi != DYNAMICS_ABI_VERSION:
            raise ValueError(
                f"xplane-ufo dynamics ABI {actual_abi}, expected {DYNAMICS_ABI_VERSION}"
            )
        self._library.xplane_ufo_default_params.argtypes = [
            ctypes.POINTER(UfoParameters)
        ]
        self._library.xplane_ufo_default_params.restype = None
        self._library.xplane_ufo_reset.argtypes = [ctypes.POINTER(UfoState)]
        self._library.xplane_ufo_reset.restype = None
        self._library.xplane_ufo_step.argtypes = [
            ctypes.POINTER(UfoParameters),
            ctypes.POINTER(UfoWrench),
            ctypes.c_double,
            ctypes.POINTER(UfoState),
        ]
        self._library.xplane_ufo_step.restype = ctypes.c_int
        self.params = UfoParameters()
        self.state = UfoState()
        self._library.xplane_ufo_default_params(ctypes.byref(self.params))
        self.reset()

    def reset(self) -> None:
        self._library.xplane_ufo_reset(ctypes.byref(self.state))

    def state_vector(self) -> np.ndarray:
        return np.asarray(
            [
                *self.state.position_ned_m,
                *self.state.velocity_ned_mps,
                *self.state.quaternion_body_to_ned,
                *self.state.angular_velocity_body_radps,
            ],
            dtype=np.float64,
        )

    def step(self, action: Sequence[float], dt_s: float) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("wrench must contain six finite values")
        wrench = UfoWrench()
        for axis in range(3):
            wrench.force_body_n[axis] = values[axis]
            wrench.moment_body_nm[axis] = values[axis + 3]
        result = self._library.xplane_ufo_step(
            ctypes.byref(self.params),
            ctypes.byref(wrench),
            float(dt_s),
            ctypes.byref(self.state),
        )
        if result != 0:
            raise RuntimeError("xplane-ufo dynamics rejected the step")
        return self.state_vector()


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
        self.default_goal = self.contract.goal.validate(goal, "goal")
        self.goal = np.asarray(self.default_goal, dtype=np.float64)
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

    def _observation(self) -> np.ndarray:
        state = self.dynamics.state_vector()
        self.contract.observation.validate(state, "observation")
        return np.asarray((*state, *self.goal), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.dynamics.reset()
        self._steps = 0
        self._return = 0.0
        self.goal = np.asarray(
            self.contract.goal.validate(options.get("goal", self.default_goal), "goal"),
            dtype=np.float64,
        )
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
        observation = self._observation()
        return observation, self._info({})

    def step(self, action):
        values = np.asarray(action, dtype=np.float64)
        self.contract.action.validate(values, "action")
        state = self.dynamics.step(values, self.dt_s)
        self._steps += 1
        terms = self._reward_terms(state, values)
        reward = float(sum(terms.values()))
        self._return += reward
        ground_contact = state[2] >= 0.0
        escaped = abs(state[0]) >= 1_000_000.0 or abs(state[1]) >= 1_000_000.0
        terminated = bool(ground_contact or escaped)
        truncated = bool(not terminated and self._steps >= self.max_steps)
        if ground_contact:
            reward -= 25.0
            terms["ground"] = -25.0
            self._return -= 25.0
        info = self._info(terms)
        if terminated or truncated:
            body_velocity = self.body_velocity(state)
            info["episode_stats"] = {
                "return": self._return,
                "time": self._steps * self.dt_s,
                "success": float(
                    np.linalg.norm(body_velocity - self.goal[:3]) < 1.0
                    and abs(state[12] - self.goal[3]) < 0.1
                    and not ground_contact
                ),
                "ground_contact": float(ground_contact),
            }
        return self._observation(), reward, terminated, truncated, info

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

    def _info(self, reward_terms: dict[str, float]) -> dict:
        return {
            "contract_hash": self.contract.digest,
            "step": self._steps,
            "dt_s": self.dt_s,
            "goal": self.goal.copy(),
            "state": self.dynamics.state_vector(),
            "reward_terms": reward_terms,
        }
