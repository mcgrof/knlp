"""Deterministic reference controllers for the standalone UFO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.envs.ufo_env import ACTION_FIELDS, UfoEnv
from rl.flight.ufo_reference import UfoReferenceParameters, velocity_target_wrench


def zero_wrench(env: UfoEnv) -> np.ndarray:
    return np.zeros(len(ACTION_FIELDS), dtype=np.float32)


@dataclass
class VelocityTargetController:
    """Proportional velocity, upright-attitude and yaw-rate controller."""

    velocity_gain: float = 1.5
    max_acceleration_mps2: float = 8.0
    attitude_gain: float = 4.0
    roll_pitch_rate_gain: float = 3.0
    yaw_rate_gain: float = 3.0

    def __call__(self, env: UfoEnv) -> np.ndarray:
        state = env.dynamics.state_vector()
        params = env.dynamics.params
        return velocity_target_wrench(
            state,
            env.goal,
            env.action_space.low,
            env.action_space.high,
            parameters=UfoReferenceParameters(
                mass_kg=float(params.mass_kg),
                inertia_kg_m2=tuple(params.inertia_kg_m2),
                linear_drag_n_per_mps=tuple(params.linear_drag_n_per_mps),
                angular_drag_nm_per_radps=tuple(params.angular_drag_nm_per_radps),
                gravity_mps2=float(params.gravity_mps2),
            ),
            velocity_gain=self.velocity_gain,
            max_acceleration_mps2=self.max_acceleration_mps2,
            attitude_gain=self.attitude_gain,
            roll_pitch_rate_gain=self.roll_pitch_rate_gain,
            yaw_rate_gain=self.yaw_rate_gain,
        )


def hover_wrench(env: UfoEnv) -> np.ndarray:
    """Hold zero body velocity and yaw rate with the reference controller."""

    saved_goal = env.goal.copy()
    try:
        env.goal[:] = 0.0
        return VelocityTargetController()(env)
    finally:
        env.goal[:] = saved_goal
