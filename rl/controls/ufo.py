"""Deterministic reference controllers for the standalone UFO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.envs.ufo_env import ACTION_FIELDS, UfoEnv
from rl.flight.geometry import quaternion_body_to_ned, roll_pitch


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
        rotation = quaternion_body_to_ned(state[6:10])
        velocity_ned = state[3:6]
        velocity_body = rotation.T @ velocity_ned
        desired_acceleration_body = np.clip(
            self.velocity_gain * (env.goal[:3] - velocity_body),
            -self.max_acceleration_mps2,
            self.max_acceleration_mps2,
        )

        mass = float(params.mass_kg)
        drag_ned = np.asarray(params.linear_drag_n_per_mps) * velocity_ned
        gravity_cancel_ned = np.asarray([0.0, 0.0, -mass * params.gravity_mps2])
        force_body = mass * desired_acceleration_body + rotation.T @ (
            drag_ned + gravity_cancel_ned
        )

        omega = state[10:13]
        roll, pitch = roll_pitch(state[6:10])
        desired_angular_acceleration = np.asarray(
            [
                -self.attitude_gain * roll - self.roll_pitch_rate_gain * omega[0],
                -self.attitude_gain * pitch - self.roll_pitch_rate_gain * omega[1],
                self.yaw_rate_gain * (env.goal[3] - omega[2]),
            ]
        )
        inertia = np.asarray(params.inertia_kg_m2)
        inertia_omega = inertia * omega
        gyroscopic = np.cross(omega, inertia_omega)
        angular_drag = np.asarray(params.angular_drag_nm_per_radps) * omega
        moment_body = inertia * desired_angular_acceleration + gyroscopic + angular_drag
        action = np.asarray((*force_body, *moment_body), dtype=np.float64)
        return np.clip(action, env.action_space.low, env.action_space.high).astype(
            np.float32
        )


def hover_wrench(env: UfoEnv) -> np.ndarray:
    """Hold zero body velocity and yaw rate with the reference controller."""

    saved_goal = env.goal.copy()
    try:
        env.goal[:] = 0.0
        return VelocityTargetController()(env)
    finally:
        env.goal[:] = saved_goal
