"""Pure velocity-target reference control for UFO state vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rl.flight.geometry import quaternion_body_to_ned, roll_pitch


@dataclass(frozen=True)
class UfoReferenceParameters:
    mass_kg: float = 1200.0
    inertia_kg_m2: tuple[float, float, float] = (1600.0, 2200.0, 2600.0)
    linear_drag_n_per_mps: tuple[float, float, float] = (80.0, 120.0, 160.0)
    angular_drag_nm_per_radps: tuple[float, float, float] = (
        300.0,
        400.0,
        500.0,
    )
    gravity_mps2: float = 9.80665


def velocity_target_wrench(
    observation: Sequence[float],
    goal: Sequence[float],
    action_low: Sequence[float],
    action_high: Sequence[float],
    *,
    parameters: UfoReferenceParameters = UfoReferenceParameters(),
    velocity_gain: float = 1.5,
    max_acceleration_mps2: float = 8.0,
    attitude_gain: float = 4.0,
    roll_pitch_rate_gain: float = 3.0,
    yaw_rate_gain: float = 3.0,
) -> np.ndarray:
    state = np.asarray(observation, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    if state.shape != (13,) or target.shape != (4,):
        raise ValueError("UFO reference state or goal has the wrong width")
    if low.shape != (6,) or high.shape != (6,):
        raise ValueError("UFO reference action bounds have the wrong width")
    values = np.asarray(
        (
            parameters.mass_kg,
            *parameters.inertia_kg_m2,
            *parameters.linear_drag_n_per_mps,
            *parameters.angular_drag_nm_per_radps,
            parameters.gravity_mps2,
            velocity_gain,
            max_acceleration_mps2,
            attitude_gain,
            roll_pitch_rate_gain,
            yaw_rate_gain,
        ),
        dtype=np.float64,
    )
    if (
        not np.isfinite(values).all()
        or parameters.mass_kg <= 0.0
        or min(parameters.inertia_kg_m2) <= 0.0
        or min(parameters.linear_drag_n_per_mps) < 0.0
        or min(parameters.angular_drag_nm_per_radps) < 0.0
        or parameters.gravity_mps2 < 0.0
        or velocity_gain <= 0.0
        or max_acceleration_mps2 <= 0.0
        or attitude_gain <= 0.0
        or roll_pitch_rate_gain < 0.0
        or yaw_rate_gain <= 0.0
    ):
        raise ValueError("UFO reference parameters must be finite and positive")

    rotation = quaternion_body_to_ned(state[6:10])
    velocity_ned = state[3:6]
    velocity_body = rotation.T @ velocity_ned
    desired_acceleration_body = np.clip(
        velocity_gain * (target[:3] - velocity_body),
        -max_acceleration_mps2,
        max_acceleration_mps2,
    )
    mass = parameters.mass_kg
    drag_ned = np.asarray(parameters.linear_drag_n_per_mps) * velocity_ned
    gravity_cancel_ned = np.asarray([0.0, 0.0, -mass * parameters.gravity_mps2])
    force_body = mass * desired_acceleration_body + rotation.T @ (
        drag_ned + gravity_cancel_ned
    )

    omega = state[10:13]
    roll, pitch = roll_pitch(state[6:10])
    desired_angular_acceleration = np.asarray(
        [
            -attitude_gain * roll - roll_pitch_rate_gain * omega[0],
            -attitude_gain * pitch - roll_pitch_rate_gain * omega[1],
            yaw_rate_gain * (target[3] - omega[2]),
        ]
    )
    inertia = np.asarray(parameters.inertia_kg_m2)
    inertia_omega = inertia * omega
    gyroscopic = np.cross(omega, inertia_omega)
    angular_drag = np.asarray(parameters.angular_drag_nm_per_radps) * omega
    moment_body = inertia * desired_angular_acceleration + gyroscopic + angular_drag
    action = np.asarray((*force_body, *moment_body), dtype=np.float64)
    return np.clip(action, low, high).astype(np.float32)
