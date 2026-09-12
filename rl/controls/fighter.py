"""Reference controller for the fixed-wing fighter surrogate."""

from __future__ import annotations

import math

import numpy as np

from rl.envs.fighter_env import FighterEnv
from rl.flight.fighter_dynamics import GRAVITY_MPS2
from rl.flight.geometry import quaternion_body_to_ned


class FighterReferenceController:
    """Track speed, climb, and coordinated turn with bounded controls."""

    def __call__(self, env: FighterEnv) -> np.ndarray:
        state = env.dynamics.state_vector()
        speed, _, turn_rate = env.tracking(state)
        desired_speed, desired_climb, desired_turn = env.goal
        rotation = quaternion_body_to_ned(state[6:10])
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        desired_roll = float(
            np.clip(
                math.atan2(speed * desired_turn, GRAVITY_MPS2),
                -math.radians(65.0),
                math.radians(65.0),
            )
        )
        desired_pitch = math.asin(
            float(np.clip(desired_climb / max(speed, 80.0), -0.45, 0.45))
        )
        cosine_roll = max(0.25, math.cos(roll))
        pitch_rate = 1.8 * (desired_pitch - pitch)
        desired_q = (pitch_rate + state[12] * math.sin(roll)) / cosine_roll
        roll_cross_coupling = math.tan(pitch) * (
            state[11] * math.sin(roll) + state[12] * math.cos(roll)
        )
        desired_p = 2.2 * (desired_roll - roll) - roll_cross_coupling
        throttle = (desired_speed / 320.0) ** 2 + 0.018 * (desired_speed - speed)
        aileron = (2.6 * desired_p + 4.0 * (desired_p - state[10])) / 5.0
        elevator = (2.4 * desired_q + 4.0 * (desired_q - state[11])) / 3.2
        rudder = 4.0 * (desired_turn - turn_rate)
        return np.asarray(
            (
                np.clip(throttle, 0.0, 1.0),
                np.clip(aileron, -1.0, 1.0),
                np.clip(elevator, -1.0, 1.0),
                np.clip(rudder, -1.0, 1.0),
            ),
            dtype=np.float32,
        )


def fighter_reference_controller(env: FighterEnv) -> FighterReferenceController:
    del env
    return FighterReferenceController()
