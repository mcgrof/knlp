"""Deterministic fixed-wing surrogate for fighter motor-policy training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.flight.geometry import quaternion_body_to_ned

GRAVITY_MPS2 = 9.80665
MINIMUM_AIRSPEED_MPS = 55.0
MAXIMUM_AIRSPEED_MPS = 380.0
MAXIMUM_ROLL_RAD = math.radians(75.0)


@dataclass(frozen=True)
class FighterEnvelope:
    """Speed terms for one version of the training surrogate."""

    maximum_airspeed_mps: float
    thrust_speed_mps: float
    acceleration_gain_mps2: float
    maximum_commanded_roll_rad: float


LEGACY_ENVELOPE = FighterEnvelope(
    maximum_airspeed_mps=MAXIMUM_AIRSPEED_MPS,
    thrust_speed_mps=320.0,
    acceleration_gain_mps2=35.0,
    maximum_commanded_roll_rad=math.radians(65.0),
)
FORMATION_ENVELOPE = FighterEnvelope(
    maximum_airspeed_mps=820.0,
    thrust_speed_mps=800.0,
    acceleration_gain_mps2=60.0,
    maximum_commanded_roll_rad=math.radians(72.0),
)


def fighter_envelope(contract_revision: int) -> FighterEnvelope:
    """Keep actors tied to the dynamics envelope they were trained on."""

    if contract_revision == 3:
        return FORMATION_ENVELOPE
    if contract_revision in {1, 2}:
        return LEGACY_ENVELOPE
    raise ValueError(
        f"unsupported fighter contract revision {contract_revision}"
    )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("fighter quaternion became invalid")
    return quaternion / norm


class FighterDynamics:
    """Small coordinated-flight model with pilot-like control inputs.

    This is an actor-training surrogate, not a claimed F-14 aerodynamic
    model. Its control ABI matches the inputs that a later X-Plane adapter
    will drive: throttle, aileron, elevator, and rudder.
    """

    source_path = Path(__file__).resolve()

    def __init__(self, envelope: FighterEnvelope = LEGACY_ENVELOPE):
        self.envelope = envelope
        self.state = np.zeros(13, dtype=np.float64)
        self.reset()

    def reset(self) -> np.ndarray:
        self.state.fill(0.0)
        self.state[2] = -1500.0
        self.state[3] = 180.0
        self.state[6] = 1.0
        return self.state_vector()

    def state_vector(self) -> np.ndarray:
        return self.state.copy()

    @staticmethod
    def _roll(quaternion: np.ndarray) -> float:
        rotation = quaternion_body_to_ned(quaternion)
        return math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))

    def step(self, action: Sequence[float], dt_s: float) -> np.ndarray:
        controls = np.asarray(action, dtype=np.float64)
        if controls.shape != (4,) or not np.isfinite(controls).all():
            raise ValueError("fighter action must contain four finite values")
        if not math.isfinite(dt_s) or dt_s <= 0.0 or dt_s > 0.1:
            raise ValueError("fighter step must use a finite dt in (0, 0.1]")
        throttle = float(np.clip(controls[0], 0.0, 1.0))
        aileron = float(np.clip(controls[1], -1.0, 1.0))
        elevator = float(np.clip(controls[2], -1.0, 1.0))
        rudder = float(np.clip(controls[3], -1.0, 1.0))

        velocity = self.state[3:6]
        speed = float(np.linalg.norm(velocity))
        speed = max(MINIMUM_AIRSPEED_MPS, speed)
        roll = float(
            np.clip(self._roll(self.state[6:10]), -MAXIMUM_ROLL_RAD, MAXIMUM_ROLL_RAD)
        )
        rates = self.state[10:13]
        coordinated_turn = GRAVITY_MPS2 * math.tan(roll) / speed
        rate_acceleration = np.asarray(
            (
                5.0 * aileron - 2.6 * rates[0],
                3.2 * elevator - 2.4 * rates[1],
                3.0 * (coordinated_turn + 0.16 * rudder - rates[2]),
            )
        )
        rates += rate_acceleration * dt_s
        np.clip(rates, -4.0, 4.0, out=rates)

        quaternion = self.state[6:10]
        omega = np.asarray((0.0, *rates))
        w1, x1, y1, z1 = quaternion
        w2, x2, y2, z2 = omega
        derivative = 0.5 * np.asarray(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )
        quaternion[:] = _normalize_quaternion(quaternion + derivative * dt_s)

        speed_acceleration = self.envelope.acceleration_gain_mps2 * (
            throttle - (speed / self.envelope.thrust_speed_mps) ** 2
        )
        new_speed = float(
            np.clip(
                speed + speed_acceleration * dt_s,
                MINIMUM_AIRSPEED_MPS,
                self.envelope.maximum_airspeed_mps,
            )
        )
        rotation = quaternion_body_to_ned(quaternion)
        desired_direction = rotation[:, 0]
        current_direction = velocity / max(float(np.linalg.norm(velocity)), 1e-9)
        direction = current_direction + 2.5 * dt_s * (
            desired_direction - current_direction
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        velocity[:] = direction * new_speed
        self.state[:3] += velocity * dt_s
        return self.state_vector()
