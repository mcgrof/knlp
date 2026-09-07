"""Small, explicit coordinate transforms used by flight environments."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def quaternion_body_to_ned(quaternion: Sequence[float]) -> np.ndarray:
    """Return the rotation matrix for a normalized ``(w, x, y, z)`` quaternion."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a body-to-NED quaternion from aerospace roll, pitch and yaw."""

    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def roll_pitch(quaternion: Sequence[float]) -> tuple[float, float]:
    """Extract roll and pitch; yaw is intentionally irrelevant to upright control."""

    rotation = quaternion_body_to_ned(quaternion)
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    return roll, pitch
