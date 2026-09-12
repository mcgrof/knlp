"""Drive multiplayer UFO poses with independent learned motor actors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.flight.contracts import EnemyPose, FlightContract, TelemetryFrame
from rl.flight.ufo_dynamics import UfoDynamics
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler
from rl.flight.shadow_ufo import ShadowPolicy

MAXIMUM_ENEMIES = 19
INITIAL_RANGE_M = 2400.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class _Enemy:
    dynamics: UfoDynamics
    policy: ShadowPolicy
    origin_ned_m: np.ndarray
    frame_yaw_rad: float

    def world_state(self) -> np.ndarray:
        state = self.dynamics.state_vector()
        cosine = math.cos(self.frame_yaw_rad)
        sine = math.sin(self.frame_yaw_rad)
        rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        state[:3] = self.origin_ned_m + rotation @ state[:3]
        state[3:6] = rotation @ state[3:6]
        frame_quaternion = quaternion_from_euler(0.0, 0.0, self.frame_yaw_rad)
        local = state[6:10].copy()
        state[6:10] = frame_quaternion[0] * local + frame_quaternion[3] * np.asarray(
            (-local[3], -local[2], local[1], local[0])
        )
        return state


class RlUfoSwarm:
    """A headless C dynamics instance and learned actor for every enemy."""

    def __init__(
        self,
        contract: FlightContract,
        model: str | Path,
        dynamics_library: str | Path,
        size: int,
    ):
        if size < 1 or size > MAXIMUM_ENEMIES:
            raise ValueError("swarm size must be between 1 and 19")
        self.contract = contract
        self.model = Path(model)
        self.dynamics_library = Path(dynamics_library)
        self.size = size
        self.episode_id: str | None = None
        self.elapsed_s = 0.0
        self.enemies: list[_Enemy] = []

    def _reset(self, frame: TelemetryFrame) -> None:
        self.enemies = []
        player_position = np.asarray(frame.observation[:3], dtype=np.float64)
        for slot in range(self.size):
            dynamics = UfoDynamics(self.dynamics_library)
            dynamics.reset()
            dynamics.state.position_ned_m[2] = -100.0
            angle = 2.0 * math.pi * slot / self.size
            radius = INITIAL_RANGE_M + 180.0 * (slot % 3)
            desired = player_position + np.asarray(
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    80.0 * ((slot % 3) - 1),
                ),
                dtype=np.float64,
            )
            delta = player_position - desired
            heading = math.atan2(float(delta[1]), float(delta[0]))
            cosine = math.cos(heading)
            sine = math.sin(heading)
            rotation = np.asarray(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
            )
            origin = desired - rotation @ dynamics.state_vector()[:3]
            self.enemies.append(
                _Enemy(
                    dynamics=dynamics,
                    policy=ShadowPolicy(self.contract, self.model),
                    origin_ned_m=origin,
                    frame_yaw_rad=heading,
                )
            )
        self.episode_id = frame.episode_id
        self.elapsed_s = 0.0

    def _goal(
        self, slot: int, enemy_state: np.ndarray, player_state: np.ndarray
    ) -> np.ndarray:
        delta_ned = player_state[:3] - enemy_state[:3]
        target_body = quaternion_body_to_ned(enemy_state[6:10]).T @ delta_ned
        target_range = float(np.linalg.norm(target_body))
        bearing = math.atan2(float(target_body[1]), float(target_body[0]))
        yaw_rate = _clamp(bearing * 2.0, -1.4, 1.4)
        if float(np.linalg.norm(enemy_state[3:6])) > 115.0:
            return np.asarray((0.0, 0.0, 0.0, yaw_rate), dtype=np.float64)
        weave = 6.0 * math.sin(self.elapsed_s * 0.9 + slot * 1.7)
        return np.asarray(
            (
                _clamp((target_range - 520.0) * 0.06, -20.0, 90.0),
                _clamp(math.degrees(bearing) * 0.2 + weave, -25.0, 25.0),
                _clamp(float(target_body[2]) * 0.05, -24.0, 24.0),
                yaw_rate,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _recenter(enemy: _Enemy) -> None:
        state = enemy.dynamics.state
        if (
            abs(state.position_ned_m[0]) < 10_000.0
            and abs(state.position_ned_m[1]) < 10_000.0
            and -10_000.0 < state.position_ned_m[2] < -25.0
        ):
            return
        world_position = enemy.world_state()[:3]
        local_position = np.asarray((0.0, 0.0, -100.0))
        cosine = math.cos(enemy.frame_yaw_rad)
        sine = math.sin(enemy.frame_yaw_rad)
        rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        enemy.origin_ned_m = world_position - rotation @ local_position
        for axis, value in enumerate(local_position):
            state.position_ned_m[axis] = value

    def update(self, frame: TelemetryFrame) -> tuple[EnemyPose, ...]:
        frame.validate(self.contract)
        if frame.episode_id != self.episode_id:
            self._reset(frame)
        player_state = np.asarray(frame.observation, dtype=np.float64)
        remaining = min(frame.dt_s, 0.1)
        while remaining > 1e-9:
            dt_s = min(self.contract.nominal_dt_s, remaining)
            for slot, enemy in enumerate(self.enemies):
                world_state = enemy.world_state()
                goal = self._goal(slot, world_state, player_state)
                local_state = enemy.dynamics.state_vector()
                actor_frame = TelemetryFrame.create(
                    self.contract,
                    episode_id=f"{frame.episode_id}-enemy-{slot}",
                    sequence=frame.sequence,
                    monotonic_ns=frame.monotonic_ns,
                    dt_s=dt_s,
                    observation=local_state,
                    goal=goal,
                )
                action, _ = enemy.policy.infer(actor_frame)
                enemy.dynamics.step(action, dt_s)
                self._recenter(enemy)
            self.elapsed_s += dt_s
            remaining -= dt_s
        return tuple(
            EnemyPose.create(
                slot=slot,
                position_ned_m=enemy.world_state()[:3],
                quaternion_body_to_ned=enemy.world_state()[6:10],
            )
            for slot, enemy in enumerate(self.enemies)
        )
