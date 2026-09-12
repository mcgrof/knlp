"""Drive multiplayer UFO poses with independent learned motor actors."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from rl.flight.contracts import (
    EnemyPose,
    EnemyShot,
    FlightContract,
    TelemetryFrame,
)
from rl.flight.ufo_dynamics import UfoDynamics
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler
from rl.flight.shadow_ufo import ShadowPolicy

MAXIMUM_ENEMIES = 19
INITIAL_RANGE_M = 4200.0
FORMATION_LONGITUDINAL_SPACING_M = 90.0
FORMATION_LATERAL_SPACING_M = 60.0
FORMATION_VERTICAL_SPACING_M = 12.0
SHOT_PERIOD_S = 4.8
SHOT_DURATION_S = 0.65
SHOT_MINIMUM_RANGE_M = 350.0
SHOT_MAXIMUM_RANGE_M = 5200.0
SHIELD_REQUEST_RANGE_M = 1800.0
SHIELD_REQUEST_DURATION_S = 0.5
SWARM_BEHAVIORS = ("combat", "formation")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class _Enemy:
    dynamics: UfoDynamics
    policy: ShadowPolicy
    origin_ned_m: np.ndarray
    frame_yaw_rad: float
    goal: np.ndarray
    goal_until_s: float = 0.0
    shot_active: bool = False
    shot_aim_ned_m: np.ndarray | None = None
    shield_used: bool = False
    shield_requested_until_s: float = 0.0

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
        behavior: str = "combat",
    ):
        if size < 1 or size > MAXIMUM_ENEMIES:
            raise ValueError("swarm size must be between 1 and 19")
        if behavior not in SWARM_BEHAVIORS:
            raise ValueError(f"swarm behavior must be one of {SWARM_BEHAVIORS}")
        self.contract = contract
        self.model = Path(model)
        self.dynamics_library = Path(dynamics_library)
        self.size = size
        self.behavior = behavior
        self.episode_id: str | None = None
        self.elapsed_s = 0.0
        self.enemies: list[_Enemy] = []
        self.shots: tuple[EnemyShot, ...] = ()
        self.player_goal = np.zeros(4, dtype=np.float64)

    @staticmethod
    def _player_heading(player_state: np.ndarray) -> float:
        body_to_ned = quaternion_body_to_ned(player_state[6:10])
        return math.atan2(float(body_to_ned[1, 0]), float(body_to_ned[0, 0]))

    def _formation_offset(self, slot: int) -> np.ndarray:
        row = 1
        row_start = 0
        while slot - row_start >= row + 1:
            row_start += row + 1
            row += 1
        row_index = slot - row_start
        row_count = min(row + 1, self.size - row_start)
        return np.asarray(
            (
                -FORMATION_LONGITUDINAL_SPACING_M * row,
                FORMATION_LATERAL_SPACING_M * (row_index - 0.5 * (row_count - 1)),
                FORMATION_VERTICAL_SPACING_M * ((slot % 3) - 1),
            ),
            dtype=np.float64,
        )

    def _initial_pose(
        self, slot: int, player_state: np.ndarray
    ) -> tuple[np.ndarray, float]:
        player_position = player_state[:3]
        if self.behavior == "formation":
            heading = self._player_heading(player_state)
            cosine = math.cos(heading)
            sine = math.sin(heading)
            rotation = np.asarray(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
            )
            return player_position + rotation @ self._formation_offset(slot), heading
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
        return desired, math.atan2(float(delta[1]), float(delta[0]))

    def _reset(self, frame: TelemetryFrame) -> None:
        self.enemies = []
        player_state = np.asarray(frame.observation, dtype=np.float64)
        for slot in range(self.size):
            dynamics = UfoDynamics(self.dynamics_library)
            dynamics.reset()
            dynamics.state.position_ned_m[2] = -100.0
            desired, heading = self._initial_pose(slot, player_state)
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
                    goal=np.zeros(4, dtype=np.float64),
                )
            )
        self.episode_id = frame.episode_id
        self.elapsed_s = 0.0
        self.shots = ()
        self.player_goal = np.zeros(4, dtype=np.float64)

    def _combat_goal(
        self,
        slot: int,
        enemy: _Enemy,
        enemy_state: np.ndarray,
        player_state: np.ndarray,
    ) -> np.ndarray:
        delta_ned = player_state[:3] - enemy_state[:3]
        target_body = quaternion_body_to_ned(enemy_state[6:10]).T @ delta_ned
        target_range = float(np.linalg.norm(target_body))
        bearing = math.atan2(float(target_body[1]), float(target_body[0]))
        yaw_rate = _clamp(bearing * 2.0, -1.4, 1.4)
        vertical = _clamp(float(target_body[2]) * 0.08, -24.0, 24.0)
        if not enemy.shield_used and target_range < SHIELD_REQUEST_RANGE_M:
            enemy.shield_used = True
            enemy.shield_requested_until_s = self.elapsed_s + SHIELD_REQUEST_DURATION_S
        if float(np.max(np.abs(enemy_state[3:6]))) > 105.0:
            enemy.goal = np.asarray(
                (0.0, 0.0, vertical, yaw_rate), dtype=np.float64
            )
            enemy.goal_until_s = self.elapsed_s + 2.0
            return enemy.goal
        if self.elapsed_s < enemy.goal_until_s:
            return enemy.goal
        direction = 1.0 if slot % 2 == 0 else -1.0
        weave = 14.0 * math.sin(self.elapsed_s * 1.4 + slot * 1.7)
        if target_range < 1000.0:
            lateral = direction * 50.0
            yaw_rate = _clamp(yaw_rate + direction * 0.9, -1.4, 1.4)
        elif target_range < 1600.0:
            lateral = direction * 38.0 + weave
        else:
            lateral = math.degrees(bearing) * 0.28 + weave
        enemy.goal = np.asarray(
            (
                75.0,
                _clamp(lateral, -50.0, 50.0),
                vertical,
                yaw_rate,
            ),
            dtype=np.float64,
        )
        enemy.goal_until_s = self.elapsed_s + 4.0
        return enemy.goal

    def _formation_goal(
        self,
        slot: int,
        enemy_state: np.ndarray,
        player_state: np.ndarray,
    ) -> np.ndarray:
        player_heading = self._player_heading(player_state)
        cosine = math.cos(player_heading)
        sine = math.sin(player_heading)
        leader_rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        desired_position = player_state[:3] + leader_rotation @ self._formation_offset(
            slot
        )
        position_error = desired_position - enemy_state[:3]
        rotating_offset = desired_position - player_state[:3]
        yaw_rate = float(player_state[12])
        slot_velocity = np.asarray(
            (
                -yaw_rate * rotating_offset[1],
                yaw_rate * rotating_offset[0],
                0.0,
            )
        )
        desired_velocity = player_state[3:6] + slot_velocity + 0.16 * position_error
        enemy_rotation = quaternion_body_to_ned(enemy_state[6:10])
        if float(np.max(np.abs(enemy_state[3:6]))) > 105.0:
            desired_velocity = np.zeros(3, dtype=np.float64)
        velocity_body = enemy_rotation.T @ desired_velocity
        leader_forward_body = enemy_rotation.T @ leader_rotation[:, 0]
        heading_error = math.atan2(
            float(leader_forward_body[1]), float(leader_forward_body[0])
        )
        return np.asarray(
            (
                _clamp(float(velocity_body[0]), -20.0, 90.0),
                _clamp(float(velocity_body[1]), -50.0, 50.0),
                _clamp(float(velocity_body[2]), -24.0, 24.0),
                _clamp(2.0 * heading_error, -1.4, 1.4),
            ),
            dtype=np.float64,
        )

    def _goal(
        self,
        slot: int,
        enemy: _Enemy,
        enemy_state: np.ndarray,
        player_state: np.ndarray,
    ) -> np.ndarray:
        if self.behavior == "formation":
            enemy.goal = self._formation_goal(slot, enemy_state, player_state)
            return enemy.goal
        return self._combat_goal(slot, enemy, enemy_state, player_state)

    def _update_shots(self, player_state: np.ndarray) -> None:
        if self.behavior != "combat":
            self.shots = ()
            return
        shots = []
        spacing_s = SHOT_PERIOD_S / self.size
        for slot, enemy in enumerate(self.enemies):
            enemy_state = enemy.world_state()
            distance_m = float(
                np.linalg.norm(player_state[:3] - enemy_state[:3])
            )
            phase_s = (self.elapsed_s - slot * spacing_s) % SHOT_PERIOD_S
            firing = (
                SHOT_MINIMUM_RANGE_M <= distance_m <= SHOT_MAXIMUM_RANGE_M
                and phase_s < SHOT_DURATION_S
            )
            if firing and not enemy.shot_active:
                lead_s = _clamp(0.35 + distance_m / 10_000.0, 0.35, 0.85)
                enemy.shot_aim_ned_m = (
                    player_state[:3] + lead_s * player_state[3:6]
                )
            enemy.shot_active = firing
            if firing and enemy.shot_aim_ned_m is not None:
                shots.append(
                    EnemyShot.create(
                        slot=slot,
                        aim_position_ned_m=enemy.shot_aim_ned_m,
                    )
                )
        self.shots = tuple(shots)

    def _player_evasive_goal(self, player_state: np.ndarray) -> np.ndarray:
        jink = 1.0 if int(self.elapsed_s / 4.0) % 2 == 0 else -1.0
        altitude_error = -320.0 - float(player_state[2])
        vertical = _clamp(
            altitude_error * 0.06 + 10.0 * math.sin(self.elapsed_s * 1.3),
            -24.0,
            18.0,
        )
        if self.shots:
            return np.asarray(
                (75.0, 40.0 * jink, vertical, 1.2 * jink),
                dtype=np.float64,
            )
        return np.asarray(
            (
                75.0,
                25.0 * math.sin(self.elapsed_s * 0.3),
                vertical,
                0.7 * math.sin(self.elapsed_s * 0.3),
            ),
            dtype=np.float64,
        )

    def control_frame(self, frame: TelemetryFrame) -> TelemetryFrame:
        """Return the live player frame with the swarm's evasive goal."""

        controlled = replace(
            frame,
            goal=tuple(float(value) for value in self.player_goal),
        )
        controlled.validate(self.contract)
        return controlled

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
                goal = self._goal(
                    slot, enemy, world_state, player_state
                )
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
        self._update_shots(player_state)
        if self.behavior == "combat":
            self.player_goal = self._player_evasive_goal(player_state)
        else:
            self.player_goal = np.zeros(4, dtype=np.float64)
        return tuple(
            EnemyPose.create(
                slot=slot,
                position_ned_m=enemy.world_state()[:3],
                quaternion_body_to_ned=enemy.world_state()[6:10],
                shield_requested=(self.elapsed_s < enemy.shield_requested_until_s),
            )
            for slot, enemy in enumerate(self.enemies)
        )
