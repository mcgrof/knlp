"""Fly native X-Plane fighters with independent learned motor actors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.flight.contracts import EnemyPose, FlightContract, TelemetryFrame
from rl.flight.fighter_dynamics import FighterDynamics, fighter_envelope
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler
from rl.flight.shadow_ufo import ShadowPolicy

MAXIMUM_FIGHTERS = 19
LONGITUDINAL_SPACING_M = 240.0
LATERAL_SPACING_M = 120.0
VERTICAL_SPACING_M = 35.0
RELOCATION_THRESHOLD_M = 1000.0
LEGACY_POSITION_GAIN_PER_S = 0.12
FAST_POSITION_GAIN_PER_S = 0.30
LEGACY_POSITION_CORRECTION_MPS = 90.0
FAST_POSITION_CORRECTION_MPS = 120.0
LEGACY_FORMATION_SPEED_MPS = 280.0
FAST_FORMATION_SPEED_MPS = 780.0
MAXIMUM_FORMATION_TURN_RADPS = 0.08
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[1] / "contracts/fighter-controls-v3.json"
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass
class _Fighter:
    dynamics: FighterDynamics
    policy: ShadowPolicy


class RlF14Swarm:
    """A fixed-wing dynamics instance and F-14 motor actor per AI slot."""

    def __init__(
        self,
        output_contract: FlightContract,
        model: str | Path,
        size: int,
        fighter_contract_path: str | Path = DEFAULT_CONTRACT,
    ):
        if size < 1 or size > MAXIMUM_FIGHTERS:
            raise ValueError("F-14 swarm size must be between 1 and 19")
        self.output_contract = output_contract
        self.fighter_contract = FlightContract.from_json(fighter_contract_path)
        self.dynamics_envelope = fighter_envelope(self.fighter_contract.revision)
        if self.fighter_contract.revision == 3:
            self.position_gain_per_s = FAST_POSITION_GAIN_PER_S
            self.maximum_position_correction_mps = FAST_POSITION_CORRECTION_MPS
            self.maximum_formation_speed_mps = FAST_FORMATION_SPEED_MPS
        else:
            self.position_gain_per_s = LEGACY_POSITION_GAIN_PER_S
            self.maximum_position_correction_mps = (
                LEGACY_POSITION_CORRECTION_MPS
            )
            self.maximum_formation_speed_mps = LEGACY_FORMATION_SPEED_MPS
        self.model = Path(model)
        identity = ShadowPolicy(self.fighter_contract, self.model)
        self.checkpoint_sha256 = identity.checkpoint_sha256
        self.model_sha256 = identity.model_sha256
        self.size = size
        self.episode_id: str | None = None
        self.fighters: list[_Fighter] = []
        self.shots = ()
        self.previous_player_position_ned_m: np.ndarray | None = None
        self.previous_player_heading_rad: float | None = None

    @staticmethod
    def _heading(state: np.ndarray) -> float:
        rotation = quaternion_body_to_ned(state[6:10])
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    @staticmethod
    def _heading_rate(state: np.ndarray) -> float:
        rotation = quaternion_body_to_ned(state[6:10])
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        q = float(state[11])
        r = float(state[12])
        return (q * math.sin(roll) + r * math.cos(roll)) / max(
            0.1, abs(math.cos(pitch))
        )

    def _offset(self, slot: int) -> np.ndarray:
        row = 1
        row_start = 0
        while slot - row_start >= row + 1:
            row_start += row + 1
            row += 1
        row_index = slot - row_start
        row_count = min(row + 1, self.size - row_start)
        return np.asarray(
            (
                -LONGITUDINAL_SPACING_M * row,
                LATERAL_SPACING_M * (row_index - 0.5 * (row_count - 1)),
                VERTICAL_SPACING_M * ((slot % 3) - 1),
            )
        )

    def _reset(self, player: np.ndarray, episode_id: str) -> None:
        heading = self._heading(player)
        cosine = math.cos(heading)
        sine = math.sin(heading)
        rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        leader_speed = float(np.linalg.norm(player[3:6]))
        if self.fighter_contract.revision == 3:
            initial_speed = _clamp(
                leader_speed,
                max(120.0, self.fighter_contract.goal.low[0]),
                min(
                    self.maximum_formation_speed_mps,
                    self.fighter_contract.goal.high[0],
                ),
            )
        else:
            initial_speed = _clamp(leader_speed, 140.0, 240.0)
        self.fighters = []
        for slot in range(self.size):
            dynamics = FighterDynamics(self.dynamics_envelope)
            dynamics.state[:3] = player[:3] + rotation @ self._offset(slot)
            dynamics.state[3:6] = rotation[:, 0] * initial_speed
            dynamics.state[6:10] = quaternion_from_euler(0.0, 0.0, heading)
            self.fighters.append(
                _Fighter(
                    dynamics=dynamics,
                    policy=ShadowPolicy(self.fighter_contract, self.model),
                )
            )
        self.episode_id = episode_id
        self.previous_player_position_ned_m = player[:3].copy()
        self.previous_player_heading_rad = heading

    def _follow_relocation(self, player: np.ndarray) -> None:
        previous = self.previous_player_position_ned_m
        if previous is None:
            return
        displacement = player[:3] - previous
        if float(np.linalg.norm(displacement)) < RELOCATION_THRESHOLD_M:
            return
        for fighter in self.fighters:
            fighter.dynamics.state[:3] += displacement

    def _goal(
        self,
        slot: int,
        fighter_state: np.ndarray,
        player: np.ndarray,
        leader_turn_rate: float,
    ) -> np.ndarray:
        player_heading = self._heading(player)
        cosine = math.cos(player_heading)
        sine = math.sin(player_heading)
        rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        desired_position = player[:3] + rotation @ self._offset(slot)
        position_error = desired_position - fighter_state[:3]
        leader_velocity = player[3:6].copy()
        if float(np.linalg.norm(leader_velocity[:2])) < 100.0:
            leader_velocity = rotation[:, 0] * 180.0
        rotated_offset = rotation @ self._offset(slot)
        slot_velocity = np.asarray(
            (
                -leader_turn_rate * rotated_offset[1],
                leader_turn_rate * rotated_offset[0],
                0.0,
            )
        )
        position_correction = self.position_gain_per_s * position_error
        correction_speed = float(np.linalg.norm(position_correction))
        if correction_speed > self.maximum_position_correction_mps:
            position_correction *= (
                self.maximum_position_correction_mps / correction_speed
            )
        desired_velocity = leader_velocity + slot_velocity + position_correction
        desired_heading = math.atan2(
            float(desired_velocity[1]), float(desired_velocity[0])
        )
        heading_error = _wrap_angle(desired_heading - self._heading(fighter_state))
        return np.asarray(
            (
                _clamp(
                    float(np.linalg.norm(desired_velocity)),
                    120.0,
                    min(
                        self.maximum_formation_speed_mps,
                        self.fighter_contract.goal.high[0],
                    ),
                ),
                _clamp(
                    -float(desired_velocity[2]),
                    self.fighter_contract.goal.low[1],
                    self.fighter_contract.goal.high[1],
                ),
                _clamp(
                    leader_turn_rate + 1.4 * heading_error,
                    -MAXIMUM_FORMATION_TURN_RADPS,
                    MAXIMUM_FORMATION_TURN_RADPS,
                ),
            )
        )

    def control_frame(self, frame: TelemetryFrame) -> TelemetryFrame:
        return frame

    def update_state(
        self,
        player_state: np.ndarray,
        *,
        episode_id: str,
        sequence: int,
        monotonic_ns: int,
        dt_s: float,
    ) -> tuple[EnemyPose, ...]:
        """Advance followers from a full player state in local NED."""

        player = np.asarray(player_state, dtype=np.float64)
        if player.shape != (13,) or not np.isfinite(player).all():
            raise ValueError("F-14 formation player state must have 13 values")
        if not episode_id or sequence < 0 or monotonic_ns < 0:
            raise ValueError("F-14 formation frame identity is invalid")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("F-14 formation dt must be positive")
        if episode_id != self.episode_id:
            self._reset(player, episode_id)
        self._follow_relocation(player)
        heading = self._heading(player)
        leader_turn_rate = _clamp(
            self._heading_rate(player),
            -MAXIMUM_FORMATION_TURN_RADPS,
            MAXIMUM_FORMATION_TURN_RADPS,
        )
        self.previous_player_position_ned_m = player[:3].copy()
        self.previous_player_heading_rad = heading
        remaining = min(dt_s, 0.1)
        while remaining > 1e-9:
            dt_s = min(self.fighter_contract.nominal_dt_s, remaining)
            for slot, fighter in enumerate(self.fighters):
                state = fighter.dynamics.state_vector()
                goal = self._goal(
                    slot, state, player, leader_turn_rate
                )
                actor_frame = TelemetryFrame.create(
                    self.fighter_contract,
                    episode_id=f"{episode_id}-f14-{slot}",
                    sequence=sequence,
                    monotonic_ns=monotonic_ns,
                    dt_s=dt_s,
                    observation=state[3:],
                    goal=goal,
                )
                action, _ = fighter.policy.infer(actor_frame)
                fighter.dynamics.step(action, dt_s)
            remaining -= dt_s
        return tuple(
            EnemyPose.create(
                slot=slot,
                position_ned_m=fighter.dynamics.state_vector()[:3],
                quaternion_body_to_ned=fighter.dynamics.state_vector()[6:10],
                native_visual=True,
            )
            for slot, fighter in enumerate(self.fighters)
        )

    def update(self, frame: TelemetryFrame) -> tuple[EnemyPose, ...]:
        frame.validate(self.output_contract)
        return self.update_state(
            np.asarray(frame.observation, dtype=np.float64),
            episode_id=frame.episode_id,
            sequence=frame.sequence,
            monotonic_ns=frame.monotonic_ns,
            dt_s=frame.dt_s,
        )
