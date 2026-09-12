"""Fly native X-Plane fighters with independent learned motor actors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.flight.contracts import EnemyPose, FlightContract, TelemetryFrame
from rl.flight.fighter_dynamics import FighterDynamics
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler
from rl.flight.shadow_ufo import ShadowPolicy

MAXIMUM_FIGHTERS = 19
LONGITUDINAL_SPACING_M = 240.0
LATERAL_SPACING_M = 120.0
VERTICAL_SPACING_M = 35.0
RELOCATION_THRESHOLD_M = 1000.0
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[1] / "contracts/fighter-controls-v1.json"
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
        self.model = Path(model)
        identity = ShadowPolicy(self.fighter_contract, self.model)
        self.checkpoint_sha256 = identity.checkpoint_sha256
        self.model_sha256 = identity.model_sha256
        self.size = size
        self.episode_id: str | None = None
        self.fighters: list[_Fighter] = []
        self.shots = ()
        self.previous_player_position_ned_m: np.ndarray | None = None

    @staticmethod
    def _heading(state: np.ndarray) -> float:
        rotation = quaternion_body_to_ned(state[6:10])
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

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

    def _reset(self, frame: TelemetryFrame) -> None:
        player = np.asarray(frame.observation, dtype=np.float64)
        heading = self._heading(player)
        cosine = math.cos(heading)
        sine = math.sin(heading)
        rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        leader_speed = float(np.linalg.norm(player[3:5]))
        initial_speed = _clamp(leader_speed, 140.0, 240.0)
        self.fighters = []
        for slot in range(self.size):
            dynamics = FighterDynamics()
            dynamics.state[:3] = player[:3] + rotation @ self._offset(slot)
            dynamics.state[3:6] = rotation[:, 0] * initial_speed
            dynamics.state[6:10] = quaternion_from_euler(0.0, 0.0, heading)
            self.fighters.append(
                _Fighter(
                    dynamics=dynamics,
                    policy=ShadowPolicy(self.fighter_contract, self.model),
                )
            )
        self.episode_id = frame.episode_id
        self.previous_player_position_ned_m = player[:3].copy()

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
        desired_velocity = leader_velocity + 0.08 * position_error
        desired_heading = math.atan2(
            float(desired_velocity[1]), float(desired_velocity[0])
        )
        heading_error = _wrap_angle(desired_heading - self._heading(fighter_state))
        return np.asarray(
            (
                _clamp(float(np.linalg.norm(desired_velocity)), 120.0, 280.0),
                _clamp(-float(desired_velocity[2]), -30.0, 30.0),
                _clamp(1.4 * heading_error, -0.08, 0.08),
            )
        )

    def control_frame(self, frame: TelemetryFrame) -> TelemetryFrame:
        return frame

    def update(self, frame: TelemetryFrame) -> tuple[EnemyPose, ...]:
        frame.validate(self.output_contract)
        if frame.episode_id != self.episode_id:
            self._reset(frame)
        player = np.asarray(frame.observation, dtype=np.float64)
        self._follow_relocation(player)
        self.previous_player_position_ned_m = player[:3].copy()
        remaining = min(frame.dt_s, 0.1)
        while remaining > 1e-9:
            dt_s = min(self.fighter_contract.nominal_dt_s, remaining)
            for slot, fighter in enumerate(self.fighters):
                state = fighter.dynamics.state_vector()
                goal = self._goal(slot, state, player)
                actor_frame = TelemetryFrame.create(
                    self.fighter_contract,
                    episode_id=f"{frame.episode_id}-f14-{slot}",
                    sequence=frame.sequence,
                    monotonic_ns=frame.monotonic_ns,
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
