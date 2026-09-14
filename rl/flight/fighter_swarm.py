"""Fly native X-Plane fighters with independent learned motor actors."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.flight.contracts import EnemyPose, FlightContract, TelemetryFrame
from rl.flight.fighter_dynamics import FighterDynamics, fighter_envelope
from rl.flight.geometry import quaternion_body_to_ned
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
FORMATION_VELOCITY_RESPONSE_MPS2 = 300.0
FORMATION_ATTITUDE_RESPONSE_S = 0.35
FORMATION_SLOT_LAG_FRACTION = 0.08
FORMATION_REACTION_DELAY_S = 0.10
FORMATION_REACTION_SLOT_STEP_S = 0.03
MAXIMUM_REACTION_DISTANCE_M = 60.0
LEADER_HISTORY_S = 0.50
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
    display_quaternion: np.ndarray


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
        self.leader_history: deque[tuple[int, np.ndarray]] = deque()

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

    def _reset(
        self, player: np.ndarray, episode_id: str, monotonic_ns: int
    ) -> None:
        heading = self._heading(player)
        rotation = quaternion_body_to_ned(player[6:10])
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
            offset = rotation @ self._offset(slot)
            dynamics.state[:3] = player[:3] + offset
            desired_velocity = self._slot_velocity(player, offset)
            velocity_norm = float(np.linalg.norm(desired_velocity))
            if velocity_norm > 1e-9:
                dynamics.state[3:6] = desired_velocity * (
                    initial_speed / velocity_norm
                )
            else:
                dynamics.state[3:6] = rotation[:, 0] * initial_speed
            dynamics.state[6:10] = player[6:10]
            dynamics.state[10:13] = player[10:13]
            self.fighters.append(
                _Fighter(
                    dynamics=dynamics,
                    policy=ShadowPolicy(self.fighter_contract, self.model),
                    display_quaternion=player[6:10].copy(),
                )
            )
        self.episode_id = episode_id
        self.previous_player_position_ned_m = player[:3].copy()
        self.previous_player_heading_rad = heading
        self.leader_history.clear()
        self.leader_history.append((monotonic_ns, player.copy()))

    @staticmethod
    def _slot_velocity(player: np.ndarray, rotated_offset: np.ndarray) -> np.ndarray:
        rotation = quaternion_body_to_ned(player[6:10])
        angular_velocity_ned = rotation @ player[10:13]
        return player[3:6] + np.cross(angular_velocity_ned, rotated_offset)

    @staticmethod
    def _blend_quaternion(
        current: np.ndarray, target: np.ndarray, fraction: float
    ) -> np.ndarray:
        """Blend along the shortest normalized quaternion arc."""

        destination = target.copy()
        if float(np.dot(current, destination)) < 0.0:
            destination *= -1.0
        blended = current + fraction * (destination - current)
        return blended / max(float(np.linalg.norm(blended)), 1e-9)

    def _feed_forward_leader_motion(
        self, player: np.ndarray, dt_s: float, monotonic_ns: int
    ) -> list[np.ndarray]:
        """Guide independent actors toward moving slots without snapping."""

        delayed_leaders = []
        for slot, fighter in enumerate(self.fighters):
            leader = self._delayed_leader(slot, player, monotonic_ns)
            delayed_leaders.append(leader)
            rotation = quaternion_body_to_ned(leader[6:10])
            offset = self._offset(slot)
            rotated_offset = rotation @ offset
            desired_position = leader[:3] + rotated_offset
            position_correction = self.position_gain_per_s * (
                desired_position - fighter.dynamics.state[:3]
            )
            correction_speed = float(np.linalg.norm(position_correction))
            if correction_speed > self.maximum_position_correction_mps:
                position_correction *= (
                    self.maximum_position_correction_mps / correction_speed
                )
            desired_velocity = (
                self._slot_velocity(leader, rotated_offset)
                + position_correction
            )
            desired_speed = float(np.linalg.norm(desired_velocity))
            if desired_speed > self.dynamics_envelope.maximum_airspeed_mps:
                desired_velocity *= (
                    self.dynamics_envelope.maximum_airspeed_mps
                    / desired_speed
                )
            lag_scale = 1.0 + FORMATION_SLOT_LAG_FRACTION * slot
            velocity_delta = (
                desired_velocity - fighter.dynamics.state[3:6]
            )
            maximum_delta = (
                FORMATION_VELOCITY_RESPONSE_MPS2 * dt_s / lag_scale
            )
            delta_norm = float(np.linalg.norm(velocity_delta))
            if delta_norm > maximum_delta:
                velocity_delta *= maximum_delta / delta_norm
            fighter.dynamics.state[3:6] += velocity_delta
            attitude_fraction = 1.0 - math.exp(
                -dt_s / (FORMATION_ATTITUDE_RESPONSE_S * lag_scale)
            )
            fighter.display_quaternion = self._blend_quaternion(
                fighter.display_quaternion,
                leader[6:10],
                attitude_fraction,
            )
            fighter.dynamics.state[6:10] = leader[6:10]
            fighter.dynamics.state[10:13] = leader[10:13]
        return delayed_leaders

    def _record_leader(self, player: np.ndarray, monotonic_ns: int) -> None:
        self.leader_history.append((monotonic_ns, player.copy()))
        oldest_ns = monotonic_ns - round(LEADER_HISTORY_S * 1e9)
        while (
            len(self.leader_history) > 2
            and self.leader_history[1][0] < oldest_ns
        ):
            self.leader_history.popleft()

    def _delayed_leader(
        self, slot: int, current: np.ndarray, monotonic_ns: int
    ) -> np.ndarray:
        speed_mps = max(1.0, float(np.linalg.norm(current[3:6])))
        delay_s = min(
            FORMATION_REACTION_DELAY_S
            + FORMATION_REACTION_SLOT_STEP_S * slot,
            MAXIMUM_REACTION_DISTANCE_M / speed_mps,
        )
        target_ns = monotonic_ns - round(delay_s * 1e9)
        before_ns, before = self.leader_history[0]
        for after_ns, after in self.leader_history:
            if after_ns >= target_ns:
                if after_ns == before_ns:
                    return after.copy()
                fraction = (target_ns - before_ns) / (after_ns - before_ns)
                delayed = before + fraction * (after - before)
                delayed[6:10] = self._blend_quaternion(
                    before[6:10], after[6:10], fraction
                )
                return delayed
            before_ns, before = after_ns, after
        return before.copy()

    def _follow_relocation(self, player: np.ndarray) -> bool:
        previous = self.previous_player_position_ned_m
        if previous is None:
            return False
        displacement = player[:3] - previous
        if float(np.linalg.norm(displacement)) < RELOCATION_THRESHOLD_M:
            return False
        for fighter in self.fighters:
            fighter.dynamics.state[:3] += displacement
        self.leader_history.clear()
        return True

    def _goal(
        self,
        slot: int,
        fighter_state: np.ndarray,
        player: np.ndarray,
        leader_turn_rate: float,
    ) -> np.ndarray:
        rotation = quaternion_body_to_ned(player[6:10])
        desired_position = player[:3] + rotation @ self._offset(slot)
        position_error = desired_position - fighter_state[:3]
        rotated_offset = rotation @ self._offset(slot)
        leader_velocity = self._slot_velocity(player, rotated_offset)
        position_correction = self.position_gain_per_s * position_error
        correction_speed = float(np.linalg.norm(position_correction))
        if correction_speed > self.maximum_position_correction_mps:
            position_correction *= (
                self.maximum_position_correction_mps / correction_speed
            )
        desired_velocity = leader_velocity + position_correction
        if float(np.linalg.norm(desired_velocity[:2])) < 1e-9:
            desired_heading = self._heading(player)
        else:
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
            self._reset(player, episode_id, monotonic_ns)
        else:
            self._follow_relocation(player)
            self._record_leader(player, monotonic_ns)
        delayed_leaders = self._feed_forward_leader_motion(
            player, dt_s, monotonic_ns
        )
        heading = self._heading(player)
        self.previous_player_position_ned_m = player[:3].copy()
        self.previous_player_heading_rad = heading
        remaining = min(dt_s, 0.1)
        while remaining > 1e-9:
            dt_s = min(self.fighter_contract.nominal_dt_s, remaining)
            for slot, fighter in enumerate(self.fighters):
                leader = delayed_leaders[slot]
                leader_turn_rate = _clamp(
                    self._heading_rate(leader),
                    -MAXIMUM_FORMATION_TURN_RADPS,
                    MAXIMUM_FORMATION_TURN_RADPS,
                )
                state = fighter.dynamics.state_vector()
                goal = self._goal(
                    slot, state, leader, leader_turn_rate
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
                quaternion_body_to_ned=fighter.display_quaternion,
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
