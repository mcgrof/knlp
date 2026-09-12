"""Versioned flight vectors and timestamped wire messages.

The simulator plugin, a standalone environment and a policy worker must agree
on more than vector width.  This module gives field order, units, bounds and
timing semantics one canonical representation and hashes that representation.
Every telemetry frame and control command carries the hash, so a stale client
fails closed instead of applying values to the wrong axes.

Advice is deliberately absent.  A shadow coach consumes ``TelemetryFrame``
objects but must use a separate advice-only transport; it cannot acquire a
``ControlCommand`` sender by changing a mode flag.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL = "knlp.flight"
PROTOCOL_VERSION = 1


def _finite_vector(values: Sequence[float], width: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != width:
        raise ValueError(f"{name} has width {len(result)}, expected {width}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class VectorSpec:
    """Ordered, bounded fields for one vector in the environment contract."""

    fields: tuple[str, ...]
    units: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> "VectorSpec":
        payload = _strict_mapping(value, {"fields", "units", "low", "high"}, name)
        fields = payload["fields"]
        units = payload["units"]
        low = payload["low"]
        high = payload["high"]
        if not isinstance(fields, (list, tuple)) or not all(
            isinstance(field, str) for field in fields
        ):
            raise ValueError(f"{name}.fields must be a string array")
        if not isinstance(units, (list, tuple)) or not all(
            isinstance(unit, str) for unit in units
        ):
            raise ValueError(f"{name}.units must be a string array")
        try:
            lower = tuple(float(item) for item in low)
            upper = tuple(float(item) for item in high)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} bounds must be numeric arrays") from error
        return cls(tuple(fields), tuple(units), lower, upper)

    def __post_init__(self) -> None:
        width = len(self.fields)
        if width == 0:
            raise ValueError("a vector must contain at least one field")
        if len(set(self.fields)) != width:
            raise ValueError("vector field names must be unique")
        if not (len(self.units) == len(self.low) == len(self.high) == width):
            raise ValueError("fields, units and bounds must have equal widths")
        if any(not field for field in self.fields):
            raise ValueError("vector field names must not be empty")
        for field, lower, upper in zip(self.fields, self.low, self.high):
            if math.isnan(lower) or math.isnan(upper) or lower >= upper:
                raise ValueError(f"invalid bounds for {field}: [{lower}, {upper}]")

    @property
    def width(self) -> int:
        return len(self.fields)

    def validate(self, values: Sequence[float], name: str) -> tuple[float, ...]:
        result = _finite_vector(values, self.width, name)
        for field, value, lower, upper in zip(self.fields, result, self.low, self.high):
            if value < lower or value > upper:
                raise ValueError(
                    f"{name}.{field}={value} is outside [{lower}, {upper}]"
                )
        return result


@dataclass(frozen=True)
class FlightContract:
    """The complete vector and timing contract for one flight environment."""

    name: str
    revision: int
    nominal_dt_s: float
    observation: VectorSpec
    goal: VectorSpec
    action: VectorSpec

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlightContract":
        payload = _strict_mapping(
            value,
            {"name", "revision", "nominal_dt_s", "observation", "goal", "action"},
            "contract",
        )
        revision = payload["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise ValueError("contract.revision must be an integer")
        try:
            nominal_dt_s = float(payload["nominal_dt_s"])
        except (TypeError, ValueError) as error:
            raise ValueError("contract.nominal_dt_s must be numeric") from error
        return cls(
            name=payload["name"],
            revision=revision,
            nominal_dt_s=nominal_dt_s,
            observation=VectorSpec.from_mapping(
                payload["observation"], "contract.observation"
            ),
            goal=VectorSpec.from_mapping(payload["goal"], "contract.goal"),
            action=VectorSpec.from_mapping(payload["action"], "contract.action"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FlightContract":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read flight contract {source}") from error
        return cls.from_mapping(value)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("contract name must not be empty")
        if self.revision < 1:
            raise ValueError("contract revision must be positive")
        if not math.isfinite(self.nominal_dt_s) or self.nominal_dt_s <= 0:
            raise ValueError("nominal_dt_s must be finite and positive")

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self), allow_nan=False, separators=(",", ":"), sort_keys=True
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class TelemetryFrame:
    """One observation sampled from a simulator or standalone environment."""

    contract_hash: str
    episode_id: str
    sequence: int
    monotonic_ns: int
    dt_s: float
    observation: tuple[float, ...]
    goal: tuple[float, ...]
    requested_action: tuple[float, ...] | None = None
    applied_action: tuple[float, ...] | None = None
    simulator_total_wrench: tuple[float, ...] | None = None
    aerodynamic_wrench: tuple[float, ...] | None = None
    vehicle_mass_kg: float | None = None
    vehicle_inertia_kg_m2: tuple[float, ...] | None = None
    terminated: bool = False
    truncated: bool = False

    @classmethod
    def create(
        cls,
        contract: FlightContract,
        *,
        episode_id: str,
        sequence: int,
        monotonic_ns: int,
        dt_s: float,
        observation: Sequence[float],
        goal: Sequence[float],
        requested_action: Sequence[float] | None = None,
        applied_action: Sequence[float] | None = None,
        simulator_total_wrench: Sequence[float] | None = None,
        aerodynamic_wrench: Sequence[float] | None = None,
        vehicle_mass_kg: float | None = None,
        vehicle_inertia_kg_m2: Sequence[float] | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> "TelemetryFrame":
        frame = cls(
            contract_hash=contract.digest,
            episode_id=episode_id,
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            dt_s=float(dt_s),
            observation=tuple(float(value) for value in observation),
            goal=tuple(float(value) for value in goal),
            requested_action=(
                tuple(float(value) for value in requested_action)
                if requested_action is not None
                else None
            ),
            applied_action=(
                tuple(float(value) for value in applied_action)
                if applied_action is not None
                else None
            ),
            simulator_total_wrench=(
                tuple(float(value) for value in simulator_total_wrench)
                if simulator_total_wrench is not None
                else None
            ),
            aerodynamic_wrench=(
                tuple(float(value) for value in aerodynamic_wrench)
                if aerodynamic_wrench is not None
                else None
            ),
            vehicle_mass_kg=(
                float(vehicle_mass_kg) if vehicle_mass_kg is not None else None
            ),
            vehicle_inertia_kg_m2=(
                tuple(float(value) for value in vehicle_inertia_kg_m2)
                if vehicle_inertia_kg_m2 is not None
                else None
            ),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        frame.validate(contract)
        return frame

    def validate(self, contract: FlightContract) -> None:
        if self.contract_hash != contract.digest:
            raise ValueError("telemetry contract hash does not match")
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.sequence < 0 or self.monotonic_ns < 0:
            raise ValueError("sequence and monotonic_ns must be non-negative")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if self.terminated and self.truncated:
            raise ValueError("a frame cannot be both terminated and truncated")
        contract.observation.validate(self.observation, "observation")
        contract.goal.validate(self.goal, "goal")
        if self.requested_action is not None:
            contract.action.validate(self.requested_action, "requested_action")
        for name, wrench in (
            ("applied_action", self.applied_action),
            ("simulator_total_wrench", self.simulator_total_wrench),
            ("aerodynamic_wrench", self.aerodynamic_wrench),
        ):
            if wrench is not None and (
                len(wrench) != contract.action.width
                or not all(math.isfinite(value) for value in wrench)
            ):
                raise ValueError(
                    f"{name} must contain {contract.action.width} finite values"
                )
        if self.vehicle_mass_kg is not None and (
            not math.isfinite(self.vehicle_mass_kg) or self.vehicle_mass_kg <= 0.0
        ):
            raise ValueError("vehicle_mass_kg must be finite and positive")
        if self.vehicle_inertia_kg_m2 is not None and (
            len(self.vehicle_inertia_kg_m2) != 3
            or not all(
                math.isfinite(value) and value > 0.0
                for value in self.vehicle_inertia_kg_m2
            )
        ):
            raise ValueError(
                "vehicle_inertia_kg_m2 must contain three positive finite values"
            )

    def to_wire(self) -> bytes:
        payload = {
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "telemetry",
            **asdict(self),
        }
        return (
            json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode()

    @classmethod
    def from_wire(cls, data: bytes, contract: FlightContract) -> "TelemetryFrame":
        payload = _wire_payload(data, "telemetry")
        frame = cls(
            **_select_fields(
                payload,
                cls.__dataclass_fields__,
                optional={
                    "requested_action",
                    "applied_action",
                    "simulator_total_wrench",
                    "aerodynamic_wrench",
                    "vehicle_mass_kg",
                    "vehicle_inertia_kg_m2",
                },
            )
        )
        frame.validate(contract)
        return frame


@dataclass(frozen=True)
class EnemyPose:
    """One actor-driven multiplayer aircraft pose in episode-relative NED."""

    slot: int
    position_ned_m: tuple[float, ...]
    quaternion_body_to_ned: tuple[float, ...]
    shield_requested: bool = False

    @classmethod
    def create(
        cls,
        *,
        slot: int,
        position_ned_m: Sequence[float],
        quaternion_body_to_ned: Sequence[float],
        shield_requested: bool = False,
    ) -> "EnemyPose":
        pose = cls(
            slot=slot,
            position_ned_m=_finite_vector(position_ned_m, 3, "enemy position"),
            quaternion_body_to_ned=_finite_vector(
                quaternion_body_to_ned, 4, "enemy quaternion"
            ),
            shield_requested=shield_requested,
        )
        pose.validate()
        return pose

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnemyPose":
        if not isinstance(value, Mapping):
            raise ValueError("enemy pose must be an object")
        required = {"slot", "position_ned_m", "quaternion_body_to_ned"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"enemy pose is missing {sorted(missing)}")
        unexpected = set(value) - required - {"shield_requested"}
        if unexpected:
            raise ValueError(f"enemy pose has unexpected fields {sorted(unexpected)}")
        return cls.create(
            slot=value["slot"],
            position_ned_m=value["position_ned_m"],
            quaternion_body_to_ned=value["quaternion_body_to_ned"],
            shield_requested=value.get("shield_requested", False),
        )

    def validate(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int):
            raise ValueError("enemy slot must be an integer")
        if self.slot < 0 or self.slot >= 19:
            raise ValueError("enemy slot must be between 0 and 18")
        if not isinstance(self.shield_requested, bool):
            raise ValueError("enemy shield request must be boolean")
        _finite_vector(self.position_ned_m, 3, "enemy position")
        quaternion = _finite_vector(self.quaternion_body_to_ned, 4, "enemy quaternion")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("enemy quaternion is not normalized")


@dataclass(frozen=True)
class EnemyShot:
    """One enemy FLUX shot aimed at an episode-relative NED point."""

    slot: int
    aim_position_ned_m: tuple[float, ...]

    @classmethod
    def create(
        cls,
        *,
        slot: int,
        aim_position_ned_m: Sequence[float],
    ) -> "EnemyShot":
        shot = cls(
            slot=slot,
            aim_position_ned_m=_finite_vector(
                aim_position_ned_m, 3, "enemy shot aim"
            ),
        )
        shot.validate()
        return shot

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnemyShot":
        payload = _strict_mapping(
            value,
            {"slot", "aim_position_ned_m"},
            "enemy shot",
        )
        return cls.create(
            slot=payload["slot"],
            aim_position_ned_m=payload["aim_position_ned_m"],
        )

    def validate(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int):
            raise ValueError("enemy shot slot must be an integer")
        if self.slot < 0 or self.slot >= 19:
            raise ValueError("enemy shot slot must be between 0 and 18")
        _finite_vector(self.aim_position_ned_m, 3, "enemy shot aim")


@dataclass(frozen=True)
class ControlCommand:
    """A bounded action tied to one unexpired telemetry frame."""

    contract_hash: str
    episode_id: str
    source_sequence: int
    issued_monotonic_ns: int
    valid_until_monotonic_ns: int
    action: tuple[float, ...]
    enemies: tuple[EnemyPose, ...] = ()
    enemy_shots: tuple[EnemyShot, ...] = ()

    @classmethod
    def create(
        cls,
        contract: FlightContract,
        *,
        episode_id: str,
        source_sequence: int,
        issued_monotonic_ns: int,
        valid_until_monotonic_ns: int,
        action: Sequence[float],
        enemies: Sequence[EnemyPose] = (),
        enemy_shots: Sequence[EnemyShot] = (),
    ) -> "ControlCommand":
        command = cls(
            contract_hash=contract.digest,
            episode_id=episode_id,
            source_sequence=source_sequence,
            issued_monotonic_ns=issued_monotonic_ns,
            valid_until_monotonic_ns=valid_until_monotonic_ns,
            action=tuple(float(value) for value in action),
            enemies=tuple(enemies),
            enemy_shots=tuple(enemy_shots),
        )
        command.validate(contract)
        return command

    def validate(
        self,
        contract: FlightContract,
        *,
        frame: TelemetryFrame | None = None,
        now_monotonic_ns: int | None = None,
    ) -> None:
        if self.contract_hash != contract.digest:
            raise ValueError("control contract hash does not match")
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.source_sequence < 0 or self.issued_monotonic_ns < 0:
            raise ValueError("sequence and timestamps must be non-negative")
        if self.valid_until_monotonic_ns <= self.issued_monotonic_ns:
            raise ValueError("control validity window must be positive")
        contract.action.validate(self.action, "action")
        slots = []
        for enemy in self.enemies:
            if not isinstance(enemy, EnemyPose):
                raise ValueError("enemies must contain EnemyPose values")
            enemy.validate()
            slots.append(enemy.slot)
        if slots != sorted(set(slots)):
            raise ValueError("enemy slots must be unique and increasing")
        shot_slots = []
        for shot in self.enemy_shots:
            if not isinstance(shot, EnemyShot):
                raise ValueError("enemy_shots must contain EnemyShot values")
            shot.validate()
            shot_slots.append(shot.slot)
        if shot_slots != sorted(set(shot_slots)):
            raise ValueError("enemy shot slots must be unique and increasing")
        if not set(shot_slots).issubset(slots):
            raise ValueError("every enemy shot must have a matching enemy pose")
        if frame is not None:
            frame.validate(contract)
            if self.episode_id != frame.episode_id:
                raise ValueError("control and telemetry episode IDs differ")
            if self.source_sequence != frame.sequence:
                raise ValueError("control does not answer the supplied telemetry frame")
            if self.issued_monotonic_ns < frame.monotonic_ns:
                raise ValueError("control was issued before its telemetry frame")
        if (
            now_monotonic_ns is not None
            and now_monotonic_ns > self.valid_until_monotonic_ns
        ):
            raise ValueError("control command has expired")

    def to_wire(self) -> bytes:
        payload = {
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "control",
            "contract_hash": self.contract_hash,
            "episode_id": self.episode_id,
            "source_sequence": self.source_sequence,
            "issued_monotonic_ns": self.issued_monotonic_ns,
            "valid_until_monotonic_ns": self.valid_until_monotonic_ns,
            "action": self.action,
        }
        if self.enemies:
            payload["enemies"] = [asdict(enemy) for enemy in self.enemies]
        if self.enemy_shots:
            payload["enemy_shots"] = [
                asdict(shot) for shot in self.enemy_shots
            ]
        return (
            json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode()

    @classmethod
    def from_wire(cls, data: bytes, contract: FlightContract) -> "ControlCommand":
        payload = _wire_payload(data, "control")
        selected = _select_fields(
            payload,
            cls.__dataclass_fields__,
            optional={"enemies", "enemy_shots"},
        )
        selected["enemies"] = tuple(
            EnemyPose.from_mapping(enemy) for enemy in selected.get("enemies", ())
        )
        selected["enemy_shots"] = tuple(
            EnemyShot.from_mapping(shot)
            for shot in selected.get("enemy_shots", ())
        )
        command = cls(**selected)
        command.validate(contract)
        return command


def _wire_payload(data: bytes, kind: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON wire message") from error
    if not isinstance(payload, dict):
        raise ValueError("wire message must be a JSON object")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("unknown flight protocol")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported flight protocol version")
    if payload.get("kind") != kind:
        raise ValueError(f"expected a {kind} message")
    return payload


def _select_fields(
    payload: Mapping[str, Any],
    fields: Mapping[str, Any],
    *,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    optional = optional or set()
    missing = set(fields) - optional - set(payload)
    if missing:
        raise ValueError(f"wire message is missing {sorted(missing)}")
    unexpected = (
        set(payload)
        - set(fields)
        - {
            "protocol",
            "protocol_version",
            "kind",
        }
    )
    if unexpected:
        raise ValueError(f"wire message has unexpected fields {sorted(unexpected)}")
    selected = {field: payload[field] for field in fields if field in payload}
    for field in (
        "observation",
        "goal",
        "action",
        "requested_action",
        "applied_action",
        "simulator_total_wrench",
        "aerodynamic_wrench",
        "vehicle_inertia_kg_m2",
    ):
        if field in selected and selected[field] is not None:
            selected[field] = tuple(selected[field])
    return selected


def _strict_mapping(
    value: Mapping[str, Any], expected: set[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"{name} is missing {sorted(missing)}")
    unexpected = set(value) - expected
    if unexpected:
        raise ValueError(f"{name} has unexpected fields {sorted(unexpected)}")
    return value
