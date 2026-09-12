"""Send explicitly selected UFO controls; this module cannot arm X-Plane."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Protocol, Sequence, TextIO

from rl.flight.contracts import ControlCommand, FlightContract, TelemetryFrame
from rl.flight.shadow_ufo import MAXIMUM_WIRE_BYTES, ShadowPolicy, connect_telemetry
from rl.flight.ufo_reference import UfoReferenceParameters, velocity_target_wrench
from rl.flight.ufo_swarm import RlUfoSwarm


class CommandSink(Protocol):
    def sendall(self, data: bytes) -> None: ...


ActionProvider = Callable[[TelemetryFrame], Sequence[float]]


@dataclass
class ControlStats:
    frames: int = 0
    sent: int = 0
    rejected_telemetry: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    interrupted: bool = False


@dataclass(frozen=True)
class LiveEnvelope:
    horizontal_position_m: float
    minimum_position_ned_z_m: float
    maximum_position_ned_z_m: float
    velocity_mps: float
    minimum_upright_cosine: float
    angular_rate_radps: float


LIVE_ENVELOPES = {
    "hover": LiveEnvelope(
        horizontal_position_m=20.0,
        minimum_position_ned_z_m=-120.0,
        maximum_position_ned_z_m=-80.0,
        velocity_mps=10.0,
        minimum_upright_cosine=0.9396926207859084,
        angular_rate_radps=1.0,
    ),
    "combat": LiveEnvelope(
        horizontal_position_m=20_000.0,
        minimum_position_ned_z_m=-5_000.0,
        maximum_position_ned_z_m=-25.0,
        velocity_mps=100.0,
        minimum_upright_cosine=0.9396926207859084,
        angular_rate_radps=1.5,
    ),
    "showcase": LiveEnvelope(
        horizontal_position_m=40_000.0,
        minimum_position_ned_z_m=-10_000.0,
        maximum_position_ned_z_m=-25.0,
        velocity_mps=130.0,
        minimum_upright_cosine=0.9396926207859084,
        angular_rate_radps=1.8,
    ),
}


def zero_action(contract: FlightContract) -> ActionProvider:
    action = tuple(0.0 for _ in range(contract.action.width))
    return lambda frame: action


def validate_live_envelope(
    frame: TelemetryFrame,
    envelope: LiveEnvelope = LIVE_ENVELOPES["hover"],
) -> None:
    position = frame.observation[:3]
    velocity = frame.observation[3:6]
    quaternion = frame.observation[6:10]
    angular_velocity = frame.observation[10:13]
    if max(abs(position[0]), abs(position[1])) > envelope.horizontal_position_m:
        raise ValueError("aircraft left the horizontal control envelope")
    if (
        position[2] < envelope.minimum_position_ned_z_m
        or position[2] > envelope.maximum_position_ned_z_m
    ):
        raise ValueError("aircraft left the vertical control envelope")
    if max(abs(value) for value in velocity) > envelope.velocity_mps:
        raise ValueError("aircraft left the velocity control envelope")
    quaternion_norm = sum(value * value for value in quaternion) ** 0.5
    if abs(quaternion_norm - 1.0) > 1e-3:
        raise ValueError("aircraft quaternion is not normalized")
    upright_cosine = 1.0 - 2.0 * (
        quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2]
    )
    if upright_cosine < envelope.minimum_upright_cosine:
        raise ValueError("aircraft tilt exceeds 20 degrees")
    if max(abs(value) for value in angular_velocity) > envelope.angular_rate_radps:
        raise ValueError("aircraft left the angular-rate control envelope")


def reference_action(
    contract: FlightContract,
    parameters: UfoReferenceParameters,
    envelope: LiveEnvelope = LIVE_ENVELOPES["hover"],
    *,
    max_acceleration_mps2: float = 8.0,
) -> ActionProvider:
    def provide(frame: TelemetryFrame) -> Sequence[float]:
        validate_live_envelope(frame, envelope)
        return velocity_target_wrench(
            frame.observation,
            frame.goal,
            contract.action.low,
            contract.action.high,
            parameters=parameters,
            max_acceleration_mps2=max_acceleration_mps2,
        )

    return provide


def actor_action(
    policy: ShadowPolicy,
    envelope: LiveEnvelope = LIVE_ENVELOPES["hover"],
) -> ActionProvider:
    def provide(frame: TelemetryFrame) -> Sequence[float]:
        validate_live_envelope(frame, envelope)
        action, _ = policy.infer(frame)
        return action

    return provide


def process_stream(
    source: BinaryIO,
    sink: CommandSink,
    output: TextIO,
    contract: FlightContract,
    action_provider: ActionProvider,
    *,
    validity_ns: int,
    max_frames: int | None = None,
    swarm: RlUfoSwarm | None = None,
) -> ControlStats:
    if validity_ns <= 0:
        raise ValueError("control validity must be positive")
    stats = ControlStats()
    last_episode = None
    last_sequence = None
    try:
        while max_frames is None or stats.frames < max_frames:
            line = source.readline(MAXIMUM_WIRE_BYTES + 1)
            if not line:
                break
            stats.frames += 1
            if len(line) > MAXIMUM_WIRE_BYTES or not line.endswith(b"\n"):
                stats.rejected_telemetry += 1
                continue
            try:
                frame = TelemetryFrame.from_wire(line, contract)
                if (
                    frame.episode_id == last_episode
                    and last_sequence is not None
                    and frame.sequence <= last_sequence
                ):
                    raise ValueError("telemetry sequence did not increase")
                action = contract.action.validate(
                    action_provider(frame), "control action"
                )
            except (TypeError, ValueError):
                stats.rejected_telemetry += 1
                continue
            enemies = swarm.update(frame) if swarm is not None else ()
            issued_ns = max(time.monotonic_ns(), frame.monotonic_ns)
            command = ControlCommand.create(
                contract,
                episode_id=frame.episode_id,
                source_sequence=frame.sequence,
                issued_monotonic_ns=issued_ns,
                valid_until_monotonic_ns=issued_ns + validity_ns,
                action=action,
                enemies=enemies,
            )
            sink.sendall(command.to_wire())
            output.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "control_sent",
                        "sent_monotonic_ns": time.monotonic_ns(),
                        "command": json.loads(command.to_wire()),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.flush()
            stats.sent += 1
            if stats.first_sequence is None:
                stats.first_sequence = frame.sequence
            stats.last_sequence = frame.sequence
            last_episode = frame.episode_id
            last_sequence = frame.sequence
    except KeyboardInterrupt:
        stats.interrupted = True
    return stats


def request_stop(signum, frame) -> None:
    del signum, frame
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("zero", "reference", "actor"), default="zero"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--swarm-size", type=int, default=0)
    parser.add_argument("--dynamics-library", type=Path)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument(
        "--reference-max-acceleration-mps2",
        type=float,
        default=8.0,
    )
    parser.add_argument("--envelope", choices=tuple(LIVE_ENVELOPES), default="hover")
    parser.add_argument("--allow-nonzero", action="store_true")
    parser.add_argument("--valid-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.valid_ms <= 0.0:
        parser.error("--valid-ms must be positive")
    if (
        not math.isfinite(args.reference_max_acceleration_mps2)
        or args.reference_max_acceleration_mps2 <= 0.0
    ):
        parser.error("--reference-max-acceleration-mps2 must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.swarm_size < 0 or args.swarm_size > 19:
        parser.error("--swarm-size must be between 0 and 19")
    contract = FlightContract.from_json(args.contract)
    envelope = LIVE_ENVELOPES[args.envelope]
    policy_identity = None
    swarm = None
    if args.mode == "zero":
        action_provider = zero_action(contract)
    elif not args.allow_nonzero:
        parser.error("nonzero control requires --allow-nonzero")
    elif args.mode == "reference":
        if args.mass_kg is None or args.mass_kg <= 0.0:
            parser.error("reference control requires a positive --mass-kg")
        action_provider = reference_action(
            contract,
            UfoReferenceParameters(mass_kg=args.mass_kg),
            envelope,
            max_acceleration_mps2=args.reference_max_acceleration_mps2,
        )
    else:
        if args.model is None:
            parser.error("actor control requires --model")
        policy = ShadowPolicy(contract, args.model)
        action_provider = actor_action(policy, envelope)
        policy_identity = {
            "checkpoint_sha256": policy.checkpoint_sha256,
            "model_sha256": policy.model_sha256,
        }
        if args.swarm_size:
            if args.dynamics_library is None:
                parser.error("actor swarm requires --dynamics-library")
            swarm = RlUfoSwarm(
                contract,
                args.model,
                args.dynamics_library,
                args.swarm_size,
            )
    if args.swarm_size and args.mode != "actor":
        parser.error("a swarm requires actor control mode")
    socket_path = args.socket or Path(f"/tmp/xplane-ufo-telemetry-{os.getuid()}.sock")
    summary_path = args.summary or args.output.with_suffix(
        args.output.suffix + ".summary.json"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with connect_telemetry(socket_path, args.connect_timeout) as client:
        with client.makefile("rb") as source, args.output.open("x") as output:
            stats = process_stream(
                source,
                client,
                output,
                contract,
                action_provider,
                validity_ns=round(args.valid_ms * 1_000_000),
                max_frames=args.max_frames,
                swarm=swarm,
            )
    summary = {
        "schema_version": 1,
        "mode": "external_control_sender",
        "socket": str(socket_path),
        "contract_hash": contract.digest,
        "valid_ms": args.valid_ms,
        "control_mode": args.mode,
        "envelope": args.envelope,
        "mass_kg": args.mass_kg,
        "reference_max_acceleration_mps2": (
            args.reference_max_acceleration_mps2 if args.mode == "reference" else None
        ),
        "policy": policy_identity,
        "swarm_size": args.swarm_size,
        **asdict(stats),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary_path.write_text(encoded)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if stats.sent else 2


if __name__ == "__main__":
    raise SystemExit(main())
