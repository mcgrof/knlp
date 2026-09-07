"""Send explicitly selected UFO controls; this module cannot arm X-Plane."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Protocol, Sequence, TextIO

from rl.flight.contracts import ControlCommand, FlightContract, TelemetryFrame
from rl.flight.shadow_ufo import MAXIMUM_WIRE_BYTES, connect_telemetry


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


def zero_action(contract: FlightContract) -> ActionProvider:
    action = tuple(0.0 for _ in range(contract.action.width))
    return lambda frame: action


def process_stream(
    source: BinaryIO,
    sink: CommandSink,
    output: TextIO,
    contract: FlightContract,
    action_provider: ActionProvider,
    *,
    validity_ns: int,
    max_frames: int | None = None,
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
            issued_ns = max(time.monotonic_ns(), frame.monotonic_ns)
            command = ControlCommand.create(
                contract,
                episode_id=frame.episode_id,
                source_sequence=frame.sequence,
                issued_monotonic_ns=issued_ns,
                valid_until_monotonic_ns=issued_ns + validity_ns,
                action=action,
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
    parser.add_argument("--mode", choices=("zero",), default="zero")
    parser.add_argument("--valid-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.valid_ms <= 0.0:
        parser.error("--valid-ms must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    contract = FlightContract.from_json(args.contract)
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
                zero_action(contract),
                validity_ns=round(args.valid_ms * 1_000_000),
                max_frames=args.max_frames,
            )
    summary = {
        "schema_version": 1,
        "mode": "external_zero_control",
        "socket": str(socket_path),
        "contract_hash": contract.digest,
        "valid_ms": args.valid_ms,
        **asdict(stats),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary_path.write_text(encoded)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if stats.sent else 2


if __name__ == "__main__":
    raise SystemExit(main())
