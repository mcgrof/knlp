"""Drive the user-selected X-Plane F-14 with a learned motor actor."""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.geometry import quaternion_from_euler
from rl.flight.shadow_ufo import ShadowPolicy
from rl.flight.xplane_udp import XPlaneUdp

DATAREFS = {
    "local_vx": "sim/flightmodel/position/local_vx",
    "local_vy": "sim/flightmodel/position/local_vy",
    "local_vz": "sim/flightmodel/position/local_vz",
    "roll_deg": "sim/flightmodel/position/phi",
    "pitch_deg": "sim/flightmodel/position/theta",
    "heading_deg": "sim/flightmodel/position/psi",
    "roll_rate": "sim/flightmodel/position/Prad",
    "pitch_rate": "sim/flightmodel/position/Qrad",
    "yaw_rate": "sim/flightmodel/position/Rrad",
    "height_agl": "sim/flightmodel/position/y_agl",
    "on_ground": "sim/flightmodel/failures/onground_any",
    "paused": "sim/time/paused",
}
OVERRIDE_JOYSTICK = "sim/operation/override/override_joystick"
OVERRIDE_THROTTLES = "sim/operation/override/override_throttles"
AILERON = "sim/joystick/yoke_roll_ratio"
ELEVATOR = "sim/joystick/yoke_pitch_ratio"
RUDDER = "sim/joystick/yoke_heading_ratio"
THROTTLE_0 = "sim/flightmodel/engine/ENGN_thro_use[0]"
THROTTLE_1 = "sim/flightmodel/engine/ENGN_thro_use[1]"
MINIMUM_HANDOFF_SPEED_MPS = 100.0
MINIMUM_HANDOFF_AGL_M = 150.0
MAXIMUM_SAMPLE_AGE_S = 0.25
SAFE_HANDOFF_SAMPLES = 10


@dataclass
class F14ControlStats:
    samples: int = 0
    commands: int = 0
    pauses: int = 0
    stale_samples: int = 0
    unsafe_samples: int = 0
    armed_transitions: int = 0
    interrupted: bool = False


def f14_observation(values: dict[str, float]) -> tuple[float, ...]:
    """Convert X-Plane local coordinates and Euler angles to fighter ABI."""

    missing = set(DATAREFS) - set(values)
    if missing:
        raise ValueError(f"missing X-Plane samples: {sorted(missing)}")
    sample = np.asarray([values[name] for name in DATAREFS], dtype=np.float64)
    if not np.isfinite(sample).all():
        raise ValueError("X-Plane sample contains a non-finite value")
    velocity_ned = (
        -values["local_vz"],
        values["local_vx"],
        -values["local_vy"],
    )
    quaternion = quaternion_from_euler(
        math.radians(values["roll_deg"]),
        math.radians(values["pitch_deg"]),
        math.radians(values["heading_deg"]),
    )
    rates = (
        values["roll_rate"],
        values["pitch_rate"],
        values["yaw_rate"],
    )
    return tuple((*velocity_ned, *quaternion, *rates))


def safe_handoff(values: dict[str, float]) -> bool:
    observation = f14_observation(values)
    speed = float(np.linalg.norm(observation[:3]))
    return bool(
        values["paused"] < 0.5
        and values["on_ground"] < 0.5
        and values["height_agl"] >= MINIMUM_HANDOFF_AGL_M
        and speed >= MINIMUM_HANDOFF_SPEED_MPS
    )


def write_overrides(client: XPlaneUdp, enabled: bool) -> None:
    value = 1.0 if enabled else 0.0
    client.write(OVERRIDE_JOYSTICK, value)
    client.write(OVERRIDE_THROTTLES, value)


def write_action(client: XPlaneUdp, action: Sequence[float]) -> None:
    throttle, aileron, elevator, rudder = action
    client.write(THROTTLE_0, throttle)
    client.write(THROTTLE_1, throttle)
    client.write(AILERON, aileron)
    client.write(ELEVATOR, elevator)
    client.write(RUDDER, rudder)


def _request_stop(signum, frame) -> None:
    del signum, frame
    raise KeyboardInterrupt


def run_controller(
    client: XPlaneUdp,
    contract: FlightContract,
    policy: ShadowPolicy,
    goal: Sequence[float],
    output,
    *,
    wait_seconds: float,
    duration_seconds: float | None,
) -> F14ControlStats:
    goal = contract.goal.validate(goal, "F-14 goal")
    stats = F14ControlStats()
    sequence = 0
    armed = False
    first_command_ns = None
    next_command_ns = 0
    safe_samples = 0
    wait_deadline = time.monotonic() + wait_seconds
    client.subscribe(DATAREFS, round(1.0 / contract.nominal_dt_s))
    try:
        while True:
            if (
                duration_seconds is not None
                and first_command_ns is not None
                and time.monotonic_ns() - first_command_ns
                >= round(duration_seconds * 1e9)
            ):
                break
            try:
                values = client.receive(min(0.2, contract.nominal_dt_s * 2.0))
            except socket.timeout:
                stats.stale_samples += 1
                safe_samples = 0
                if armed:
                    write_overrides(client, False)
                    armed = False
                if first_command_ns is None and time.monotonic() >= wait_deadline:
                    raise TimeoutError("timed out waiting for airborne F-14 data")
                continue
            stats.samples += 1
            if not client.fresh(MAXIMUM_SAMPLE_AGE_S):
                stats.stale_samples += 1
                safe_samples = 0
                continue
            if values["paused"] >= 0.5:
                stats.pauses += 1
                safe_samples = 0
                if armed:
                    write_overrides(client, False)
                    armed = False
                continue
            if not safe_handoff(values):
                stats.unsafe_samples += 1
                safe_samples = 0
                if armed:
                    write_overrides(client, False)
                    armed = False
                    raise RuntimeError("F-14 left the live actor handoff envelope")
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError(
                        "F-14 did not become airborne above 100 m/s and 150 m AGL"
                    )
                continue
            safe_samples += 1
            if not armed and safe_samples < SAFE_HANDOFF_SAMPLES:
                continue
            now_ns = time.monotonic_ns()
            if now_ns < next_command_ns:
                continue
            observation = contract.observation.validate(
                f14_observation(values), "F-14 observation"
            )
            frame = TelemetryFrame.create(
                contract,
                episode_id="xplane-f14-player",
                sequence=sequence,
                monotonic_ns=now_ns,
                dt_s=contract.nominal_dt_s,
                observation=observation,
                goal=goal,
            )
            action, latency_ns = policy.infer(frame)
            if not armed:
                write_overrides(client, True)
                armed = True
                stats.armed_transitions += 1
            write_action(client, action)
            record = {
                "schema_version": 1,
                "kind": "xplane_f14_control",
                "monotonic_ns": now_ns,
                "sequence": sequence,
                "observation": observation,
                "goal": goal,
                "action": action,
                "inference_latency_ns": latency_ns,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            stats.commands += 1
            sequence += 1
            next_command_ns = now_ns + round(contract.nominal_dt_s * 1e9)
            if first_command_ns is None:
                first_command_ns = now_ns
    except KeyboardInterrupt:
        stats.interrupted = True
    finally:
        if armed:
            write_overrides(client, False)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=49000)
    parser.add_argument("--airspeed", type=float, default=180.0)
    parser.add_argument("--climb-rate", type=float, default=0.0)
    parser.add_argument("--turn-rate", type=float, default=0.0)
    parser.add_argument("--wait-seconds", type=float, default=300.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0.0:
        parser.error("--wait-seconds must be positive")
    if args.duration_seconds is not None and (
        not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0.0
    ):
        parser.error("--duration-seconds must be positive")
    contract = FlightContract.from_json(args.contract)
    goal = contract.goal.validate(
        (args.airspeed, args.climb_rate, args.turn_rate), "F-14 goal"
    )
    policy = ShadowPolicy(contract, args.model)
    summary_path = args.summary or args.output.with_suffix(
        args.output.suffix + ".summary.json"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    with XPlaneUdp(args.host, args.port) as client:
        with args.output.open("x", encoding="utf-8") as output:
            stats = run_controller(
                client,
                contract,
                policy,
                goal,
                output,
                wait_seconds=args.wait_seconds,
                duration_seconds=args.duration_seconds,
            )
    summary = {
        "schema_version": 1,
        "mode": "xplane_f14_player",
        "host": args.host,
        "port": args.port,
        "contract_hash": contract.digest,
        "model_sha256": policy.model_sha256,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "goal": goal,
        **asdict(stats),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if stats.commands else 2


if __name__ == "__main__":
    raise SystemExit(main())
