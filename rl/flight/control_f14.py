"""Drive the user-selected X-Plane F-14 through a guarded live pilot."""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.geometry import quaternion_from_euler
from rl.flight.shadow_ufo import ShadowPolicy
from rl.flight.xplane_udp import XPlaneUdp
from rl.flight.xplane_web import XPlaneRest, XPlaneWeb

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
MINIMUM_HANDOFF_SPEED_MPS = 90.0
MINIMUM_HANDOFF_AGL_M = 30.0
INITIAL_MINIMUM_SPEED_MPS = 90.0
INITIAL_MAXIMUM_SPEED_MPS = 500.0
INITIAL_MINIMUM_AGL_M = 50.0
INITIAL_LOW_ALTITUDE_M = 150.0
INITIAL_LOW_ALTITUDE_MINIMUM_CLIMB_MPS = -5.0
INITIAL_MAXIMUM_CLIMB_MPS = 120.0
INITIAL_MAXIMUM_ROLL_DEG = 120.0
INITIAL_MAXIMUM_PITCH_DEG = 70.0
INITIAL_MAXIMUM_RATE_RADPS = 2.0
MAXIMUM_LIVE_SPEED_MPS = 500.0
MAXIMUM_LIVE_CLIMB_MPS = 150.0
MAXIMUM_LIVE_ROLL_DEG = 120.0
MAXIMUM_LIVE_PITCH_DEG = 70.0
MAXIMUM_LIVE_RATE_RADPS = 2.5
PROTECTIVE_THROTTLE_SPEED_MPS = 170.0
MINIMUM_CADENCE_FRACTION = 0.8
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
    handoff_hz: float = 0.0
    transport_disconnects: int = 0
    interrupted: bool = False
    release_reason: str = ""


@dataclass
class F14LiveLimiter:
    """Blend into bounded, slew-limited live control surface commands."""

    surface_limit: float = 0.25
    throttle_slew_per_s: float = 0.35
    surface_slew_per_s: float = 0.8
    blend_seconds: float = 2.0
    previous: list[float] = field(
        default_factory=lambda: [0.9, 0.0, 0.0, 0.0]
    )
    started_ns: int | None = None
    previous_ns: int | None = None

    def apply(self, action: Sequence[float], now_ns: int) -> tuple[float, ...]:
        raw = tuple(float(value) for value in action)
        if len(raw) != 4 or not all(math.isfinite(value) for value in raw):
            raise ValueError(
                "F-14 actor action must contain four finite values"
            )
        if self.started_ns is None:
            self.started_ns = now_ns
            self.previous_ns = now_ns
        elapsed_s = max(0.0, (now_ns - self.started_ns) / 1e9)
        dt_s = min(0.25, max(0.0, (now_ns - self.previous_ns) / 1e9))
        blend = min(1.0, elapsed_s / self.blend_seconds)
        target = [
            min(1.0, max(0.0, raw[0])),
            *(
                min(
                    self.surface_limit,
                    max(-self.surface_limit, value * blend),
                )
                for value in raw[1:]
            ),
        ]
        limits = [self.throttle_slew_per_s, *([self.surface_slew_per_s] * 3)]
        for index, slew_per_s in enumerate(limits):
            maximum_delta = slew_per_s * dt_s
            delta = min(
                maximum_delta,
                max(-maximum_delta, target[index] - self.previous[index]),
            )
            self.previous[index] += delta
        self.previous_ns = now_ns
        return tuple(self.previous)


@dataclass
class F14LiveReference:
    """Conservative inner loop for collecting real F-14 demonstrations."""

    surface_limit: float = 0.20

    def action(
        self, values: dict[str, float], goal: Sequence[float]
    ) -> tuple[float, ...]:
        speed = math.sqrt(
            values["local_vx"] ** 2
            + values["local_vy"] ** 2
            + values["local_vz"] ** 2
        )
        desired_speed, desired_climb, desired_turn = goal
        roll = math.radians(values["roll_deg"])
        pitch = math.radians(values["pitch_deg"])
        desired_roll = clamp(
            math.atan2(speed * desired_turn, 9.80665),
            -math.radians(30.0),
            math.radians(30.0),
        )
        desired_pitch = clamp(
            0.012 * (desired_climb - values["local_vy"]),
            -math.radians(12.0),
            math.radians(12.0),
        )
        throttle = clamp(0.88 + 0.004 * (desired_speed - speed), 0.65, 1.0)
        if speed < PROTECTIVE_THROTTLE_SPEED_MPS:
            throttle = 1.0
        aileron = clamp(
            1.2 * (desired_roll - roll) - 0.9 * values["roll_rate"],
            -self.surface_limit,
            self.surface_limit,
        )
        elevator = clamp(
            1.8 * (desired_pitch - pitch) - 1.0 * values["pitch_rate"],
            -self.surface_limit,
            self.surface_limit,
        )
        rudder = clamp(
            1.5 * (desired_turn - values["yaw_rate"]),
            -0.10,
            0.10,
        )
        return throttle, aileron, elevator, rudder


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def showcase_goal(
    base_goal: Sequence[float], elapsed_s: float, height_agl_m: float = 1000.0
) -> tuple[float, ...]:
    """Build safe altitude, then fly a visible repeating maneuver."""

    airspeed = float(base_goal[0])
    if elapsed_s < 2.0:
        return airspeed, 0.0, 0.0
    phase_s = (elapsed_s - 2.0) % 32.0
    turn_sign = 1.0 if phase_s < 16.0 else -1.0
    if height_agl_m < 500.0:
        return airspeed, 25.0, 0.025 * turn_sign
    if phase_s < 8.0:
        return airspeed, 15.0, 0.040
    if phase_s < 16.0:
        return airspeed, 5.0, -0.050
    if phase_s < 24.0:
        return airspeed, -10.0, 0.040
    if phase_s < 30.0:
        return airspeed, 12.0, -0.040
    return airspeed, 0.0, 0.0


def f14_observation(
    values: dict[str, float], revision: int = 1
) -> tuple[float, ...]:
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
    if revision == 1:
        return tuple((*velocity_ned, *quaternion, *rates))
    if revision == 2:
        speed = math.sqrt(sum(value * value for value in velocity_ned))
        return (
            speed,
            values["local_vy"],
            math.radians(values["roll_deg"]),
            math.radians(values["pitch_deg"]),
            *rates,
        )
    raise ValueError(f"unsupported fighter contract revision {revision}")


def safe_handoff(values: dict[str, float], *, initial: bool = False) -> bool:
    observation = f14_observation(values)
    speed = float(np.linalg.norm(observation[:3]))
    base_safe = bool(
        values["paused"] < 0.5
        and values["on_ground"] < 0.5
        and values["height_agl"] >= MINIMUM_HANDOFF_AGL_M
        and speed >= MINIMUM_HANDOFF_SPEED_MPS
    )
    if not base_safe:
        return False
    climb_rate = values["local_vy"]
    if initial:
        return bool(
            INITIAL_MINIMUM_SPEED_MPS <= speed <= INITIAL_MAXIMUM_SPEED_MPS
            and values["height_agl"] >= INITIAL_MINIMUM_AGL_M
            and (
                values["height_agl"] >= INITIAL_LOW_ALTITUDE_M
                or climb_rate >= INITIAL_LOW_ALTITUDE_MINIMUM_CLIMB_MPS
            )
            and abs(climb_rate) <= INITIAL_MAXIMUM_CLIMB_MPS
            and abs(values["roll_deg"]) <= INITIAL_MAXIMUM_ROLL_DEG
            and abs(values["pitch_deg"]) <= INITIAL_MAXIMUM_PITCH_DEG
            and max(abs(value) for value in observation[7:])
            <= INITIAL_MAXIMUM_RATE_RADPS
        )
    return bool(
        speed <= MAXIMUM_LIVE_SPEED_MPS
        and abs(climb_rate) <= MAXIMUM_LIVE_CLIMB_MPS
        and abs(values["roll_deg"]) <= MAXIMUM_LIVE_ROLL_DEG
        and abs(values["pitch_deg"]) <= MAXIMUM_LIVE_PITCH_DEG
        and max(abs(value) for value in observation[7:])
        <= MAXIMUM_LIVE_RATE_RADPS
    )


def sample_rate_hz(sample_times_ns: Sequence[int]) -> float:
    if len(sample_times_ns) < 2:
        return 0.0
    duration_s = (sample_times_ns[-1] - sample_times_ns[0]) / 1e9
    if duration_s <= 0.0:
        return 0.0
    return (len(sample_times_ns) - 1) / duration_s


def write_overrides(client: XPlaneUdp, enabled: bool) -> None:
    value = 1 if enabled else 0
    client.write_many(
        (
            (OVERRIDE_JOYSTICK, value),
            (OVERRIDE_THROTTLES, value),
        )
    )


def write_action(client: XPlaneUdp, action: Sequence[float]) -> None:
    throttle, aileron, elevator, rudder = action
    client.write_many(
        (
            (THROTTLE_0, throttle),
            (THROTTLE_1, throttle),
            (AILERON, aileron),
            (ELEVATOR, elevator),
            (RUDDER, rudder),
        )
    )


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
    showcase: bool = False,
) -> F14ControlStats:
    goal = contract.goal.validate(goal, "F-14 goal")
    stats = F14ControlStats()
    sequence = 0
    armed = False
    first_command_ns = None
    captured_goal = None
    next_command_ns = 0
    next_wait_report_ns = 0
    safe_samples = 0
    safe_sample_times: deque[int] = deque(maxlen=SAFE_HANDOFF_SAMPLES)
    limiter = F14LiveLimiter()
    live_reference = F14LiveReference()
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
                values = client.receive(max(0.2, contract.nominal_dt_s * 2.0))
            except ConnectionError:
                stats.transport_disconnects += 1
                safe_samples = 0
                safe_sample_times.clear()
                if armed or not isinstance(client, XPlaneWeb):
                    raise
                client.close()
                if time.monotonic() >= wait_deadline:
                    raise
                time.sleep(0.25)
                client.subscribe(
                    DATAREFS, round(1.0 / contract.nominal_dt_s)
                )
                continue
            except socket.timeout:
                stats.stale_samples += 1
                safe_samples = 0
                if armed:
                    write_overrides(client, False)
                    armed = False
                    limiter = F14LiveLimiter()
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
                safe_sample_times.clear()
                if armed:
                    write_overrides(client, False)
                    armed = False
                    limiter = F14LiveLimiter()
                continue
            if not safe_handoff(values, initial=not armed):
                stats.unsafe_samples += 1
                safe_samples = 0
                safe_sample_times.clear()
                if armed:
                    write_overrides(client, False)
                    armed = False
                    speed = math.sqrt(
                        values["local_vx"] ** 2
                        + values["local_vy"] ** 2
                        + values["local_vz"] ** 2
                    )
                    stats.release_reason = (
                        "guarded envelope exit: "
                        f"speed={speed:.1f}m/s "
                        f"agl={values['height_agl']:.0f}m "
                        f"climb={values['local_vy']:.1f}m/s "
                        f"roll={values['roll_deg']:.1f}deg "
                        f"pitch={values['pitch_deg']:.1f}deg"
                    )
                    print(stats.release_reason, flush=True)
                    break
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError(
                        "F-14 never entered the broad airborne handoff envelope"
                    )
                now_ns = time.monotonic_ns()
                if now_ns >= next_wait_report_ns:
                    speed = math.sqrt(
                        values["local_vx"] ** 2
                        + values["local_vy"] ** 2
                        + values["local_vz"] ** 2
                    )
                    print(
                        "handoff wait: "
                        f"speed={speed:.1f}m/s "
                        f"agl={values['height_agl']:.0f}m "
                        f"climb={values['local_vy']:.1f}m/s "
                        f"roll={values['roll_deg']:.1f}deg "
                        f"pitch={values['pitch_deg']:.1f}deg",
                        flush=True,
                    )
                    next_wait_report_ns = now_ns + 2_000_000_000
                continue
            safe_samples += 1
            now_ns = time.monotonic_ns()
            if not armed:
                safe_sample_times.append(now_ns)
                if safe_samples < SAFE_HANDOFF_SAMPLES:
                    continue
                stats.handoff_hz = sample_rate_hz(tuple(safe_sample_times))
                required_hz = MINIMUM_CADENCE_FRACTION / contract.nominal_dt_s
                if stats.handoff_hz < required_hz:
                    raise RuntimeError(
                        "F-14 telemetry cadence "
                        f"{stats.handoff_hz:.2f} Hz is below "
                        f"{required_hz:.2f} Hz"
                    )
            if now_ns < next_command_ns:
                continue
            observation = contract.observation.validate(
                f14_observation(values, contract.revision), "F-14 observation"
            )
            if captured_goal is None:
                if showcase:
                    captured_goal = (
                        clamp(observation[0], goal[0], contract.goal.high[0]),
                        0.0,
                        0.0,
                    )
                else:
                    captured_goal = goal
            elapsed_s = 0.0 if first_command_ns is None else (
                now_ns - first_command_ns
            ) / 1e9
            if showcase:
                applied_goal = showcase_goal(
                    captured_goal, elapsed_s, values["height_agl"]
                )
            else:
                applied_goal = captured_goal
            frame = TelemetryFrame.create(
                contract,
                episode_id="xplane-f14-player",
                sequence=sequence,
                monotonic_ns=now_ns,
                dt_s=contract.nominal_dt_s,
                observation=observation,
                goal=applied_goal,
            )
            actor_action, latency_ns = policy.infer(frame)
            reference_action = live_reference.action(values, applied_goal)
            action = limiter.apply(reference_action, now_ns)
            if not armed:
                write_overrides(client, True)
                armed = True
                stats.armed_transitions += 1
                print(
                    "F-14 live pilot armed: "
                    f"speed={observation[0]:.1f}m/s "
                    f"goal={captured_goal[0]:.1f}m/s; "
                    "two-second stabilization started",
                    flush=True,
                )
            write_action(client, action)
            record = {
                "schema_version": 1,
                "kind": "xplane_f14_control",
                "monotonic_ns": now_ns,
                "sequence": sequence,
                "observation": observation,
                "goal": applied_goal,
                "action": action,
                "actor_action": actor_action,
                "reference_action": reference_action,
                "controller": "live_reference_actor_shadow",
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
            try:
                write_overrides(client, False)
            except Exception:
                pass
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--transport", choices=("rest", "web", "udp"), default="rest"
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--airspeed", type=float, default=180.0)
    parser.add_argument("--climb-rate", type=float, default=0.0)
    parser.add_argument("--turn-rate", type=float, default=0.0)
    parser.add_argument("--wait-seconds", type=float, default=300.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--showcase", action="store_true")
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
    port = args.port or (49000 if args.transport == "udp" else 8086)
    client_type = {
        "rest": XPlaneRest,
        "web": XPlaneWeb,
        "udp": XPlaneUdp,
    }[args.transport]
    with client_type(args.host, port) as client:
        with args.output.open("x", encoding="utf-8") as output:
            stats = run_controller(
                client,
                contract,
                policy,
                goal,
                output,
                wait_seconds=args.wait_seconds,
                duration_seconds=args.duration_seconds,
                showcase=args.showcase,
            )
    summary = {
        "schema_version": 1,
        "mode": "xplane_f14_player",
        "transport": args.transport,
        "host": args.host,
        "port": port,
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
