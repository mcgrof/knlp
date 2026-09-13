"""Fly configured X-Plane AI F-14s around a player-selected F-14."""

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

from rl.flight.contracts import EnemyPose, FlightContract
from rl.flight.control_f14 import DATAREFS, safe_handoff
from rl.flight.fighter_swarm import DEFAULT_CONTRACT, RlF14Swarm
from rl.flight.geometry import quaternion_from_euler, roll_pitch_yaw
from rl.flight.xplane_web import XPlaneWeb

FORMATION_DATAREFS = {
    **DATAREFS,
    "local_x": "sim/flightmodel/position/local_x",
    "local_y": "sim/flightmodel/position/local_y",
    "local_z": "sim/flightmodel/position/local_z",
}
OVERRIDE_PLANEPATH = "sim/operation/override/override_planepath"
MAXIMUM_SAMPLE_AGE_S = 0.25


@dataclass
class F14FormationStats:
    samples: int = 0
    frames: int = 0
    pauses: int = 0
    stale_samples: int = 0
    unsafe_samples: int = 0
    armed_transitions: int = 0
    override_releases: int = 0
    transport_disconnects: int = 0
    maximum_leader_speed_mps: float = 0.0
    interrupted: bool = False


def f14_player_state(values: dict[str, float]) -> np.ndarray:
    """Convert X-Plane local coordinates into a full NED fighter state."""

    missing = set(FORMATION_DATAREFS) - set(values)
    if missing:
        raise ValueError(f"missing X-Plane formation samples: {sorted(missing)}")
    sample = np.asarray(
        [values[name] for name in FORMATION_DATAREFS], dtype=np.float64
    )
    if not np.isfinite(sample).all():
        raise ValueError("X-Plane formation sample contains a non-finite value")
    return np.asarray(
        (
            -values["local_z"],
            values["local_x"],
            -values["local_y"],
            -values["local_vz"],
            values["local_vx"],
            -values["local_vy"],
            *quaternion_from_euler(
                math.radians(values["roll_deg"]),
                math.radians(values["pitch_deg"]),
                math.radians(values["heading_deg"]),
            ),
            values["roll_rate"],
            values["pitch_rate"],
            values["yaw_rate"],
        ),
        dtype=np.float64,
    )


def formation_pose_writes(
    poses: Sequence[EnemyPose],
) -> tuple[tuple[str, float], ...]:
    """Map episode-local NED poses to X-Plane multiplayer datarefs."""

    writes = []
    for pose in poses:
        index = pose.slot + 1
        north, east, down = pose.position_ned_m
        roll, pitch, yaw = roll_pitch_yaw(pose.quaternion_body_to_ned)
        prefix = f"sim/multiplayer/position/plane{index}"
        writes.extend(
            (
                (f"{prefix}_x", east),
                (f"{prefix}_y", -down),
                (f"{prefix}_z", -north),
                (f"{prefix}_phi", math.degrees(roll)),
                (f"{prefix}_the", math.degrees(pitch)),
                (f"{prefix}_psi", math.degrees(yaw) % 360.0),
            )
        )
    return tuple(writes)


def write_formation_overrides(client: XPlaneWeb, size: int, enabled: bool) -> None:
    value = 1.0 if enabled else 0.0
    client.write_many(
        tuple(
            (f"{OVERRIDE_PLANEPATH}[{index}]", value)
            for index in range(1, size + 1)
        )
    )


def _request_stop(signum, frame) -> None:
    del signum, frame
    raise KeyboardInterrupt


def run_formation(
    client: XPlaneWeb,
    swarm: RlF14Swarm,
    output,
    *,
    wait_seconds: float,
    duration_seconds: float | None,
) -> F14FormationStats:
    """Drive AI aircraft paths while leaving the player controls untouched."""

    stats = F14FormationStats()
    armed = False
    sequence = 0
    generation = 0
    episode_id = ""
    first_frame_ns = None
    previous_frame_ns = None
    wait_deadline = time.monotonic() + wait_seconds
    client.subscribe(FORMATION_DATAREFS, 20)

    def release() -> None:
        nonlocal armed
        if not armed:
            return
        try:
            write_formation_overrides(client, swarm.size, False)
        except Exception:
            client.close()
            write_formation_overrides(client, swarm.size, False)
        armed = False
        stats.override_releases += 1

    try:
        while True:
            if (
                duration_seconds is not None
                and first_frame_ns is not None
                and time.monotonic_ns() - first_frame_ns
                >= round(duration_seconds * 1e9)
            ):
                break
            try:
                values = client.receive(0.3)
            except ConnectionError:
                stats.transport_disconnects += 1
                break
            except socket.timeout:
                stats.stale_samples += 1
                release()
                previous_frame_ns = None
                if first_frame_ns is None and time.monotonic() >= wait_deadline:
                    raise TimeoutError("timed out waiting for an airborne F-14")
                continue
            stats.samples += 1
            if not client.fresh(MAXIMUM_SAMPLE_AGE_S):
                stats.stale_samples += 1
                release()
                previous_frame_ns = None
                continue
            if values["paused"] >= 0.5:
                stats.pauses += 1
                release()
                previous_frame_ns = None
                continue
            if not safe_handoff(values, initial=True):
                stats.unsafe_samples += 1
                release()
                previous_frame_ns = None
                if first_frame_ns is None and time.monotonic() >= wait_deadline:
                    raise TimeoutError(
                        "F-14 never entered the airborne formation envelope"
                    )
                continue

            now_ns = time.monotonic_ns()
            if not armed:
                generation += 1
                episode_id = f"xplane-f14-formation-{now_ns}-{generation}"
                previous_frame_ns = None
            dt_s = (
                swarm.fighter_contract.nominal_dt_s
                if previous_frame_ns is None
                else min(0.1, max(0.001, (now_ns - previous_frame_ns) / 1e9))
            )
            player_state = f14_player_state(values)
            leader_speed_mps = float(np.linalg.norm(player_state[3:6]))
            stats.maximum_leader_speed_mps = max(
                stats.maximum_leader_speed_mps, leader_speed_mps
            )
            poses = swarm.update_state(
                player_state,
                episode_id=episode_id,
                sequence=sequence,
                monotonic_ns=now_ns,
                dt_s=dt_s,
            )
            if not armed:
                write_formation_overrides(client, swarm.size, True)
                armed = True
                stats.armed_transitions += 1
                print(
                    f"F-14 formation armed with {swarm.size} followers; "
                    "player controls remain manual",
                    flush=True,
                )
            client.write_many(formation_pose_writes(poses))
            output.write(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "xplane_f14_formation",
                        "monotonic_ns": now_ns,
                        "sequence": sequence,
                        "player_state_ned": player_state.tolist(),
                        "player_speed_mps": leader_speed_mps,
                        "followers": [asdict(pose) for pose in poses],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.flush()
            stats.frames += 1
            sequence += 1
            previous_frame_ns = now_ns
            if first_frame_ns is None:
                first_frame_ns = now_ns
    except KeyboardInterrupt:
        stats.interrupted = True
    finally:
        try:
            release()
        except Exception:
            pass
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--wait-seconds", type=float, default=300.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.size < 1 or args.size > 19:
        parser.error("--size must be between 1 and 19")
    if not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0.0:
        parser.error("--wait-seconds must be positive")
    if args.duration_seconds is not None and (
        not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0.0
    ):
        parser.error("--duration-seconds must be positive")

    contract = FlightContract.from_json(args.contract)
    swarm = RlF14Swarm(contract, args.model, args.size, args.contract)
    summary_path = args.summary or args.output.with_suffix(
        args.output.suffix + ".summary.json"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    with XPlaneWeb(args.host, args.port) as client:
        with args.output.open("x", encoding="utf-8") as output:
            stats = run_formation(
                client,
                swarm,
                output,
                wait_seconds=args.wait_seconds,
                duration_seconds=args.duration_seconds,
            )
    summary = {
        "schema_version": 1,
        "mode": "xplane_f14_formation",
        "host": args.host,
        "port": args.port,
        "size": args.size,
        "contract_hash": contract.digest,
        "model_sha256": swarm.model_sha256,
        "checkpoint_sha256": swarm.checkpoint_sha256,
        **asdict(stats),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if stats.frames else 2


if __name__ == "__main__":
    raise SystemExit(main())
