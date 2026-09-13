"""Evaluate learned F-14 followers against sustained formation turns."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from rl.flight.contracts import FlightContract
from rl.flight.fighter_swarm import DEFAULT_CONTRACT, RlF14Swarm
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(path), "rev-parse", "HEAD"), text=True
    ).strip()


def parse_turn_rates(value: str) -> tuple[float, ...]:
    try:
        rates = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("turn rates must be numbers") from error
    if not rates or not all(math.isfinite(rate) for rate in rates):
        raise argparse.ArgumentTypeError("turn rates must be finite")
    if any(abs(rate) > 0.08 for rate in rates):
        raise argparse.ArgumentTypeError("turn rates must be within 0.08 rad/s")
    return rates


def parse_speeds(value: str) -> tuple[float, ...]:
    try:
        speeds = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("speeds must be numbers") from error
    if not speeds or not all(math.isfinite(speed) for speed in speeds):
        raise argparse.ArgumentTypeError("speeds must be finite")
    if any(speed <= 0.0 for speed in speeds):
        raise argparse.ArgumentTypeError("speeds must be positive")
    return speeds


def evaluate_turn(
    swarm: RlF14Swarm,
    turn_rate: float,
    *,
    seconds: float,
    settle_seconds: float,
    speed_mps: float = 180.0,
) -> dict:
    dt_s = swarm.fighter_contract.nominal_dt_s
    steps = math.ceil(seconds / dt_s)
    settle_steps = math.ceil(settle_seconds / dt_s)
    player = np.zeros(13, dtype=np.float64)
    player[2] = -1500.0
    player[3] = speed_mps
    player[6] = 1.0
    errors = []
    minimum_separation = math.inf
    episode_id = f"sustained-turn-{turn_rate:+.5f}"
    for sequence in range(steps):
        yaw = turn_rate * sequence * dt_s
        player[6:10] = quaternion_from_euler(0.0, 0.0, yaw)
        rotation = quaternion_body_to_ned(player[6:10])
        player[3:6] = rotation[:, 0] * speed_mps
        if sequence:
            player[:3] += player[3:6] * dt_s
        poses = swarm.update_state(
            player,
            episode_id=episode_id,
            sequence=sequence,
            monotonic_ns=round(sequence * dt_s * 1e9),
            dt_s=dt_s,
        )
        positions = [np.asarray(pose.position_ned_m) for pose in poses]
        if len(positions) > 1:
            minimum_separation = min(
                minimum_separation,
                *(
                    float(np.linalg.norm(first - second))
                    for index, first in enumerate(positions)
                    for second in positions[index + 1 :]
                ),
            )
        if sequence >= settle_steps:
            heading_rotation = np.asarray(
                (
                    (math.cos(yaw), -math.sin(yaw), 0.0),
                    (math.sin(yaw), math.cos(yaw), 0.0),
                    (0.0, 0.0, 1.0),
                )
            )
            errors.extend(
                float(
                    np.linalg.norm(
                        player[:3]
                        + heading_rotation @ swarm._offset(slot)
                        - position
                    )
                )
                for slot, position in enumerate(positions)
            )
    values = np.asarray(errors)
    return {
        "speed_mps": speed_mps,
        "turn_rate_radps": turn_rate,
        "samples": len(errors),
        "slot_error_rmse_m": float(np.sqrt(np.mean(values * values))),
        "slot_error_p95_m": float(np.quantile(values, 0.95)),
        "slot_error_max_m": float(np.max(values)),
        "minimum_pair_separation_m": (
            minimum_separation if math.isfinite(minimum_separation) else None
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument(
        "--turn-rates", type=parse_turn_rates, default=(-0.04, 0.04)
    )
    parser.add_argument(
        "--speeds", type=parse_speeds
    )
    parser.add_argument("--maximum-p95-error-m", type=float, default=80.0)
    parser.add_argument("--minimum-separation-m", type=float, default=80.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--knlp-commit")
    args = parser.parse_args(argv)
    if args.size < 1 or args.size > 19:
        parser.error("--size must be between 1 and 19")
    if args.seconds <= 0.0 or args.settle_seconds < 0.0:
        parser.error("evaluation times must be non-negative")
    if args.settle_seconds >= args.seconds:
        parser.error("settle time must be shorter than evaluation time")
    if args.maximum_p95_error_m <= 0.0 or args.minimum_separation_m <= 0.0:
        parser.error("formation thresholds must be positive")

    contract = FlightContract.from_json(args.contract)
    speeds = args.speeds or (
        (180.0, 400.0, 700.0) if contract.revision == 3 else (180.0,)
    )
    if any(
        speed < contract.goal.low[0] or speed > contract.goal.high[0]
        for speed in speeds
    ):
        parser.error("--speeds must be within the contract airspeed bounds")
    swarm = RlF14Swarm(contract, args.model, args.size, args.contract)
    scenarios = [
        evaluate_turn(
            swarm,
            rate,
            seconds=args.seconds,
            settle_seconds=args.settle_seconds,
            speed_mps=speed,
        )
        for speed in speeds
        for rate in args.turn_rates
    ]
    passed = all(
        scenario["slot_error_p95_m"] <= args.maximum_p95_error_m
        and scenario["minimum_pair_separation_m"] >= args.minimum_separation_m
        for scenario in scenarios
    )
    report = {
        "schema_version": 1,
        "knlp_commit": args.knlp_commit
        or _git_head(Path(__file__).resolve().parents[1]),
        "environment": "f14:sustained-formation-turns",
        "dynamics_fidelity": "generic fixed-wing training surrogate",
        "size": args.size,
        "seconds": args.seconds,
        "settle_seconds": args.settle_seconds,
        "model_sha256": swarm.model_sha256,
        "checkpoint_sha256": swarm.checkpoint_sha256,
        "maximum_p95_error_m": args.maximum_p95_error_m,
        "minimum_separation_m": args.minimum_separation_m,
        "machine_gate_passed": passed,
        "scenarios": scenarios,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    else:
        print(encoded, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
