"""Summarize an X-Plane read-only shadow trace and its transfer quality."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

from rl.flight.contracts import FlightContract, TelemetryFrame


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return float(ordered[index])


def distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": float(min(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": float(max(values)),
    }


def _root_mean_square(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def analyze_trace(path: Path, contract: FlightContract) -> dict:
    frames: list[TelemetryFrame] = []
    actions: list[tuple[float, ...]] = []
    inference_us: list[float] = []
    checkpoint_hashes: set[str] = set()
    model_hashes: set[str] = set()

    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
                if record.get("kind") != "shadow_action":
                    raise ValueError("record is not a shadow action")
                telemetry = json.dumps(
                    record["telemetry"], allow_nan=False, separators=(",", ":")
                ).encode()
                frame = TelemetryFrame.from_wire(telemetry, contract)
                action = contract.action.validate(
                    record["proposed_action"], "proposed_action"
                )
                inference_ns = int(record["inference_ns"])
                if inference_ns < 0:
                    raise ValueError("inference_ns must be non-negative")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid shadow record on line {line_number}"
                ) from error
            frames.append(frame)
            actions.append(action)
            inference_us.append(inference_ns / 1000.0)
            checkpoint_hashes.add(str(record["checkpoint_sha256"]))
            model_hashes.add(str(record["model_sha256"]))

    if not frames:
        raise ValueError("shadow trace contains no frames")
    if len(checkpoint_hashes) != 1 or len(model_hashes) != 1:
        raise ValueError("shadow trace changes policy identity")

    sequence_gaps = 0
    monotonic_errors = 0
    kinematic_errors: list[float] = []
    action_deltas: list[float] = []
    for previous, current, previous_action, action in zip(
        frames, frames[1:], actions, actions[1:]
    ):
        if current.episode_id == previous.episode_id:
            if current.sequence != previous.sequence + 1:
                sequence_gaps += 1
            elapsed = (current.monotonic_ns - previous.monotonic_ns) / 1e9
            if elapsed <= 0.0:
                monotonic_errors += 1
            else:
                for axis in range(3):
                    derived_velocity = (
                        current.observation[axis] - previous.observation[axis]
                    ) / elapsed
                    reported_velocity = 0.5 * (
                        current.observation[axis + 3] + previous.observation[axis + 3]
                    )
                    kinematic_errors.append(derived_velocity - reported_velocity)
            normalized_delta = sum(
                ((right - left) / ((high - low) / 2.0)) ** 2
                for left, right, low, high in zip(
                    previous_action,
                    action,
                    contract.action.low,
                    contract.action.high,
                )
            )
            action_deltas.append(math.sqrt(normalized_delta))

    observations = list(zip(*(frame.observation for frame in frames)))
    action_channels = list(zip(*actions))
    applied_actions = [
        frame.applied_action for frame in frames if frame.applied_action is not None
    ]
    vehicle_masses = [
        frame.vehicle_mass_kg for frame in frames if frame.vehicle_mass_kg is not None
    ]
    quaternion_norm_errors = [
        abs(math.sqrt(sum(value * value for value in frame.observation[6:10])) - 1.0)
        for frame in frames
    ]
    dt_values = [frame.dt_s for frame in frames]
    monotonic_elapsed = (frames[-1].monotonic_ns - frames[0].monotonic_ns) / 1e9
    observation_ranges = {
        field: {"minimum": float(min(values)), "maximum": float(max(values))}
        for field, values in zip(contract.observation.fields, observations)
    }
    action_ranges = {}
    for field, values, low, high in zip(
        contract.action.fields,
        action_channels,
        contract.action.low,
        contract.action.high,
    ):
        midpoint = (low + high) / 2.0
        half_range = (high - low) / 2.0
        action_ranges[field] = {
            "minimum": float(min(values)),
            "maximum": float(max(values)),
            "maximum_absolute_fraction": float(
                max(abs(value - midpoint) / half_range for value in values)
            ),
        }
    applied_action_ranges = None
    if applied_actions:
        applied_action_ranges = {
            field: {"minimum": float(min(values)), "maximum": float(max(values))}
            for field, values in zip(contract.action.fields, zip(*applied_actions))
        }

    return {
        "schema_version": 1,
        "source": str(path),
        "contract_hash": contract.digest,
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "model_sha256": next(iter(model_hashes)),
        "frames": len(frames),
        "episodes": len({frame.episode_id for frame in frames}),
        "first_sequence": frames[0].sequence,
        "last_sequence": frames[-1].sequence,
        "sequence_gaps": sequence_gaps,
        "monotonic_errors": monotonic_errors,
        "monotonic_elapsed_s": monotonic_elapsed,
        "dt_s": distribution(dt_values),
        "inference_us": distribution(inference_us),
        "quaternion_maximum_norm_error": max(quaternion_norm_errors),
        "position_velocity_consistency_rmse_mps": _root_mean_square(kinematic_errors),
        "action_normalized_delta": (
            distribution(action_deltas) if action_deltas else None
        ),
        "observation_ranges": observation_ranges,
        "action_ranges": action_ranges,
        "applied_action_frames": len(applied_actions),
        "applied_action_ranges": applied_action_ranges,
        "vehicle_mass_frames": len(vehicle_masses),
        "vehicle_mass_kg": distribution(vehicle_masses) if vehicle_masses else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze_trace(args.trace, FlightContract.from_json(args.contract))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
