"""Identify UFO mass/inertia and test whether linear drag is supported."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.geometry import quaternion_body_to_ned


def _load_frames(path: Path, contract: FlightContract) -> list[TelemetryFrame]:
    frames = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
                if record.get("kind") != "shadow_action":
                    raise ValueError("record is not a shadow action")
                frame = TelemetryFrame.from_wire(
                    json.dumps(
                        record["telemetry"], allow_nan=False, separators=(",", ":")
                    ).encode(),
                    contract,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid shadow record on line {line_number}"
                ) from error
            frames.append(frame)
    if not frames:
        raise ValueError("shadow trace contains no frames")
    missing = [
        frame.sequence
        for frame in frames
        if frame.applied_action is None
        or frame.aerodynamic_wrench is None
        or frame.simulator_total_wrench is None
        or frame.vehicle_mass_kg is None
    ]
    if missing:
        raise ValueError(
            f"response diagnostics are missing from {len(missing)} telemetry frames"
        )
    episode_ids = {frame.episode_id for frame in frames}
    if len(episode_ids) != 1:
        raise ValueError("response trace must contain exactly one episode")
    for previous, current in zip(frames, frames[1:]):
        if current.sequence != previous.sequence + 1:
            raise ValueError("response trace contains a sequence gap")
        if current.monotonic_ns <= previous.monotonic_ns:
            raise ValueError("response trace timestamps are not strictly increasing")
    return frames


def _airborne_mask(
    frames: Sequence[TelemetryFrame], closure_threshold_n: float
) -> list[bool]:
    result = []
    for frame in frames:
        total = np.asarray(frame.simulator_total_wrench, dtype=np.float64)
        aerodynamic = np.asarray(frame.aerodynamic_wrench, dtype=np.float64)
        applied = np.asarray(frame.applied_action, dtype=np.float64)
        result.append(
            float(np.linalg.norm(total[:3] - aerodynamic[:3] - applied[:3]))
            <= closure_threshold_n
        )
    return result


def _windows(
    frames: Sequence[TelemetryFrame],
    airborne: Sequence[bool],
    *,
    target_s: float,
    maximum_step_s: float,
    gravity_mps2: float,
) -> list[dict[str, np.ndarray | float]]:
    result = []
    start = 1
    while start < len(frames):
        first_dt = (frames[start].monotonic_ns - frames[start - 1].monotonic_ns) / 1e9
        if (
            not airborne[start]
            or not airborne[start - 1]
            or first_dt <= 0.0
            or first_dt > maximum_step_s
        ):
            start += 1
            continue
        end = start
        while end < len(frames):
            duration = (frames[end].monotonic_ns - frames[start - 1].monotonic_ns) / 1e9
            step = (frames[end].monotonic_ns - frames[end - 1].monotonic_ns) / 1e9
            if not airborne[end] or step <= 0.0 or step > maximum_step_s:
                break
            if duration >= target_s:
                break
            end += 1
        if end >= len(frames) or not airborne[end]:
            start = end + 1
            continue
        step = (frames[end].monotonic_ns - frames[end - 1].monotonic_ns) / 1e9
        if step <= 0.0 or step > maximum_step_s:
            start = end + 1
            continue
        duration = (frames[end].monotonic_ns - frames[start - 1].monotonic_ns) / 1e9
        if duration < target_s or duration > 1.5 * target_s:
            start = end + 1
            continue

        acceleration_impulse = np.zeros(3)
        moment_impulse = np.zeros(3)
        cross_impulse = np.zeros(3)
        previous_ns = frames[start - 1].monotonic_ns
        for index in range(start, end + 1):
            frame = frames[index]
            dt_s = (frame.monotonic_ns - previous_ns) / 1e9
            previous_ns = frame.monotonic_ns
            observation = np.asarray(frame.observation, dtype=np.float64)
            total = np.asarray(frame.simulator_total_wrench, dtype=np.float64)
            rotation = quaternion_body_to_ned(observation[6:10])
            acceleration_impulse += (
                rotation @ total[:3] / float(frame.vehicle_mass_kg)
            ) * dt_s
            moment_impulse += total[3:] * dt_s
            omega = observation[10:13]
            cross_impulse += (
                np.asarray(
                    [omega[1] * omega[2], omega[2] * omega[0], omega[0] * omega[1]]
                )
                * dt_s
            )

        before = np.asarray(frames[start - 1].observation, dtype=np.float64)
        after = np.asarray(frames[end].observation, dtype=np.float64)
        actual_acceleration = (after[3:6] - before[3:6]) / duration
        predicted_acceleration = acceleration_impulse / duration
        predicted_acceleration[2] += gravity_mps2
        result.append(
            {
                "actual_acceleration": actual_acceleration,
                "predicted_acceleration": predicted_acceleration,
                "angular_acceleration": (after[10:13] - before[10:13]) / duration,
                "moment": moment_impulse / duration,
                "omega_cross": cross_impulse / duration,
            }
        )
        start = end + 1
    return result


def _coefficient_of_determination(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    centered = float(np.sum((actual - np.mean(actual, axis=0)) ** 2))
    return 1.0 - residual / centered if centered > 0.0 else 0.0


def _fit_inertia(windows: Sequence[dict[str, np.ndarray | float]]) -> dict:
    design = []
    response = []
    for window in windows:
        alpha = np.asarray(window["angular_acceleration"])
        cross = np.asarray(window["omega_cross"])
        moment = np.asarray(window["moment"])
        design.extend(
            (
                [alpha[0], -cross[0], cross[0]],
                [cross[1], alpha[1], -cross[1]],
                [-cross[2], cross[2], alpha[2]],
            )
        )
        response.extend(moment)
    matrix = np.asarray(design, dtype=np.float64)
    observed = np.asarray(response, dtype=np.float64)
    inertia, _residuals, rank, singular_values = np.linalg.lstsq(
        matrix, observed, rcond=None
    )
    predicted = matrix @ inertia
    return {
        "inertia_kg_m2": [float(value) for value in inertia],
        "rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "moment_rmse_nm": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "coefficient_of_determination": _coefficient_of_determination(
            observed, predicted
        ),
    }


def _fit_drag(
    frames: Sequence[TelemetryFrame], airborne: Sequence[bool], *, body_axes: bool
) -> list[dict[str, float]]:
    velocities = []
    forces = []
    for frame, is_airborne in zip(frames, airborne):
        if not is_airborne:
            continue
        observation = np.asarray(frame.observation, dtype=np.float64)
        rotation = quaternion_body_to_ned(observation[6:10])
        velocity = observation[3:6]
        force = np.asarray(frame.aerodynamic_wrench[:3], dtype=np.float64)
        if body_axes:
            velocity = rotation.T @ velocity
        else:
            force = rotation @ force
        velocities.append(velocity)
        forces.append(force)
    velocity_values = np.asarray(velocities)
    force_values = np.asarray(forces)
    result = []
    for axis in range(3):
        velocity = velocity_values[:, axis]
        force = force_values[:, axis]
        denominator = float(velocity @ velocity)
        if denominator <= np.finfo(np.float64).eps:
            result.append(
                {
                    "coefficient_n_per_mps": 0.0,
                    "force_rmse_n": float(np.sqrt(np.mean(force**2))),
                    "coefficient_of_determination": 0.0,
                }
            )
            continue
        coefficient = -float(velocity @ force) / denominator
        predicted = -coefficient * velocity
        result.append(
            {
                "coefficient_n_per_mps": coefficient,
                "force_rmse_n": float(np.sqrt(np.mean((force - predicted) ** 2))),
                "coefficient_of_determination": _coefficient_of_determination(
                    force, predicted
                ),
            }
        )
    return result


def _fit_angular_drag(
    frames: Sequence[TelemetryFrame], airborne: Sequence[bool]
) -> list[dict[str, float]]:
    angular_velocities = []
    moments = []
    for frame, is_airborne in zip(frames, airborne):
        if not is_airborne:
            continue
        observation = np.asarray(frame.observation, dtype=np.float64)
        angular_velocities.append(observation[10:13])
        moments.append(np.asarray(frame.aerodynamic_wrench[3:], dtype=np.float64))
    angular_velocity_values = np.asarray(angular_velocities)
    moment_values = np.asarray(moments)
    result = []
    for axis in range(3):
        angular_velocity = angular_velocity_values[:, axis]
        moment = moment_values[:, axis]
        denominator = float(angular_velocity @ angular_velocity)
        if denominator <= np.finfo(np.float64).eps:
            result.append(
                {
                    "coefficient_nm_per_radps": 0.0,
                    "moment_rmse_nm": float(np.sqrt(np.mean(moment**2))),
                    "coefficient_of_determination": 0.0,
                }
            )
            continue
        coefficient = -float(angular_velocity @ moment) / denominator
        predicted = -coefficient * angular_velocity
        result.append(
            {
                "coefficient_nm_per_radps": coefficient,
                "moment_rmse_nm": float(np.sqrt(np.mean((moment - predicted) ** 2))),
                "coefficient_of_determination": _coefficient_of_determination(
                    moment, predicted
                ),
            }
        )
    return result


def analyze_response(
    path: Path,
    contract: FlightContract,
    *,
    window_s: float = 0.1,
    maximum_step_s: float = 0.02,
    closure_threshold_n: float = 10.0,
    gravity_mps2: float = 9.80665,
) -> dict:
    frames = _load_frames(path, contract)
    airborne = _airborne_mask(frames, closure_threshold_n)
    durations = (0.5 * window_s, window_s, 2.0 * window_s)
    window_sets = {
        duration: _windows(
            frames,
            airborne,
            target_s=duration,
            maximum_step_s=maximum_step_s,
            gravity_mps2=gravity_mps2,
        )
        for duration in durations
    }
    if any(len(windows) < 3 for windows in window_sets.values()):
        raise ValueError("trace has too few contiguous airborne response windows")

    inertia_fits = {
        duration: _fit_inertia(windows) for duration, windows in window_sets.items()
    }
    main_windows = window_sets[window_s]
    actual = np.asarray([window["actual_acceleration"] for window in main_windows])
    predicted = np.asarray(
        [window["predicted_acceleration"] for window in main_windows]
    )
    errors = actual - predicted
    force_rmse_axes = np.sqrt(np.mean(errors**2, axis=0))
    force_rmse = float(np.sqrt(np.mean(errors**2)))
    force_r2 = _coefficient_of_determination(actual, predicted)

    inertia_values = np.asarray([fit["inertia_kg_m2"] for fit in inertia_fits.values()])
    inertia_spread = np.ptp(inertia_values, axis=0) / np.median(inertia_values, axis=0)
    main_inertia = inertia_fits[window_s]
    ned_drag = _fit_drag(frames, airborne, body_axes=False)
    body_drag = _fit_drag(frames, airborne, body_axes=True)
    angular_drag = _fit_angular_drag(frames, airborne)

    force_gate_thresholds = {
        "minimum_windows": 100,
        "minimum_coefficient_of_determination": 0.99,
        "maximum_acceleration_rmse_mps2": 0.2,
    }
    inertia_gate_thresholds = {
        "required_rank": 3,
        "maximum_condition_number": 100.0,
        "minimum_coefficient_of_determination": 0.99,
        "require_positive_components": True,
        "maximum_relative_window_spread": 0.02,
    }
    drag_gate_thresholds = {
        "require_nonnegative_coefficients": True,
        "minimum_axis_coefficient_of_determination": 0.8,
    }

    force_gate = (
        len(main_windows) >= force_gate_thresholds["minimum_windows"]
        and force_r2 >= force_gate_thresholds["minimum_coefficient_of_determination"]
        and force_rmse <= force_gate_thresholds["maximum_acceleration_rmse_mps2"]
    )
    inertia_gate = (
        main_inertia["rank"] == 3
        and main_inertia["condition_number"] <= 100.0
        and main_inertia["coefficient_of_determination"] >= 0.99
        and min(main_inertia["inertia_kg_m2"]) > 0.0
        and float(np.max(inertia_spread)) <= 0.02
    )
    ned_drag_gate = all(
        fit["coefficient_n_per_mps"] >= 0.0
        and fit["coefficient_of_determination"] >= 0.8
        for fit in ned_drag
    )
    angular_drag_gate = all(
        fit["coefficient_nm_per_radps"] >= 0.0
        and fit["coefficient_of_determination"] >= 0.8
        for fit in angular_drag
    )
    masses = [float(frame.vehicle_mass_kg) for frame in frames if frame.vehicle_mass_kg]
    return {
        "schema_version": 1,
        "source": str(path),
        "contract_sha256": contract.digest,
        "frames": len(frames),
        "airborne_frames": sum(airborne),
        "airborne_fraction": sum(airborne) / len(frames),
        "airborne_force_closure_threshold_n": closure_threshold_n,
        "vehicle_mass_kg": {
            "minimum": min(masses),
            "median": float(statistics.median(masses)),
            "maximum": max(masses),
        },
        "window_s": window_s,
        "response_windows": len(main_windows),
        "response_windows_by_window_s": {
            f"{duration:g}": len(windows) for duration, windows in window_sets.items()
        },
        "total_force_consistency": {
            "acceleration_rmse_mps2": force_rmse,
            "axis_rmse_mps2": [float(value) for value in force_rmse_axes],
            "coefficient_of_determination": force_r2,
            "gate_passed": force_gate,
        },
        "inertia": {
            **main_inertia,
            "fits_by_window_s": {
                f"{duration:g}": fit for duration, fit in inertia_fits.items()
            },
            "relative_spread_across_half_to_double_window": [
                float(value) for value in inertia_spread
            ],
            "gate_passed": inertia_gate,
        },
        "aerodynamic_linear_drag_ned": ned_drag,
        "aerodynamic_linear_drag_body": body_drag,
        "aerodynamic_angular_drag_body": angular_drag,
        "linear_drag_gate_passed": ned_drag_gate,
        "angular_drag_gate_passed": angular_drag_gate,
        "gate_thresholds": {
            "total_force_consistency": force_gate_thresholds,
            "inertia": inertia_gate_thresholds,
            "aerodynamic_linear_drag_ned": drag_gate_thresholds,
            "aerodynamic_angular_drag_body": drag_gate_thresholds,
        },
        "mass_and_inertia_gate_passed": force_gate and inertia_gate,
        "full_standalone_model_gate_passed": (
            force_gate and inertia_gate and ned_drag_gate and angular_drag_gate
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-s", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.window_s <= 0.0 or not math.isfinite(args.window_s):
        parser.error("--window-s must be finite and positive")
    report = analyze_response(
        args.trace,
        FlightContract.from_json(args.contract),
        window_s=args.window_s,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
