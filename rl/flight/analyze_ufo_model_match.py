"""Test whether a compensated X-Plane UFO follows the standalone model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.flight.analyze_ufo_response import (
    _airborne_mask,
    _coefficient_of_determination,
    _load_frames,
)
from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.geometry import quaternion_body_to_ned
from rl.flight.ufo_reference import UfoReferenceParameters


def _windows(
    frames: Sequence[TelemetryFrame],
    airborne: Sequence[bool],
    parameters: UfoReferenceParameters,
    *,
    target_s: float,
    maximum_step_s: float,
) -> list[dict[str, np.ndarray]]:
    result = []
    start = 1
    while start < len(frames):
        if not airborne[start] or not airborne[start - 1]:
            start += 1
            continue
        end = start
        while end < len(frames):
            step_s = (
                frames[end].monotonic_ns - frames[end - 1].monotonic_ns
            ) / 1e9
            duration_s = (
                frames[end].monotonic_ns - frames[start - 1].monotonic_ns
            ) / 1e9
            if not airborne[end] or step_s <= 0.0 or step_s > maximum_step_s:
                break
            if duration_s >= target_s:
                break
            end += 1
        if end >= len(frames) or not airborne[end]:
            start = end + 1
            continue
        step_s = (
            frames[end].monotonic_ns - frames[end - 1].monotonic_ns
        ) / 1e9
        duration_s = (
            frames[end].monotonic_ns - frames[start - 1].monotonic_ns
        ) / 1e9
        if (
            step_s <= 0.0
            or step_s > maximum_step_s
            or duration_s < target_s
            or duration_s > 1.5 * target_s
        ):
            start = end + 1
            continue

        velocity_delta = np.zeros(3)
        angular_velocity_delta = np.zeros(3)
        inertia = np.asarray(parameters.inertia_kg_m2, dtype=np.float64)
        linear_drag = np.asarray(
            parameters.linear_drag_n_per_mps, dtype=np.float64
        )
        angular_drag = np.asarray(
            parameters.angular_drag_nm_per_radps, dtype=np.float64
        )
        previous_ns = frames[start - 1].monotonic_ns
        for index in range(start, end + 1):
            frame = frames[index]
            dt_s = (frame.monotonic_ns - previous_ns) / 1e9
            previous_ns = frame.monotonic_ns
            state = np.asarray(frame.observation, dtype=np.float64)
            requested = np.asarray(frame.requested_action, dtype=np.float64)
            rotation = quaternion_body_to_ned(state[6:10])
            velocity_delta += (
                rotation @ requested[:3] / parameters.mass_kg
                - linear_drag * state[3:6] / parameters.mass_kg
                + np.asarray((0.0, 0.0, parameters.gravity_mps2))
            ) * dt_s
            omega = state[10:13]
            angular_velocity_delta += (
                requested[3:]
                - np.cross(omega, inertia * omega)
                - angular_drag * omega
            ) / inertia * dt_s

        before = np.asarray(frames[start - 1].observation, dtype=np.float64)
        after = np.asarray(frames[end].observation, dtype=np.float64)
        result.append(
            {
                "linear_actual": (after[3:6] - before[3:6]) / duration_s,
                "linear_predicted": velocity_delta / duration_s,
                "angular_actual": (after[10:13] - before[10:13]) / duration_s,
                "angular_predicted": angular_velocity_delta / duration_s,
            }
        )
        start = end + 1
    return result


def _metrics(windows: Sequence[dict[str, np.ndarray]], prefix: str) -> dict:
    actual = np.asarray([window[f"{prefix}_actual"] for window in windows])
    predicted = np.asarray(
        [window[f"{prefix}_predicted"] for window in windows]
    )
    error = actual - predicted
    return {
        "windows": len(windows),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "axis_rmse": [
            float(value) for value in np.sqrt(np.mean(error**2, axis=0))
        ],
        "coefficient_of_determination": _coefficient_of_determination(
            actual, predicted
        ),
    }


def analyze_model_match(
    path: Path,
    contract: FlightContract,
    *,
    parameters: UfoReferenceParameters = UfoReferenceParameters(),
    window_s: float = 0.1,
    maximum_step_s: float = 0.02,
    closure_threshold_n: float = 10.0,
) -> dict:
    frames = _load_frames(path, contract)
    missing = [frame.sequence for frame in frames if frame.requested_action is None]
    if missing:
        raise ValueError(
            f"requested_action is missing from {len(missing)} telemetry frames"
        )
    airborne = _airborne_mask(frames, closure_threshold_n)
    durations = (0.5 * window_s, window_s, 2.0 * window_s)
    window_sets = {
        duration: _windows(
            frames,
            airborne,
            parameters,
            target_s=duration,
            maximum_step_s=maximum_step_s,
        )
        for duration in durations
    }
    if any(len(windows) < 3 for windows in window_sets.values()):
        raise ValueError("trace has too few contiguous airborne model-match windows")

    linear_by_window = {
        duration: _metrics(windows, "linear")
        for duration, windows in window_sets.items()
    }
    angular_by_window = {
        duration: _metrics(windows, "angular")
        for duration, windows in window_sets.items()
    }
    linear = linear_by_window[window_s]
    angular = angular_by_window[window_s]
    thresholds = {
        "minimum_windows": 100,
        "minimum_coefficient_of_determination": 0.99,
        "maximum_linear_acceleration_rmse_mps2": 0.2,
        "maximum_angular_acceleration_rmse_radps2": 0.05,
    }

    def passes(metrics_by_window: dict[float, dict], maximum_rmse: float) -> bool:
        main = metrics_by_window[window_s]
        return (
            main["windows"] >= thresholds["minimum_windows"]
            and all(
                metrics["coefficient_of_determination"]
                >= thresholds["minimum_coefficient_of_determination"]
                and metrics["rmse"] <= maximum_rmse
                for metrics in metrics_by_window.values()
            )
        )

    linear_passed = passes(
        linear_by_window, thresholds["maximum_linear_acceleration_rmse_mps2"]
    )
    angular_passed = passes(
        angular_by_window,
        thresholds["maximum_angular_acceleration_rmse_radps2"],
    )
    return {
        "schema_version": 1,
        "source": str(path),
        "contract_sha256": contract.digest,
        "frames": len(frames),
        "airborne_frames": sum(airborne),
        "airborne_fraction": sum(airborne) / len(frames),
        "airborne_force_closure_threshold_n": closure_threshold_n,
        "target_model": {
            "mass_kg": parameters.mass_kg,
            "inertia_kg_m2": list(parameters.inertia_kg_m2),
            "linear_drag_n_per_mps": list(parameters.linear_drag_n_per_mps),
            "angular_drag_nm_per_radps": list(
                parameters.angular_drag_nm_per_radps
            ),
            "gravity_mps2": parameters.gravity_mps2,
        },
        "window_s": window_s,
        "linear_response": {
            **linear,
            "by_window_s": {
                f"{duration:g}": metrics
                for duration, metrics in linear_by_window.items()
            },
            "gate_passed": linear_passed,
        },
        "angular_response": {
            **angular,
            "by_window_s": {
                f"{duration:g}": metrics
                for duration, metrics in angular_by_window.items()
            },
            "gate_passed": angular_passed,
        },
        "gate_thresholds": thresholds,
        "model_match_gate_passed": linear_passed and angular_passed,
        "note": (
            "vehicle_inertia_kg_m2 is diagnostic only; this gate compares "
            "observed motion directly with the declared target model"
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
    report = analyze_model_match(
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
