# SPDX-License-Identifier: MIT
"""Balanced Gemma SavedModel runtime comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .compare_outputs import compare_roots

TIMING_PATTERN = re.compile(r"steps=(\d+) seconds=([0-9.]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_binary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--binary must be LABEL=/path/to/runner")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("--binary must be LABEL=/path/to/runner")
    return label, Path(raw_path).expanduser().resolve()


def parse_case(case_name: str) -> tuple[int, int]:
    match = re.fullmatch(r"b(\d+)_s(\d+)", case_name)
    if not match:
        raise ValueError(f"invalid case name: {case_name}")
    return int(match.group(1)), int(match.group(2))


def median_abs_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        action="append",
        type=parse_binary,
        required=True,
        help="Repeat as LABEL=/path/to/gemma_savedmodel_runner",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feeds", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--fetch", default="StatefulPartitionedCall:0")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["b1_s32", "b4_s32", "b1_s128"],
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional runtime environment as NAME=VALUE",
    )
    args = parser.parse_args()

    binaries = dict(args.binary)
    if len(binaries) != len(args.binary):
        raise SystemExit("binary labels must be unique")
    for label, path in binaries.items():
        if not path.is_file():
            raise SystemExit(f"{label} binary does not exist: {path}")

    environment = os.environ.copy()
    for assignment in args.env:
        if "=" not in assignment:
            raise SystemExit("--env must be NAME=VALUE")
        key, value = assignment.split("=", 1)
        environment[key] = value

    args.results.mkdir(parents=True, exist_ok=True)
    logs = args.results / "logs"
    outputs = args.results / "outputs"
    logs.mkdir(exist_ok=True)
    samples = {label: {case_name: [] for case_name in args.cases} for label in binaries}
    labels = list(binaries)
    started = utc_now()

    for trial in range(args.trials):
        if trial % 4 in {1, 2}:
            build_order = list(reversed(labels))
        else:
            build_order = labels
        case_order = (
            args.cases[trial % len(args.cases) :]
            + args.cases[: trial % len(args.cases)]
        )
        for case_name in case_order:
            batch, sequence = parse_case(case_name)
            shape = f"{batch}x{sequence}"
            for label in build_order:
                output_dir = outputs / label / case_name
                output_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs / f"trial{trial + 1}_{case_name}_{label}.log"
                command = [
                    str(binaries[label]),
                    f"--model={args.model}",
                    (
                        "--feed=serve_token_ids:0,float32,"
                        f"{shape},{args.feeds}/token_ids_{case_name}.bin"
                    ),
                    (
                        "--feed=serve_padding_mask:0,float32,"
                        f"{shape},{args.feeds}/padding_mask_{case_name}.bin"
                    ),
                    f"--fetch={args.fetch}",
                    f"--output-dir={output_dir}",
                    f"--warmup={args.warmup}",
                    f"--steps={args.steps}",
                ]
                print(
                    f"trial={trial + 1} case={case_name} build={label}",
                    flush=True,
                )
                with log_path.open("w") as log:
                    completed = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        check=False,
                    )
                (log_path.with_suffix(".status")).write_text(
                    f"{completed.returncode}\n"
                )
                if completed.returncode:
                    print(log_path.read_text(), file=sys.stderr)
                    raise SystemExit(
                        f"{label} failed for {case_name}: " f"{completed.returncode}"
                    )
                matches = TIMING_PATTERN.findall(log_path.read_text())
                if not matches:
                    raise SystemExit(f"timing missing from {log_path}")
                steps, seconds = matches[-1]
                samples[label][case_name].append(float(seconds) / int(steps))

    summary: dict[str, object] = {
        "started_utc": started,
        "finished_utc": utc_now(),
        "protocol": {
            "trials": args.trials,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "order": "ABBA by trial, rotated case order",
        },
        "binaries": {label: str(path) for label, path in binaries.items()},
        "cases": {},
    }
    for case_name in args.cases:
        case_summary = {}
        for label in labels:
            values = samples[label][case_name]
            center = statistics.median(values)
            case_summary[label] = {
                "seconds_per_step": values,
                "median_seconds_per_step": center,
                "mad_seconds_per_step": median_abs_deviation(values),
                "minimum_seconds_per_step": min(values),
                "maximum_seconds_per_step": max(values),
            }
        if len(labels) == 2:
            left, right = labels
            ratios = [
                right_value / left_value
                for left_value, right_value in zip(
                    samples[left][case_name],
                    samples[right][case_name],
                )
            ]
            case_summary["paired_right_over_left_ratios"] = ratios
            case_summary["median_paired_percent"] = (
                statistics.median(ratios) - 1.0
            ) * 100.0
        summary["cases"][case_name] = case_summary

    if len(labels) == 2:
        left, right = labels
        ratios = []
        for case_name in args.cases:
            ratios.extend(
                right_value / left_value
                for left_value, right_value in zip(
                    samples[left][case_name],
                    samples[right][case_name],
                )
            )
        geometric_mean = math.exp(
            sum(math.log(value) for value in ratios) / len(ratios)
        )
        summary["overall_right_over_left_geomean_ratio"] = geometric_mean
        summary["overall_right_over_left_geomean_percent"] = (
            geometric_mean - 1.0
        ) * 100.0
        summary["correctness"] = compare_roots(
            outputs / left,
            outputs / right,
            args.cases,
        )
    output = args.results / "runtime-summary.json"
    write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary: {output}")


if __name__ == "__main__":
    main()
