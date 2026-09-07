"""Capture a live UFO shadow flight and emit its machine acceptance verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rl.flight.analyze_shadow import analyze_trace
from rl.flight.analyze_ufo_model_match import analyze_model_match
from rl.flight.analyze_ufo_response import analyze_response
from rl.flight.contracts import FlightContract
from rl.flight.shadow_ufo import default_socket_path, main as capture_main


def _write_json(path: Path, report: dict) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def build_acceptance_verdict(
    summary: dict,
    analysis: dict,
    response: dict,
    model_match: dict,
    *,
    expected_frames: int,
) -> dict:
    transport_passed = (
        summary.get("frames") == expected_frames
        and summary.get("rejected") == 0
        and analysis.get("sequence_gaps") == 0
        and analysis.get("monotonic_errors") == 0
    )
    required_diagnostics = (
        "requested_action_frames",
        "applied_action_frames",
        "simulator_total_wrench_frames",
        "aerodynamic_wrench_frames",
        "vehicle_mass_frames",
        "vehicle_inertia_frames",
    )
    diagnostics_passed = all(
        analysis.get(field) == expected_frames for field in required_diagnostics
    )
    response_identification_passed = bool(
        response.get("mass_and_inertia_gate_passed")
    )
    target_model_passed = bool(model_match.get("model_match_gate_passed"))
    machine_gate_passed = (
        transport_passed
        and diagnostics_passed
        and response_identification_passed
        and target_model_passed
    )
    return {
        "schema_version": 1,
        "expected_frames": expected_frames,
        "transport_gate_passed": transport_passed,
        "diagnostic_coverage_gate_passed": diagnostics_passed,
        "response_identification_gate_passed": response_identification_passed,
        "target_model_gate_passed": target_model_passed,
        "machine_gate_passed": machine_gate_passed,
        "human_gates": {
            "handling_safe": None,
            "audio_clean_in_headset": None,
            "halo_visible_without_artifacts": None,
        },
        "note": (
            "machine_gate_passed does not authorize external control; the "
            "pilot must record the three human gates separately"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=20_000)
    parser.add_argument("--connect-timeout", type=float, default=43_200.0)
    args = parser.parse_args(argv)
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")

    capture_status = capture_main(
        [
            "--socket",
            str(args.socket),
            "--contract",
            str(args.contract),
            "--model",
            str(args.model),
            "--output",
            str(args.output),
            "--max-frames",
            str(args.max_frames),
            "--connect-timeout",
            str(args.connect_timeout),
        ]
    )
    if capture_status != 0:
        return capture_status

    contract = FlightContract.from_json(args.contract)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    analysis_path = args.output.with_suffix(args.output.suffix + ".analysis.json")
    response_path = args.output.with_suffix(args.output.suffix + ".response.json")
    model_match_path = args.output.with_suffix(
        args.output.suffix + ".model-match.json"
    )
    verdict_path = args.output.with_suffix(args.output.suffix + ".verdict.json")
    summary = json.loads(summary_path.read_text())
    analysis = analyze_trace(args.output, contract)
    response = analyze_response(args.output, contract)
    model_match = analyze_model_match(args.output, contract)
    verdict = build_acceptance_verdict(
        summary,
        analysis,
        response,
        model_match,
        expected_frames=args.max_frames,
    )
    _write_json(analysis_path, analysis)
    _write_json(response_path, response)
    _write_json(model_match_path, model_match)
    _write_json(verdict_path, verdict)
    print(json.dumps(verdict, sort_keys=True), flush=True)
    return 0 if verdict["machine_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
