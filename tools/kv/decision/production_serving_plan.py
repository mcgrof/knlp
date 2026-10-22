# SPDX-License-Identifier: MIT
"""Freeze native-only pilots, then fixed serving traces before candidate timing.

Plan generation never starts a worker. Selection replays complete raw native
evidence; the bounded fallback sequence and load grid are prospective rules.
"""

import argparse
import json
from pathlib import Path

from .production import read_inputs, sha
from .production_plan import PILOT_RULES, derive_slo, orders, stress_load, trace
from .production_timing import save, validate_plan
from .production_timing_analysis import invocation, read


def plan(protocol, digest, quality_digest, arm, purpose, cells):
    return {
        "source_sha": protocol["source_sha"],
        "protocol_sha256": digest,
        "quality_analysis_sha256": quality_digest,
        "arm": arm,
        "purpose": purpose,
        "cells": cells,
    }


def one_cell(data, label, length, rate, seed, slo, *, seconds=120):
    return {
        "label": label,
        "trace": trace(data, length, rate, seed, label, seconds=seconds),
        "slo": slo,
    }


def emit(output, name, value, quality, protocol, digest, data):
    validate_plan(value, quality, protocol, digest, data)
    path = output / (name + ".json")
    if path.exists():
        raise ValueError("Frozen invocation plan already exists")
    save(path, value)
    return {
        "name": name,
        "plan_file": path.name,
        "plan_sha256": sha(path),
        "arm": value["arm"],
    }


def checked_manifest(folder, expected, purpose, protocol, digest, quality_digest):
    if sha(folder / "MANIFEST.json") != expected:
        raise ValueError("Previous phase manifest changed")
    manifest = read(folder / "MANIFEST.json")
    if any(
        manifest[k] != v
        for k, v in {
            "purpose": purpose,
            "protocol_sha256": digest,
            "source_sha": protocol["source_sha"],
            "quality_analysis_sha256": quality_digest,
        }.items()
    ):
        raise ValueError("Previous serving phase identity differs")
    if manifest["pilot_rules"] != PILOT_RULES:
        raise ValueError("Prospective pilot rules changed")
    for entry in manifest["invocations"]:
        if (
            entry["plan_file"] != entry["name"] + ".json"
            or sha(folder / entry["plan_file"]) != entry["plan_sha256"]
        ):
            raise ValueError("Frozen invocation changed")
    return manifest


def audit_entry(folder, root, entry, quality, quality_digest, protocol, digest, data):
    value = read(folder / entry["plan_file"])
    result = invocation(
        root / entry["name"],
        value,
        entry["plan_sha256"],
        quality,
        quality_digest,
        protocol,
        digest,
        data,
    )
    return value, result


def initial(output, protocol, digest, quality, quality_digest, data):
    entries = []
    cells = [
        one_cell(
            data,
            "soak",
            8192,
            PILOT_RULES["soak_rate"],
            20261101,
            None,
            seconds=PILOT_RULES["soak_arrival_seconds"],
        )
    ]
    entries.append(
        emit(
            output,
            "soak",
            plan(protocol, digest, quality_digest, "native_fa3", "soak", cells),
            quality,
            protocol,
            digest,
            data,
        )
    )
    for i, rate in enumerate(PILOT_RULES["slo_calibration_rates_in_order"]):
        name = f"calibration-{i:02d}"
        cells = [one_cell(data, name, 8192, rate, 20261200 + i, None)]
        entries.append(
            emit(
                output,
                name,
                plan(
                    protocol,
                    digest,
                    quality_digest,
                    "native_fa3",
                    "slo_calibration",
                    cells,
                ),
                quality,
                protocol,
                digest,
                data,
            )
        )
    return {
        "invocations": entries,
        "execution_rule": "Soak once, then calibration invocations in order; stop at the first stable native pilot. Later calibration plans remain unexecuted. No retries.",
    }


def select_slo(folder, root, previous, quality, quality_digest, protocol, digest, data):
    entries = previous["invocations"]
    expected = ["soak", *(f"calibration-{i:02d}" for i in range(4))]
    if [r["name"] for r in entries] != expected:
        raise ValueError("Native calibration sequence differs")
    _, soak = audit_entry(
        folder, root, entries[0], quality, quality_digest, protocol, digest, data
    )
    check = soak["cells"][0]
    if not check["harness_screen_passed"] or not all(
        r["valid_completion"]
        and r["client_output_tokens"] == 128
        and r["finish_reasons"] == ["length"]
        for r in check["boundaries"]["requests"]
    ):
        raise ValueError(
            "Connection-boundary soak did not complete with a valid harness"
        )
    observations, selected = [], None
    for i, entry in enumerate(entries[1:]):
        if selected is not None:
            if (root / entry["name"]).exists():
                raise ValueError("Calibration continued after a stable pilot")
            continue
        value, result = audit_entry(
            folder, root, entry, quality, quality_digest, protocol, digest, data
        )
        cell = result["cells"][0]
        expected_rate = PILOT_RULES["slo_calibration_rates_in_order"][i]
        if value["cells"][0]["trace"]["nominal_rate"] != expected_rate:
            raise ValueError("Calibration rate order changed")
        observations.append(
            {
                "name": entry["name"],
                "rate": expected_rate,
                "stable": cell["native_calibration_stability"],
            }
        )
        if cell["native_calibration_stability"]["passed"]:
            capture_path = (
                root / entry["name"] / "timing" / (cell["label"] + "-client.json")
            )
            selected = {
                "name": entry["name"],
                "rate": expected_rate,
                "slo": derive_slo(
                    read(capture_path), cell["boundaries"], cell["client_resources"]
                ),
                "capture_sha256": sha(capture_path),
                "scope": "Baseline-relative limits: 110% of native FA3 p95, frozen before any candidate timing",
            }
    if selected is None:
        raise ValueError("No stable native calibration pilot in the bounded sequence")
    return {"selected": selected, "observations": observations}


def load_plan(output, selection, protocol, digest, quality, quality_digest, data):
    cells = [
        one_cell(
            data,
            f"load-{i:02d}",
            8192,
            rate,
            20261300 + i,
            selection["selected"]["slo"],
        )
        for i, rate in enumerate(PILOT_RULES["load_selection_rates"])
    ]
    entry = emit(
        output,
        "loads",
        plan(protocol, digest, quality_digest, "native_fa3", "load_selection", cells),
        quality,
        protocol,
        digest,
        data,
    )
    return {"invocations": [entry], "slo_selection": selection}


def confirmation(
    output,
    previous,
    native_plan,
    native_audit,
    protocol,
    digest,
    quality,
    quality_digest,
    data,
):
    if len(native_plan["cells"]) != 9 or len(native_audit["cells"]) != 9:
        raise ValueError("Every bounded native load must be retained")
    pilots = [
        {
            "arm": "native_fa3",
            "rate": spec["trace"]["nominal_rate"],
            "slo_fraction": audit["summary"]["slo_fraction_of_all_offered"],
            "harness_passed": audit["harness_screen_passed"],
            "resource_passed": True,
        }
        for spec, audit in zip(native_plan["cells"], native_audit["cells"])
    ]
    stress = stress_load(pilots)
    slo = previous["slo_selection"]["selected"]["slo"]
    if any(c["slo"] != slo for c in native_plan["cells"]):
        raise ValueError("Load-selection SLOs differ from frozen calibration")
    arms = ["native_fi", "AF", "native_fa3"]
    if "SD" in quality["serving_eligible_candidates"]:
        arms.append("SD")
    arm_orders = orders(arms)
    rates = sorted(
        set(
            PILOT_RULES["initial_confirmation_rates"]
            + ([stress["rate"]] if stress["rate"] is not None else [])
        )
    )
    entries = []
    for block, arm_order in enumerate(arm_orders):
        rates_order = rates[block % len(rates) :] + rates[: block % len(rates)]
        cells = [
            one_cell(
                data,
                f"b{block:02d}-8k-r{round(rate * 10):02d}",
                8192,
                rate,
                20262000 + block * 100 + round(rate * 10),
                slo,
            )
            for rate in rates_order
        ]
        cells.append(
            one_cell(
                data,
                f"b{block:02d}-1k",
                1024,
                PILOT_RULES["secondary_1k_rate"],
                20265000 + block,
                slo,
            )
        )
        for arm in arm_order:
            name = f"block-{block:02d}-{arm}"
            entry = emit(
                output,
                name,
                plan(protocol, digest, quality_digest, arm, "confirmation", cells),
                quality,
                protocol,
                digest,
                data,
            )
            entries.append({**entry, "block": block})
    return {
        "invocations": entries,
        "arm_orders": arm_orders,
        "slo_selection": previous["slo_selection"],
        "native_load_pilots": pilots,
        "primary_stress_load": stress,
        "rates_8k": rates,
        "secondary_1k_rate": PILOT_RULES["secondary_1k_rate"],
        "inference": {
            "candidate": "AF",
            "controls": [a for a in arms if a != "AF"],
            "blocks": 10,
            "alpha": 0.05,
            "median_improvement_screen_percent": 5,
            "method": "Paired block sign tests with Holm correction; simultaneous Bonferroni distribution-free median bounds",
            "primary_scope": "Only the frozen 8K stress cell supports inference; all other cells are descriptive. With no informative stress cell, no general serving improvement claim is tested.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("initial", "loads", "confirmation"), required=True
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--quality-analysis", type=Path, required=True)
    parser.add_argument("--previous-plans", type=Path)
    parser.add_argument("--previous-manifest-sha256")
    parser.add_argument("--previous-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, data = read_inputs(args.inputs, args.protocol_sha256)
    digest, qdigest = args.protocol_sha256, sha(args.quality_analysis)
    quality = read(args.quality_analysis)
    prior = [args.previous_plans, args.previous_manifest_sha256, args.previous_results]
    if (
        args.phase == "initial"
        and any(x is not None for x in prior)
        or args.phase != "initial"
        and not all(x is not None for x in prior)
    ):
        parser.error(
            "Later phases require all three previous-phase arguments; initial requires none"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase == "initial":
        detail = initial(args.output, protocol, digest, quality, qdigest, data)
    else:
        previous = checked_manifest(
            args.previous_plans,
            args.previous_manifest_sha256,
            "initial" if args.phase == "loads" else "loads",
            protocol,
            digest,
            qdigest,
        )
        if args.phase == "loads":
            selected = select_slo(
                args.previous_plans,
                args.previous_results,
                previous,
                quality,
                qdigest,
                protocol,
                digest,
                data,
            )
            detail = load_plan(
                args.output, selected, protocol, digest, quality, qdigest, data
            )
        else:
            if len(previous["invocations"]) != 1:
                raise ValueError("Exactly one fixed load-grid invocation is required")
            value, audit = audit_entry(
                args.previous_plans,
                args.previous_results,
                previous["invocations"][0],
                quality,
                qdigest,
                protocol,
                digest,
                data,
            )
            detail = confirmation(
                args.output,
                previous,
                value,
                audit,
                protocol,
                digest,
                quality,
                qdigest,
                data,
            )
    save(
        args.output / "MANIFEST.json",
        {
            "schema": 1,
            "purpose": args.phase,
            "source_sha": protocol["source_sha"],
            "protocol_sha256": digest,
            "quality_analysis_sha256": qdigest,
            "pilot_rules": PILOT_RULES,
            "previous_manifest_sha256": args.previous_manifest_sha256,
            **detail,
        },
    )


if __name__ == "__main__":
    main()
