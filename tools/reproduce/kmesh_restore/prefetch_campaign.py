#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recheck complete saved experiments before admission or interpretation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from prefetch_analysis import choose_lead, confirm, verify_clock
from prefetch_verify import verify_block
from restore_common import write_json


def verify_capture(root):
    """Verify every declared block, including warmups, failures and controls."""
    root = Path(root)
    config = json.loads((root / "CONFIG.json").read_text())
    phase = config["phase"]
    if phase not in ("qualification", "pilot", "confirmation"):
        raise ValueError("unknown campaign phase")
    blocks = {"qualification": 1, "pilot": 2, "confirmation": 10}[phase]
    samples = {"qualification": 1, "pilot": 5, "confirmation": 20}[phase]
    fanouts = (4, 8) if phase == "confirmation" else (4,)
    if (
        config["blocks_per_path_fanout"] != blocks
        or config["measured_groups_per_arm_block"] != samples
        or config["warmups_per_arm_block"] != 3
        or config["arrival_offset_ms"] != 250
    ):
        raise ValueError("campaign differs from frozen sampling plan")
    if (
        (root / "FAILURE.json").exists()
        or list(root.glob("*/failed-*.json"))
        or list(root.glob("*/*-failed.json"))
    ):
        raise ValueError("campaign contains a failed trial")
    expected = {
        f"{phase}-{mode}-n{fanout}-b{block:02d}": (mode, fanout, block)
        for block in range(blocks)
        for mode in ("staged", "direct")
        for fanout in fanouts
    }
    actual = {
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(phase + "-")
    }
    if actual != expected.keys():
        raise ValueError("missing or extra engine-process block")
    rows, identities, verified, overlaps = [], set(), [], {"staged": 0, "direct": 0}
    for name, (mode, fanout, index) in sorted(expected.items()):
        cell = root / name
        cfg = json.loads((cell / "config.json").read_text())
        if (
            cfg["mode"] != mode
            or cfg["fanouts"] != [fanout]
            or cfg["block_id"] != name
            or cfg["block_index"] != index
            or cfg["phase"] != phase
            or cfg["serving_samples"] != samples
        ):
            raise ValueError("process block differs from declared campaign")
        for key in (
            "model",
            "arena_bytes",
            "decode_tokens",
            "external_singleflight",
            "family_prefetch",
            "retained_fill",
            "max_num_seqs",
            "selected_lead_ms",
            "gpu_provider",
        ):
            if cfg[key] != config[key]:
                raise ValueError("process configuration changed: " + key)
        receipt = verify_block(cell)
        process = receipt["process"]
        identity = (process["boot_id"], process["pid"], process["start_ticks"])
        if identity in identities:
            raise ValueError("engine process incarnation reused across blocks")
        identities.add(identity)
        rows.extend(receipt["rows"])
        for row in receipt["rows"]:
            overlaps[mode] += (
                row["after"]["prefetch_controller"]["overlap_joined_total"]
                - row["before"]["prefetch_controller"]["overlap_joined_total"]
            )
        verified.append({k: v for k, v in receipt.items() if k != "rows"})
    result = {
        "phase": phase,
        "accepted": True,
        "blocks": len(verified),
        "verified_blocks": verified,
        "overlap_joins_by_path": overlaps,
    }
    if phase == "qualification":
        for mode in ("staged", "direct"):
            control = json.loads(
                (root / f"corruption-{mode}-rejected.json").read_text()
            )
            if (
                control["returncode"] == 0
                or not control["expected_payload_rejection"]
                or "restored GPU consumer bytes differ"
                not in (root / f"corruption-{mode}.log").read_text()
            ):
                raise ValueError("missing native deliberate-corruption rejection")
            if overlaps[mode] <= 0:
                raise ValueError(
                    "native qualification never joined an overlapping fill"
                )
    elif phase == "pilot":
        result["selection"] = choose_lead(rows)
    else:
        result["confirmation"] = confirm(
            rows, selected_lead_ms=config["selected_lead_ms"]
        )
    return result, rows


def export_csv(path, rows):
    """Export every actual child observation, retaining warmup labels."""
    fields = [
        "mode",
        "block_id",
        "fanout",
        "arm",
        "lead_ms",
        "sample",
        "warmup",
        "request_id",
        "submission_delay_ms",
        "deadline_first_token_ms",
        "deadline_completion_ms",
        "ttft_ms",
        "deadline_group_completion_ms",
    ]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            verify_clock(row)
            for child in row["rows"]:
                writer.writerow(
                    {key: child[key] if key in child else row[key] for key in fields}
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    result, rows = verify_capture(args.capture)
    if args.report:
        write_json(args.report, result)
    if args.csv:
        export_csv(args.csv, rows)
    print(json.dumps(result, indent=2, allow_nan=False))
