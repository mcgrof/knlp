#!/usr/bin/env python3
"""Paired raw-row analysis for the four-pair KV-Lingo checkpoint grid."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from . import coqa
from .kv_lingo_data import write_json
from .kv_lingo_eval import attach_native_reference, read_jsonl

CHECKPOINT_PAIRS = (
    "forward_1000_reverse_1000",
    "forward_5000_reverse_5000",
    "forward_5000_reverse_1000",
    "forward_1000_reverse_5000",
)
STARTING_MODELS = ("4B", "8B")
REFERENCES = {
    "same_transcript_native": "same_transcript_native_f1",
    "alternating_native_trajectory": "native_trajectory_f1",
}
PAIRS = (
    (
        "forward_1000_reverse_1000",
        "forward_5000_reverse_1000",
        "forward_1000_to_5000_at_reverse_1000",
    ),
    (
        "forward_1000_reverse_1000",
        "forward_1000_reverse_5000",
        "reverse_1000_to_5000_at_forward_1000",
    ),
    (
        "forward_5000_reverse_1000",
        "forward_5000_reverse_5000",
        "reverse_1000_to_5000_at_forward_5000",
    ),
    (
        "forward_1000_reverse_5000",
        "forward_5000_reverse_5000",
        "forward_1000_to_5000_at_reverse_5000",
    ),
    (
        "forward_1000_reverse_1000",
        "forward_5000_reverse_5000",
        "both_1000_to_5000",
    ),
    (
        "forward_5000_reverse_1000",
        "forward_1000_reverse_5000",
        "mixed_pair_contrast",
    ),
)
KEY_FIELDS = ("conversation_id", "domain", "starting_model", "receiver", "turn")


def row_key(row: dict) -> tuple:
    return tuple(row[field] for field in KEY_FIELDS)


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    index = {row_key(row): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate row identity")
    return index


def load_cell(raw_path: Path, native_path: Path) -> dict[tuple, dict]:
    translated = read_jsonl(raw_path)
    native = read_jsonl(native_path)
    native_index = {
        (row["conversation_id"], row["starting_model"], row["turn"]): row
        for row in native
    }
    joined = attach_native_reference(translated, native)
    for row in joined:
        reference = native_index[
            (row["conversation_id"], row["starting_model"], row["turn"])
        ]
        row["native_trajectory_raw_text"] = reference["native_trajectory_raw_text"]
        row["native_trajectory_token_ids"] = reference["native_trajectory_token_ids"]
    return index_rows(joined)


def prompt_history_hash(index: dict[tuple, dict]) -> str:
    digest = hashlib.sha256()
    for key, row in sorted(index.items()):
        digest.update(json.dumps(key, separators=(",", ":")).encode())
        digest.update(row["prompt_token_ids_sha256"].encode())
    return digest.hexdigest()


def turn1_control(cells: dict[str, dict[tuple, dict]]) -> dict:
    fields = (
        "prompt_token_ids_sha256",
        "translated_raw_text",
        "translated_token_ids",
        "translated_f1",
        "same_transcript_native_raw_text",
        "same_transcript_native_token_ids",
        "same_transcript_native_f1",
        "native_trajectory_raw_text",
        "native_trajectory_token_ids",
        "native_trajectory_f1",
    )
    mismatches = []
    baseline = cells[CHECKPOINT_PAIRS[0]]
    for key in sorted(baseline):
        if key[-1] != 1:
            continue
        for cell in CHECKPOINT_PAIRS[1:]:
            changed = [
                field
                for field in fields
                if cells[cell][key][field] != baseline[key][field]
            ]
            if changed:
                mismatches.append({"key": list(key), "cell": cell, "fields": changed})
    return {
        "rows": sum(key[-1] == 1 for key in baseline),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def paired_comparison(
    before: dict[tuple, dict],
    after: dict[tuple, dict],
    *,
    start: str,
    reference_field: str,
    draws: int,
    seed: int,
) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for key in sorted(before):
        row_before = before[key]
        if row_before["starting_model"] != start:
            continue
        row_after = after[key]
        grouped[row_before["domain"]][row_before["conversation_id"]].append(
            (
                row_after["translated_f1"] - row_before["translated_f1"],
                row_after[reference_field] - row_before[reference_field],
                (row_after[reference_field] - row_after["translated_f1"])
                - (row_before[reference_field] - row_before["translated_f1"]),
            )
        )
    if set(grouped) != set(coqa.DOMAINS):
        raise ValueError("paired comparison does not contain every frozen domain")

    domain_arrays = []
    domain_point = {}
    for domain in coqa.DOMAINS:
        conversations = grouped[domain]
        values = np.asarray(
            [np.mean(conversations[name], axis=0) for name in sorted(conversations)]
        )
        domain_arrays.append(values)
        domain_point[domain] = {
            "conversations": len(values),
            "treatment_f1_change": float(values[:, 0].mean()),
            "reference_f1_change": float(values[:, 1].mean()),
            "deficit_change": float(values[:, 2].mean()),
        }
    point = np.mean([values.mean(axis=0) for values in domain_arrays], axis=0)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 3), dtype=np.float64)
    for draw in range(draws):
        samples[draw] = np.mean(
            [
                values[rng.integers(0, len(values), size=len(values))].mean(axis=0)
                for values in domain_arrays
            ],
            axis=0,
        )
    names = ("treatment_f1_change", "reference_f1_change", "deficit_change")
    return {
        "estimand": "after minus before, paired by complete conversation; equal weight per domain",
        "draws": draws,
        "seed": seed,
        "point": {name: float(value) for name, value in zip(names, point)},
        "interval_95": {
            name: [
                float(np.quantile(samples[:, position], 0.025)),
                float(np.quantile(samples[:, position], 0.975)),
            ]
            for position, name in enumerate(names)
        },
        "by_domain": domain_point,
    }


def analyze(cells: dict[str, dict[tuple, dict]], *, draws: int = 10_000) -> dict:
    if set(cells) != set(CHECKPOINT_PAIRS):
        raise ValueError(
            f"expected exactly {CHECKPOINT_PAIRS}, got {tuple(sorted(cells))}"
        )
    baseline_keys = set(cells[CHECKPOINT_PAIRS[0]])
    if any(set(cells[name]) != baseline_keys for name in CHECKPOINT_PAIRS[1:]):
        raise ValueError("checkpoint-pair row identities differ")
    comparisons = {}
    seed = 20261005
    for before, after, label in PAIRS:
        by_condition = {}
        for start in STARTING_MODELS:
            for reference, reference_field in REFERENCES.items():
                by_condition[f"{start}/{reference}"] = paired_comparison(
                    cells[before],
                    cells[after],
                    start=start,
                    reference_field=reference_field,
                    draws=draws,
                    seed=seed,
                )
                seed += 1
        comparisons[f"{before}_to_{after}"] = {
            "label": label,
            "before": before,
            "after": after,
            "conditions": by_condition,
        }
    control = turn1_control(cells)
    return {
        "schema": "kv_lingo_checkpoint_grid_paired_analysis_v1",
        "rows_per_cell": len(baseline_keys),
        "prompt_history_sha256": {
            name: prompt_history_hash(cells[name]) for name in CHECKPOINT_PAIRS
        },
        "turn1_control": control,
        "paired_comparisons": comparisons,
        "passed_integrity_checks": control["passed"],
        "Generated-by": "OpenAI Codex",
    }


def parse_cell(value: str) -> tuple[str, tuple[Path, Path]]:
    try:
        name, paths = value.split("=", 1)
        raw, native = paths.split(",", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected CHECKPOINT_PAIR=RAW.jsonl,NATIVE.jsonl"
        ) from error
    if name not in CHECKPOINT_PAIRS:
        raise argparse.ArgumentTypeError(f"unknown checkpoint pair {name!r}")
    return name, (Path(raw), Path(native))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", type=parse_cell, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = dict(args.cell)
    if len(paths) != len(args.cell):
        parser.error("duplicate --cell name")
    cells = {name: load_cell(*paths[name]) for name in CHECKPOINT_PAIRS}
    write_json(args.out, analyze(cells, draws=args.draws))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
