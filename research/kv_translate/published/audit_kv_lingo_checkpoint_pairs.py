#!/usr/bin/env python3
"""Audit saved step-1,000/5,000 retained KV-Lingo evaluations.

This is the CPU gate for the mixed-checkpoint experiment.  It hashes the
actual checkpoint payloads, regrades every persisted answer, reconstructs the
token histories, validates the ownership ledgers, rebuilds the frozen quality
summaries, and compares the two checkpoint pairs by conversation.

The script is deliberately specific to the corrected Qwen3-4B/8B study.
It does not generate model outputs and it never opens the untouched reserve.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import coqa
from .kv_lingo_eval import summarize

PINS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}
STAGE2_STEPS = 5_000
WARMUP_STEPS = 250
EFFECTIVE_BATCH = 8
LEARNING_RATE = 3e-5
CAPTURE_CUT = "pre-k_norm keys and v_proj values; target k_norm then RoPE"
LOSS_REDUCTION = "mean_token_forward_kl_then_mean_of_8_samples"
MAP_ARCHITECTURE = "per_layer_dense_key_and_value_maps_fp32"
DISTRIBUTED_EXECUTION_SCHEMA = "kv_lingo_distributed_execution_v1"
DISTRIBUTED_ASSIGNMENT = "length_balanced_within_fixed_global_batch_v1"


EXPECTED_SOURCE_COMMIT = "7222b884e0bf52ef8fa5f147894c21becff845c5"
EXPECTED_IMPLEMENTATION_SHA256 = (
    "1354b2c8199508615e75d3b1be3fd8ae1e7597e8b2927739a476619da9fb4e86"
)
EXPECTED_CHECKPOINTS = {
    "F1000": {
        "direction": "4b-to-8b",
        "step": 1000,
        "directory": "checkpoints-v2/forward-step1000",
        "payload": "stage2-4b-to-8b-step1000.pt",
        "receipt": "STAGE2_4B-TO-8B_STEP1000.json",
        "contract": "RESUME_CONTRACT_4B-TO-8B_STEP1000.json",
        "sha256": "25d55491ed7e8aa60650dd998f4e5da81e4c33cee651ffda18787634896eb051",
    },
    "R1000": {
        "direction": "8b-to-4b",
        "step": 1000,
        "directory": "checkpoints-v2/reverse-step1000",
        "payload": "stage2-8b-to-4b-step1000.pt",
        "receipt": "STAGE2_8B-TO-4B_STEP1000.json",
        "contract": "RESUME_CONTRACT_8B-TO-4B_STEP1000.json",
        "sha256": "37aed10cc47af43cfacc22a2257a73f8b053a82998c8f8774056795e7ab81afb",
    },
    "F5000": {
        "direction": "4b-to-8b",
        "step": 5000,
        "directory": "checkpoints-v2/forward-step5000",
        "payload": "stage2-4b-to-8b-step5000.pt",
        "receipt": "STAGE2_4B-TO-8B_STEP5000.json",
        "contract": "RESUME_CONTRACT_4B-TO-8B_STEP5000.json",
        "sha256": "2c83eb2fc27bd5a6bfc5cf88c89245fb0ef240b65d09a619ca963ffa2e89c963",
    },
    "R5000": {
        "direction": "8b-to-4b",
        "step": 5000,
        "directory": "checkpoints-v2/reverse-step5000",
        "payload": "stage2-8b-to-4b-step5000.pt",
        "receipt": "STAGE2_8B-TO-4B_STEP5000.json",
        "contract": "RESUME_CONTRACT_8B-TO-4B_STEP5000.json",
        "sha256": "98b58a1f15e40d04616e429a3ea76738a7bc42cc080864458a226494f2d60f68",
    },
}
RUNS = {
    "forward_1000_reverse_1000": "results/forward-1000-reverse-1000",
    "forward_5000_reverse_5000": "results/forward-5000-reverse-5000",
}
REFERENCES = {
    "same_transcript_native": "same_transcript_native",
    "alternating_native_trajectory": "native_trajectory",
}
KEY_FIELDS = ("conversation_id", "starting_model", "turn")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def ids_sha256(ids: Iterable[int]) -> str:
    return hashlib.sha256(
        b"".join(int(token).to_bytes(8, "little", signed=False) for token in ids)
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict) -> tuple[str, str, int]:
    return tuple(row[field] for field in KEY_FIELDS)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def equal(actual: Any, expected: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(
            float(actual), float(expected), rel_tol=0, abs_tol=tolerance
        )
    return actual == expected


class Audit:
    def __init__(self):
        self.failures: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []

    def require(self, condition: bool, label: str, **details: Any) -> None:
        if not condition:
            self.failures.append({"check": label, **details})

    def require_equal(self, actual: Any, expected: Any, label: str) -> None:
        self.require(
            equal(actual, expected),
            label,
            actual=actual,
            expected=expected,
        )

    def require_file(self, path: Path, label: str) -> bool:
        if path.is_file():
            return True
        self.gaps.append({"artifact": label, "path": str(path)})
        return False


def input_receipt(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def checkpoint_audit(root: Path, audit: Audit) -> tuple[dict, list[dict]]:
    results = {}
    inputs = []
    data_paths = {
        "train_rows": root / "frozen-v2/train.jsonl",
        "train_tokens": root / "frozen-v2/train.tokens.u32",
        "validation_rows": root / "frozen-v2/validation.jsonl",
        "validation_tokens": root / "frozen-v2/validation.tokens.u32",
    }
    data_hashes = {}
    for label, path in data_paths.items():
        if audit.require_file(path, label):
            receipt = input_receipt(root, path)
            inputs.append(receipt)
            data_hashes[label] = receipt["sha256"]

    for label, spec in EXPECTED_CHECKPOINTS.items():
        directory = root / spec["directory"]
        paths = {
            "payload": directory / spec["payload"],
            "receipt": directory / spec["receipt"],
            "contract": directory / spec["contract"],
            "marker": directory / "COPIED_TO_PRUNE.sha256",
        }
        if not all(
            audit.require_file(path, f"{label}:{name}") for name, path in paths.items()
        ):
            continue
        records = {name: input_receipt(root, path) for name, path in paths.items()}
        inputs.extend(records.values())
        receipt = load_json(paths["receipt"])
        contract = load_json(paths["contract"])
        marker = paths["marker"].read_text(encoding="utf-8").strip()
        actual = records["payload"]["sha256"]
        audit.require_equal(actual, spec["sha256"], f"{label}:payload expected hash")
        audit.require_equal(marker, actual, f"{label}:copy marker")
        audit.require_equal(
            receipt["checkpoint"]["sha256"], actual, f"{label}:receipt payload hash"
        )
        audit.require_equal(
            receipt["checkpoint"]["path"],
            spec["payload"],
            f"{label}:receipt payload path",
        )
        audit.require_equal(
            receipt["resume_contract"]["sha256"],
            records["contract"]["sha256"],
            f"{label}:contract hash",
        )
        audit.require_equal(
            receipt["direction"], spec["direction"], f"{label}:direction"
        )
        audit.require_equal(receipt["global_step"], spec["step"], f"{label}:step")
        audit.require_equal(
            receipt["sample_cursor"], spec["step"] * EFFECTIVE_BATCH, f"{label}:cursor"
        )
        topology = receipt["execution_topology"]
        for field, expected in {
            "schema": DISTRIBUTED_EXECUTION_SCHEMA,
            "world_size": 4,
            "backend": "nccl",
            "assignment": DISTRIBUTED_ASSIGNMENT,
            "global_batch": EFFECTIVE_BATCH,
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        }.items():
            audit.require_equal(
                topology.get(field), expected, f"{label}:topology:{field}"
            )
        for field, expected in {
            "direction": spec["direction"],
            "global_step": spec["step"],
            "sample_cursor": spec["step"] * EFFECTIVE_BATCH,
            "model_revisions": PINS,
            "tokenizer_revision": PINS["Qwen/Qwen3-4B"],
            "capture_cut": CAPTURE_CUT,
            "map_architecture": MAP_ARCHITECTURE,
            "loss_reduction": LOSS_REDUCTION,
            "effective_batch": EFFECTIVE_BATCH,
            "warmup_steps": WARMUP_STEPS,
            "total_schedule_steps": STAGE2_STEPS,
            "peak_learning_rate": LEARNING_RATE,
            "train_rows_sha256": data_hashes.get("train_rows"),
            "train_tokens_sha256": data_hashes.get("train_tokens"),
            "validation_rows_sha256": data_hashes.get("validation_rows"),
            "validation_tokens_sha256": data_hashes.get("validation_tokens"),
        }.items():
            audit.require_equal(
                contract.get(field), expected, f"{label}:contract:{field}"
            )
        audit.require_equal(
            contract.get("checkpoint_sha256"),
            actual,
            f"{label}:contract checkpoint hash",
        )
        audit.require_equal(
            contract.get("execution_topology"), topology, f"{label}:contract topology"
        )
        results[label] = {
            "direction": spec["direction"],
            "step": spec["step"],
            "payload": records["payload"],
            "receipt": records["receipt"],
            "contract": records["contract"],
            "marker": records["marker"],
            "execution_topology": topology,
        }
    return results, inputs


def regrade_fields(row: dict, prefix: str, references: list[str]) -> dict:
    score = coqa.turn_score(references, row[f"{prefix}_answer"])
    health = coqa.health_events(
        row[f"{prefix}_answer"],
        row[f"{prefix}_raw_text"],
        stop_reason=row[f"{prefix}_stop_reason"],
        new_tokens=len(row[f"{prefix}_token_ids"]),
    )
    return {"f1": score["f1"], "em": score["em"], "health": health}


def verify_scored_row(
    row: dict,
    prefix: str,
    references: list[str],
    audit: Audit,
    label: str,
) -> None:
    rebuilt = regrade_fields(row, prefix, references)
    for field, value in rebuilt.items():
        audit.require_equal(
            row[f"{prefix}_{field}"], value, f"{label}:{prefix}:{field}"
        )


def reconstruct_history(
    story: dict, rows: list[dict], prefix: str, audit: Audit, label: str
) -> list[dict]:
    ledger: list[int] = []
    prior_stop = None
    result = []
    for expected_turn, row in enumerate(sorted(rows, key=lambda item: item["turn"]), 1):
        audit.require_equal(row["turn"], expected_turn, f"{label}:turn sequence")
        suffix = (
            list(story["turn_1_token_ids"])
            if expected_turn == 1
            else list(
                story["next_turn_suffix_token_ids"][str(expected_turn)][
                    "after_eos" if prior_stop == "eos" else "after_other"
                ]
            )
        )
        ledger.extend(suffix)
        digest = ids_sha256(ledger)
        audit.require_equal(
            row["prompt_token_ids_sha256"],
            digest,
            f"{label}:turn{expected_turn}:prompt hash",
        )
        result.append(
            {
                "turn": expected_turn,
                "prompt_tokens": len(ledger),
                "prompt_sha256": digest,
            }
        )
        ledger.extend(row[f"{prefix}_token_ids"])
        prior_stop = row[f"{prefix}_stop_reason"]
    return result


def verify_ownership(record: dict, audit: Audit, label: str) -> dict:
    audit.require_equal(record.get("models"), ["4B", "8B"], f"{label}:models")
    spans = record.get("spans", [])
    translations = record.get("translations", [])
    translated = set()
    expected_start = 0
    by_serial = {}
    for serial, span in enumerate(spans):
        audit.require_equal(span.get("serial"), serial, f"{label}:span serial")
        audit.require_equal(span.get("start"), expected_start, f"{label}:span start")
        audit.require(
            span.get("writer") in ("4B", "8B"), f"{label}:span writer", span=span
        )
        audit.require(
            span.get("end", 0) > span.get("start", 0),
            f"{label}:positive span",
            span=span,
        )
        expected_start = span.get("end", expected_start)
        by_serial[serial] = span
    for item in translations:
        pair = (item.get("span_serial"), item.get("receiver"))
        audit.require(pair not in translated, f"{label}:unique translation", pair=pair)
        translated.add(pair)
        span = by_serial.get(item.get("span_serial"))
        audit.require(span is not None, f"{label}:known translated span", item=item)
        if span is not None:
            audit.require(
                item.get("receiver") != span["writer"],
                f"{label}:opposite receiver",
                item=item,
            )
    coverage = {}
    for model in ("4B", "8B"):
        frontier = 0
        for span in spans:
            available = span["writer"] == model or (span["serial"], model) in translated
            if not available:
                break
            audit.require_equal(
                span["start"], frontier, f"{label}:{model}:contiguous coverage"
            )
            frontier = span["end"]
        coverage[model] = frontier
    audit.require_equal(
        record.get("total_tokens"), expected_start, f"{label}:total tokens"
    )
    audit.require_equal(record.get("line_covered"), coverage, f"{label}:line coverage")
    missing = sorted(
        span["serial"]
        for span in spans
        if (span["serial"], "4B" if span["writer"] == "8B" else "8B") not in translated
    )
    audit.require_equal(
        record.get("remaining_auxiliary_spans"), missing, f"{label}:remaining auxiliary"
    )
    return {
        "spans": len(spans),
        "translations": len(translations),
        "total_tokens": expected_start,
        "line_covered": coverage,
        "remaining_auxiliary_spans": missing,
    }


def deep_compare(actual: Any, expected: Any, audit: Audit, label: str) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        audit.require_equal(sorted(actual), sorted(expected), f"{label}:keys")
        for name in sorted(set(actual) & set(expected)):
            deep_compare(actual[name], expected[name], audit, f"{label}.{name}")
    elif isinstance(expected, list) and isinstance(actual, list):
        audit.require_equal(len(actual), len(expected), f"{label}:length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            deep_compare(left, right, audit, f"{label}[{index}]")
    else:
        audit.require_equal(actual, expected, label)


def run_audit(
    root: Path, label: str, relative: str, stories: dict, audit: Audit
) -> tuple[dict, list[dict]]:
    directory = root / relative
    paths = {
        "raw": directory / "run/RAW.jsonl",
        "native": directory / "run/NATIVE_TRAJECTORY.jsonl",
        "ownership": directory / "run/OWNERSHIP.jsonl",
        "result": directory / "run/RESULT.json",
        "summary": directory / "SUMMARY.json",
    }
    for name, path in paths.items():
        if not audit.require_file(path, f"{label}:{name}"):
            return {}, []
    inputs = [input_receipt(root, path) for path in paths.values()]
    raw = read_jsonl(paths["raw"])
    native = read_jsonl(paths["native"])
    ownership = read_jsonl(paths["ownership"])
    expected_keys = {
        (conversation_id, start, turn)
        for conversation_id in stories
        for start in ("4B", "8B")
        for turn in range(1, 11)
    }
    raw_index = {key(row): row for row in raw}
    native_index = {key(row): row for row in native}
    audit.require_equal(len(raw_index), len(raw), f"{label}:unique raw keys")
    audit.require_equal(len(native_index), len(native), f"{label}:unique native keys")
    audit.require_equal(set(raw_index), expected_keys, f"{label}:complete raw join")
    audit.require_equal(
        set(native_index), expected_keys, f"{label}:complete native join"
    )
    ownership_index = {
        (row["conversation_id"], row["starting_model"]): row for row in ownership
    }
    expected_ownership = {
        (conversation_id, start)
        for conversation_id in stories
        for start in ("4B", "8B")
    }
    audit.require_equal(
        len(ownership_index), len(ownership), f"{label}:unique ownership keys"
    )
    audit.require_equal(
        set(ownership_index), expected_ownership, f"{label}:complete ownership join"
    )

    prompt_metadata = {}
    ownership_summaries = {}
    for conversation_id, story in stories.items():
        for start in ("4B", "8B"):
            trajectory = [
                raw_index[(conversation_id, start, turn)] for turn in range(1, 11)
            ]
            native_trajectory = [
                native_index[(conversation_id, start, turn)] for turn in range(1, 11)
            ]
            for row in trajectory:
                expected_receiver = (
                    start if row["turn"] % 2 else ("8B" if start == "4B" else "4B")
                )
                row_label = f"{label}:{conversation_id}:{start}:turn{row['turn']}"
                audit.require_equal(
                    row["receiver"], expected_receiver, f"{row_label}:receiver"
                )
                audit.require_equal(
                    row["domain"], story["source"], f"{row_label}:domain"
                )
                references = coqa.answer_references(story, row["turn"])
                verify_scored_row(row, "translated", references, audit, row_label)
                verify_scored_row(
                    row, "same_transcript_native", references, audit, row_label
                )
                native_row = native_index[key(row)]
                audit.require_equal(
                    native_row["receiver"],
                    expected_receiver,
                    f"{row_label}:native receiver",
                )
                audit.require_equal(
                    native_row["domain"], story["source"], f"{row_label}:native domain"
                )
                verify_scored_row(
                    native_row, "native_trajectory", references, audit, row_label
                )
            translated_history = reconstruct_history(
                story,
                trajectory,
                "translated",
                audit,
                f"{label}:{conversation_id}:{start}:translated",
            )
            native_history = reconstruct_history(
                story,
                native_trajectory,
                "native_trajectory",
                audit,
                f"{label}:{conversation_id}:{start}:native",
            )
            prompt_metadata[(conversation_id, start)] = {
                "translated": translated_history,
                "native": native_history,
            }
            ownership_summary = verify_ownership(
                ownership_index[(conversation_id, start)],
                audit,
                f"{label}:{conversation_id}:{start}:ownership",
            )
            last = trajectory[-1]
            audit.require_equal(
                last["cache_line_tokens"],
                ownership_summary["line_covered"],
                f"{label}:{conversation_id}:{start}:final cache lines",
            )
            ownership_summaries[(conversation_id, start)] = ownership_summary

    rebuilt = summarize(raw, native)
    recorded = load_json(paths["summary"])
    deep_compare(rebuilt, recorded, audit, f"{label}:summary rebuild")
    result = load_json(paths["result"])
    audit.require_equal(
        result.get("conversations"), len(stories), f"{label}:result conversations"
    )
    audit.require_equal(result.get("raw_rows"), len(raw), f"{label}:result raw rows")
    audit.require_equal(
        result.get("native_trajectory_rows"), len(native), f"{label}:result native rows"
    )
    audit.require_equal(
        result.get("ownership_rows"), len(ownership), f"{label}:result ownership rows"
    )

    joined = []
    for row in raw:
        native_row = native_index[key(row)]
        meta = prompt_metadata[(row["conversation_id"], row["starting_model"])][
            "translated"
        ][row["turn"] - 1]
        joined.append(
            {
                **row,
                "native_trajectory_f1": native_row["native_trajectory_f1"],
                "native_trajectory_health": native_row["native_trajectory_health"],
                "native_trajectory_answer": native_row["native_trajectory_answer"],
                "native_trajectory_token_ids": native_row[
                    "native_trajectory_token_ids"
                ],
                "native_trajectory_prompt_token_ids_sha256": native_row[
                    "prompt_token_ids_sha256"
                ],
                "prompt_tokens": meta["prompt_tokens"],
            }
        )
    health = {}
    for start in ("4B", "8B"):
        selected = [row for row in joined if row["starting_model"] == start]
        health[start] = {}
        for name, prefix in REFERENCES.items():
            health[start][name] = {
                "rows": len(selected),
                "treatment_unhealthy": sum(
                    bool(row["translated_health"]) for row in selected
                ),
                "reference_unhealthy": sum(
                    bool(row[f"{prefix}_health"]) for row in selected
                ),
            }
    return {
        "input_receipts": inputs,
        "recorded_result": result,
        "rebuilt_summary": rebuilt,
        "absolute_health_counts": health,
        "ownership": {
            "records": len(ownership_summaries),
            "total_spans": sum(item["spans"] for item in ownership_summaries.values()),
            "total_translations": sum(
                item["translations"] for item in ownership_summaries.values()
            ),
        },
    }, joined


def domain_equal_metrics(rows: list[dict], reference: str) -> dict:
    by_domain = {}
    for domain in coqa.DOMAINS:
        selected = [row for row in rows if row["domain"] == domain]
        treatment = mean(row["translated_f1"] for row in selected)
        native = mean(row[f"{reference}_f1"] for row in selected)
        by_domain[domain] = {
            "conversations": len({row["conversation_id"] for row in selected}),
            "rows": len(selected),
            "treatment_f1": treatment,
            "reference_f1": native,
            "reference_minus_treatment_f1": native - treatment,
        }
    return {
        "treatment_f1": mean(value["treatment_f1"] for value in by_domain.values()),
        "reference_f1": mean(value["reference_f1"] for value in by_domain.values()),
        "reference_minus_treatment_f1": mean(
            value["reference_minus_treatment_f1"] for value in by_domain.values()
        ),
        "by_domain": by_domain,
    }


def paired_checkpoint_bootstrap(
    old: list[dict], new: list[dict], reference: str, draws: int = 10_000
) -> dict:
    indexes = [{key(row): row for row in rows} for rows in (old, new)]
    grouped = defaultdict(list)
    for row_key in sorted(set(indexes[0]) & set(indexes[1])):
        before, after = indexes[0][row_key], indexes[1][row_key]
        grouped[before["domain"], before["conversation_id"]].append(
            (after[f"{reference}_f1"] - after["translated_f1"])
            - (before[f"{reference}_f1"] - before["translated_f1"])
        )
    rng = np.random.default_rng(20261005)
    samples = []
    for _ in range(draws):
        domains = []
        for domain in coqa.DOMAINS:
            ids = sorted(
                conversation_id
                for item_domain, conversation_id in grouped
                if item_domain == domain
            )
            picked = rng.choice(ids, size=len(ids), replace=True)
            domains.append(
                mean(
                    value
                    for conversation_id in picked
                    for value in grouped[domain, str(conversation_id)]
                )
            )
        samples.append(mean(domains))
    return {
        "draws": draws,
        "seed": 20261005,
        "unit": "complete conversations resampled within domain and paired by identity",
        "step5000_minus_step1000_deficit_interval_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def leave_one_out(rows: list[dict], reference: str) -> dict:
    result = {}
    for domain in coqa.DOMAINS:
        selected = [row for row in rows if row["domain"] == domain]
        ids = sorted({row["conversation_id"] for row in selected})
        deficits = []
        for omitted in ids:
            retained = [row for row in selected if row["conversation_id"] != omitted]
            deficits.append(
                {
                    "omitted": omitted,
                    "deficit": mean(
                        row[f"{reference}_f1"] - row["translated_f1"]
                        for row in retained
                    ),
                }
            )
        values = [item["deficit"] for item in deficits]
        result[domain] = {
            "conversations": len(ids),
            "full_deficit": mean(
                row[f"{reference}_f1"] - row["translated_f1"] for row in selected
            ),
            "minimum": min(values),
            "maximum": max(values),
            "passes_5pp_after_omission": sum(value <= 0.05 for value in values),
            "details": deficits,
        }
    return result


def correlation(rows: list[dict], reference: str, field: str) -> float | None:
    x = np.asarray([row[field] for row in rows], dtype=np.float64)
    y = np.asarray(
        [row[f"{reference}_f1"] - row["translated_f1"] for row in rows],
        dtype=np.float64,
    )
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def comparison_analysis(old: list[dict], new: list[dict], audit: Audit) -> dict:
    old_index, new_index = ({key(row): row for row in rows} for rows in (old, new))
    audit.require_equal(
        set(old_index), set(new_index), "checkpoint-pair row identities"
    )
    turn1_differences = []
    first_divergence = []
    for trajectory_key in sorted(
        {(row["conversation_id"], row["starting_model"]) for row in old}
    ):
        conversation_id, start = trajectory_key
        first = None
        for turn in range(1, 11):
            before = old_index[(conversation_id, start, turn)]
            after = new_index[(conversation_id, start, turn)]
            same_answer = (
                before["translated_token_ids"] == after["translated_token_ids"]
            )
            if turn == 1:
                fields = (
                    "prompt_token_ids_sha256",
                    "translated_token_ids",
                    "translated_answer",
                    "translated_f1",
                    "same_transcript_native_token_ids",
                    "same_transcript_native_f1",
                )
                changed = [field for field in fields if before[field] != after[field]]
                if changed:
                    turn1_differences.append(
                        {
                            "conversation_id": conversation_id,
                            "starting_model": start,
                            "fields": changed,
                        }
                    )
            if first is None and not same_answer:
                first = {
                    "conversation_id": conversation_id,
                    "domain": before["domain"],
                    "starting_model": start,
                    "turn": turn,
                    "receiver": before["receiver"],
                    "direction": (
                        "4b-to-8b" if before["receiver"] == "8B" else "8b-to-4b"
                    ),
                    "same_prompt_history": before["prompt_token_ids_sha256"]
                    == after["prompt_token_ids_sha256"],
                    "prompt_tokens_step1000": before["prompt_tokens"],
                    "prompt_tokens_step5000": after["prompt_tokens"],
                    "catch_up_tokens_step1000": before["catch_up"]["tokens"],
                    "catch_up_tokens_step5000": after["catch_up"]["tokens"],
                }
        if first is not None:
            first_divergence.append(first)
    audit.require(
        not turn1_differences,
        "turn 1 differs despite no incoming translation",
        differences=turn1_differences,
    )

    cells = {}
    for start in ("4B", "8B"):
        cells[start] = {}
        old_start = [row for row in old if row["starting_model"] == start]
        new_start = [row for row in new if row["starting_model"] == start]
        for name, reference in REFERENCES.items():
            old_metrics = domain_equal_metrics(old_start, reference)
            new_metrics = domain_equal_metrics(new_start, reference)
            same_prompt = sum(
                old_index[key(row)]["prompt_token_ids_sha256"]
                == row["prompt_token_ids_sha256"]
                for row in new_start
            )
            cells[start][name] = {
                "step1000": old_metrics,
                "step5000": new_metrics,
                "step5000_minus_step1000": {
                    "treatment_f1": new_metrics["treatment_f1"]
                    - old_metrics["treatment_f1"],
                    "reference_f1": new_metrics["reference_f1"]
                    - old_metrics["reference_f1"],
                    "deficit": new_metrics["reference_minus_treatment_f1"]
                    - old_metrics["reference_minus_treatment_f1"],
                },
                "paired_bootstrap": paired_checkpoint_bootstrap(
                    old_start, new_start, reference
                ),
                "leave_one_conversation_out": {
                    "step1000": leave_one_out(old_start, reference),
                    "step5000": leave_one_out(new_start, reference),
                },
                "same_prompt_history_rows": same_prompt,
                "changed_prompt_history_rows": len(new_start) - same_prompt,
                "pearson_deficit_vs_prompt_tokens": {
                    "step1000": correlation(old_start, reference, "prompt_tokens"),
                    "step5000": correlation(new_start, reference, "prompt_tokens"),
                },
            }

    by_receiver_domain = {}
    for receiver in ("4B", "8B"):
        for domain in coqa.DOMAINS:
            name = f"{receiver}:{domain}"
            by_receiver_domain[name] = {}
            for cell, rows in (
                ("forward_1000_reverse_1000", old),
                ("forward_5000_reverse_5000", new),
            ):
                selected = [
                    row
                    for row in rows
                    if row["receiver"] == receiver and row["domain"] == domain
                ]
                by_receiver_domain[name][cell] = {
                    "rows": len(selected),
                    "same_transcript_native_minus_translated_f1": mean(
                        row["same_transcript_native_f1"] - row["translated_f1"]
                        for row in selected
                    ),
                    "native_history_effect_f1": mean(
                        row["native_trajectory_f1"] - row["same_transcript_native_f1"]
                        for row in selected
                    ),
                }
    return {
        "turn1_native_control": {
            "rows": sum(row["turn"] == 1 for row in old),
            "differences": turn1_differences,
            "passed": not turn1_differences,
        },
        "first_translated_answer_divergence": {
            "trajectories": len(
                {(row["conversation_id"], row["starting_model"]) for row in old}
            ),
            "trajectories_with_divergence": len(first_divergence),
            "records": first_divergence,
        },
        "cells": cells,
        "representation_and_history_decomposition": by_receiver_domain,
    }


def method_crosswalk(root: Path) -> dict:
    manifest = load_json(root / "frozen-v2/MANIFEST.json")
    return {
        "paper": {
            "version": "arXiv:2609.32610v1",
            "models": ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"],
            "model_revisions": "not reported",
            "translator": "separate token-wise linear key/value maps per target layer; corresponding layer",
            "capture": "pre-k-norm keys and pre-cache values; target normalization and RoPE after translation",
            "stage1": "fp64 moments over 400 conversations; symmetric eigensolve; relative tolerance 1e-8",
            "stage2": "forward-KL self-distillation; AdamW; 5000 steps; batch 8; 5% warmup; cosine; no weight decay; clip 1.0",
            "selected_learning_rate_4b_8b_both_directions": 3e-5,
            "training_stream": "author stream and exact sample identities not released",
            "multi_turn": "10-turn reasoning-off CoQA; both starting models; three translator seeds; retain one cache line per model and translate each new span at most once",
        },
        "implementation": {
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "models": PINS,
            "translator": MAP_ARCHITECTURE,
            "capture": CAPTURE_CUT,
            "stage1_samples": manifest["stage1"]["samples"],
            "stage1_prefix_positions": manifest["stage1"]["prefix_positions"],
            "stage1_tolerance": 1e-8,
            "stage2": {
                "objective": LOSS_REDUCTION,
                "optimizer": "AdamW",
                "steps": STAGE2_STEPS,
                "effective_batch": EFFECTIVE_BATCH,
                "warmup_steps": WARMUP_STEPS,
                "peak_learning_rate": LEARNING_RATE,
                "weight_decay": 0,
                "gradient_clip": 1.0,
            },
            "training_stream": {
                "dataset": manifest["dataset"],
                "construction": manifest["construction"],
                "method_identity": manifest["method_identity"],
            },
            "evaluation": {
                "cohort": "32 exposed development conversations; 6-7 per domain",
                "turns": 10,
                "reasoning": "off; empty Qwen3 thinking marker prefilled",
                "decode": "greedy; stop at EOS, newline, or 64-token cap",
                "references": [
                    "same generated transcript",
                    "independently evolving alternating native trajectory",
                ],
                "scorer": "official CoQA leave-one-reference-out token F1",
                "seeds": 1,
            },
        },
        "material_differences": [
            "The exact author training row manifest and loader were unavailable, so the stream is an independent construction.",
            "The paper reports three translator seeds for this pair; this study trained one seed.",
            "The 32-conversation development cohort and strict per-domain/reference gates are local qualification rules, not the paper's reported aggregate protocol.",
            "The independent native-trajectory reference is an added local robustness test.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="root containing frozen-v2, coqa, checkpoints-v2, and results",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = Audit()

    cohort_path = root / "coqa/DEVELOPMENT_CONVERSATIONS.json"
    manifest_path = root / "coqa/COHORT_MANIFEST.json"
    frozen_manifest_path = root / "frozen-v2/MANIFEST.json"
    for label, path in {
        "development cohort": cohort_path,
        "cohort manifest": manifest_path,
        "frozen stream manifest": frozen_manifest_path,
    }.items():
        audit.require_file(path, label)
    if audit.gaps:
        write_json(
            args.out_dir / "CHECKPOINT_AUDIT.json",
            {
                "state": "MISSING_ARTIFACT",
                "artifact_gaps": audit.gaps,
                "Generated-by": "OpenAI Codex",
            },
        )
        return 2

    stories_list = load_json(cohort_path)
    stories = {story["id"]: story for story in stories_list}
    audit.require_equal(
        len(stories), len(stories_list), "unique development conversation ids"
    )
    audit.require_equal(len(stories), 32, "development conversation count")
    audit.require_equal(
        {
            domain: sum(story["source"] == domain for story in stories_list)
            for domain in coqa.DOMAINS
        },
        load_json(manifest_path)["development"]["by_domain"],
        "development domain counts",
    )

    checkpoint_results, checkpoint_inputs = checkpoint_audit(root, audit)
    run_results = {}
    joined = {}
    all_inputs = checkpoint_inputs + [
        input_receipt(root, cohort_path),
        input_receipt(root, manifest_path),
        input_receipt(root, frozen_manifest_path),
    ]
    for label, relative in RUNS.items():
        run_results[label], joined[label] = run_audit(
            root, label, relative, stories, audit
        )
        all_inputs.extend(run_results[label].pop("input_receipts", []))

    comparison = (
        comparison_analysis(
            joined.get("forward_1000_reverse_1000", []),
            joined.get("forward_5000_reverse_5000", []),
            audit,
        )
        if all(joined.get(pair) for pair in RUNS)
        else {}
    )
    crosswalk = method_crosswalk(root)
    state = (
        "AUDIT_PASS"
        if not audit.gaps and not audit.failures
        else ("MISSING_ARTIFACT" if audit.gaps else "AUDIT_FAILED")
    )
    report = {
        "schema": "kv_lingo_checkpoint_pair_audit_v1",
        "state": state,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "archive_root": str(root),
        "checkpoints": checkpoint_results,
        "runs": run_results,
        "paired_checkpoint_analysis": comparison,
        "method_crosswalk": crosswalk,
        "input_manifest": sorted(all_inputs, key=lambda item: item["path"]),
        "artifact_gaps": audit.gaps,
        "failures": audit.failures,
        "reserve_opened": False,
        "gpu_work_performed": False,
        "Generated-by": "OpenAI Codex",
    }
    write_json(args.out_dir / "CHECKPOINT_AUDIT.json", report)
    write_json(args.out_dir / "METHOD_CROSSWALK.json", crosswalk)
    for label, result in run_results.items():
        write_json(
            args.out_dir / f"{label}_REBUILT_SUMMARY.json", result["rebuilt_summary"]
        )
    write_json(args.out_dir / "PAIRED_CHECKPOINT_ANALYSIS.json", comparison)
    print(
        json.dumps(
            {"state": state, "failures": len(audit.failures), "gaps": len(audit.gaps)},
            indent=2,
        )
    )
    return 0 if state == "AUDIT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
