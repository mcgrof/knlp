#!/usr/bin/env python3
"""Freeze CPU inputs for the CoQA reusable-passage-cache experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from . import coqa, coqa_reuse
from .freeze_coqa import compact_story, file_sha256, row_for


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ids_sha256(ids):
    return hashlib.sha256(
        b"".join(int(token).to_bytes(8, "little", signed=False) for token in ids)
    ).hexdigest()


def add_cuts(row: dict, tokenizer) -> dict:
    encoded = tokenizer(
        row["prompt"], add_special_tokens=False, return_offsets_mapping=True
    )
    token_ids = list(encoded["input_ids"])
    if token_ids != row["token_ids"]:
        raise ValueError(f"token IDs changed for frozen row {row['row_id']}")
    cut = coqa_reuse.passage_cut(row["prompt"], list(encoded["offset_mapping"]))
    early = cut["early_cut"]
    late = len(token_ids) - 1
    if early >= late:
        raise ValueError(f"early cut is not before late cut for {row['row_id']}")
    return {
        **row,
        **cut,
        "late_cut": late,
        "early_prefix_token_ids_sha256": ids_sha256(token_ids[:early]),
        "early_suffix_token_ids_sha256": ids_sha256(token_ids[early:]),
        "late_prefix_token_ids_sha256": ids_sha256(token_ids[:late]),
        "late_suffix_token_ids_sha256": ids_sha256(token_ids[late:]),
    }


def artifact_info(path: Path):
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def git_commit_exists(repo: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-q0", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--guidance-commit", required=True)
    parser.add_argument("--source-repo")
    parser.add_argument("--gpu-rate", type=float, required=True)
    parser.add_argument("--cpu-rate", type=float, required=True)
    parser.add_argument("--memory-rate", type=float, required=True)
    args = parser.parse_args(argv)

    previous = Path(args.previous_q0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    data_path = previous / "inputs" / "coqa-dev-v1.0.json"
    evaluator_path = previous / "inputs" / "evaluate-v1.0.py"
    dataset = read_json(data_path)
    previous_conversations = read_json(previous / "FROZEN_CONVERSATIONS.json")
    previous_q1 = read_jsonl(previous / "Q1_FROZEN_ROWS.jsonl")
    previous_smoke = read_jsonl(previous / "SMOKE_FROZEN_ROWS.jsonl")

    from transformers import AutoTokenizer

    source_name = "Qwen/Qwen3-14B"
    source_revision = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    target_name = "Qwen/Qwen3-32B"
    target_revision = "9216db5781bf21249d130ec9da846c4624c16137"
    source_tok = AutoTokenizer.from_pretrained(source_name, revision=source_revision)
    target_tok = AutoTokenizer.from_pretrained(target_name, revision=target_revision)

    r1_rows = [add_cuts(row, target_tok) for row in previous_q1]
    smoke_rows = [add_cuts(row, target_tok) for row in previous_smoke]
    excluded_ids = {story["id"] for story in previous_conversations}
    replication = coqa_reuse.select_replication_conversations(
        dataset, excluded_ids, per_domain=20
    )
    r2_rows = [
        add_cuts(row_for(story, turn, source_tok, target_tok, "r2"), target_tok)
        for story in replication["selected"]
        for turn in coqa.Q1_TURNS
        if turn <= len(story["questions"])
    ]
    r2_conversations = [
        compact_story(story, ["r2"]) for story in replication["selected"]
    ]
    r2_conversations.sort(
        key=lambda row: (row["source"], row["selection_sha256"], row["id"])
    )

    row_paths = {
        "R1_FROZEN_ROWS.jsonl": r1_rows,
        "SMOKE_FROZEN_ROWS.jsonl": smoke_rows,
        "R2_FROZEN_ROWS.jsonl": r2_rows,
    }
    for name, rows in row_paths.items():
        write_jsonl(out / name, rows)
    write_json(out / "R2_FROZEN_CONVERSATIONS.json", r2_conversations)

    shared_prefix_checks = []
    for rows in (r1_rows, smoke_rows, r2_rows):
        by_conversation = {}
        for row in rows:
            by_conversation.setdefault(row["conversation_id"], []).append(row)
        for conversation_id, conversation_rows in by_conversation.items():
            cuts = {row["early_cut"] for row in conversation_rows}
            hashes = {row["early_prefix_token_ids_sha256"] for row in conversation_rows}
            shared_prefix_checks.append(
                {
                    "conversation_id": conversation_id,
                    "row_count": len(conversation_rows),
                    "one_early_cut": len(cuts) == 1,
                    "one_early_prefix_hash": len(hashes) == 1,
                    "early_cut": next(iter(cuts)) if len(cuts) == 1 else None,
                    "early_prefix_token_ids_sha256": (
                        next(iter(hashes)) if len(hashes) == 1 else None
                    ),
                }
            )
    if not all(
        row["one_early_cut"] and row["one_early_prefix_hash"]
        for row in shared_prefix_checks
    ):
        raise RuntimeError("turns do not share an identical passage prefix")

    boundary_protocol = {
        "schema": "coqa_passage_reuse_boundary_protocol_v1",
        "guidance_commit": args.guidance_commit,
        "inherited_protocol_sha256": file_sha256(previous / "PROTOCOL.json"),
        "models": {
            "source": {"name": source_name, "revision": source_revision},
            "receiver": {"name": target_name, "revision": target_revision},
        },
        "dtype": "bfloat16",
        "mapper": "inherited saved forward Full-Head k=8 coefficients; no fitting",
        "early_cut": (
            "largest token boundary at or before the first Question marker; "
            "a token crossing that character boundary remains in the suffix"
        ),
        "late_cut": "full frozen prompt except the final token",
        "full_prompt_identity": "preserve every inherited token ID exactly",
        "arms": [
            "native_early",
            "translated_early",
            "native_late",
            "translated_late",
        ],
        "reuse": (
            "one canonical 14B passage cache, one canonical 32B passage cache, "
            "and one mapped passage cache per conversation; independent clones "
            "for all three suffix branches"
        ),
        "segmentation": (
            "early arms consume the complete suffix in one receiver call; late "
            "arms extend a canonical-cache clone through the late prefix and "
            "then consume the final token"
        ),
        "turns": list(coqa.Q1_TURNS),
        "history": "primary official reference answers remain in the receiver suffix",
        "decoding": read_json(previous / "PROTOCOL.json")["decoding"],
        "statistics": {
            "bootstrap_unit": "complete conversations resampled within domain",
            "seed": coqa.BOOTSTRAP_SEED,
            "replicates": coqa.BOOTSTRAP_REPLICATES,
            "interval_use": "descriptive only",
        },
        "screens": {
            "native_cut_stability": {"mean": 0.01, "each_domain": 0.02, "health": 0.01},
            "late_positive_control": {
                "mean": 0.03,
                "each_domain": 0.05,
                "health": 0.02,
            },
            "early_reuse": {
                "mean": 0.03,
                "each_domain": 0.05,
                "health": 0.02,
                "turn_1": 0.03,
            },
        },
        "forbidden": [
            "mapper fitting or repair",
            "additional cuts, prompts, tasks, or model pairs",
            "recursive transfer",
            "storage or transport benchmarking",
            "outcome-driven cohort changes",
        ],
    }
    write_json(out / "BOUNDARY_PROTOCOL.json", boundary_protocol)

    cut_manifest = {
        "schema": "coqa_passage_reuse_cut_manifest_v1",
        "artifacts": {},
        "rows": {
            name: [
                {
                    key: row[key]
                    for key in (
                        "row_id",
                        "conversation_id",
                        "domain",
                        "turn",
                        "token_ids_sha256",
                        "early_cut",
                        "late_cut",
                        "first_question_character_offset",
                        "token_crossing_boundary",
                        "crossing_token_left_in_suffix",
                        "early_prefix_token_ids_sha256",
                        "early_suffix_token_ids_sha256",
                        "late_prefix_token_ids_sha256",
                        "late_suffix_token_ids_sha256",
                    )
                }
                for row in rows
            ]
            for name, rows in row_paths.items()
        },
        "shared_prefix_checks": shared_prefix_checks,
    }
    for name in row_paths:
        cut_manifest["artifacts"][name] = artifact_info(out / name)
    cut_manifest["artifacts"]["R2_FROZEN_CONVERSATIONS.json"] = artifact_info(
        out / "R2_FROZEN_CONVERSATIONS.json"
    )
    write_json(out / "CUT_MANIFEST.json", cut_manifest)

    previous_mapper = read_json(previous / "MAPPER_MANIFEST.json")
    previous_input = read_json(previous / "INPUT_MANIFEST.json")
    source_repo = (
        Path(args.source_repo)
        if args.source_repo
        else Path(__file__).resolve().parents[3]
    )
    commit_exists = git_commit_exists(source_repo, args.source_commit)
    if not commit_exists:
        raise ValueError(f"source commit {args.source_commit} is unavailable")
    source_manifest = {
        "schema": "coqa_passage_reuse_source_manifest_v1",
        "knlp_commit": args.source_commit,
        "commit_exists_in_source_repo": commit_exists,
        "inherited": {
            "protocol": artifact_info(previous / "PROTOCOL.json"),
            "input_manifest": artifact_info(previous / "INPUT_MANIFEST.json"),
            "scorer_check": artifact_info(previous / "SCORER_CHECK.json"),
            "official_data": artifact_info(data_path),
            "official_evaluator": artifact_info(evaluator_path),
            "forward_mapper": previous_mapper["forward"],
            "source_model": previous_input["models"]["source"],
            "target_model": previous_input["models"]["target"],
        },
        "entrypoints": [
            "research/kv_translate/published/freeze_coqa_reuse.py",
            "research/kv_translate/published/coqa_reuse.py",
            "research/kv_translate/published/coqa_reuse_runner.py",
            "research/kv_translate/published/coqa_reuse_modal_app.py",
        ],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "SOURCE_MANIFEST.json", source_manifest)

    replication_manifest = {
        "schema": "coqa_passage_reuse_replication_manifest_v1",
        "selection_salt": coqa.SELECTION_SALT,
        "ordering": "ascending SHA256(salt + conversation ID), then ID",
        "normalization": "Unicode casefold, collapse whitespace; no other changes",
        "excluded_previous_conversation_ids": sorted(excluded_ids),
        "eligible_unique_passages_by_domain": replication[
            "eligible_unique_passages_by_domain"
        ],
        "requested_per_domain": replication["requested_per_domain"],
        "sufficient": replication["sufficient"],
        "selected": [
            {
                "conversation_id": story["id"],
                "domain": story["source"],
                "selection_sha256": coqa.selection_digest(story["id"]),
                "raw_passage_sha256": coqa_reuse.passage_sha256(story["story"]),
                "normalized_passage_sha256": coqa_reuse.passage_sha256(
                    coqa_reuse.normalize_passage(story["story"])
                ),
            }
            for story in replication["selected"]
        ],
        "exclusions": replication["exclusions"],
        "rows": len(r2_rows),
        "maximum_answers": 2 * len(r2_rows),
    }
    write_json(out / "REPLICATION_MANIFEST.json", replication_manifest)

    combined_rate = args.gpu_rate + 4 * args.cpu_rate + 48 * args.memory_rate
    cost_forecast = {
        "schema": "coqa_passage_reuse_cost_forecast_v1",
        "provider": "modal",
        "resource": {"gpu": "H200", "cpu_cores": 4, "memory_gib": 48},
        "live_rates_usd_per_hour": {
            "gpu": args.gpu_rate,
            "cpu_per_core": args.cpu_rate,
            "memory_per_gib": args.memory_rate,
            "combined_requested": combined_rate,
        },
        "planning_gpu_hours_range": [0.5, 0.75],
        "planning_usd_range": [
            round(0.5 * combined_rate, 4),
            round(0.75 * combined_rate, 4),
        ],
        "answer_bounds": {
            "smoke": 4 * len(smoke_rows),
            "r1": 4 * len(r1_rows),
            "conditional_r2": 2 * len(r2_rows),
        },
        "first_allocation": "smoke only",
        "reforecast": "replace stage projections with measured smoke suffix work",
        "lifecycle_checkpoint_hours": 1,
        "watchdog": "stage-specific timeout and durable conversation checkpoints",
        "not_a_completion_time_or_spending_cap": True,
    }
    write_json(out / "COST_FORECAST.json", cost_forecast)

    cpu_checks = {
        "schema": "coqa_passage_reuse_cpu_checks_v1",
        "status": "PASS",
        "inherited_q1_token_ids_unchanged": all(
            row["token_ids"] == old["token_ids"]
            for row, old in zip(r1_rows, previous_q1, strict=True)
        ),
        "all_rows_cut_before_first_question": all(
            row["early_cut"] < row["late_cut"] and row["crossing_token_left_in_suffix"]
            for rows in row_paths.values()
            for row in rows
        ),
        "all_three_turns_share_early_prefix": all(
            row["one_early_cut"] and row["one_early_prefix_hash"]
            for row in shared_prefix_checks
        ),
        "r1_rows": len(r1_rows),
        "smoke_rows": len(smoke_rows),
        "r2_rows": len(r2_rows),
        "replication_inventory_sufficient": replication["sufficient"],
        "scorer_compatibility_inherited_pass": read_json(
            previous / "SCORER_CHECK.json"
        )["passed"],
        "gpu_hours": 0,
        "model_runs": 0,
    }
    write_json(out / "CPU_CHECKS.json", cpu_checks)
    validation = {
        "schema": "coqa_passage_reuse_r0_validation_v1",
        "status": "PASS",
        "required_receipts": [
            "BOUNDARY_PROTOCOL.json",
            "CUT_MANIFEST.json",
            "SOURCE_MANIFEST.json",
            "REPLICATION_MANIFEST.json",
            "COST_FORECAST.json",
            "CPU_CHECKS.json",
        ],
        "receipts_present": True,
        "next_state": "SMOKE",
    }
    write_json(out / "R0_VALIDATION.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
