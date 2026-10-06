#!/usr/bin/env python3
"""Census and structurally verify every row in a frozen KV-Lingo v1 stream."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

from .kv_lingo_data import (
    CONTINUATION_CAP,
    PREFIX_CAP,
    apply_chat_template_ids,
    candidates_from_record,
    file_sha256,
    focal_answer_text,
    historical_messages,
    log_length_bucket,
    mixture_cell,
    substantive_reasoning,
    write_json,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {message}", flush=True)


def token_blob(values) -> bytes:
    payload = array("I", (int(value) for value in values))
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tobytes()


def blob_tokens(value: bytes) -> np.ndarray:
    return np.frombuffer(value, dtype="<u4")


def token_hash(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<u4", copy=False).tobytes()).hexdigest()


def identity_parts(identity: str) -> tuple[str, int, str]:
    uuid, assistant_index, lane = identity.rsplit(":", 2)
    return uuid, int(assistant_index), lane


def collect_rows(path: Path) -> tuple[list[dict], set[str]]:
    rows, identities = [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                identities.update(row["constituents"])
    return rows, identities


def build_candidate_index(
    database: Path,
    tokenizer,
    dataset_dir: Path,
    identities: set[str],
) -> dict:
    wanted: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for identity in identities:
        uuid, assistant_index, lane = identity_parts(identity)
        wanted[lane][uuid].add(assistant_index)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS candidates (
        identity TEXT PRIMARY KEY, uuid TEXT, assistant_index INTEGER, lane TEXT,
        prefix BLOB, capped BLOB, required_focal BLOB, complete_history BLOB,
        v1_uncapped_tokens INTEGER, required_uncapped_tokens INTEGER,
        source_reasoning INTEGER,
        v1_retains_reasoning INTEGER, v1_omits_current_reasoning INTEGER,
        empty_placeholder INTEGER
        )""")
    found = set()
    lane_receipts = {}
    for lane in ("off", "on"):
        path = dataset_dir / f"reasoning_{lane}.jsonl"
        scanned = selected = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                scanned += 1
                record = json.loads(line)
                uuid = str(record["uuid"])
                assistant_indices = wanted[lane].get(uuid)
                if not assistant_indices:
                    continue
                by_index = {
                    row.assistant_index: row
                    for row in candidates_from_record(tokenizer, record, lane)
                }
                for assistant_index in assistant_indices:
                    candidate = by_index.get(assistant_index)
                    if candidate is None:
                        raise RuntimeError(
                            f"archived constituent cannot be reconstructed: "
                            f"{uuid}:{assistant_index}:{lane}"
                        )
                    message = record["messages"][assistant_index]
                    content = message["content"]
                    v1_uncapped = tokenizer.encode(
                        content + "<|im_end|>\n", add_special_tokens=False
                    )
                    required_uncapped = tokenizer.encode(
                        focal_answer_text(message) + "<|im_end|>\n",
                        add_special_tokens=False,
                    )
                    complete = apply_chat_template_ids(
                        tokenizer,
                        historical_messages(record["messages"][: assistant_index + 1]),
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    identity = candidate.identity()
                    connection.execute(
                        "INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            identity,
                            uuid,
                            assistant_index,
                            lane,
                            token_blob(candidate.prefix_ids),
                            token_blob(candidate.continuation_ids),
                            token_blob(required_uncapped[:CONTINUATION_CAP]),
                            token_blob(complete),
                            len(v1_uncapped),
                            len(required_uncapped),
                            int(substantive_reasoning(message)),
                            int(substantive_reasoning(content)),
                            int(
                                substantive_reasoning(message)
                                and not substantive_reasoning(content)
                            ),
                            int(content.lstrip().startswith("<think>\n\n</think>")),
                        ),
                    )
                    found.add(identity)
                    selected += 1
                    if selected % 1000 == 0:
                        connection.commit()
                        log(f"{lane}: indexed {selected:,} selected turns")
        connection.commit()
        lane_receipts[lane] = {
            "source": str(path),
            "source_sha256": file_sha256(path),
            "records_scanned": scanned,
            "selected_constituents": selected,
        }
    missing = sorted(identities - found)
    if missing:
        raise RuntimeError(f"{len(missing)} constituents absent from source records")
    connection.close()
    return lane_receipts


def inspect_stream(
    name: str,
    rows: list[dict],
    tokens_path: Path,
    database: Path,
    tokenizer,
    boundary_stream,
) -> dict:
    store = np.memmap(tokens_path, mode="r", dtype="<u4")
    connection = sqlite3.connect(database)
    cell_requested = defaultdict(Counter)
    cell_realized = defaultdict(Counter)
    cell_continuations = defaultdict(Counter)
    cell_modes = defaultdict(Counter)
    affected_truncated_rows = set()
    affected_reasoning_rows = set()
    omitted_current_reasoning_rows = set()
    truncated_unique = set()
    reasoning_unique = set()
    truncated_occurrences = reasoning_occurrences = 0
    omitted_current_reasoning_occurrences = 0
    fallback = Counter()
    focal_failures = []
    hash_failures = []
    cap_failures = []
    placeholder_rows = []
    representatives = {
        "truncated_filler": [],
        "retained_reasoning": [],
        "omitted_current_reasoning": [],
    }
    constituent_ids = set()
    constituent_uuids = set()

    def lookup(identity: str):
        value = connection.execute(
            "SELECT prefix,capped,required_focal,complete_history,v1_uncapped_tokens,"
            "required_uncapped_tokens,"
            "source_reasoning,v1_retains_reasoning,v1_omits_current_reasoning,"
            "empty_placeholder FROM candidates WHERE identity=?",
            (identity,),
        ).fetchone()
        if value is None:
            raise KeyError(identity)
        return value

    for serial, row in enumerate(rows):
        index = int(row["index"])
        expected_kind, expected_lane = mixture_cell(index)
        if (row["kind"], row["reasoning"]) != (expected_kind, expected_lane):
            focal_failures.append({"row": index, "failure": "mixture cell"})
        cell = f"{row['kind']}:{row['reasoning']}"
        prefix = np.asarray(
            store[
                int(row["prefix_offset_uint32"]) : int(row["prefix_offset_uint32"])
                + int(row["prefix_tokens"])
            ]
        )
        continuation = np.asarray(
            store[
                int(row["continuation_offset_uint32"]) : int(
                    row["continuation_offset_uint32"]
                )
                + int(row["continuation_tokens"])
            ]
        )
        if token_hash(prefix) != row["prefix_sha256"]:
            hash_failures.append({"row": index, "side": "prefix"})
        if token_hash(continuation) != row["continuation_sha256"]:
            hash_failures.append({"row": index, "side": "continuation"})
        if len(prefix) > PREFIX_CAP or len(continuation) > CONTINUATION_CAP:
            cap_failures.append(index)

        constituents = row["constituents"]
        constituent_ids.update(constituents)
        constituent_uuids.update(identity_parts(value)[0] for value in constituents)
        focal_identity = constituents[-1]
        (
            focal_prefix_blob,
            focal_cont_blob,
            _required_focal_blob,
            _,
            _,
            required_focal_tokens,
            focal_source_reasoning,
            _,
            focal_omits_reasoning,
            focal_empty,
        ) = lookup(focal_identity)
        focal_prefix = blob_tokens(focal_prefix_blob)
        focal_cont = blob_tokens(focal_cont_blob)
        if (
            focal_identity
            != f"{row['focal_uuid']}:{row['focal_assistant_index']}:{row['reasoning']}"
            or len(prefix) < len(focal_prefix)
            or not np.array_equal(prefix[-len(focal_prefix) :], focal_prefix)
            or not np.array_equal(continuation, focal_cont)
        ):
            focal_failures.append({"row": index, "failure": "focal placement/content"})
        if focal_empty:
            placeholder_rows.append(index)
        if focal_omits_reasoning:
            omitted_current_reasoning_occurrences += 1
            omitted_current_reasoning_rows.add(index)
            if len(representatives["omitted_current_reasoning"]) < 6:
                required_focal = blob_tokens(_required_focal_blob)
                representatives["omitted_current_reasoning"].append(
                    {
                        "split": name,
                        "row": index,
                        "identity": focal_identity,
                        "archived_continuation_tokens": len(focal_cont),
                        "required_uncapped_continuation_tokens": required_focal_tokens,
                        "archived_start_decoded": tokenizer.decode(focal_cont[:96]),
                        "required_start_decoded": tokenizer.decode(required_focal[:96]),
                    }
                )

        desired = index % 32
        realized = log_length_bucket(len(focal_prefix))
        cell_requested[cell][str(desired)] += 1
        cell_realized[cell][str(realized)] += 1
        cell_continuations[cell][str(len(continuation))] += 1
        cell_modes[cell]["on" if focal_source_reasoning else "off"] += 1
        fallback[cell] += int(desired != realized)

        cursor = 0
        for position, identity in enumerate(constituents[:-1]):
            (
                prefix_blob,
                capped_blob,
                _,
                complete_blob,
                uncapped,
                _,
                _,
                v1_retains_reasoning,
                _,
                _,
            ) = lookup(identity)
            old_filler = np.concatenate(
                (blob_tokens(prefix_blob), blob_tokens(capped_blob))
            )
            end = cursor + len(old_filler)
            if end > len(prefix) or not np.array_equal(prefix[cursor:end], old_filler):
                focal_failures.append(
                    {
                        "row": index,
                        "failure": "constituent boundary",
                        "position": position,
                    }
                )
                break
            issue = None
            if uncapped > CONTINUATION_CAP:
                issue = "capped_unclosed_historical_filler"
                truncated_occurrences += 1
                truncated_unique.add(identity)
                affected_truncated_rows.add(index)
            if v1_retains_reasoning:
                reasoning_occurrences += 1
                reasoning_unique.add(identity)
                affected_reasoning_rows.add(index)
                issue = (
                    "capped_unclosed_and_substantive_reasoning_retained"
                    if issue
                    else "substantive_reasoning_retained_in_history"
                )
            if issue:
                record = {
                    "split": name,
                    "row": index,
                    "constituent_position": position,
                    "identity": identity,
                    "start_token": cursor,
                    "end_token_exclusive": end,
                    "v1_filler_tokens": len(old_filler),
                    "uncapped_continuation_tokens": uncapped,
                    "required_clean_history_tokens": len(blob_tokens(complete_blob)),
                    "issue": issue,
                }
                boundary_stream.write(json.dumps(record, sort_keys=True) + "\n")
                key = (
                    "truncated_filler"
                    if uncapped > CONTINUATION_CAP
                    else "retained_reasoning"
                )
                if len(representatives[key]) < 6:
                    clean = blob_tokens(complete_blob)
                    representatives[key].append(
                        {
                            **record,
                            "v1_tail_decoded": tokenizer.decode(old_filler[-48:]),
                            "next_tokens_decoded": tokenizer.decode(
                                prefix[end : end + 48]
                            ),
                            "required_clean_tail_decoded": tokenizer.decode(
                                clean[-48:]
                            ),
                        }
                    )
            cursor = end
        if row["kind"] == "packed" and cursor != len(prefix) - len(focal_prefix):
            focal_failures.append({"row": index, "failure": "packed cut"})
        if row["kind"] == "natural" and constituents[:-1]:
            focal_failures.append({"row": index, "failure": "natural has filler"})
        if (serial + 1) % 2000 == 0:
            log(f"{name}: audited {serial + 1:,}/{len(rows):,} rows")
    connection.close()
    return {
        "rows": len(rows),
        "rows_sha256": None,
        "tokens": str(tokens_path),
        "tokens_sha256": file_sha256(tokens_path),
        "constituent_identity_count": len(constituent_ids),
        "constituent_uuid_count": len(constituent_uuids),
        "constituent_identities": constituent_ids,
        "constituent_uuids": constituent_uuids,
        "hash_failures": hash_failures,
        "cap_failures": cap_failures,
        "focal_or_boundary_failures": focal_failures,
        "accidentally_scored_empty_placeholder_rows": placeholder_rows,
        "length_sampling": {
            "declared_policy": "log-spaced, not the paper's approximately uniform natural lengths",
            "requested_bucket_by_cell": {
                key: dict(value) for key, value in cell_requested.items()
            },
            "realized_bucket_by_cell": {
                key: dict(value) for key, value in cell_realized.items()
            },
            "fallback_count_by_cell": dict(fallback),
            "continuation_length_histogram_by_cell": {
                key: dict(value) for key, value in cell_continuations.items()
            },
            "actual_answer_reasoning_mode_by_cell": {
                key: dict(value) for key, value in cell_modes.items()
            },
        },
        "historical_filler_findings": {
            "truncated_occurrences": truncated_occurrences,
            "truncated_unique_constituents": len(truncated_unique),
            "truncated_affected_row_ids": sorted(affected_truncated_rows),
            "substantive_reasoning_occurrences": reasoning_occurrences,
            "substantive_reasoning_unique_constituents": len(reasoning_unique),
            "substantive_reasoning_affected_row_ids": sorted(affected_reasoning_rows),
            "omitted_current_reasoning_occurrences": omitted_current_reasoning_occurrences,
            "omitted_current_reasoning_affected_row_ids": sorted(
                omitted_current_reasoning_rows
            ),
            "representative_decoded_spans": representatives,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifest = json.loads((args.frozen / "MANIFEST.json").read_text(encoding="utf-8"))
    train_rows, train_ids = collect_rows(args.frozen / "train.jsonl")
    validation_rows, validation_ids = collect_rows(args.frozen / "validation.jsonl")
    database = args.out / "CANDIDATE_INDEX_V2.sqlite3"
    lane_receipts = build_candidate_index(
        database, tokenizer, args.dataset, train_ids | validation_ids
    )
    boundaries = args.out / "AFFECTED_BOUNDARIES.jsonl"
    with boundaries.open("w", encoding="utf-8") as stream:
        train = inspect_stream(
            "train",
            train_rows,
            args.frozen / "train.tokens.u32",
            database,
            tokenizer,
            stream,
        )
        validation = inspect_stream(
            "validation",
            validation_rows,
            args.frozen / "validation.tokens.u32",
            database,
            tokenizer,
            stream,
        )
    for name, value in (("train", train), ("validation", validation)):
        rows_path = args.frozen / f"{name}.jsonl"
        value["rows_path"] = str(rows_path)
        value["rows_sha256"] = file_sha256(rows_path)
    overlap_ids = sorted(
        train["constituent_identities"] & validation["constituent_identities"]
    )
    overlap_uuids = sorted(train["constituent_uuids"] & validation["constituent_uuids"])
    for value in (train, validation):
        del value["constituent_identities"]
        del value["constituent_uuids"]
    expected = manifest["training"], manifest["validation"]
    hashes_match_manifest = all(
        (
            train["rows_sha256"] == expected[0]["rows_sha256"],
            train["tokens_sha256"] == expected[0]["tokens_sha256"],
            validation["rows_sha256"] == expected[1]["rows_sha256"],
            validation["tokens_sha256"] == expected[1]["tokens_sha256"],
        )
    )
    result = {
        "schema": "kv_lingo_data_census_v1",
        "frozen_schema": manifest["schema"],
        "coverage": "all 40000 training and 64 validation samples plus every underlying constituent",
        "manifest_sha256": file_sha256(args.frozen / "MANIFEST.json"),
        "archived_hashes_match_manifest": hashes_match_manifest,
        "sources": lane_receipts,
        "train": train,
        "validation": validation,
        "split_overlap": {
            "constituent_identities": overlap_ids,
            "constituent_uuids": overlap_uuids,
            "passed": not overlap_ids and not overlap_uuids,
        },
        "affected_boundaries": {
            "path": boundaries.name,
            "sha256": file_sha256(boundaries),
            "format": "one structural filler boundary per affected occurrence",
        },
        "procedural_deviations": [
            "v1 packed fillers cap answers at 2048 tokens before historical reuse",
            "reachable inline reasoning would be retained in v1 filler tokens",
            "v1 omits Nemotron reasoning_content from selected current answers",
            "mixture index modulo 4 aliases each cell to eight of 32 requested bins",
            "natural lengths are log-spaced rather than approximately uniform",
            "the exposed one-hop freeze was late and remains a diagnostic deviation",
        ],
        "training_integrity": {
            "teacher_student_token_alignment": "same frozen prefix/scored token tensors by construction",
            "substantive_failure": not (
                hashes_match_manifest
                and not overlap_ids
                and not overlap_uuids
                and not train["hash_failures"]
                and not validation["hash_failures"]
                and not train["focal_or_boundary_failures"]
                and not validation["focal_or_boundary_failures"]
                and not train["historical_filler_findings"][
                    "omitted_current_reasoning_affected_row_ids"
                ]
                and not validation["historical_filler_findings"][
                    "omitted_current_reasoning_affected_row_ids"
                ]
            ),
            "decision": "R1B_CORRECTED_SCREEN",
            "rationale": "the reasoning-on half omits structured current-answer reasoning_content from the scored continuation, a wrong-objective-input failure",
        },
        "Generated-by": "OpenAI Codex",
    }
    write_json(args.out / "DATA_CENSUS.json", result)
    log(f"wrote {args.out / 'DATA_CENSUS.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
