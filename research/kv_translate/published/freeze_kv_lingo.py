#!/usr/bin/env python3
"""Freeze the independent KV-Lingo train/validation token stream on CPU."""

from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path
import resource
import time

from .kv_lingo_data import (
    ALGORITHM,
    ALGORITHM_V2,
    CONTINUATION_CAP,
    DATASET_REVISION,
    PREFIX_CAP,
    SCHEMA,
    SCHEMA_V2,
    TOKENIZER_REVISION,
    CandidatePool,
    TokenStoreWriter,
    build_packed,
    build_packed_v2,
    candidate_receipt,
    candidates_from_record,
    candidates_from_record_v2,
    file_sha256,
    mixture_cell,
    mixture_cell_v2,
    stable_int,
    token_sha256,
    uniform_length_bucket,
    write_json,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {message}", flush=True)


def best_records(
    path: Path, *, seed: int, keep: int, lane: str, algorithm: str = ALGORITHM
) -> list[dict]:
    """Keep the ``keep`` records with the smallest frozen hash in one pass."""

    heap: list[tuple[int, str, dict]] = []
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            uuid = str(row["uuid"])
            score = stable_int(algorithm, seed, lane, uuid)
            item = (-score, uuid, row)
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            rows += 1
            if rows % 100_000 == 0:
                log(f"{path.name}: scanned {rows:,} rows")
    selected = [item[2] for item in heap]
    selected.sort(
        key=lambda row: (stable_int(algorithm, seed, lane, row["uuid"]), row["uuid"])
    )
    log(f"{path.name}: selected {len(selected):,} of {rows:,} records for {lane}")
    return selected


def split_records(
    path: Path, *, seed: int, train_keep: int, val_keep: int, algorithm: str = ALGORITHM
):
    # Select a few extra training records, then remove every validation UUID.
    validation = best_records(
        path, seed=seed, keep=val_keep, lane="validation", algorithm=algorithm
    )
    validation_ids = {row["uuid"] for row in validation}
    training = best_records(
        path,
        seed=seed,
        keep=train_keep + val_keep,
        lane="training",
        algorithm=algorithm,
    )
    training = [row for row in training if row["uuid"] not in validation_ids][
        :train_keep
    ]
    if len(training) != train_keep:
        raise RuntimeError("validation exclusion left too few training records")
    return training, validation


def make_pool(
    tokenizer,
    records: list[dict],
    reasoning: str,
    *,
    seed: int,
    stream_version: int = 1,
):
    candidates = []
    for index, record in enumerate(records, 1):
        if stream_version == 1:
            candidates.extend(candidates_from_record(tokenizer, record, reasoning))
        else:
            rows = candidates_from_record_v2(tokenizer, record)
            candidates.extend(row for row in rows if row.reasoning == reasoning)
        if index % 500 == 0:
            log(
                f"tokenized {index:,}/{len(records):,} {reasoning} records; "
                f"{len(candidates):,} eligible turns"
            )
    if not candidates:
        raise RuntimeError(f"no eligible {reasoning} candidates")
    if stream_version == 1:
        return CandidatePool(candidates, seed=seed)
    return CandidatePool(
        candidates,
        seed=seed,
        algorithm=ALGORITHM_V2,
        bucket_function=uniform_length_bucket,
    )


def append_sample(
    token_store: TokenStoreWriter,
    pools: dict[str, CandidatePool],
    *,
    index: int,
    split: str,
) -> dict:
    kind, reasoning = mixture_cell(index)
    pool = pools[reasoning]
    focal = pool.natural(index)
    if kind == "natural":
        prefix = list(focal.prefix_ids)
        continuation = list(focal.continuation_ids)
        constituents = [focal.identity()]
    else:
        prefix, continuation, constituents = build_packed(focal, pool, serial=index)
    if len(prefix) > PREFIX_CAP or len(continuation) > CONTINUATION_CAP:
        raise RuntimeError("sample exceeds the frozen token cap")
    prefix_offset, prefix_tokens = token_store.append(prefix)
    continuation_offset, continuation_tokens = token_store.append(continuation)
    return {
        "schema": SCHEMA,
        "index": index,
        "split": split,
        "kind": kind,
        "reasoning": reasoning,
        "focal_uuid": focal.uuid,
        "focal_assistant_index": focal.assistant_index,
        "constituents": constituents,
        "prefix_offset_uint32": prefix_offset,
        "prefix_tokens": prefix_tokens,
        "continuation_offset_uint32": continuation_offset,
        "continuation_tokens": continuation_tokens,
        "prefix_sha256": token_sha256(prefix),
        "continuation_sha256": token_sha256(continuation),
    }


def append_sample_v2(
    token_store: TokenStoreWriter,
    pools: dict[str, CandidatePool],
    *,
    index: int,
    split: str,
) -> dict:
    kind, reasoning, cell_serial = mixture_cell_v2(index)
    pool = pools[reasoning]
    focal = pool.natural(cell_serial)
    if kind == "natural":
        prefix = list(focal.prefix_ids)
        continuation = list(focal.continuation_ids)
        constituents = [focal.identity()]
    else:
        prefix, continuation, constituents = build_packed_v2(
            focal, pool, serial=cell_serial
        )
    if len(prefix) > PREFIX_CAP or len(continuation) > CONTINUATION_CAP:
        raise RuntimeError("sample exceeds the frozen token cap")
    prefix_offset, prefix_tokens = token_store.append(prefix)
    continuation_offset, continuation_tokens = token_store.append(continuation)
    return {
        "schema": SCHEMA_V2,
        "index": index,
        "split": split,
        "kind": kind,
        "reasoning": reasoning,
        "cell_serial": cell_serial,
        "focal_uuid": focal.uuid,
        "focal_assistant_index": focal.assistant_index,
        "constituents": constituents,
        "prefix_offset_uint32": prefix_offset,
        "prefix_tokens": prefix_tokens,
        "continuation_offset_uint32": continuation_offset,
        "continuation_tokens": continuation_tokens,
        "prefix_sha256": token_sha256(prefix),
        "continuation_sha256": token_sha256(continuation),
    }


def write_stream(
    out: Path,
    name: str,
    pools: dict[str, CandidatePool],
    *,
    samples: int,
    index_base: int,
    stream_version: int = 1,
) -> dict:
    token_path = out / f"{name}.tokens.u32"
    rows_path = out / f"{name}.jsonl"
    rows_tmp = rows_path.with_suffix(".jsonl.tmp")
    token_writer = TokenStoreWriter(token_path)
    prefix_positions = 0
    translated_prefix_positions = 0
    counts: dict[str, int] = {}
    prefix_lengths = []
    continuation_lengths = []
    try:
        with rows_tmp.open("w", encoding="utf-8") as stream:
            for serial in range(samples):
                index = index_base + serial
                append = append_sample if stream_version == 1 else append_sample_v2
                row = append(token_writer, pools, index=index, split=name)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                cell = f"{row['kind']}:{row['reasoning']}"
                counts[cell] = counts.get(cell, 0) + 1
                prefix_positions += row["prefix_tokens"]
                # The final prefix token seeds continuation scoring and is not
                # part of the Stage-1/Stage-2 translated cache span.
                translated_prefix_positions += row["prefix_tokens"] - 1
                prefix_lengths.append(row["prefix_tokens"])
                continuation_lengths.append(row["continuation_tokens"])
                if (serial + 1) % 1000 == 0:
                    log(f"{name}: froze {serial + 1:,}/{samples:,} samples")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(rows_tmp, rows_path)
    finally:
        token_writer.close()
    return {
        "rows": rows_path.name,
        "rows_sha256": file_sha256(rows_path),
        "tokens": token_path.name,
        "tokens_sha256": file_sha256(token_path),
        "samples": samples,
        "uint32_tokens": token_writer.tokens,
        "bytes": token_path.stat().st_size,
        "mixture_counts": counts,
        "prefix_positions": prefix_positions,
        "translated_prefix_positions": translated_prefix_positions,
        "prefix_tokens_min": min(prefix_lengths),
        "prefix_tokens_max": max(prefix_lengths),
        "prefix_tokens_mean": prefix_positions / samples,
        "continuation_tokens_min": min(continuation_lengths),
        "continuation_tokens_max": max(continuation_lengths),
        "continuation_tokens_mean": sum(continuation_lengths) / samples,
    }


def source_receipt(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=40_000)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--stage1-samples", type=int, default=400)
    parser.add_argument("--train-records-per-reasoning", type=int, default=4096)
    parser.add_argument("--validation-records-per-reasoning", type=int, default=128)
    parser.add_argument("--stream-version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.stage1_samples > args.train_samples:
        parser.error("stage 1 must be a prefix of the frozen training stream")

    from transformers import AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )
    train_records, val_records = {}, {}
    source_files = {}
    algorithm = ALGORITHM if args.stream_version == 1 else ALGORITHM_V2
    for reasoning in ("off", "on"):
        path = args.dataset_dir / f"reasoning_{reasoning}.jsonl"
        source_files[reasoning] = source_receipt(path)
        train_records[reasoning], val_records[reasoning] = split_records(
            path,
            seed=args.seed,
            train_keep=args.train_records_per_reasoning,
            val_keep=args.validation_records_per_reasoning,
            algorithm=algorithm,
        )

    train_pools, val_pools = {}, {}
    for reasoning in ("off", "on"):
        train_pools[reasoning] = make_pool(
            tokenizer,
            train_records[reasoning],
            reasoning,
            seed=args.seed,
            stream_version=args.stream_version,
        )
        val_pools[reasoning] = make_pool(
            tokenizer,
            val_records[reasoning],
            reasoning,
            seed=args.seed + 1_000_000,
            stream_version=args.stream_version,
        )
    train_uuids = {
        candidate.uuid for pool in train_pools.values() for candidate in pool.all
    }
    val_uuids = {
        candidate.uuid for pool in val_pools.values() for candidate in pool.all
    }
    if train_uuids & val_uuids:
        raise RuntimeError("training and validation UUIDs overlap")

    args.out.mkdir(parents=True, exist_ok=True)
    training = write_stream(
        args.out,
        "train",
        train_pools,
        samples=args.train_samples,
        index_base=0,
        stream_version=args.stream_version,
    )
    validation = write_stream(
        args.out,
        "validation",
        val_pools,
        samples=args.validation_samples,
        index_base=1_000_000_000,
        stream_version=args.stream_version,
    )
    stage1_rows = []
    with (args.out / training["rows"]).open(encoding="utf-8") as stream:
        for _ in range(args.stage1_samples):
            stage1_rows.append(json.loads(next(stream)))
    stage1 = {
        "samples": args.stage1_samples,
        "sample_indices": [row["index"] for row in stage1_rows],
        "prefix_positions": sum(row["prefix_tokens"] for row in stage1_rows),
        "translated_prefix_positions": sum(
            row["prefix_tokens"] - 1 for row in stage1_rows
        ),
        "train_rows_sha256": training["rows_sha256"],
    }

    tokenizer_files = []
    for path in sorted(args.tokenizer.iterdir()):
        if path.is_file():
            tokenizer_files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    manifest = {
        "schema": SCHEMA if args.stream_version == 1 else SCHEMA_V2,
        "algorithm": algorithm,
        "method_identity": "independent construction; author row manifest and loader unavailable",
        "seed": args.seed,
        "dataset": {
            "name": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
            "revision": DATASET_REVISION,
            "source_files": source_files,
        },
        "tokenizer": {
            "name": "Qwen/Qwen3-4B",
            "revision": TOKENIZER_REVISION,
            "files": tokenizer_files,
        },
        "construction": {
            "mixture": "50% natural / 50% packed; reasoning status alternates",
            "natural_selection": (
                "32 log-spaced 10..10000 token strata; deterministic nearest nonempty stratum"
                if args.stream_version == 1
                else "32 equal-width 10..10000 token strata with independent per-cell serials"
            ),
            "packed_selection": (
                "capped v1 samples concatenated before one focal question; 12000..16000 tokens"
                if args.stream_version == 1
                else "complete reasoning-stripped historical renderings before one focal question; 12000..16000 tokens"
            ),
            "continuation_cap": CONTINUATION_CAP,
            "prefix_cap": PREFIX_CAP,
            "deviation": "exact author sample identities, length sampler and packing heuristic were not published",
        },
        "candidate_pools": {
            "training_records_per_reasoning": args.train_records_per_reasoning,
            "validation_records_per_reasoning": args.validation_records_per_reasoning,
            "training_candidates": {
                key: len(pool.all) for key, pool in train_pools.items()
            },
            "validation_candidates": {
                key: len(pool.all) for key, pool in val_pools.items()
            },
            "training_uuid_count": len(train_uuids),
            "validation_uuid_count": len(val_uuids),
            "uuid_overlap": 0,
            "training_candidate_digest": token_sha256(
                [
                    stable_int(candidate.identity()) & 0xFFFFFFFF
                    for pool in train_pools.values()
                    for candidate in pool.all
                ]
            ),
            "validation_candidate_digest": token_sha256(
                [
                    stable_int(candidate.identity()) & 0xFFFFFFFF
                    for pool in val_pools.values()
                    for candidate in pool.all
                ]
            ),
            "examples": {
                key: [candidate_receipt(row) for row in pool.all[:2]]
                for key, pool in train_pools.items()
            },
        },
        "training": training,
        "stage1": stage1,
        "validation": validation,
        "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "wall_seconds": time.time() - started,
        "Generated-by": "OpenAI Codex",
    }
    write_json(args.out / "MANIFEST.json", manifest)
    log(
        f"wrote {args.out / 'MANIFEST.json'}; stage1 positions="
        f"{stage1['translated_prefix_positions']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
