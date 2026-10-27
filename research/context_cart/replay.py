# SPDX-License-Identifier: MIT
"""Cold-start prefix checkpoint replay.

``producer`` runs inside a process that holds the writer: it exports one
cartridge per world and answers every question with the exact rows still in
memory (the live-prefix reference), twice, to measure the unmodified-reader
variability. ``consume`` runs in a separate process that loads only the pinned
base reader, the artifacts, and the question texts. It installs the rows as
the initial cache, answers its assigned question range at batch one, and
records traces. ``compare`` scores the two consumers against the reference.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import artifact as A
from . import dataset as D
from . import evaluate as E
from . import model as M
from . import prompts as P
from . import runs as R


def _answer_one(
    reader: M.Reader,
    tokenizer: Any,
    prefix: list[tuple[torch.Tensor, torch.Tensor]],
    question: str,
    first_position: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = P.prompt_ids(tokenizer, P.user_message("learned", question))
    ids = torch.tensor([prompt], dtype=torch.long, device=reader.device)
    mask = torch.ones_like(ids)
    reader.traces.clear()
    reader.record_traces = True
    out = reader.greedy(
        prefix, ids, mask, first_position, max_new_tokens, keep_logits=True
    )
    reader.record_traces = False
    trace = reader.traces[0]
    gen = out["ids"][0].tolist()
    return {
        "pred_ids": M.strip_generation(gen, reader.eos_ids, reader.pad_id),
        "raw_ids": gen,
        "pred": E.decode_prediction(tokenizer, gen, reader.eos_ids, reader.pad_id),
        "prompt_tokens": len(prompt),
        "trace": dataclasses.asdict(trace),
        "step_logits": out["step_logits"][0].cpu().numpy().astype(np.float32),
        "prefill_seconds": out["prefill_seconds"],
        "decode_seconds": out["decode_seconds"],
    }


def _save_answer(out_dir: Path, key: str, answer: dict[str, Any]) -> dict[str, Any]:
    logits_dir = out_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{key}.npy", answer["step_logits"])
    row = {k: v for k, v in answer.items() if k != "step_logits"}
    row["logits_file"] = f"logits/{key}.npy"
    return row


@torch.no_grad()
def produce(
    writer: M.Writer,
    reader: M.Reader,
    tokenizer: Any,
    identity: dict[str, Any],
    worlds: list[dict[str, Any]],
    out_dir: Path,
    source_revision: str,
    checkpoint_digest: str,
    namespace: str,
    max_new_tokens: int,
    reference: bool = True,
) -> dict[str, Any]:
    """Export cartridges and, with ``reference``, write the live-prefix
    reference answers twice (the second pass measures reader variability)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "artifacts": {},
        "namespace": namespace,
        "worlds": [w["world_id"] for w in worlds],
    }
    live_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for world in worlds:
        source = torch.tensor(
            [world["source_ids"]], dtype=torch.long, device=reader.device
        )
        prefix, length = writer.encode(source)
        prefix = [(k.detach(), v.detach()) for k, v in prefix]
        rows = int(prefix[0][0].shape[2])
        # memory rows were produced right after the source: positions L..L+M-1
        positions = range(length, length + rows)
        first_position = length + rows
        result = A.export_prefix(
            out_dir / "cartridges" / world["world_id"],
            prefix,
            identity,
            positions,
            first_position,
            source_revision,
            checkpoint_digest,
            namespace,
            {
                "world_id": world["world_id"],
                "source_digest": world["source_digest"],
                "source_tokens": length,
            },
        )
        index["artifacts"][world["world_id"]] = {
            "artifact_id": result.artifact_id,
            "payload_digest": result.payload_digest,
            "payload_bytes": result.payload_bytes,
            "tensor_bytes": result.tensor_bytes,
            "tensor_digest": M.prefix_digest(prefix),
            "first_position": first_position,
            "positions": [positions[0], positions[-1]],
            "rows": rows,
            "source_digest": world["source_digest"],
        }
        if not reference:
            continue
        for pass_name, rows in (("live", live_rows), ("live_repeat", repeat_rows)):
            for q in world["questions"]:
                answer = _answer_one(
                    reader, tokenizer, prefix, q["text"], first_position, max_new_tokens
                )
                key = f"{world['world_id']}__{q['qid']}".replace(":", "_").replace(
                    "/", "_"
                )
                row = _save_answer(out_dir / pass_name, key, answer)
                row.update(
                    {
                        "world_id": world["world_id"],
                        "qid": q["qid"],
                        "family": q["family"],
                        "attribute": q["attribute"],
                        "gold": q["answer"],
                    }
                )
                row["em"] = D.exact_match(row["pred"], q["answer"], q["attribute"])
                rows.append(row)
        if (
            M.prefix_digest(prefix)
            != index["artifacts"][world["world_id"]]["tensor_digest"]
        ):
            raise RuntimeError("prefix rows changed during live answering")
    if reference:
        E.write_rows(out_dir / "live" / "predictions.jsonl", live_rows)
        E.write_rows(out_dir / "live_repeat" / "predictions.jsonl", repeat_rows)
    index["identity"] = identity
    index["environment"] = R.environment()
    R.dump_json(out_dir / "index.json", index)
    return index


def consume(
    artifacts_dir: Path,
    questions_path: Path,
    out_dir: Path,
    spec: M.ModelSpec,
    question_range: tuple[int, int],
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Reader-only consumer. Receives artifacts and question text only."""
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    R.configure_threads()
    tokenizer = M.load_tokenizer(spec)
    t0 = time.perf_counter()
    base = M.load_base(spec, device)
    if any(p.requires_grad for p in base.parameters()):
        raise RuntimeError("reader must be frozen")
    load_seconds = time.perf_counter() - t0
    eos = tuple(
        int(i)
        for i in (
            [tokenizer.eos_token_id]
            if isinstance(tokenizer.eos_token_id, int)
            else tokenizer.eos_token_id
        )
    )
    reader = M.Reader(
        base,
        eos_ids=eos
        + ((int(tokenizer.pad_token_id),) if tokenizer.pad_token_id not in eos else ()),
        pad_id=int(tokenizer.pad_token_id),
    )
    identity = M.reader_identity(base, tokenizer, spec)
    adapters = [n for n, _ in base.named_modules() if "lora" in n.lower()]
    if adapters:
        raise RuntimeError("reader carries adapter modules")
    questions = D.read_jsonl(questions_path)
    by_world: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        by_world.setdefault(q["world_id"], []).append(q)
    # the consumer answers exactly the worlds it holds artifacts for
    artifact_worlds = sorted(
        p.name for p in artifacts_dir.iterdir() if (p / A.MANIFEST).exists()
    )
    if not artifact_worlds:
        raise RuntimeError(f"no artifacts under {artifacts_dir}")
    missing = [w for w in artifact_worlds if w not in by_world]
    if missing:
        raise RuntimeError(f"no questions for artifact worlds: {missing[:4]}")
    lo, hi = question_range
    rows: list[dict[str, Any]] = []
    artifact_records: dict[str, Any] = {}
    totals = {"read": 0.0, "validate": 0.0, "copy": 0.0, "prefill": 0.0, "decode": 0.0}
    for world_id in artifact_worlds:
        qs = by_world[world_id]
        loaded = A.load_prefix(artifacts_dir / world_id, identity, device)
        totals["read"] += loaded.read_seconds
        totals["validate"] += loaded.validate_seconds
        totals["copy"] += loaded.copy_seconds
        artifact_records[world_id] = {
            "artifact_id": loaded.manifest.artifact_id,
            "payload_digest": loaded.payload_digest,
            "tensor_digest": loaded.tensor_digest,
            "rows": loaded.rows,
            "first_position": loaded.first_position,
            "positions": [
                loaded.manifest.metadata.positions[0],
                loaded.manifest.metadata.positions[-1],
            ],
        }
        assert loaded.rows == int(loaded.prefix[0][0].shape[2])
        for q in qs[lo:hi]:
            answer = _answer_one(
                reader,
                tokenizer,
                loaded.prefix,
                q["text"],
                loaded.first_position,
                max_new_tokens,
            )
            totals["prefill"] += answer["prefill_seconds"]
            totals["decode"] += answer["decode_seconds"]
            trace = answer["trace"]
            if (
                trace["cache_rows_before"] != loaded.rows
                or trace["source_tokens_processed"] != 0
            ):
                raise RuntimeError("consumer processed unexpected rows")
            if (
                trace["first_position"] != loaded.first_position
                or trace["input_tokens"] != answer["prompt_tokens"]
            ):
                raise RuntimeError("consumer position contract violated")
            key = f"{world_id}__{q['qid']}".replace(":", "_").replace("/", "_")
            row = _save_answer(out_dir, key, answer)
            row.update(
                {
                    "world_id": world_id,
                    "qid": q["qid"],
                    "family": q.get("family"),
                    "attribute": q.get("attribute"),
                }
            )
            rows.append(row)
        if M.prefix_digest(loaded.prefix) != loaded.tensor_digest:
            raise RuntimeError("artifact rows changed during use")
        artifact_records[world_id]["tensor_digest_after_use"] = M.prefix_digest(
            loaded.prefix
        )
    E.write_rows(out_dir / "predictions.jsonl", rows)
    record = {
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "question_range": [lo, hi],
        "answers": len(rows),
        "identity": identity,
        "adapter_modules": adapters,
        "artifacts": artifact_records,
        "model_load_seconds": load_seconds,
        "seconds": totals,
        "wall_seconds": time.perf_counter() - t_start,
        "environment": R.environment(),
        "inputs": {
            "artifacts_dir": str(artifacts_dir),
            "questions": str(questions_path),
        },
    }
    R.dump_json(out_dir / "consumer.json", record)
    return record


def _load_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = D.read_jsonl(path / "predictions.jsonl")
    return {(r["world_id"], r["qid"]): r for r in rows}


def _logit_diff(
    path_a: Path, row_a: dict[str, Any], path_b: Path, row_b: dict[str, Any]
) -> dict[str, float]:
    a = np.load(path_a / row_a["logits_file"])
    b = np.load(path_b / row_b["logits_file"])
    steps = min(a.shape[0], b.shape[0])
    diff = np.abs(a[:steps] - b[:steps])
    return {
        "first_token_max_abs": float(diff[0].max()),
        "scored_tokens_max_abs": float(diff.max()),
        "steps_compared": int(steps),
        "argmax_equal_all_steps": bool(
            (a[:steps].argmax(-1) == b[:steps].argmax(-1)).all()
        ),
    }


def compare(
    producer_dir: Path,
    consumer_dirs: list[Path],
    out_path: Path,
    tolerance_factor: float = 1.0,
) -> dict[str, Any]:
    """Score restored answers against the live-prefix reference."""
    live = _load_rows(producer_dir / "live")
    repeat = _load_rows(producer_dir / "live_repeat")
    index = R.load_json(producer_dir / "index.json")
    envelope = 0.0
    repeat_mismatch = 0
    for key, row in live.items():
        diff = _logit_diff(
            producer_dir / "live", row, producer_dir / "live_repeat", repeat[key]
        )
        envelope = max(envelope, diff["scored_tokens_max_abs"])
        repeat_mismatch += int(row["pred_ids"] != repeat[key]["pred_ids"])
    pairs = []
    restored_total = 0
    token_mismatch = 0
    worst_first = 0.0
    worst_all = 0.0
    consumers = []
    for consumer_dir in consumer_dirs:
        record = R.load_json(consumer_dir / "consumer.json")
        consumers.append(
            {
                "dir": str(consumer_dir),
                "process_id": record["process_id"],
                "answers": record["answers"],
                "question_range": record["question_range"],
                "seconds": record["seconds"],
                "model_load_seconds": record["model_load_seconds"],
            }
        )
        for key, row in _load_rows(consumer_dir).items():
            ref = live[key]
            diff = _logit_diff(producer_dir / "live", ref, consumer_dir, row)
            same = row["pred_ids"] == ref["pred_ids"]
            token_mismatch += int(not same)
            restored_total += 1
            worst_first = max(worst_first, diff["first_token_max_abs"])
            worst_all = max(worst_all, diff["scored_tokens_max_abs"])
            art = record["artifacts"][key[0]]
            producer_art = index["artifacts"][key[0]]
            pairs.append(
                {
                    "world_id": key[0],
                    "qid": key[1],
                    "consumer": str(consumer_dir.name),
                    "live_pred": ref["pred"],
                    "restored_pred": row["pred"],
                    "tokens_equal": same,
                    "gold": ref.get("gold"),
                    "live_em": ref.get("em"),
                    "restored_em": (
                        D.exact_match(row["pred"], ref["gold"], ref["attribute"])
                        if ref.get("gold") is not None
                        else None
                    ),
                    "artifact_id_equal": art["artifact_id"]
                    == producer_art["artifact_id"],
                    "tensor_digest_equal": art["tensor_digest"]
                    == producer_art["tensor_digest"]
                    == art["tensor_digest_after_use"],
                    "trace": row["trace"],
                    **diff,
                }
            )
    identity_equal = all(
        R.load_json(d / "consumer.json")["identity"] == index["identity"]
        for d in consumer_dirs
    )
    digests_equal = all(
        p["artifact_id_equal"] and p["tensor_digest_equal"] for p in pairs
    )
    traces_clean = all(
        p["trace"]["source_tokens_processed"] == 0
        and p["trace"]["cache_rows_before"] == index["artifacts"][p["world_id"]]["rows"]
        and p["trace"]["first_position"]
        == index["artifacts"][p["world_id"]]["first_position"]
        for p in pairs
    )
    threshold = max(envelope * tolerance_factor, 1e-6)
    verdict = {
        "live_answers": len(live),
        "restored_answers": restored_total,
        "token_sequences_equal": restored_total - token_mismatch,
        "token_mismatches": token_mismatch,
        "live_repeat_token_mismatches": repeat_mismatch,
        "unmodified_reader_envelope_max_abs_logit": envelope,
        "restored_first_token_max_abs_logit": worst_first,
        "restored_scored_tokens_max_abs_logit": worst_all,
        "logit_threshold": threshold,
        "within_envelope": worst_all <= threshold,
        "reader_identity_equal": identity_equal,
        "artifact_digests_equal": digests_equal,
        "traces_clean": traces_clean,
        "consumers": consumers,
        "pass": token_mismatch == 0
        and restored_total == len(live)
        and identity_equal
        and digests_equal
        and worst_all <= threshold,
    }
    R.dump_json(out_path, {"verdict": verdict, "pairs": pairs})
    return verdict
