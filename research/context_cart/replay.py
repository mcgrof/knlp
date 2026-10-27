# SPDX-License-Identifier: MIT
"""Cold-start prefix checkpoint replay.

``produce`` runs inside a process that holds the writer (or an imported
native cartridge): it exports one cartridge per world and answers every
question with the exact rows still in memory (the live-prefix reference),
twice, to measure the unmodified-reader variability. ``consume`` runs in a
separate process that loads only the pinned base reader, the artifacts, and
the question texts. It installs the rows as the initial cache, answers its
assigned question range at batch one, and records traces plus the exact
number of token ids it processed. ``compare`` scores the consumers against
the reference and accepts only complete, non-overlapping, clean evidence.
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
from . import longhealth as LH
from . import model as M
from . import prompts as P
from . import runs as R

Prefix = list[tuple[torch.Tensor, torch.Tensor]]

# fields a reader-only consumer may receive in a question row
CONSUMER_QUESTION_FIELDS = {
    "world_id",
    "qid",
    "text",
    "family",
    "attribute",
    "stratum",
    "template",
    "entity",
    "quartile",
    "options",
    "patient_id",
    "question_id",
    "question",
    "style",
}
FORBIDDEN_QUESTION_FIELDS = {"source_text", "source_ids", "answer", "support", "gold"}


def question_prompt_ids(tokenizer: Any, row: dict[str, Any], style: str) -> list[int]:
    if style == "ledger":
        return P.prompt_ids(tokenizer, P.user_message("learned", row["text"]))
    if style == "longhealth":
        return LH.prompt_ids(tokenizer, LH.question_from_row(row))
    raise ValueError(f"unknown prompt style {style}")


def _answer_one(
    reader: M.Reader,
    tokenizer: Any,
    prefix: Prefix,
    row: dict[str, Any],
    first_position: int,
    max_new_tokens: int,
    style: str,
) -> dict[str, Any]:
    prompt = question_prompt_ids(tokenizer, row, style)
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
        "prompt_ids": prompt,
        "trace": dataclasses.asdict(trace),
        "cache_rows_final": out["cache_rows"],
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
    row["logits_steps"] = int(answer["step_logits"].shape[0])
    return row


def _key(world_id: str, qid: str) -> str:
    return f"{world_id}__{qid}".replace(":", "_").replace("/", "_")


@torch.no_grad()
def produce_prefixes(
    reader: M.Reader,
    tokenizer: Any,
    identity: dict[str, Any],
    entries: list[dict[str, Any]],
    out_dir: Path,
    source_revision: str,
    checkpoint_digest: str,
    namespace: str,
    max_new_tokens: int,
    style: str = "ledger",
    reference: bool = True,
) -> dict[str, Any]:
    """Export one cartridge per entry and, with ``reference``, answer each
    entry's questions twice from the rows still in memory.

    Each entry holds ``world_id``, ``prefix`` (single-batch rows), the exact
    ``positions`` of those rows, the ``reader_start``, the question rows
    (``questions``, consumer-safe fields plus ``answer``/``gold`` when known)
    and ``extra`` provenance for the payload metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "artifacts": {},
        "namespace": namespace,
        "style": style,
        "worlds": [e["world_id"] for e in entries],
        "process_id": os.getpid(),
        "expected_keys": [],
    }
    live_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for entry in entries:
        world_id = entry["world_id"]
        prefix = [(k.detach(), v.detach()) for k, v in entry["prefix"]]
        positions = list(entry["positions"])
        reader_start = int(entry["reader_start"])
        result = A.export_prefix(
            out_dir / "cartridges" / world_id,
            prefix,
            identity,
            positions,
            reader_start,
            source_revision,
            checkpoint_digest,
            namespace,
            dict(entry.get("extra", {}), world_id=world_id),
        )
        index["artifacts"][world_id] = {
            "artifact_id": result.artifact_id,
            "payload_digest": result.payload_digest,
            "payload_bytes": result.payload_bytes,
            "tensor_bytes": result.tensor_bytes,
            "tensor_digest": M.prefix_digest(prefix),
            "first_position": reader_start,
            "positions": [positions[0], positions[-1]],
            "rows": int(prefix[0][0].shape[2]),
            **{k: v for k, v in entry.get("extra", {}).items() if k != "world_id"},
        }
        for q in entry["questions"]:
            index["expected_keys"].append([world_id, q["qid"]])
        if not reference:
            continue
        for pass_name, rows in (("live", live_rows), ("live_repeat", repeat_rows)):
            for q in entry["questions"]:
                answer = _answer_one(
                    reader, tokenizer, prefix, q, reader_start, max_new_tokens, style
                )
                row = _save_answer(
                    out_dir / pass_name, _key(world_id, q["qid"]), answer
                )
                row.update(
                    {
                        "world_id": world_id,
                        "qid": q["qid"],
                        "family": q.get("family"),
                        "attribute": q.get("attribute"),
                        "gold": q.get("answer", q.get("gold")),
                    }
                )
                if row["gold"] is not None and q.get("attribute"):
                    row["em"] = D.exact_match(row["pred"], row["gold"], q["attribute"])
                rows.append(row)
        if M.prefix_digest(prefix) != index["artifacts"][world_id]["tensor_digest"]:
            raise RuntimeError("prefix rows changed during live answering")
    if reference:
        E.write_rows(out_dir / "live" / "predictions.jsonl", live_rows)
        E.write_rows(out_dir / "live_repeat" / "predictions.jsonl", repeat_rows)
    index["identity"] = identity
    index["environment"] = R.environment()
    R.dump_json(out_dir / "index.json", index)
    return index


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
    """Writer-produced cartridges: memory rows sit right after the source at
    positions ``L .. L + M - 1`` and the reader starts at ``L + M``."""
    entries = []
    for world in worlds:
        source = torch.tensor(
            [world["source_ids"]], dtype=torch.long, device=reader.device
        )
        prefix, length = writer.encode(source)
        rows = int(prefix[0][0].shape[2])
        entries.append(
            {
                "world_id": world["world_id"],
                "prefix": [(k.detach(), v.detach()) for k, v in prefix],
                "positions": range(length, length + rows),
                "reader_start": length + rows,
                "questions": world["questions"],
                "extra": {
                    "source_digest": world["source_digest"],
                    "source_tokens": length,
                },
            }
        )
    return produce_prefixes(
        reader,
        tokenizer,
        identity,
        entries,
        out_dir,
        source_revision,
        checkpoint_digest,
        namespace,
        max_new_tokens,
        "ledger",
        reference,
    )


def check_consumer_questions(rows: list[dict[str, Any]]) -> None:
    """A consumer may see questions and options, never sources or answers."""
    for row in rows:
        leaked = FORBIDDEN_QUESTION_FIELDS & set(row)
        if leaked:
            raise RuntimeError(f"consumer input leaks {sorted(leaked)}")
        unknown = set(row) - CONSUMER_QUESTION_FIELDS
        if unknown:
            raise RuntimeError(
                f"consumer input has unexpected fields {sorted(unknown)}"
            )


def consume(
    artifacts_dir: Path,
    questions_path: Path,
    out_dir: Path,
    spec: M.ModelSpec,
    question_range: tuple[int, int],
    max_new_tokens: int,
    device: str,
    style: str = "ledger",
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
    check_consumer_questions(questions)
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
    tokens_expected = 0
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
            "positions": [loaded.positions[0], loaded.positions[-1]],
        }
        assert loaded.rows == int(loaded.prefix[0][0].shape[2])
        for q in qs[lo:hi]:
            answer = _answer_one(
                reader,
                tokenizer,
                loaded.prefix,
                q,
                loaded.first_position,
                max_new_tokens,
                style,
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
                or trace["cache_rows_after"] != loaded.rows + answer["prompt_tokens"]
            ):
                raise RuntimeError("consumer position contract violated")
            generated = len(answer["raw_ids"])
            # the prefill sees the prompt; every decode step but the last
            # feeds one generated token back through the reader
            tokens_expected += answer["prompt_tokens"] + max(generated - 1, 0)
            row = _save_answer(out_dir, _key(world_id, q["qid"]), answer)
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
    if reader.tokens_seen != tokens_expected:
        raise RuntimeError(
            f"reader processed {reader.tokens_seen} tokens, expected {tokens_expected}"
        )
    E.write_rows(out_dir / "predictions.jsonl", rows)
    record = {
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "question_range": [lo, hi],
        "answers": len(rows),
        "keys": [[r["world_id"], r["qid"]] for r in rows],
        "identity": identity,
        "adapter_modules": adapters,
        "artifacts": artifact_records,
        "model_load_seconds": load_seconds,
        "seconds": totals,
        "wall_seconds": time.perf_counter() - t_start,
        "tokens_seen": reader.tokens_seen,
        "tokens_expected": tokens_expected,
        "style": style,
        "environment": R.environment(),
        "inputs": {
            "artifacts_dir": str(artifacts_dir),
            "questions": str(questions_path),
            "question_fields": sorted({k for q in questions for k in q}),
        },
    }
    R.dump_json(out_dir / "consumer.json", record)
    return record


def _load_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = D.read_jsonl(path / "predictions.jsonl")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (r["world_id"], r["qid"])
        if key in out:
            raise RuntimeError(f"duplicate answer for {key} under {path}")
        out[key] = r
    return out


def _logit_diff(
    path_a: Path, row_a: dict[str, Any], path_b: Path, row_b: dict[str, Any]
) -> dict[str, Any]:
    a = np.load(path_a / row_a["logits_file"])
    b = np.load(path_b / row_b["logits_file"])
    same_shape = a.shape == b.shape
    steps = min(a.shape[0], b.shape[0])
    diff = np.abs(a[:steps] - b[:steps])
    return {
        "first_token_max_abs": float(diff[0].max()) if steps else float("inf"),
        "scored_tokens_max_abs": float(diff.max()) if steps else float("inf"),
        "steps_compared": int(steps),
        "steps_equal": bool(same_shape),
        "argmax_equal_all_steps": bool(
            same_shape and (a.argmax(-1) == b.argmax(-1)).all()
        ),
    }


def compare(
    producer_dir: Path,
    consumer_dirs: list[Path],
    out_path: Path,
    tolerance_factor: float = 1.0,
) -> dict[str, Any]:
    """Score restored answers against the live-prefix reference.

    Acceptance needs the consumers to cover exactly the producer's expected
    (world, question) keys once each, from distinct processes other than the
    producer's, with clean traces, the same logit step counts, equal token
    sequences, equal reader identity and artifact digests, and logits within
    the unmodified-reader envelope.
    """
    live = _load_rows(producer_dir / "live")
    repeat = _load_rows(producer_dir / "live_repeat")
    index = R.load_json(producer_dir / "index.json")
    expected = {tuple(k) for k in index.get("expected_keys", [])} or set(live)
    problems: list[str] = []
    if set(live) != expected or set(repeat) != expected:
        problems.append("reference answers do not cover the expected keys")
    envelope = 0.0
    repeat_mismatch = 0
    for key, row in live.items():
        if key not in repeat:
            continue
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
    seen_keys: dict[tuple[str, str], str] = {}
    process_ids: list[int] = []
    seen_dirs: set[str] = set()
    steps_mismatch = 0
    for consumer_dir in consumer_dirs:
        resolved = str(Path(consumer_dir).resolve())
        if resolved in seen_dirs:
            problems.append(f"consumer listed twice: {consumer_dir}")
            continue
        seen_dirs.add(resolved)
        record = R.load_json(consumer_dir / "consumer.json")
        process_ids.append(int(record["process_id"]))
        if int(record["process_id"]) == int(index.get("process_id", -1)):
            problems.append(f"consumer {consumer_dir} ran in the producer process")
        if record.get("tokens_seen") != record.get("tokens_expected"):
            problems.append(f"consumer {consumer_dir} token accounting mismatch")
        if record.get("style", "ledger") != index.get("style", "ledger"):
            problems.append(f"consumer {consumer_dir} used another prompt style")
        consumers.append(
            {
                "dir": str(consumer_dir),
                "process_id": record["process_id"],
                "answers": record["answers"],
                "question_range": record["question_range"],
                "seconds": record["seconds"],
                "model_load_seconds": record["model_load_seconds"],
                "tokens_seen": record.get("tokens_seen"),
            }
        )
        rows = _load_rows(consumer_dir)
        if len(rows) != record["answers"]:
            problems.append(f"consumer {consumer_dir} answer count disagrees")
        for key, row in rows.items():
            if key in seen_keys:
                problems.append(f"{key} answered by two consumers")
                continue
            seen_keys[key] = str(consumer_dir)
            if key not in live:
                problems.append(f"{key} has no live reference")
                continue
            ref = live[key]
            diff = _logit_diff(producer_dir / "live", ref, consumer_dir, row)
            same = (
                row["pred_ids"] == ref["pred_ids"] and row["raw_ids"] == ref["raw_ids"]
            )
            token_mismatch += int(not same)
            steps_mismatch += int(not diff["steps_equal"])
            restored_total += 1
            worst_first = max(worst_first, diff["first_token_max_abs"])
            worst_all = max(worst_all, diff["scored_tokens_max_abs"])
            art = record["artifacts"][key[0]]
            producer_art = index["artifacts"][key[0]]
            trace = row["trace"]
            trace_clean = (
                trace["source_tokens_processed"] == 0
                and trace["cache_rows_before"] == producer_art["rows"]
                and trace["first_position"] == producer_art["first_position"]
                and trace["input_tokens"] == row["prompt_tokens"]
                and row["prompt_ids"] == ref["prompt_ids"]
                and trace["cache_rows_after"]
                == producer_art["rows"] + row["prompt_tokens"]
            )
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
                        if ref.get("gold") is not None and ref.get("attribute")
                        else None
                    ),
                    "artifact_id_equal": art["artifact_id"]
                    == producer_art["artifact_id"],
                    "tensor_digest_equal": art["tensor_digest"]
                    == producer_art["tensor_digest"]
                    == art["tensor_digest_after_use"],
                    "trace_clean": bool(trace_clean),
                    "trace": trace,
                    **diff,
                }
            )
    missing = sorted(expected - set(seen_keys))
    if missing:
        problems.append(f"{len(missing)} expected keys were not restored")
    if len(set(process_ids)) != len(process_ids):
        problems.append("two consumers report the same process id")
    identity_equal = all(
        R.load_json(d / "consumer.json")["identity"] == index["identity"]
        for d in consumer_dirs
    )
    digests_equal = bool(pairs) and all(
        p["artifact_id_equal"] and p["tensor_digest_equal"] for p in pairs
    )
    traces_clean = bool(pairs) and all(p["trace_clean"] for p in pairs)
    threshold = max(envelope * tolerance_factor, 1e-6)
    verdict = {
        "expected_answers": len(expected),
        "live_answers": len(live),
        "restored_answers": restored_total,
        "token_sequences_equal": restored_total - token_mismatch,
        "token_mismatches": token_mismatch,
        "logit_step_mismatches": steps_mismatch,
        "live_repeat_token_mismatches": repeat_mismatch,
        "unmodified_reader_envelope_max_abs_logit": envelope,
        "restored_first_token_max_abs_logit": worst_first,
        "restored_scored_tokens_max_abs_logit": worst_all,
        "logit_threshold": threshold,
        "within_envelope": worst_all <= threshold,
        "reader_identity_equal": identity_equal,
        "artifact_digests_equal": digests_equal,
        "traces_clean": traces_clean,
        "consumer_processes_distinct": len(set(process_ids)) == len(process_ids),
        "consumers": consumers,
        "missing_keys": missing[:16],
        "problems": problems,
        "pass": (
            not problems
            and token_mismatch == 0
            and steps_mismatch == 0
            and restored_total == len(expected) > 0
            and identity_equal
            and digests_equal
            and traces_clean
            and worst_all <= threshold
        ),
    }
    R.dump_json(out_path, {"verdict": verdict, "pairs": pairs})
    return verdict
