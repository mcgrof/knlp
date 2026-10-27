# SPDX-License-Identifier: MIT
"""Replay acceptance: the checker refuses incomplete or dirty evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.context_cart import replay as RP
from research.context_cart import runs as R

WORLD = "w1"
QIDS = ("q1", "q2", "q3", "q4")
ROWS = 4
START = 10


def _logits(seed: int, steps: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((steps, 6)).astype(np.float32)


def _answer(qid: str, seed: int, first_position: int = START) -> dict:
    prompt = [5, 6, 7]
    return {
        "world_id": WORLD,
        "qid": qid,
        "pred_ids": [1 + seed, 2],
        "raw_ids": [1 + seed, 2, 0],
        "pred": f"p{seed}",
        "prompt_tokens": len(prompt),
        "prompt_ids": prompt,
        "trace": {
            "input_tokens": len(prompt),
            "cache_rows_before": ROWS,
            "cache_rows_after": ROWS + len(prompt),
            "first_position": first_position,
            "source_tokens_processed": 0,
        },
        "logits_steps": 3,
        "gold": None,
        "family": None,
        "attribute": None,
    }


def _write_rows(directory: Path, answers: list[dict]) -> None:
    (directory / "logits").mkdir(parents=True, exist_ok=True)
    with (directory / "predictions.jsonl").open("w") as stream:
        for a in answers:
            key = f"{a['world_id']}__{a['qid']}"
            np.save(directory / "logits" / f"{key}.npy", _logits(QIDS.index(a["qid"])))
            a = dict(a, logits_file=f"logits/{key}.npy")
            stream.write(json.dumps(a) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    producer = tmp_path / "producer"
    answers = [_answer(q, i) for i, q in enumerate(QIDS)]
    _write_rows(producer / "live", answers)
    _write_rows(producer / "live_repeat", answers)
    identity = {"model": "tiny", "weights_digest": "x" * 64}
    R.dump_json(
        producer / "index.json",
        {
            "artifacts": {
                WORLD: {
                    "artifact_id": "art",
                    "tensor_digest": "td",
                    "rows": ROWS,
                    "first_position": START,
                }
            },
            "identity": identity,
            "process_id": 1000,
            "style": "ledger",
            "expected_keys": [[WORLD, q] for q in QIDS],
        },
    )
    consumers = []
    for n, (lo, hi, pid) in enumerate(((0, 2, 2001), (2, 4, 2002))):
        directory = tmp_path / f"consumer_{n}"
        rows = [_answer(q, i) for i, q in enumerate(QIDS) if lo <= i < hi]
        _write_rows(directory, rows)
        R.dump_json(
            directory / "consumer.json",
            {
                "process_id": pid,
                "question_range": [lo, hi],
                "answers": len(rows),
                "identity": identity,
                "artifacts": {
                    WORLD: {
                        "artifact_id": "art",
                        "tensor_digest": "td",
                        "tensor_digest_after_use": "td",
                    }
                },
                "model_load_seconds": 0.1,
                "seconds": {},
                "tokens_seen": 5 * len(rows),
                "tokens_expected": 5 * len(rows),
                "style": "ledger",
            },
        )
        consumers.append(directory)
    return producer, consumers


def test_complete_clean_replay_passes(tmp_path: Path):
    producer, consumers = _fixture(tmp_path)
    verdict = RP.compare(producer, consumers, tmp_path / "out.json")
    assert verdict["pass"], verdict
    assert verdict["restored_answers"] == 4 and verdict["token_mismatches"] == 0
    assert verdict["traces_clean"] and verdict["consumer_processes_distinct"]


def test_duplicate_consumer_cannot_stand_in_for_coverage(tmp_path: Path):
    producer, consumers = _fixture(tmp_path)
    verdict = RP.compare(producer, [consumers[0], consumers[0]], tmp_path / "o.json")
    assert not verdict["pass"]
    assert any("listed twice" in p for p in verdict["problems"])
    assert any("not restored" in p for p in verdict["problems"])
    # a copy under another name is still the same keys twice and leaves two
    # expected keys unanswered
    import shutil

    copy = tmp_path / "consumer_copy"
    shutil.copytree(consumers[0], copy)
    verdict = RP.compare(producer, [consumers[0], copy], tmp_path / "o2.json")
    assert not verdict["pass"]
    assert any("answered by two consumers" in p for p in verdict["problems"])


def test_dirty_trace_fails_even_with_equal_tokens(tmp_path: Path):
    producer, consumers = _fixture(tmp_path)
    path = consumers[1] / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["trace"]["first_position"] = START + 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    verdict = RP.compare(producer, consumers, tmp_path / "o.json")
    assert verdict["token_mismatches"] == 0
    assert not verdict["traces_clean"] and not verdict["pass"]


def test_producer_process_and_step_coverage_are_checked(tmp_path: Path):
    producer, consumers = _fixture(tmp_path)
    record = R.load_json(consumers[0] / "consumer.json")
    record["process_id"] = 1000
    R.dump_json(consumers[0] / "consumer.json", record)
    verdict = RP.compare(producer, consumers, tmp_path / "o.json")
    assert not verdict["pass"]
    assert any("producer process" in p for p in verdict["problems"])
    # fewer logit steps than the reference is a coverage failure
    producer, consumers = _fixture(tmp_path / "b")
    np.save(consumers[1] / "logits" / f"{WORLD}__q3.npy", _logits(2)[:2])
    verdict = RP.compare(producer, consumers, tmp_path / "b" / "o.json")
    assert verdict["logit_step_mismatches"] == 1 and not verdict["pass"]
