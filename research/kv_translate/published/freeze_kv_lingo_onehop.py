#!/usr/bin/env python3
"""Freeze the fixed-history CoQA one-hop diagnostic for KV-Lingo C1/C2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import coqa
from .freeze_kv_lingo_coqa import prompt_messages
from .kv_lingo_data import TOKENIZER_REVISION, file_sha256, token_sha256, write_json


def freeze_rows(tokenizer, stories: list[dict]) -> list[dict]:
    rows = []
    for story in stories:
        supplied = coqa.primary_answers(story)
        for turn in coqa.Q1_TURNS:
            messages = prompt_messages(story, turn, supplied)
            token_ids = list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            if len(token_ids) < 2:
                raise ValueError(f"one-hop row {story['id']}:{turn} is too short")
            rows.append(
                {
                    "schema": "kv_lingo_onehop_row_v1",
                    "row_id": f"{story['id']}:{turn}",
                    "conversation_id": story["id"],
                    "domain": story["source"],
                    "turn": turn,
                    "token_ids": token_ids,
                    "token_ids_sha256": token_sha256(token_ids),
                    "prefix_tokens": len(token_ids) - 1,
                    "references": coqa.answer_references(story, turn),
                    "primary_reference_history": supplied[: turn - 1],
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    stories = json.loads(args.conversations.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = freeze_rows(tokenizer, stories)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ONE_HOP_ROWS.jsonl"
    write_jsonl(path, rows)
    manifest = {
        "schema": "kv_lingo_onehop_manifest_v1",
        "construction": (
            "turns 1,5,10; primary CoQA answers supply prior history; Qwen3 "
            "reasoning-off chat template; final prompt token is receiver-scored"
        ),
        "procedural_deviation": (
            "derived after C1 retained outputs because the C0 one-hop artifact "
            "was omitted; turns, supplied-history rule, cohort and gates were "
            "already fixed by 7edfdad9 and no C2 output was inspected"
        ),
        "input": {
            "path": args.conversations.name,
            "bytes": args.conversations.stat().st_size,
            "sha256": file_sha256(args.conversations),
        },
        "tokenizer": {
            "name": "Qwen/Qwen3-4B",
            "revision": TOKENIZER_REVISION,
        },
        "rows": len(rows),
        "turn_counts": {
            str(turn): sum(row["turn"] == turn for row in rows)
            for turn in coqa.Q1_TURNS
        },
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in coqa.DOMAINS
        },
        "artifact": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        },
        "Generated-by": "OpenAI Codex",
    }
    write_json(args.out / "ONE_HOP_MANIFEST.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
