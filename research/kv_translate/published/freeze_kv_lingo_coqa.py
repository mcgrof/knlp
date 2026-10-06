#!/usr/bin/env python3
"""Freeze exposed development and untouched CoQA cohorts for KV-Lingo."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from . import coqa
from .coqa_reuse import normalize_passage
from .kv_lingo_data import file_sha256, stable_int, token_sha256, write_json

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".csv",
    ".yaml",
    ".yml",
}
TOKEN = re.compile(r"[0-9a-z]{20,}")


def historical_ids(root: Path, valid_ids: set[str]) -> tuple[set[str], list[dict]]:
    found: set[str] = set()
    receipts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # The archived official input contains every conversation and is not
        # evidence that its answers or model outputs were inspected.
        if path.name == "coqa-dev-v1.0.json":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = set(TOKEN.findall(text)) & valid_ids
        if hits:
            found.update(hits)
            receipts.append(
                {
                    "path": str(path.relative_to(root)),
                    "ids": len(hits),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
    return found, receipts


def round_robin_select(by_domain: dict[str, list[dict]], count: int, salt: str):
    ordered = {}
    for domain in coqa.DOMAINS:
        ordered[domain] = sorted(
            by_domain[domain],
            key=lambda row: (stable_int(salt, row["id"]), row["id"]),
        )
    result = []
    cursor = {domain: 0 for domain in coqa.DOMAINS}
    while len(result) < count:
        progressed = False
        for domain in coqa.DOMAINS:
            index = cursor[domain]
            if index < len(ordered[domain]) and len(result) < count:
                result.append(ordered[domain][index])
                cursor[domain] += 1
                progressed = True
        if not progressed:
            break
    return result


def prompt_messages(story: dict, turn: int, prior_answers: list[str]):
    content = [f"Passage: {story['story']}"]
    for index in range(turn):
        content.append(f"Question: {story['questions'][index]['input_text']}")
        if index < turn - 1:
            content.append(f"Answer: {prior_answers[index]}")
    return [
        {"role": "system", "content": coqa.INSTRUCTION},
        {"role": "user", "content": "\n\n".join(content)},
    ]


def frozen_story(tokenizer, story: dict) -> dict:
    messages = prompt_messages(story, 1, [])
    turn_one = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    suffixes = {}
    for turn in range(2, 11):
        question = story["questions"][turn - 1]["input_text"]
        bodies = {
            "after_eos": f"\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "after_other": f"<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        }
        suffixes[str(turn)] = {
            key: tokenizer.encode(text, add_special_tokens=False)
            for key, text in bodies.items()
        }
    return {
        **story,
        "turn_1_token_ids": list(turn_one),
        "turn_1_token_ids_sha256": token_sha256(turn_one),
        "next_turn_suffix_token_ids": suffixes,
        "normalized_passage_sha256": hashlib.sha256(
            normalize_passage(story["story"]).encode("utf-8")
        ).hexdigest(),
    }


def write_stories(path: Path, stories: list[dict]):
    write_json(path, stories)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development", type=int, default=32)
    parser.add_argument("--reserve", type=int, default=256)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    dataset = json.loads(args.data.read_text())
    stories = dataset["data"]
    valid_ids = {story["id"] for story in stories}
    exposed, sources = historical_ids(args.history_root, valid_ids)
    if not exposed:
        raise RuntimeError("historical scan found no previously exposed CoQA IDs")
    by_id = {story["id"]: story for story in stories}
    exposed_passages = {
        normalize_passage(by_id[conversation_id]["story"])
        for conversation_id in exposed
    }

    exposed_eligible, untouched_eligible = defaultdict(list), defaultdict(list)
    for story in stories:
        if story["source"] not in coqa.DOMAINS or len(story["questions"]) < 10:
            continue
        passage = normalize_passage(story["story"])
        if story["id"] in exposed:
            exposed_eligible[story["source"]].append(story)
        elif passage not in exposed_passages:
            untouched_eligible[story["source"]].append(story)

    development = round_robin_select(
        exposed_eligible, args.development, "kv-lingo-coqa-development-v1"
    )
    reserve = round_robin_select(
        untouched_eligible, args.reserve, "kv-lingo-coqa-reserve-v1"
    )
    if len(development) != args.development:
        raise RuntimeError(
            f"only {len(development)} exposed ten-turn conversations qualify"
        )
    if not reserve:
        raise RuntimeError("no untouched reserve conversations qualify")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    frozen_development = [frozen_story(tokenizer, story) for story in development]
    frozen_reserve = [frozen_story(tokenizer, story) for story in reserve]
    first_by_domain = {}
    for story in frozen_development:
        first_by_domain.setdefault(story["source"], story)
    mechanics = [first_by_domain[domain] for domain in coqa.DOMAINS]

    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "development": write_stories(
            args.out / "DEVELOPMENT_CONVERSATIONS.json", frozen_development
        ),
        "mechanics": write_stories(
            args.out / "MECHANICS_CONVERSATIONS.json", mechanics
        ),
        "reserve": write_stories(
            args.out / "RESERVE_CONVERSATIONS.json", frozen_reserve
        ),
    }
    development_ids = {story["id"] for story in development}
    reserve_ids = {story["id"] for story in reserve}
    development_passages = {normalize_passage(story["story"]) for story in development}
    reserve_passages = {normalize_passage(story["story"]) for story in reserve}
    manifest = {
        "schema": "kv_lingo_coqa_cohorts_v1",
        "official_data": {
            "path": str(args.data),
            "bytes": args.data.stat().st_size,
            "sha256": file_sha256(args.data),
            "version": dataset.get("version"),
        },
        "history_scan": {
            "root": str(args.history_root),
            "exposed_ids": len(exposed),
            "source_files_with_ids": sources,
        },
        "eligibility": "public five CoQA domains, >=10 turns, passage not historically exposed",
        "development": {
            "requested": args.development,
            "selected": len(development),
            "ids": [story["id"] for story in development],
            "by_domain": {
                domain: sum(story["source"] == domain for story in development)
                for domain in coqa.DOMAINS
            },
            "historically_exposed": development_ids <= exposed,
        },
        "reserve": {
            "requested": args.reserve,
            "selected": len(reserve),
            "deviation": (
                None
                if len(reserve) == args.reserve
                else "largest deterministic eligible reserve is smaller than paper target"
            ),
            "ids": [story["id"] for story in reserve],
            "by_domain": {
                domain: sum(story["source"] == domain for story in reserve)
                for domain in coqa.DOMAINS
            },
            "all_ids_untouched": not bool(reserve_ids & exposed),
        },
        "checks": {
            "development_reserve_ids_disjoint": not bool(development_ids & reserve_ids),
            "development_reserve_passages_disjoint": not bool(
                development_passages & reserve_passages
            ),
            "reserve_passages_disjoint_from_all_exposed": not bool(
                reserve_passages & exposed_passages
            ),
            "all_selected_have_ten_turns": all(
                len(story["questions"]) >= 10 for story in development + reserve
            ),
        },
        "protocol": {
            "turns": 10,
            "starting_models": ["Qwen3-4B", "Qwen3-8B"],
            "reasoning": "off; empty Qwen3 thinking marker is prefilled",
            "decode": "greedy, max64, stop at EOS/newline/cap",
            "scoring": "official CoQA leave-one-reference-out token F1",
            "bootstrap": {"draws": 10_000, "seed": 20261002, "cluster": "conversation"},
        },
        "artifacts": artifacts,
        "Generated-by": "OpenAI Codex",
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError(f"cohort integrity failed: {manifest['checks']}")
    write_json(args.out / "COHORT_MANIFEST.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
