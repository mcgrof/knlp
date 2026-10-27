# SPDX-License-Identifier: MIT
"""Frozen prompt text for every reader arm.

All arms share one instruction, one question format, the pinned chat template
with thinking disabled, and the same answer cap. The arms differ only in what
precedes the question: nothing (empty, and every key/value memory arm), the
full ledger text, the last 64 source tokens re-encoded as text, or a text
summary that the frozen reader wrote before seeing any question.
"""

from __future__ import annotations

import hashlib
from typing import Any

INSTRUCTION = (
    "You answer questions about a staff ledger. Your entire reply must be the "
    "answer value only: a team name, a locker number, a badge code, or a "
    "person's full name. A question that asks for two things gets both values "
    "separated by a comma. A correction later in the ledger replaces the "
    "earlier value. Do not repeat the ledger sentence and do not explain."
)
ANSWER_SUFFIX = "Answer (value only):"

SUMMARY_INSTRUCTION = (
    "Write a compact summary of the following staff ledger in at most 60 words. "
    "Keep every specific fact: names, teams, locker numbers, badge codes, "
    "mentors, and corrections. Output only the summary."
)

ARMS = (
    "full",
    "empty",
    "tail",
    "summary",
    "untrained",
    "learned",
    "wrong",
)
MEMORY_ARMS = ("untrained", "learned", "wrong")

IM_END = "<|im_end|>"


def user_message(
    arm: str, question: str, source_text: str = "", context: str = ""
) -> str:
    """Text arms carry their context in the user turn. Every other arm name
    (empty, untrained, learned, wrong, and seed-suffixed variants) gets the
    instruction and question only; its context, if any, is the cache."""
    tail = f"Question: {question}\n{ANSWER_SUFFIX}"
    if arm == "full":
        return f"{INSTRUCTION}\n\nLedger:\n{source_text}\n\n{tail}"
    if arm == "tail":
        return f"{INSTRUCTION}\n\nLedger excerpt:\n{context}\n\n{tail}"
    if arm == "summary":
        return f"{INSTRUCTION}\n\nLedger summary:\n{context}\n\n{tail}"
    return f"{INSTRUCTION}\n\n{tail}"


def render_prompt(tokenizer: Any, content: str) -> str:
    """Chat-template prompt text with thinking disabled; ends at the point
    where the assistant answer starts."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_ids(tokenizer: Any, content: str) -> list[int]:
    return list(
        tokenizer(render_prompt(tokenizer, content), add_special_tokens=False)[
            "input_ids"
        ]
    )


def answer_ids(tokenizer: Any, answer: str) -> list[int]:
    """Teacher-forced answer tokens followed by the end-of-turn token."""
    return list(tokenizer(answer + IM_END, add_special_tokens=False)["input_ids"])


def summary_prompt_ids(tokenizer: Any, source_text: str) -> list[int]:
    return prompt_ids(tokenizer, f"{SUMMARY_INSTRUCTION}\n\n{source_text}")


def template_digest(tokenizer: Any) -> str:
    """Digest of the exact chat template text plus the frozen instruction."""
    template = tokenizer.chat_template or ""
    return hashlib.sha256(
        (template + "\n--\n" + INSTRUCTION + "\n--\n" + SUMMARY_INSTRUCTION).encode()
    ).hexdigest()
