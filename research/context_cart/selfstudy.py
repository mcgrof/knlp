# SPDX-License-Identifier: MIT
"""Self-Study conversations as reader training examples.

A Self-Study parquet row holds one synthetic conversation about a section
of a document: a user turn and an assistant turn whose token ids and
top-k teacher log-probabilities were captured when the teacher answered
with the section in its system prompt. The cartridge trainer of the
Cartridges at Scale line feeds the conversation to the reader without
that system prompt, so the cartridge has to stand in for the section.

This module reproduces that convention exactly: Qwen chat start and end
tokens around each message, captured assistant token ids kept as they
are, end-of-turn handling for truncated generations, and one top-k target
row per assistant token at the position of that token. The teacher
log-probabilities are used as given; nothing is renormalised. It also
splits a patient's rows by whole prompt group, selects groups
deterministically, and audits lexical overlap with evaluation questions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Qwen chat template pieces the native trainer hard-codes:
# <|im_start|>{role}\n ... <|im_end|>\n
MESSAGE_START = {
    "user": [151644, 872, 198],
    "assistant": [151644, 77091, 198],
    "system": [151644, 8948, 198],
}
MESSAGE_END = [151645]
MESSAGE_EXTRA_END = [198]
TOP_K = 20


@dataclasses.dataclass
class Element:
    """One conversation as the reader sees it."""

    row_index: int
    prompt: str
    input_ids: list[int]
    target_positions: list[int]  # index of the predicted assistant token
    target_ids: np.ndarray  # [T, K] int64, -1 pads
    target_logprobs: np.ndarray  # [T, K] float32, -1000 pads
    truncated: bool
    kind: str
    note_id: str

    @property
    def scored_tokens(self) -> int:
        return len(self.target_positions)

    def teacher_mass(self) -> float:
        """Mean over scored tokens of the probability mass the top-k rows
        retain; the loss weights are exactly these probabilities."""
        if not self.target_positions:
            return 0.0
        weights = np.exp(self.target_logprobs.astype(np.float64))
        weights[self.target_ids < 0] = 0.0
        return float(weights.sum(-1).mean())


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a Self-Study parquet with their captured token fields."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _flat_topk(block: dict[str, Any], tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """The captured ``top_logprobs`` block is flat: ``token_idx`` (which
    assistant token), ``token_id`` and ``logprobs`` per entry, with
    ``shape`` ``[T, K]``. Rebuild the ``[T, K]`` matrices."""
    shape = list(block["shape"])
    if len(shape) != 2:
        raise ValueError(f"unexpected top_logprobs shape {shape}")
    rows, k = int(shape[0]), int(shape[1])
    idx = np.asarray(block["token_idx"], dtype=np.int64)
    ids = np.asarray(block["token_id"], dtype=np.int64)
    lps = np.asarray(block["logprobs"], dtype=np.float32)
    if not (len(idx) == len(ids) == len(lps) == rows * k):
        raise ValueError("top_logprobs entries do not match the recorded shape")
    if rows != tokens:
        raise ValueError(f"{rows} target rows for {tokens} assistant tokens")
    target_ids = np.full((rows, k), -1, dtype=np.int64)
    target_lps = np.full((rows, k), -1000.0, dtype=np.float32)
    counts = np.zeros(rows, dtype=np.int64)
    for i, t in enumerate(idx):
        slot = counts[t]
        if slot >= k:
            raise ValueError("more than K entries for one token")
        target_ids[t, slot] = ids[i]
        target_lps[t, slot] = lps[i]
        counts[t] += 1
    return target_ids, target_lps


def element_from_row(
    row: dict[str, Any], row_index: int, max_tokens: int, tokenizer: Any = None
) -> Element:
    """Mirror of the native message converter for Qwen models, followed by
    the native truncation rule: the sequence is cut at ``max_tokens`` and
    targets past the cut are dropped."""
    messages = row["messages"]
    input_ids: list[int] = []
    positions: list[int] = []
    ids_blocks: list[np.ndarray] = []
    lps_blocks: list[np.ndarray] = []
    prompt = ""
    for i, message in enumerate(messages):
        role = message["role"]
        if role == "user" and not prompt:
            prompt = message["content"]
        if message.get("token_ids") is None:
            if tokenizer is None:
                raise ValueError("message without token ids needs a tokenizer")
            msg_ids = list(
                tokenizer.encode(message["content"], add_special_tokens=False)
            )
        else:
            msg_ids = [int(t) for t in message["token_ids"]]
        msg_input = MESSAGE_START[role] + msg_ids
        end = MESSAGE_END
        ends_with_eot = len(msg_input) >= len(end) and msg_input[-len(end) :] == end
        last = i == len(messages) - 1
        if not ends_with_eot and not last:
            msg_input = msg_input + end + MESSAGE_EXTRA_END
        elif ends_with_eot and not last:
            msg_input = msg_input + MESSAGE_EXTRA_END
        block = message.get("top_logprobs")
        if block is not None:
            offset = len(input_ids) + len(MESSAGE_START[role])
            t_ids, t_lps = _flat_topk(block, len(msg_ids))
            positions.extend(int(offset + t) for t in range(len(msg_ids)))
            ids_blocks.append(t_ids)
            lps_blocks.append(t_lps)
        input_ids.extend(msg_input)
    target_ids = (
        np.concatenate(ids_blocks) if ids_blocks else np.zeros((0, TOP_K), np.int64)
    )
    target_lps = (
        np.concatenate(lps_blocks) if lps_blocks else np.zeros((0, TOP_K), np.float32)
    )
    truncated = len(input_ids) > max_tokens
    if truncated:
        input_ids = input_ids[:max_tokens]
        keep = np.asarray(positions) < max_tokens
        positions = [p for p, k in zip(positions, keep) if k]
        target_ids = target_ids[keep]
        target_lps = target_lps[keep]
    if any(p <= 0 for p in positions):
        raise ValueError("a target cannot sit at position 0")
    meta = row.get("metadata") or {}
    return Element(
        row_index=row_index,
        prompt=prompt,
        input_ids=input_ids,
        target_positions=positions,
        target_ids=target_ids,
        target_logprobs=target_lps,
        truncated=truncated,
        kind=str(row.get("type", "")),
        note_id=str(meta.get("note_id", "")),
    )


def prompt_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    """Row indices grouped by the first user turn, the unit the native
    split keeps together so a paraphrase family never straddles train and
    held-out."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        prompt = next(
            (m["content"] for m in row["messages"] if m["role"] == "user"), ""
        )
        groups.setdefault(prompt, []).append(index)
    return groups


def select_groups(groups: dict[str, list[int]], seed: int) -> list[str]:
    """A deterministic order of prompt groups: sorted prompts shuffled by
    ``seed``. Callers take the first N for each role."""
    prompts = sorted(groups)
    random.Random(seed).shuffle(prompts)
    return prompts


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def overlap_audit(
    prompts: Iterable[str], questions: Iterable[str]
) -> list[dict[str, Any]]:
    """Highest token-set Jaccard overlap between each evaluation question
    and any selected training prompt, with the closest prompt."""
    prompt_sets = [(p, _tokens(p)) for p in prompts]
    out = []
    for q in questions:
        qs = _tokens(q)
        best, best_prompt = 0.0, ""
        for p, ps in prompt_sets:
            union = len(qs | ps)
            score = len(qs & ps) / union if union else 0.0
            if score > best:
                best, best_prompt = score, p
        out.append({"question": q, "jaccard": best, "closest_prompt": best_prompt})
    return out


def digest_rows(rows: list[dict[str, Any]], indices: Iterable[int]) -> str:
    """Digest over the selected rows' message contents and token ids."""
    digest = hashlib.sha256()
    for index in indices:
        row = rows[index]
        for message in row["messages"]:
            digest.update(message["role"].encode())
            digest.update(message["content"].encode())
            if message.get("token_ids") is not None:
                digest.update(
                    json.dumps([int(t) for t in message["token_ids"]]).encode()
                )
    return digest.hexdigest()


def cartridge_rows(
    doc_tokens: int, divisor: int = 20, lo: int = 256, hi: int = 2048
) -> int:
    """The native per-document cartridge size: ceil(doc_tokens / 20),
    clamped, where ``doc_tokens`` counts the record text's tokens."""
    return max(lo, min(hi, math.ceil(doc_tokens / divisor)))


@dataclasses.dataclass
class PatientSplit:
    patient_id: str
    parquet: str
    parquet_digest: str
    rows: int
    unique_prompts: int
    duplicate_rows: int
    seed: int
    train_groups: list[str]
    train_rows: list[int]
    question_groups: list[str]
    question_rows: list[int]
    holdout_groups: list[str]
    holdout_rows: list[int]
    train_digest: str
    question_digest: str
    holdout_digest: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def split_patient(
    patient_id: str,
    parquet: Path,
    rows: list[dict[str, Any]],
    seed: int,
    train_groups: int,
    question_groups: int,
    holdout_groups: int,
    rows_per_group: int = 1,
) -> PatientSplit:
    """Disjoint prompt groups for training, question generalisation and the
    held-out loss panel, chosen before any outcome is seen. ``rows_per_group``
    caps how many rows of a group enter a split (1 keeps panels small)."""
    groups = prompt_groups(rows)
    order = select_groups(groups, seed)
    needed = train_groups + question_groups + holdout_groups
    if len(order) < needed:
        raise ValueError(f"{patient_id}: {len(order)} groups, need {needed}")
    train = order[:train_groups]
    question = order[train_groups : train_groups + question_groups]
    holdout = order[train_groups + question_groups : needed]

    def rows_of(names: list[str]) -> list[int]:
        return [i for n in names for i in groups[n][:rows_per_group]]

    train_rows = [i for n in train for i in groups[n]]
    question_rows = rows_of(question)
    holdout_rows = rows_of(holdout)
    seen = set()
    duplicates = 0
    for row in rows:
        key = json.dumps([(m["role"], m["content"]) for m in row["messages"]])
        if key in seen:
            duplicates += 1
        seen.add(key)
    return PatientSplit(
        patient_id=patient_id,
        parquet=str(parquet),
        parquet_digest=hashlib.sha256(Path(parquet).read_bytes()).hexdigest(),
        rows=len(rows),
        unique_prompts=len(groups),
        duplicate_rows=duplicates,
        seed=seed,
        train_groups=train,
        train_rows=train_rows,
        question_groups=question,
        question_rows=question_rows,
        holdout_groups=holdout,
        holdout_rows=holdout_rows,
        train_digest=digest_rows(rows, train_rows),
        question_digest=digest_rows(rows, question_rows),
        holdout_digest=digest_rows(rows, holdout_rows),
    )
