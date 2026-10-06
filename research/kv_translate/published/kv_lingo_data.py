#!/usr/bin/env python3
"""Frozen data objects for the independent KV-Lingo reproduction.

The paper does not publish its loader or row manifest.  This module therefore
owns an explicit, versioned construction which can be replayed from the pinned
Nemotron JSONL files and Qwen3 tokenizer.  Large token payloads live outside
git in a little-endian uint32 file; JSONL rows carry offsets, constituent UUIDs
and hashes so a run never samples data after it has started.
"""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Iterable, Iterator, Sequence

SCHEMA = "kv_lingo_frozen_stream_v1"
ALGORITHM = "knlp-kv-lingo-stream-v1"
SCHEMA_V2 = "kv_lingo_frozen_stream_v2"
ALGORITHM_V2 = "knlp-kv-lingo-stream-v2"
DATASET_REVISION = "1a9454ed054b8544503ab8d8c0a519d141a44c5b"
TOKENIZER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PREFIX_CAP = 16_384
CONTINUATION_CAP = 2_048
NATURAL_MIN = 10
NATURAL_MAX = 10_000
PACKED_MIN = 12_000
PACKED_MAX = 16_000


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def token_sha256(tokens: Sequence[int]) -> str:
    payload = array("I", (int(token) for token in tokens))
    if sys.byteorder != "little":
        payload.byteswap()
    return sha256_bytes(payload.tobytes())


def file_sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def assistant_turns(messages: Sequence[dict]) -> list[int]:
    """Assistant indices which are immediately preceded by a user turn."""

    return [
        index
        for index, message in enumerate(messages)
        if index > 0
        and message.get("role") == "assistant"
        and messages[index - 1].get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"]
    ]


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def apply_chat_template_ids(tokenizer, messages: Sequence[dict], **kwargs) -> list[int]:
    """Normalize Transformers 4.x list and 5.x BatchEncoding returns."""

    value = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("KV-Lingo expects one rendered conversation")
        value = value[0]
    return list(value)


def render_turn(tokenizer, record: dict, assistant_index: int, reasoning: str):
    """Render one paper-shaped natural ``(prefix, continuation)`` pair.

    Qwen3's template removes old thinking blocks.  With reasoning disabled it
    also puts the empty thinking marker into the generation prompt, which the
    paper keeps in the translated prefix rather than scoring as continuation.
    """

    messages = record["messages"]
    if assistant_index not in assistant_turns(messages):
        raise ValueError("selected message is not an eligible assistant turn")
    prefix_messages = messages[:assistant_index]
    enable_thinking = reasoning == "on"
    prefix = apply_chat_template_ids(
        tokenizer,
        prefix_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    continuation_text = messages[assistant_index]["content"] + "<|im_end|>\n"
    continuation = _encode(tokenizer, continuation_text)[:CONTINUATION_CAP]
    if not prefix or not continuation:
        raise ValueError("rendered sample has an empty side")
    return list(prefix), continuation


def substantive_reasoning(value: str | dict) -> bool:
    """Classify the selected answer, excluding empty template placeholders."""

    if isinstance(value, dict):
        reasoning_content = value.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return True
        content = value.get("content", "")
    else:
        content = value
    opening, closing = "<think>", "</think>"
    if opening not in content or closing not in content:
        return False
    body = content.split(opening, 1)[1].split(closing, 1)[0]
    return bool(body.strip())


def focal_answer_text(message: dict) -> str:
    """Render the selected current answer, including structured reasoning."""

    content = message["content"]
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.lstrip()}"
    return content


def historical_messages(messages: Sequence[dict]) -> list[dict]:
    """Return structured history with completed reasoning traces removed.

    This mirrors the Qwen3 template's treatment once an assistant answer has
    become history, but does so before rendering so a filler is one complete,
    closed conversation rather than a capped answer fragment.
    """

    result = []
    for message in messages:
        row = dict(message)
        row.pop("reasoning_content", None)
        content = row.get("content")
        if row.get("role") == "assistant" and isinstance(content, str):
            if "<think>" in content and "</think>" in content:
                content = content.rsplit("</think>", 1)[1].lstrip("\n")
            row["content"] = content
        result.append(row)
    return result


@dataclass(frozen=True)
class CandidateV2:
    uuid: str
    reasoning: str
    assistant_index: int
    prefix_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]
    uncapped_continuation_tokens: int
    complete_filler_ids: tuple[int, ...]

    @property
    def prefix_tokens(self) -> int:
        return len(self.prefix_ids)

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.complete_filler_ids

    def identity(self) -> str:
        return f"{self.uuid}:{self.assistant_index}:{self.reasoning}"


def candidates_from_record_v2(tokenizer, record: dict) -> list[CandidateV2]:
    """Build v2 candidates with separate focal and complete-history forms."""

    rows = []
    messages = record.get("messages", [])
    for assistant_index in assistant_turns(messages):
        message = messages[assistant_index]
        reasoning = "on" if substantive_reasoning(message) else "off"
        prefix = apply_chat_template_ids(
            tokenizer,
            messages[:assistant_index],
            add_generation_prompt=True,
            enable_thinking=reasoning == "on",
        )
        uncapped = _encode(tokenizer, focal_answer_text(message) + "<|im_end|>\n")
        complete = apply_chat_template_ids(
            tokenizer,
            historical_messages(messages[: assistant_index + 1]),
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if NATURAL_MIN <= len(prefix) <= NATURAL_MAX and uncapped and complete:
            rows.append(
                CandidateV2(
                    uuid=str(record["uuid"]),
                    reasoning=reasoning,
                    assistant_index=assistant_index,
                    prefix_ids=tuple(prefix),
                    continuation_ids=tuple(uncapped[:CONTINUATION_CAP]),
                    uncapped_continuation_tokens=len(uncapped),
                    complete_filler_ids=tuple(complete),
                )
            )
    return rows


@dataclass(frozen=True)
class Candidate:
    uuid: str
    reasoning: str
    assistant_index: int
    prefix_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]

    @property
    def prefix_tokens(self) -> int:
        return len(self.prefix_ids)

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.prefix_ids + self.continuation_ids

    def identity(self) -> str:
        return f"{self.uuid}:{self.assistant_index}:{self.reasoning}"


def candidates_from_record(tokenizer, record: dict, reasoning: str) -> list[Candidate]:
    rows = []
    for assistant_index in assistant_turns(record.get("messages", [])):
        prefix, continuation = render_turn(
            tokenizer, record, assistant_index, reasoning
        )
        if NATURAL_MIN <= len(prefix) <= NATURAL_MAX:
            rows.append(
                Candidate(
                    uuid=str(record["uuid"]),
                    reasoning=reasoning,
                    assistant_index=assistant_index,
                    prefix_ids=tuple(prefix),
                    continuation_ids=tuple(continuation),
                )
            )
    return rows


def log_length_bucket(length: int, buckets: int = 32) -> int:
    """Map 10..10k to the frozen log-spaced natural-length strata."""

    if not NATURAL_MIN <= length <= NATURAL_MAX:
        raise ValueError(f"natural prefix length {length} is outside the contract")
    import math

    low, high = math.log(NATURAL_MIN), math.log(NATURAL_MAX + 1)
    value = int((math.log(length) - low) / (high - low) * buckets)
    return min(buckets - 1, max(0, value))


def uniform_length_bucket(length: int, buckets: int = 32) -> int:
    """Map natural lengths into equal-width token intervals for stream v2."""

    if not NATURAL_MIN <= length <= NATURAL_MAX:
        raise ValueError(f"natural prefix length {length} is outside the contract")
    width = NATURAL_MAX - NATURAL_MIN + 1
    return min(buckets - 1, (length - NATURAL_MIN) * buckets // width)


class CandidatePool:
    """Deterministic candidate selection without outcome-dependent sampling."""

    def __init__(
        self,
        candidates: Iterable[Candidate],
        *,
        seed: int,
        algorithm: str = ALGORITHM,
        bucket_function=log_length_bucket,
    ):
        self.seed = int(seed)
        self.algorithm = algorithm
        self.bucket_function = bucket_function
        self.all = sorted(
            candidates,
            key=lambda row: (
                stable_int(self.algorithm, seed, row.identity()),
                row.identity(),
            ),
        )
        if not self.all:
            raise ValueError("candidate pool is empty")
        self.by_bucket: dict[int, list[Candidate]] = {index: [] for index in range(32)}
        for row in self.all:
            self.by_bucket[self.bucket_function(row.prefix_tokens)].append(row)
        self.nonempty = [index for index, rows in self.by_bucket.items() if rows]
        if not self.nonempty:
            raise ValueError("candidate pool has no natural-length bucket")

    def natural(self, serial: int) -> Candidate:
        desired = serial % 32
        bucket = min(self.nonempty, key=lambda value: (abs(value - desired), value))
        rows = self.by_bucket[bucket]
        return rows[
            stable_int(self.algorithm, self.seed, "natural", serial) % len(rows)
        ]

    def any(self, serial: int) -> Candidate:
        return self.all[
            stable_int(self.algorithm, self.seed, "any", serial) % len(self.all)
        ]


def mixture_cell(index: int) -> tuple[str, str]:
    """Alternate reasoning status while balancing natural/packed in four rows."""

    return (
        ("natural", "off"),
        ("packed", "on"),
        ("packed", "off"),
        ("natural", "on"),
    )[index % 4]


def mixture_cell_v2(index: int) -> tuple[str, str, int]:
    """Return the cell and its independent serial for unaliased selection."""

    kind, reasoning = mixture_cell(index)
    return kind, reasoning, index // 4


def build_packed(
    focal: Candidate,
    fillers: CandidatePool,
    *,
    serial: int,
) -> tuple[list[int], list[int], list[str]]:
    """Build a deterministic 12k--16k packed prefix ending at ``focal``.

    Complete natural samples are concatenated as filler.  The focal question
    remains at the end and its answer remains the continuation, matching the
    geometry disclosed by the paper.  The exact author packing heuristic is
    unavailable, so this implementation and every constituent are recorded.
    """

    rng = random.Random(stable_int(ALGORITHM, fillers.seed, "pack", serial))
    target = rng.randint(PACKED_MIN, PACKED_MAX)
    budget = target - focal.prefix_tokens
    if budget < 0:
        raise ValueError("focal prefix exceeds packed target")

    chosen: list[Candidate] = []
    used = 0
    attempts = 0
    while used < budget and attempts < 4096:
        candidate = fillers.any(serial * 4096 + attempts)
        attempts += 1
        width = len(candidate.full_ids)
        if width <= budget - used:
            chosen.append(candidate)
            used += width
        if budget - used < 32:
            break

    # Message boundaries can leave a small gap under the randomly chosen
    # target.  Continue against the published 16k ceiling until the 12k floor
    # is actually reached; never pad or truncate a conversation.
    while used + focal.prefix_tokens < PACKED_MIN and attempts < 8192:
        candidate = fillers.any(serial * 8192 + attempts)
        attempts += 1
        width = len(candidate.full_ids)
        if width <= PACKED_MAX - focal.prefix_tokens - used:
            chosen.append(candidate)
            used += width

    prefix = []
    for candidate in chosen:
        prefix.extend(candidate.full_ids)
    prefix.extend(focal.prefix_ids)

    # If discrete message boundaries leave the prefix just below 12k, choose a
    # different deterministic target/filler path rather than truncating chat.
    if not PACKED_MIN <= len(prefix) <= PACKED_MAX:
        raise RuntimeError(
            f"could not construct a packed sample: {len(prefix)} tokens after "
            f"{attempts} attempts"
        )
    constituents = [candidate.identity() for candidate in chosen]
    constituents.append(focal.identity())
    return prefix, list(focal.continuation_ids), constituents


def build_packed_v2(
    focal: CandidateV2,
    fillers: CandidatePool,
    *,
    serial: int,
) -> tuple[list[int], list[int], list[str]]:
    """Pack only complete, uncapped historical renderings for stream v2."""

    rng = random.Random(stable_int(ALGORITHM_V2, fillers.seed, "pack", serial))
    target = rng.randint(PACKED_MIN, PACKED_MAX)
    budget = target - focal.prefix_tokens
    if budget < 0:
        raise ValueError("focal prefix exceeds packed target")
    chosen = []
    used = 0
    attempts = 0
    while attempts < 8192:
        candidate = fillers.any(serial * 8192 + attempts)
        attempts += 1
        width = len(candidate.complete_filler_ids)
        remaining = budget - used
        if width <= remaining:
            chosen.append(candidate)
            used += width
        if remaining < 32 or used + focal.prefix_tokens >= PACKED_MIN:
            break
    while used + focal.prefix_tokens < PACKED_MIN and attempts < 16_384:
        candidate = fillers.any(serial * 16_384 + attempts)
        attempts += 1
        width = len(candidate.complete_filler_ids)
        if width <= PACKED_MAX - focal.prefix_tokens - used:
            chosen.append(candidate)
            used += width
    prefix = [token for row in chosen for token in row.complete_filler_ids]
    prefix.extend(focal.prefix_ids)
    if not PACKED_MIN <= len(prefix) <= PACKED_MAX:
        raise RuntimeError(
            f"could not construct a v2 packed sample: {len(prefix)} tokens after "
            f"{attempts} attempts"
        )
    return (
        prefix,
        list(focal.continuation_ids),
        [row.identity() for row in chosen] + [focal.identity()],
    )


class TokenStoreWriter:
    def __init__(self, path: Path):
        if sys.byteorder != "little":
            raise RuntimeError("the frozen uint32 store requires little-endian host")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("wb")
        self.tokens = 0

    def append(self, values: Sequence[int]) -> tuple[int, int]:
        offset = self.tokens
        payload = array("I", (int(value) for value in values))
        payload.tofile(self.stream)
        self.tokens += len(payload)
        return offset, len(payload)

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


class TokenStore:
    def __init__(self, path: Path):
        self.path = path

    def read(self, offset: int, count: int) -> list[int]:
        payload = array("I")
        with self.path.open("rb") as stream:
            stream.seek(int(offset) * 4)
            payload.fromfile(stream, int(count))
        if sys.byteorder != "little":
            payload.byteswap()
        if len(payload) != count:
            raise EOFError(f"wanted {count} tokens at {offset}, got {len(payload)}")
        return payload.tolist()


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def candidate_receipt(row: Candidate) -> dict:
    value = asdict(row)
    del value["prefix_ids"]
    del value["continuation_ids"]
    value.pop("complete_filler_ids", None)
    value.update(
        {
            "prefix_tokens": row.prefix_tokens,
            "continuation_tokens": len(row.continuation_ids),
            "prefix_sha256": token_sha256(row.prefix_ids),
            "continuation_sha256": token_sha256(row.continuation_ids),
        }
    )
    return value
