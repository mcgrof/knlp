# SPDX-License-Identifier: MIT
"""Mooncake (FAST25) trace ingest, token-ID synthesis, and arrival scheduling.

Neutral serving/benchmark infrastructure for replaying a real production
request trace through vLLM + LMCache and measuring timing / prefix-cache
reuse / KV-offload storage behaviour.

The Mooncake public traces (conversation / toolagent / synthetic) contain
NO tokens or text -- only, per JSONL record:

    timestamp      relative arrival time in milliseconds
    input_length   number of prompt (prefill) tokens
    output_length  number of decode tokens
    hash_ids       list[int] of remapped 512-token prefix block hashes

Two records that share leading ``hash_ids`` share that prefix; an identical
``hash_id`` denotes an identical (reusable) 512-token KV block.  To run this
on a real model we synthesise token-ID sequences that PRESERVE the hash
structure: the same ``hash_id`` deterministically maps to the same block of
512 token IDs, so real cross-request prefix-cache hits occur.  The content is
synthetic; the reuse/length/arrival *structure* is faithful.

Design constraints (baked in):

  * Stable hashing (hashlib SHA256), NOT Python ``hash()`` -- reproducible
    across processes and machines.
  * Token IDs are bounded to a configurable vocab size and skip a reserved
    low-ID range (special tokens: BOS/EOS/PAD/...).
  * When a request's concatenated hash blocks are SHORTER than
    ``input_length`` the tail is filled with REQUEST-UNIQUE deterministic
    noise -- never a shared pad token (shared padding manufactures fake
    cache hits).
  * Requests carry raw ``prompt_token_ids`` for direct submission via the
    Completions API -- we never build a text string (that would incur
    detokenize / re-tokenize drift and break hash->block identity).

This module is pure: it imports no GPU / vLLM / LMCache packages at module
top and is unit-testable on CPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# The content-neutral primitives (submittable request record, open-loop
# scheduler, manifest builder, shared size defaults) now live in
# serving_replay.py and are shared with the content-trace stage.  They are
# re-exported here so existing importers (the mooncake stage + tests) keep
# working unchanged.
from .serving_replay import (  # noqa: F401
    DEFAULT_BLOCK_SIZE,
    DEFAULT_RESERVED_IDS,
    DEFAULT_VOCAB_SIZE,
    ScheduledRequest,
    SynthRequest,
    Wait,
    build_config_manifest,
    schedule_arrivals,
    sha256_file,
)

log = logging.getLogger(__name__)


# ── Typed records ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MooncakeRecord:
    """One parsed FAST25 trace record."""

    index: int
    timestamp_ms: float
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]
    malformed: bool = False
    malformed_reason: str = ""


# ── Parser ─────────────────────────────────────────────────────────────────


def _validate_raw(
    obj: dict,
    index: int,
    block_size: int,
    hash_count_tol: int,
) -> tuple[bool, str]:
    """Return (malformed, reason) for one raw JSON record."""
    required = ("timestamp", "input_length", "output_length", "hash_ids")
    for key in required:
        if key not in obj:
            return True, f"missing field {key!r}"

    try:
        input_length = int(obj["input_length"])
        output_length = int(obj["output_length"])
    except (TypeError, ValueError):
        return True, "non-integer input_length/output_length"

    if input_length <= 0:
        return True, f"non-positive input_length={input_length}"
    if output_length < 0:
        return True, f"negative output_length={output_length}"

    hash_ids = obj["hash_ids"]
    if not isinstance(hash_ids, (list, tuple)):
        return True, "hash_ids is not a list"
    if not all(isinstance(h, int) for h in hash_ids):
        return True, "hash_ids contains non-integer entries"

    # Block-count sanity: len(hash_ids) should be ~ceil(input_length/block).
    # Fewer hashes than blocks is legitimate (tail is noise-filled); MORE
    # hashes than blocks (beyond a small rounding tolerance) is corrupt.
    expected = math.ceil(input_length / block_size)
    if len(hash_ids) > expected + hash_count_tol:
        return True, (
            f"hash count wildly off: {len(hash_ids)} hashes for "
            f"input_length={input_length} (expected ~{expected})"
        )
    return False, ""


def parse_trace(
    path: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_records: Optional[int] = None,
    hash_count_tol: int = 1,
    skip_malformed: bool = False,
) -> list[MooncakeRecord]:
    """Parse a FAST25 JSONL Mooncake trace into typed records.

    Malformed records are flagged (``MooncakeRecord.malformed=True``) and, by
    default, still returned so callers can count/inspect them.  Set
    ``skip_malformed=True`` to drop them.
    """
    p = Path(path)
    records: list[MooncakeRecord] = []
    with p.open("r") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_records is not None and len(records) >= max_records:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("trace line %d: invalid JSON: %s", index, e)
                if skip_malformed:
                    continue
                records.append(
                    MooncakeRecord(
                        index=index,
                        timestamp_ms=0.0,
                        input_length=0,
                        output_length=0,
                        hash_ids=(),
                        malformed=True,
                        malformed_reason=f"invalid JSON: {e}",
                    )
                )
                continue

            malformed, reason = _validate_raw(obj, index, block_size, hash_count_tol)
            if malformed:
                log.warning("trace record %d malformed: %s", index, reason)
                if skip_malformed:
                    continue
                records.append(
                    MooncakeRecord(
                        index=index,
                        timestamp_ms=float(obj.get("timestamp", 0.0) or 0.0),
                        input_length=int(obj.get("input_length", 0) or 0),
                        output_length=int(obj.get("output_length", 0) or 0),
                        hash_ids=tuple(
                            h for h in obj.get("hash_ids", []) if isinstance(h, int)
                        ),
                        malformed=True,
                        malformed_reason=reason,
                    )
                )
                continue

            records.append(
                MooncakeRecord(
                    index=index,
                    timestamp_ms=float(obj["timestamp"]),
                    input_length=int(obj["input_length"]),
                    output_length=int(obj["output_length"]),
                    hash_ids=tuple(int(h) for h in obj["hash_ids"]),
                )
            )
    return records


# ── Token-ID synthesis ─────────────────────────────────────────────────────


def _stable_seed(*parts: object) -> int:
    """Derive a stable 64-bit seed from arbitrary parts via SHA256.

    Reproducible across processes/machines -- unlike Python ``hash()``.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


@dataclass
class TokenSynthesizer:
    """Deterministic hash_id -> 512-token-block and request-unique noise.

    All draws are from ``[reserved_ids, vocab_size)`` so no special-token IDs
    are ever emitted.  ``seed`` namespaces an entire run; within a run the
    same ``hash_id`` always yields the identical block.
    """

    seed: int = 0
    block_size: int = DEFAULT_BLOCK_SIZE
    vocab_size: int = DEFAULT_VOCAB_SIZE
    reserved_ids: int = DEFAULT_RESERVED_IDS
    # Namespaces keep the block RNG stream and the noise RNG stream disjoint
    # so noise tails can never accidentally reproduce a real block.
    _block_ns: str = field(default="mooncake.block", init=False)
    _noise_ns: str = field(default="mooncake.noise", init=False)
    _block_cache: dict[int, tuple[int, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.reserved_ids < 0 or self.reserved_ids >= self.vocab_size:
            raise ValueError(
                f"reserved_ids={self.reserved_ids} out of range for "
                f"vocab_size={self.vocab_size}"
            )

    @property
    def low(self) -> int:
        return self.reserved_ids

    @property
    def high(self) -> int:
        return self.vocab_size

    def block_for_hash(self, hash_id: int) -> tuple[int, ...]:
        """Return the fixed 512-token block for ``hash_id`` (memoised)."""
        cached = self._block_cache.get(hash_id)
        if cached is not None:
            return cached
        rng = np.random.default_rng(_stable_seed(self._block_ns, self.seed, hash_id))
        block = tuple(
            int(x)
            for x in rng.integers(self.low, self.high, size=self.block_size)
        )
        self._block_cache[hash_id] = block
        return block

    def noise_tail(self, request_index: int, count: int) -> list[int]:
        """Request-unique deterministic noise tokens (never shared)."""
        if count <= 0:
            return []
        rng = np.random.default_rng(
            _stable_seed(self._noise_ns, self.seed, request_index)
        )
        return [int(x) for x in rng.integers(self.low, self.high, size=count)]

    def prompt_token_ids(self, record: MooncakeRecord) -> list[int]:
        """Synthesise ``input_length`` token IDs preserving hash structure.

        Concatenate each ``hash_id``'s 512-token block; then trim to
        ``input_length`` if longer, or fill the remainder with
        request-unique noise if shorter.
        """
        ids: list[int] = []
        for hid in record.hash_ids:
            ids.extend(self.block_for_hash(hid))

        target = record.input_length
        if len(ids) >= target:
            return ids[:target]
        ids.extend(self.noise_tail(record.index, target - len(ids)))
        return ids

    def synth_request(self, record: MooncakeRecord) -> SynthRequest:
        return SynthRequest(
            index=record.index,
            timestamp_ms=record.timestamp_ms,
            prompt_token_ids=self.prompt_token_ids(record),
            output_length=record.output_length,
            input_length=record.input_length,
            hash_ids=record.hash_ids,
        )


def synth_requests(
    records: Iterable[MooncakeRecord],
    synth: TokenSynthesizer,
    *,
    skip_malformed: bool = True,
) -> list[SynthRequest]:
    """Turn parsed records into submittable requests.

    Malformed records are skipped by default (with a log line).
    """
    out: list[SynthRequest] = []
    for rec in records:
        if rec.malformed:
            log.warning(
                "skipping malformed record %d: %s", rec.index, rec.malformed_reason
            )
            if skip_malformed:
                continue
        out.append(synth.synth_request(rec))
    return out
