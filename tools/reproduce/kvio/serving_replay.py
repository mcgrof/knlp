# SPDX-License-Identifier: MIT
"""Content-neutral serving-replay primitives shared by the trace stages.

This module holds the pieces that are NOT specific to any one trace source:
the submittable-request record, the open-loop arrival scheduler, and the
reproducibility-manifest builder.  Both the Mooncake stage
(``mooncake_trace.py``) and the content-bearing stage
(``content_trace.py``) import from here so the scheduler / submit path
and the JSON manifest schema stay identical across trace sources.

The module is pure: it imports no GPU / vLLM / LMCache packages at module
top and is unit-testable on CPU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

# vLLM's KV block size is 512 tokens for the Mooncake remapped hashes; kept
# here as the shared default so both stages agree on block accounting.
DEFAULT_BLOCK_SIZE = 512
# Typical open-vocab model vocab; overridable per model.
DEFAULT_VOCAB_SIZE = 32000
# Reserve the low ID range for special tokens (BOS/EOS/PAD/UNK/...).
DEFAULT_RESERVED_IDS = 1000


# ── Typed records ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SynthRequest:
    """A record turned into a submittable request.

    ``prompt_token_ids`` are submitted directly via the Completions API with
    ``max_tokens = output_length`` and ``ignore_eos = True`` to force the
    decode length.  ``input_length`` and ``hash_ids`` are trace-source
    specific and default to empty for sources (e.g. content datasets) that do
    not carry a hash structure.
    """

    index: int
    timestamp_ms: float
    prompt_token_ids: list[int]
    output_length: int
    input_length: int = 0
    hash_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScheduledRequest:
    """A request paired with its open-loop release time (seconds)."""

    release_s: float
    request: SynthRequest


@dataclass(frozen=True)
class Wait:
    """Yielded by the scheduler when the in-flight cap is reached.

    The caller should wait for in-flight requests to complete and then
    ``send()`` the updated completed count.
    """

    in_flight: int
    max_in_flight: int


# ── Arrival scheduler ──────────────────────────────────────────────────────


def schedule_arrivals(
    requests: Sequence[SynthRequest],
    *,
    ms_to_s: float = 0.001,
    speedup: float = 1.0,
    max_in_flight: Optional[int] = None,
) -> Iterator[object]:
    """Yield ``ScheduledRequest`` in timestamp order for open-loop replay.

    ``release_s = timestamp_ms * ms_to_s / speedup`` (a larger ``speedup``
    compresses the replay).  Requests are emitted sorted by timestamp.

    Admission control (open-loop with a cap): when ``max_in_flight`` is set,
    the caller MUST drive the generator with ``send(completed_count)`` after
    each yield.  If the number of in-flight requests (yielded minus
    completed) has reached ``max_in_flight`` the generator yields a ``Wait``
    sentinel instead of the next request; the caller waits for completions
    and sends the updated count.  When ``max_in_flight`` is None the
    generator is a plain iterator (sends are ignored).
    """
    if speedup <= 0:
        raise ValueError(f"speedup must be > 0, got {speedup}")

    ordered = sorted(requests, key=lambda r: r.timestamp_ms)
    completed = 0
    yielded = 0

    for req in ordered:
        release_s = req.timestamp_ms * ms_to_s / speedup
        if max_in_flight is not None:
            # Gate until there is room for one more in-flight request.
            while (yielded - completed) >= max_in_flight:
                sent = yield Wait(
                    in_flight=yielded - completed, max_in_flight=max_in_flight
                )
                if sent is not None:
                    completed = max(completed, int(sent))
        sent = yield ScheduledRequest(release_s=release_s, request=req)
        yielded += 1
        if sent is not None:
            completed = max(completed, int(sent))


# ── Config manifest ────────────────────────────────────────────────────────


def sha256_file(path: str | Path) -> str:
    """SHA256 of a file's contents (empty string if missing)."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_config_manifest(
    *,
    model: str,
    tokenizer: str = "",
    trace_path: str | Path = "",
    seed: int = 0,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    reserved_ids: int = DEFAULT_RESERVED_IDS,
    ms_to_s: float = 0.001,
    speedup: float = 1.0,
    max_in_flight: Optional[int] = None,
    max_requests: Optional[int] = None,
    # Filled in at run time when the stack is present; placeholders otherwise.
    vllm_version: str = "",
    lmcache_version: str = "",
    tensor_parallel_size: int = 1,
    kv_cache_dtype: str = "",
    vllm_block_size: int = DEFAULT_BLOCK_SIZE,
    vllm_apc_enabled: bool = False,
    lmcache_chunk_size: int = 256,
    lmcache_save_decode_cache: bool = False,
    lmcache_save_unfull_chunk: bool = False,
    lmcache_compression: str = "off",
    lmcache_local_cpu: bool = True,
    lmcache_l2_capacity: str = "",
    lmcache_eviction_policy: str = "",
) -> dict:
    """Assemble the run's reproducibility manifest (written beside results).

    ``vllm_apc_enabled`` should be False for a storage-faithful measurement:
    vLLM's own automatic prefix cache can satisfy a hit in-GPU so LMCache L2
    never records the store/load you want for the kvio dataset.  LMCache
    content-dependent paths (serde/compression/blending, save_decode_cache,
    save_unfull_chunk) must be OFF for KV-byte geometry to match real content.
    """
    return {
        "model": model,
        "tokenizer": tokenizer or model,
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path) if trace_path else "",
        "seed": seed,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "reserved_ids": reserved_ids,
        "arrival": {
            "ms_to_s": ms_to_s,
            "speedup": speedup,
            "max_in_flight": max_in_flight,
            "max_requests": max_requests,
        },
        "vllm": {
            "version": vllm_version,
            "tensor_parallel_size": tensor_parallel_size,
            "kv_cache_dtype": kv_cache_dtype,
            "block_size": vllm_block_size,
            "automatic_prefix_caching": vllm_apc_enabled,
        },
        "lmcache": {
            "version": lmcache_version,
            "chunk_size": lmcache_chunk_size,
            "save_decode_cache": lmcache_save_decode_cache,
            "save_unfull_chunk": lmcache_save_unfull_chunk,
            "compression": lmcache_compression,
            "local_cpu": lmcache_local_cpu,
            "l2_capacity": lmcache_l2_capacity,
            "eviction_policy": lmcache_eviction_policy,
        },
    }
