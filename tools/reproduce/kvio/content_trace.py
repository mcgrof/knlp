# SPDX-License-Identifier: MIT
"""Content-bearing trace replay: real tokenization + repeated-prefix builder.

Neutral serving/benchmark infrastructure for replaying *real* conversation
(LMSYS-Chat-1M) and long-context (LongBench) datasets through vLLM + LMCache
and measuring timing / prefix-cache reuse / KV-offload storage behaviour --
and capturing the resulting KV corpus + kvio semantic trace for offline,
GPU-free KV-geometry / reuse / storage analysis.

Unlike the Mooncake stage (``mooncake_trace.py``), which synthesises token IDs
to preserve a *hash structure*, this module tokenizes the **real dataset
content** with the target model's tokenizer, so the prefixes and their KV
bytes are the ones the model actually produces.

Repeated-prefix construction (this is what yields real prefix-cache hits):

  * LMSYS multi-turn -- for one conversation, request *k* is the tokenized
    concatenation of turns ``1..k``.  Successive requests therefore share a
    genuine GROWING token prefix: request ``k+1``'s ``prompt_token_ids`` has
    request ``k``'s as a strict prefix.  To make that invariant EXACT
    (independent of BPE merges across a turn boundary) we tokenize each turn
    independently and concatenate the token-ID sequences -- KV blocks are
    per-token, so this is the faithful realization of "prompt = tokens of
    turns 1..k" and it guarantees the strict-prefix property the cache needs.

  * LongBench long-context -- every question about one document is
    ``tokens(document) ++ tokens(question)``; all questions about a document
    share the same long document token prefix.

Datasets have no timestamps, so arrival times are SYNTHESISED (fixed-rate or
seeded Poisson) and fed to the SAME open-loop scheduler used by the Mooncake
stage (``serving_replay.schedule_arrivals``).  Requests are returned as
``serving_replay.SynthRequest`` objects so the scheduler / submit path is
reused unchanged: submit ``prompt_token_ids`` directly via the Completions
API (no chat template, no special tokens) with ``max_tokens = output_length``
+ ``ignore_eos`` to force the decode length.

This module is pure: it imports no GPU / vLLM / LMCache / transformers
packages at module top (transformers is imported lazily inside
``load_hf_tokenizer``) so it imports and is unit-testable on a CPU-only box.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .serving_replay import (
    DEFAULT_RESERVED_IDS,
    DEFAULT_VOCAB_SIZE,
    SynthRequest,
    build_config_manifest,
)

log = logging.getLogger(__name__)

# Default forced decode length when a dataset record carries no target.
DEFAULT_OUTPUT_LENGTH = 32


# ── Tokenization ───────────────────────────────────────────────────────────
#
# A ``Tokenizer`` here is any object exposing ``encode(text) -> list[int]``.
# The real run uses the model's HF tokenizer (``load_hf_tokenizer``); the
# CPU-only smoke path and the unit tests use ``HashTokenizer`` so the pure
# request-building / prefix / arrival logic runs without ``transformers``.


def _stable_seed(*parts: object) -> int:
    """Derive a stable 64-bit seed from arbitrary parts via SHA256.

    Reproducible across processes/machines -- unlike Python ``hash()``.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


@dataclass
class HashTokenizer:
    """Deterministic whitespace-word -> token-ID map (no transformers needed).

    Each whitespace-delimited word maps to a fixed ID in
    ``[reserved_ids, vocab_size)`` via a stable SHA256 hash, so the same word
    always yields the same ID across processes/machines.  This is NOT a real
    tokenizer -- it exists only so the request-builder / prefix / arrival
    logic (and its tests) run on a CPU box without ``transformers``, while
    preserving the property that matters for these tests: identical text
    prefixes produce identical token-ID prefixes.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    reserved_ids: int = DEFAULT_RESERVED_IDS

    def _word_id(self, word: str) -> int:
        span = self.vocab_size - self.reserved_ids
        return self.reserved_ids + (_stable_seed("word", self.seed, word) % span)

    def encode(self, text: str) -> list[int]:
        return [self._word_id(w) for w in text.split()]


def load_hf_tokenizer(model: str):
    """Load the model's HF tokenizer (``transformers`` imported lazily).

    Returned object exposes ``encode(text) -> list[int]`` via a thin wrapper
    so callers stay tokenizer-agnostic.  We always encode with
    ``add_special_tokens=False`` -- the repeated-prefix invariant and the
    direct Completions submission both require raw content tokens with no
    BOS/EOS injected.
    """
    from transformers import AutoTokenizer  # lazy: keeps module CPU-importable

    tok = AutoTokenizer.from_pretrained(model)

    class _Wrap:
        def __init__(self, t):
            self._t = t

        def encode(self, text: str) -> list[int]:
            return list(self._t(text, add_special_tokens=False)["input_ids"])

    return _Wrap(tok)


# ── Repeated-prefix request builders ───────────────────────────────────────


def _concat_turn_ids(
    tokenizer, turns: Sequence[dict], *, joiner: str = "\n"
) -> list[list[int]]:
    """Tokenize each turn's text independently; return the per-turn ID lists.

    ``joiner`` is appended to every turn's text before tokenizing so turns do
    not run together; because each turn is tokenized on its own and the lists
    are concatenated by the caller, the growing-prefix invariant holds exactly.
    """
    out: list[list[int]] = []
    for turn in turns:
        text = str(turn.get("content", ""))
        out.append(tokenizer.encode(text + joiner))
    return out


def build_lmsys_requests(
    conversations: Sequence[dict],
    tokenizer,
    *,
    default_output_length: int = DEFAULT_OUTPUT_LENGTH,
    start_index: int = 0,
) -> list[SynthRequest]:
    """Build growing-prefix requests from normalized LMSYS conversations.

    ``conversations`` is a list of normalized records (see
    ``content_datasets.lmsys.normalize_lmsys``): each has ``turns`` =
    ``[{"role", "content"}, ...]``.  For a conversation with turns
    ``t1..tn`` we emit one request per turn ``k`` whose ``prompt_token_ids``
    is ``tokens(t1) ++ ... ++ tokens(tk)`` -- so within a conversation each
    request's prompt is a strict prefix of the next.  Distinct conversations
    produce distinct (independent) prefix chains.

    ``timestamp_ms`` is left 0.0 here; call ``assign_arrivals`` (or set it
    yourself) before scheduling.
    """
    requests: list[SynthRequest] = []
    idx = start_index
    for conv in conversations:
        turns = conv.get("turns", [])
        if not turns:
            continue
        per_turn = _concat_turn_ids(tokenizer, turns)
        running: list[int] = []
        out_len = int(conv.get("output_length", default_output_length) or
                      default_output_length)
        for turn_ids in per_turn:
            running = running + turn_ids
            if not running:
                continue
            requests.append(
                SynthRequest(
                    index=idx,
                    timestamp_ms=0.0,
                    prompt_token_ids=list(running),
                    output_length=out_len,
                    input_length=len(running),
                )
            )
            idx += 1
    return requests


def build_longbench_requests(
    documents: Sequence[dict],
    tokenizer,
    *,
    default_output_length: int = DEFAULT_OUTPUT_LENGTH,
    start_index: int = 0,
    joiner: str = "\n\n",
) -> list[SynthRequest]:
    """Build shared-document-prefix requests from normalized LongBench records.

    ``documents`` is a list of normalized records (see
    ``content_datasets.longbench.normalize_longbench``): each has
    ``document`` (long context text) and ``questions`` (list[str]).  Each
    question becomes one request whose ``prompt_token_ids`` is
    ``tokens(document) ++ tokens(question)`` -- so every request about one
    document shares the document's token prefix (real prefix-cache reuse).
    """
    requests: list[SynthRequest] = []
    idx = start_index
    for doc in documents:
        document = str(doc.get("document", ""))
        questions = doc.get("questions", [])
        if not document or not questions:
            continue
        doc_ids = tokenizer.encode(document + joiner)
        out_len = int(doc.get("output_length", default_output_length) or
                      default_output_length)
        for q in questions:
            q_ids = tokenizer.encode(str(q))
            prompt = doc_ids + q_ids
            requests.append(
                SynthRequest(
                    index=idx,
                    timestamp_ms=0.0,
                    prompt_token_ids=list(prompt),
                    output_length=out_len,
                    input_length=len(prompt),
                )
            )
            idx += 1
    return requests


# ── Arrival synthesis ──────────────────────────────────────────────────────


def synthesize_arrival_times(
    n: int,
    *,
    mode: str = "fixed",
    rate_hz: float = 10.0,
    seed: int = 0,
) -> list[float]:
    """Synthesize ``n`` monotonically non-decreasing arrival times (ms).

    ``mode="fixed"``  -> evenly spaced at ``1000/rate_hz`` ms.
    ``mode="poisson"`` -> exponential inter-arrivals (mean ``1000/rate_hz``
    ms) from a seeded ``numpy`` generator; deterministic under ``seed``.

    Datasets carry no real timestamps, so these are a *synthesised* arrival
    process, recorded in the manifest as such.
    """
    if n <= 0:
        return []
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
    period_ms = 1000.0 / rate_hz
    if mode == "fixed":
        return [i * period_ms for i in range(n)]
    if mode == "poisson":
        rng = np.random.default_rng(seed)
        gaps = rng.exponential(scale=period_ms, size=n)
        times = np.cumsum(gaps) - gaps[0]  # start at t=0
        return [float(t) for t in times]
    raise ValueError(f"unknown arrival mode {mode!r} (want 'fixed' or 'poisson')")


def assign_arrivals(
    requests: Sequence[SynthRequest],
    *,
    mode: str = "fixed",
    rate_hz: float = 10.0,
    seed: int = 0,
) -> list[SynthRequest]:
    """Return copies of ``requests`` with synthesised ``timestamp_ms`` set.

    Times are assigned in the given request order (request 0 arrives first).
    """
    times = synthesize_arrival_times(len(requests), mode=mode, rate_hz=rate_hz,
                                     seed=seed)
    out: list[SynthRequest] = []
    for req, ts in zip(requests, times):
        out.append(
            SynthRequest(
                index=req.index,
                timestamp_ms=float(ts),
                prompt_token_ids=req.prompt_token_ids,
                output_length=req.output_length,
                input_length=req.input_length,
                hash_ids=req.hash_ids,
            )
        )
    return out


# ── Reproducibility manifest ───────────────────────────────────────────────


def build_content_manifest(
    *,
    model: str,
    dataset: str,
    dataset_revision: str = "",
    seed: int = 0,
    speedup: float = 1.0,
    max_requests: Optional[int] = None,
    arrival_mode: str = "fixed",
    arrival_rate_hz: float = 10.0,
    kvio_trace_path: str = "",
    l2_backend: str = "raw_block",
    l2_path: str = "",
    l2_capacity: str = "",
    kv_cache_dtype: str = "bf16",
    chunk_size: int = 256,
    save_decode_cache: bool = False,
    save_unfull_chunk: bool = False,
    tensor_parallel_size: int = 1,
    eviction_policy: str = "off",
) -> dict:
    """Extend the shared manifest with content + capture-discipline fields.

    Bakes in the storage-faithful capture discipline (see the module and stage
    docstrings): vLLM automatic prefix caching OFF, KV dtype BF16/FP16 (never
    FP8), no serde/compression/blending, ``save_unfull_chunk=false``, TP=1,
    L2 eviction off / capacity large, and the kvio semantic-trace path so the
    identity join is captured alongside the KV corpus.
    """
    manifest = build_config_manifest(
        model=model,
        seed=seed,
        speedup=speedup,
        max_requests=max_requests,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
        vllm_apc_enabled=False,
        lmcache_chunk_size=chunk_size,
        lmcache_save_decode_cache=save_decode_cache,
        lmcache_save_unfull_chunk=save_unfull_chunk,
        lmcache_compression="off",
        lmcache_l2_capacity=l2_capacity,
        lmcache_eviction_policy=eviction_policy,
    )
    manifest["dataset"] = {
        "name": dataset,
        "revision": dataset_revision,
        "arrival_mode": arrival_mode,
        "arrival_rate_hz": arrival_rate_hz,
        "synthesised_arrivals": True,
    }
    # Capture-discipline block: the L2 corpus + kvio semantic trace this run
    # is configured to produce, plus the storage-faithfulness flags.
    manifest["capture"] = {
        "kvio_trace_path": kvio_trace_path,
        "l2_backend": l2_backend,
        "l2_path": l2_path,
        "l2_capacity": l2_capacity,
        "kv_cache_dtype": kv_cache_dtype,
        "serde": "off",
        "compression": "off",
        "blending": "off",
        "mla": False,
        "full_attention": True,
        "tensor_parallel_size": tensor_parallel_size,
        "eviction_policy": eviction_policy,
        "vllm_automatic_prefix_caching": False,
    }
    return manifest


# ── Phase-2 Q-probe hook (STUB -- intentionally not implemented) ────────────


def q_probe_hook(
    *args,
    enabled: bool = False,
    **kwargs,
) -> Optional[dict]:
    """Documented hook for a FUTURE offline-attention-analysis extension.

    NOT IMPLEMENTED in this phase -- this is a deliberate stub so the call
    site and the manifest schema exist before the capture is wired up.

    A Phase-2 extension would, for a SAMPLED set of decode positions, capture:

      * the post-RoPE query vectors at those positions,
      * the query-head -> kv-head mapping (GQA grouping) for the model, and
      * the attention softmax scale,

    so that attention scores against the captured KV corpus can be
    reconstructed offline on CPU (no GPU) for KV-geometry analysis.  The
    sampling keeps the captured Q volume small relative to the KV corpus.

    Wiring this requires a model-forward hook on a GPU run and is intentionally
    left for a later, deliberate change; it is out of scope for the
    capture-by-default KV-corpus + kvio-trace path implemented now.
    """
    if enabled:
        raise NotImplementedError(
            "q_probe_hook is a Phase-2 stub; Q capture is not implemented yet"
        )
    return None


def default_tokenizer(model: str, *, seed: int = 0) -> object:
    """Return the HF tokenizer for ``model``; fall back to ``HashTokenizer``.

    The fallback keeps the CPU-only offline smoke path (and anyone without
    ``transformers``) working, at the cost of not being a real tokenizer.  The
    fallback is signalled to callers by the returned object's type.
    """
    try:
        return load_hf_tokenizer(model)
    except Exception as e:  # ImportError, network/model errors, ...
        log.warning(
            "falling back to HashTokenizer (transformers/model unavailable: %s)", e
        )
        return HashTokenizer(seed=seed)


def is_real_tokenizer(tokenizer: object) -> bool:
    """True when ``tokenizer`` is a real HF tokenizer (not the hash fallback)."""
    return not isinstance(tokenizer, HashTokenizer)


BuilderFn = Callable[..., list[SynthRequest]]
