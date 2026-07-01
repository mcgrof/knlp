# SPDX-License-Identifier: GPL-2.0
"""Adapters for ten published KV-cache compression methods.

These wrap methods from the KVPress leaderboard (StreamingLLM, Knorm, SnapKV,
Expected Attention, TOVA, H2O/Observed-Attention, PyramidKV) plus three widely
benchmarked ones from their own papers (Quest, Ada-KV, KIVI). The point is not
to reimplement their kernels -- it is to run each through the one narrow PIA slot
and read off where it is safe to deploy.

Each adapter models the method's *contract behavior*, which is a design fact of
the method, not a property of any particular attention values:

  - granularity: does it keep whole blocks (pages) or individual tokens? Token
    -level selection leaves partial blocks that block-hash prefix caching cannot
    reuse without a custom connector.
  - query dependence: does the kept set depend on the query / prompt tail /
    generation? An adaptive method picks different KV for different queries, so
    two requests that share a prefix no longer agree on the prefix object.
  - per head: do different heads keep different positions? Then no block is fully
    intact across all heads, which deepens the partial-block problem.
  - shape: does it change dtype / add residual metadata (a codec)?

The contract axis (CPU, no model) classifies each method from these properties.
The drift axis (GPU) needs each method's real selection over a real model and is
a separate follow-up; the faithful path is to drive it from the KVPress presses.
The `LEADERBOARD` table records provenance so the classification is auditable.
"""

from __future__ import annotations

import hashlib
import random

from .datatypes import BlockManifest, Mode


LEADERBOARD = {
    "streaming_llm": ("StreamingLLM", "Xiao et al. 2023", "KVPress"),
    "knorm": ("Knorm (key-norm)", "Devoto et al. 2024", "KVPress"),
    "snapkv": ("SnapKV", "Li et al. 2024", "KVPress"),
    "expected_attention": ("Expected Attention", "NVIDIA KVPress 2024", "KVPress"),
    "tova": ("TOVA", "Oren et al. 2024", "KVPress"),
    "h2o": ("H2O (heavy-hitter)", "Zhang et al. 2023", "KVPress"),
    "pyramidkv": ("PyramidKV", "Cai et al. 2024", "KVPress"),
    "quest": ("Quest", "Tang et al. 2024", "paper"),
    "adakv": ("Ada-KV", "Feng et al. 2024", "paper"),
    "kivi": ("KIVI (2-bit)", "Liu et al. 2024", "paper"),
}


def _query_seed(request: dict) -> int:
    q = str(request.get("query", request.get("id", "")))
    return int(hashlib.sha256(q.encode()).hexdigest()[:8], 16)


def _whole_block_manifest(num_blocks, block_size, blocks):
    return BlockManifest.from_selected(num_blocks, blocks, block_size=block_size)


def _scatter_token_manifest(
    num_blocks,
    block_size,
    budget_k,
    request,
    *,
    query_dependent,
    sink=1,
    recent=2,
    seed=0,
):
    """Keep the sink and recent windows as whole blocks, then scatter the middle
    token budget across individual positions. Token-level selection rarely fills
    a whole middle block, so most kept middle blocks come back PARTIAL -- exactly
    the pattern a block-hash cache cannot reuse. When query_dependent, the
    scatter is seeded by the query so it differs request to request.
    """
    prefix_len = num_blocks * block_size
    kept = set()
    for t in range(min(sink, num_blocks) * block_size):
        kept.add(t)
    for t in range(max(0, (num_blocks - recent) * block_size), prefix_len):
        kept.add(t)
    mid_budget = max(0, budget_k - sink - recent) * block_size
    lo = sink * block_size
    hi = max(lo, (num_blocks - recent) * block_size)
    if mid_budget > 0 and hi > lo:
        s = seed + (_query_seed(request) if query_dependent else 0)
        rng = random.Random(s)
        kept.update(rng.sample(range(lo, hi), min(mid_budget, hi - lo)))
    return BlockManifest.from_token_mask(num_blocks, block_size, kept)


class _LibraryMethod:
    """Shared plumbing. Subclasses set the contract properties and manifest()."""

    name = "library"
    mode = Mode.SELECTOR.value
    policy = "prefix_cache"
    cache_key_fields = ("prefix_hash",)
    has_custom_restore_path = False
    shape_preserved = True
    # descriptive (surfaced in reports, not gates)
    query_dependent = False
    generation_dependent = False
    per_head = False
    granularity = "block"  # block | token

    def __init__(
        self, sink: int = 1, recent: int = 2, declare_query_hash: bool = False
    ):
        self.sink = sink
        self.recent = recent
        if declare_query_hash:
            self.cache_key_fields = ("prefix_hash", "query_hash")

    def manifest(self, request, num_blocks, budget_k, block_size=1):
        raise NotImplementedError

    def select_blocks(self, request, num_blocks, budget_k):
        return self.manifest(request, num_blocks, budget_k, 1).selected


# --- Structural, query-independent (the prefix-cache-clean case) ----------


class StreamingLLM(_LibraryMethod):
    """Attention sinks (first tokens) plus a sliding recent window. No attention
    scores, no query -- purely positional. The one method here that is a clean
    prefix-cache citizen: deterministic, query-independent, whole-block.
    """

    name = "streaming_llm"
    granularity = "block"

    def manifest(self, request, num_blocks, budget_k, block_size=1):
        keep = list(range(min(self.sink, num_blocks)))
        recent = min(max(0, budget_k - self.sink), num_blocks)
        keep += list(range(num_blocks - recent, num_blocks))
        return _whole_block_manifest(num_blocks, block_size, keep)


# --- Query-independent but token-level / codec (custom-connector cases) ---


class Knorm(_LibraryMethod):
    """Keep tokens by key L2 norm -- a query-free heuristic. Query-independent,
    so it is prefix-stable; but it selects per head at token granularity, so the
    kept blocks are partial and cannot be reused by plain block-hash caching.
    """

    name = "knorm"
    granularity = "token"
    per_head = True

    def manifest(self, request, num_blocks, budget_k, block_size=1):
        # query_dependent=False -> same scatter every request (key norms are
        # fixed by the prefix), seeded by prefix only.
        return _scatter_token_manifest(
            num_blocks,
            block_size,
            budget_k,
            request,
            query_dependent=False,
            sink=self.sink,
            recent=self.recent,
        )


class KIVI(_LibraryMethod):
    """2-bit quantization: per-channel keys, per-token values, plus a full
    -precision residual window. Every block stays present, so it is not a
    selector -- but it changes dtype and adds residual metadata, so the stored
    object is not the original layout. Query-independent; reloadable only through
    a KIVI-aware connector.
    """

    name = "kivi"
    mode = Mode.CODEC.value
    granularity = "block"
    shape_preserved = False
    has_custom_restore_path = True

    def manifest(self, request, num_blocks, budget_k, block_size=1):
        # codec keeps all blocks; the shape change is expressed via shape_preserved
        return _whole_block_manifest(num_blocks, block_size, range(num_blocks))


# --- Adaptive: query / prompt / generation dependent (the routing class) --


class _AdaptiveTokenMethod(_LibraryMethod):
    """Base for adaptive, query-dependent, per-head, token-level selectors. These
    are the methods PIA is built to flag: different kept KV per query under a
    prefix-only key means two requests disagree on the prefix object.
    """

    granularity = "token"
    per_head = True
    query_dependent = True

    def manifest(self, request, num_blocks, budget_k, block_size=1):
        return _scatter_token_manifest(
            num_blocks,
            block_size,
            budget_k,
            request,
            query_dependent=True,
            sink=self.sink,
            recent=self.recent,
        )


class SnapKV(_AdaptiveTokenMethod):
    """Selects prefix KV using the attention of an observation window (the prompt
    tail) after prefill, per head. The observation window is query content, so
    the selection is query-dependent.
    """

    name = "snapkv"


class ExpectedAttention(_AdaptiveTokenMethod):
    """Scores prefix KV by expected attention from a model of future queries,
    conditioned on the prompt. Different suffix over the same prefix gives a
    different prompt and a different selection.
    """

    name = "expected_attention"


class TOVA(_AdaptiveTokenMethod):
    """Keeps top tokens by the current decode token's attention, updated every
    step. Query- and generation-dependent.
    """

    name = "tova"
    generation_dependent = True


class H2O(_AdaptiveTokenMethod):
    """Heavy-hitter eviction: keep tokens with high accumulated attention plus a
    recent window, evicting during decode. Query- and generation-dependent, and
    non-stationary across the decode.
    """

    name = "h2o"
    generation_dependent = True


class PyramidKV(_AdaptiveTokenMethod):
    """SnapKV-style selection with a per-layer budget pyramid (more in lower
    layers). Query-dependent, per-head, and now also per-layer heterogeneous.
    """

    name = "pyramidkv"


class AdaKV(_AdaptiveTokenMethod):
    """Adaptive budget allocation across heads on top of a SnapKV/Pyramid base.
    Query-dependent with per-head variable budgets.
    """

    name = "adakv"


class Quest(_LibraryMethod):
    """Query-aware page selection at decode: keep the full prefix in memory, but
    per query estimate each page's attention upper bound and read only the top
    pages. It does not change the stored prefix -- it is a router. Modeled as
    whole-page (block) selection, query-dependent, honestly labeled routing.
    """

    name = "quest"
    granularity = "block"
    per_head = True
    query_dependent = True
    policy = "routing"

    def manifest(self, request, num_blocks, budget_k, block_size=1):
        sink = list(range(min(self.sink, num_blocks)))
        recent = list(range(max(0, num_blocks - self.recent), num_blocks))
        mid = [b for b in range(num_blocks) if b not in set(sink + recent)]
        rng = random.Random(_query_seed(request))
        k = min(max(0, budget_k - len(sink) - len(recent)), len(mid))
        pages = rng.sample(mid, k) if k else []
        return _whole_block_manifest(num_blocks, block_size, sink + recent + pages)


LIBRARY = {
    "streaming_llm": StreamingLLM,
    "knorm": Knorm,
    "kivi": KIVI,
    "snapkv": SnapKV,
    "expected_attention": ExpectedAttention,
    "tova": TOVA,
    "h2o": H2O,
    "pyramidkv": PyramidKV,
    "adakv": AdaKV,
    "quest": Quest,
}


def make(name: str, **kwargs):
    """Factory for the CLI's package.module:factory spec, e.g.
    routing.prefix_integrity.library_adapters:make with config {"name": "snapkv"}.
    """
    key = name.lower()
    if key not in LIBRARY:
        raise KeyError(f"unknown method '{name}'; known: {sorted(LIBRARY)}")
    ctor_kwargs = {k: v for k, v in kwargs.items() if k != "name"}
    return LIBRARY[key](**ctor_kwargs)
