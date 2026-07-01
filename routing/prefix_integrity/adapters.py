# SPDX-License-Identifier: GPL-2.0
"""Adapters: the one narrow slot every candidate algorithm goes through.

The point of PIA is to stuff arbitrary KV-cache inventions through a single
interface and watch them break consistently. An adapter expresses an algorithm
as one of three things: a block-survival selector, a KV codec, or both. The
harness never needs to know whether the algorithm is KRI, a quantizer, an
eviction policy, or a clustering scheme -- it only sees the manifest and the
(optional) transformed cartridge.

Built-in adapters wrap the baselines the routing stack already has (full,
random middle, recency, anchor+recency / A1R2K13 structural pinning, an offline
KRI per-block prior, and a deliberately query-aware selector) so a new algorithm
is graded against known-good and known-dangerous references. Custom algorithms
load from a "package.module:factory" spec.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from typing import Optional, Protocol, runtime_checkable

from .datatypes import BlockManifest, CandidateArtifact, Mode


@runtime_checkable
class PrefixAlgorithmAdapter(Protocol):
    name: str
    mode: str  # one of Mode values

    def select_blocks(self, request: dict, num_blocks: int, budget_k: int) -> list:
        """Return the ordered block ids this algorithm keeps for this request."""
        ...


def parse_pins(spec: Optional[str]) -> Optional[tuple]:
    """Parse a structural pin like 'A1R2K13' -> (anchor=1, recent=2, kri=13).

    Returns None if spec is falsy. Missing letters default to 0.
    """
    if not spec:
        return None
    a = r = k = 0
    for letter, val in re.findall(r"([ARK])(\d+)", spec.upper()):
        if letter == "A":
            a = int(val)
        elif letter == "R":
            r = int(val)
        elif letter == "K":
            k = int(val)
    return (a, r, k)


def _query_seed(request: dict) -> int:
    q = str(request.get("query", request.get("id", "")))
    return int(hashlib.sha256(q.encode()).hexdigest()[:8], 16)


def _middle_eligible(num_blocks: int, anchor: int, recent: int) -> list:
    """Blocks that are neither anchor nor recent (the routable middle)."""
    lo = anchor
    hi = num_blocks - recent
    return list(range(max(0, lo), max(lo, hi)))


class _BaseSelector:
    """Mixin providing a manifest/prepare wrapper around select_blocks."""

    name = "base"
    mode = Mode.SELECTOR.value
    # Declared intent the harness checks observed behavior against.
    policy = "prefix_cache"
    cache_key_fields = ("prefix_hash",)
    has_custom_restore_path = False

    def manifest(
        self, request: dict, num_blocks: int, budget_k: int, block_size: int = 1
    ) -> BlockManifest:
        sel = self.select_blocks(request, num_blocks, budget_k)
        return BlockManifest.from_selected(num_blocks, sel, block_size=block_size)

    def artifact(
        self, request, num_blocks, budget_k, config_hash=""
    ) -> CandidateArtifact:
        sel = sorted(set(self.select_blocks(request, num_blocks, budget_k)))
        digest = hashlib.sha256(
            (self.name + "|" + ",".join(map(str, sel))).encode()
        ).hexdigest()[:16]
        return CandidateArtifact(
            algorithm_id=self.name,
            config_hash=config_hash,
            mode=self.mode,
            artifact_digest=digest,
            cache_key_fields={k: "1" for k in self.cache_key_fields},
        )


class FullAdapter(_BaseSelector):
    """Keep every block. The trivially-safe reference: deterministic, query
    independent, whole-block, PRE 1.0.
    """

    name = "full"

    def select_blocks(self, request, num_blocks, budget_k):
        return list(range(num_blocks))


class RecencyAdapter(_BaseSelector):
    """Keep the last K blocks. Query independent and deterministic, so its
    geometry is prefix-safe; it just drops the middle of the document.
    """

    name = "recency"

    def select_blocks(self, request, num_blocks, budget_k):
        k = min(budget_k, num_blocks)
        return list(range(num_blocks - k, num_blocks))


class RandomMiddleAdapter(_BaseSelector):
    """Random middle blocks. If `per_query` the seed mixes the query (selection
    changes per request -> query dependent); otherwise it is fixed (query
    independent but content-blind). Used as the unstable reference.
    """

    name = "random"

    def __init__(
        self, seed: int = 0, per_query: bool = True, anchor: int = 1, recent: int = 2
    ):
        self.seed = seed
        self.per_query = per_query
        self.anchor = anchor
        self.recent = recent
        if per_query:
            self.cache_key_fields = ("prefix_hash", "query_hash")

    def select_blocks(self, request, num_blocks, budget_k):
        import random

        elig = _middle_eligible(num_blocks, self.anchor, self.recent)
        s = self.seed + (_query_seed(request) if self.per_query else 0)
        rng = random.Random(s)
        k = min(budget_k, len(elig))
        return sorted(rng.sample(elig, k)) if k else []


class AnchorRecencyAdapter(_BaseSelector):
    """Structural pinning: keep `anchor` first blocks, `recent` last blocks, and
    fill the rest of the budget with evenly spaced middle blocks. Query
    independent and deterministic -> prefix-safe geometry. This is the A1R2K13
    structural baseline when the middle is filled structurally rather than by
    KRI scores.
    """

    name = "anchor_recency"

    def __init__(self, anchor: int = 1, recent: int = 2):
        self.anchor = anchor
        self.recent = recent

    def select_blocks(self, request, num_blocks, budget_k):
        anc = list(range(min(self.anchor, num_blocks)))
        rec = list(range(max(0, num_blocks - self.recent), num_blocks))
        pinned = sorted(set(anc + rec))
        remaining = budget_k - len(pinned)
        elig = [b for b in _middle_eligible(num_blocks, self.anchor, self.recent)]
        mid = []
        if remaining > 0 and elig:
            step = max(1, len(elig) // remaining)
            mid = elig[::step][:remaining]
        return sorted(set(pinned + mid))


class KRIPriorAdapter(_BaseSelector):
    """Offline KRI per-block prior: load block_affinities[L,H,N] from a .pt,
    collapse over layers and heads, keep the top-K blocks. The prior is computed
    once per prefix and does not depend on the decode query, so it is query
    independent and prefix-safe -- the production KRI-D-kv-sum shape.
    """

    name = "kri_prior"

    def __init__(
        self, prior_path: str, anchor: int = 1, recent: int = 2, reduce: str = "mean"
    ):
        self.prior_path = prior_path
        self.anchor = anchor
        self.recent = recent
        self.reduce = reduce
        self._scores = None

    def _block_scores(self, num_blocks: int):
        if self._scores is not None:
            return self._scores
        import torch  # lazy

        obj = torch.load(self.prior_path, map_location="cpu", weights_only=False)
        aff = obj["block_affinities"] if isinstance(obj, dict) else obj
        # [L, H, N] -> [N]
        red = (
            aff.float().mean(dim=(0, 1))
            if self.reduce == "mean"
            else aff.float().amax(dim=(0, 1))
        )
        self._scores = red[:num_blocks].tolist()
        return self._scores

    def select_blocks(self, request, num_blocks, budget_k):
        scores = self._block_scores(num_blocks)
        anc = list(range(min(self.anchor, num_blocks)))
        rec = list(range(max(0, num_blocks - self.recent), num_blocks))
        pinned = set(anc + rec)
        remaining = budget_k - len(pinned)
        elig = [b for b in _middle_eligible(num_blocks, self.anchor, self.recent)]
        elig.sort(key=lambda b: scores[b] if b < len(scores) else 0.0, reverse=True)
        mid = elig[: max(0, remaining)]
        return sorted(pinned | set(mid))


class QueryAwareSelectorAdapter(_BaseSelector):
    """A query-conditioned selector (KRI-Q shape): the kept blocks depend on the
    decode query. Declared here as policy='prefix_cache' on purpose -- this is
    the adapter that should be SHAMED as DANGEROUS_FOR_PREFIX_SHARING when keyed
    by prefix_hash alone, and demoted to SAFE_ONLY_WITH_EXTENDED_CACHE_KEY once
    query_hash is declared. Selection here is a deterministic function of the
    query string (no model needed) so the determinism vs query-dependence axes
    are cleanly separated.
    """

    name = "query_aware"

    def __init__(
        self, anchor: int = 1, recent: int = 2, declare_query_hash: bool = False
    ):
        self.anchor = anchor
        self.recent = recent
        if declare_query_hash:
            self.cache_key_fields = ("prefix_hash", "query_hash")

    def select_blocks(self, request, num_blocks, budget_k):
        import random

        elig = _middle_eligible(num_blocks, self.anchor, self.recent)
        anc = list(range(min(self.anchor, num_blocks)))
        rec = list(range(max(0, num_blocks - self.recent), num_blocks))
        pinned = set(anc + rec)
        remaining = budget_k - len(pinned)
        # Deterministic per query, different across queries.
        rng = random.Random(_query_seed(request))
        elig = [b for b in elig if b not in pinned]
        k = min(max(0, remaining), len(elig))
        mid = rng.sample(elig, k) if k else []
        return sorted(pinned | set(mid))


_BUILTINS = {
    "full": FullAdapter,
    "recency": RecencyAdapter,
    "random": RandomMiddleAdapter,
    "anchor_recency": AnchorRecencyAdapter,
    "a1r2k13": lambda: AnchorRecencyAdapter(anchor=1, recent=2),
    "kri_prior": KRIPriorAdapter,
    "query_aware": QueryAwareSelectorAdapter,
}


def load_adapter(spec: str, config: Optional[dict] = None):
    """Resolve an adapter from a built-in name or a 'package.module:factory'
    spec. The factory is called with **config (filtered to its needs by the
    caller). Built-ins that need args (kri_prior) read them from config.
    """
    config = config or {}
    if ":" in spec:
        mod_name, _, factory = spec.partition(":")
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, factory)
        return fn(**config) if config else fn()
    key = spec.lower()
    if key in _BUILTINS:
        ctor = _BUILTINS[key]
        try:
            return ctor(**config) if config else ctor()
        except TypeError:
            return ctor()
    # Published-method library (SnapKV, H2O, StreamingLLM, ...).
    from .library_adapters import LIBRARY

    if key in LIBRARY:
        ctor = LIBRARY[key]
        try:
            return ctor(**config) if config else ctor()
        except TypeError:
            return ctor()
    known = sorted(_BUILTINS) + sorted(LIBRARY)
    raise KeyError(f"unknown adapter '{spec}'; known: {known}")
