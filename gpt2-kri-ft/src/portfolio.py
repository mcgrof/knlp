"""Phase-A protected-portfolio composition (ordered-quota router).

Per the Codex-pinned grid spec (knlp-key-results lattice-kri/
portfolio-router-20260612/PLAN.md): compose a budget of K blocks from ordered
candidate lists with hard quotas, deduplicating in priority order

    R reserve (m picks, round-robin over the two relevance probe lists)
    -> H reserve (h picks by exact attention mass)
    -> D fill (remaining, residual-diversity order restricted to a
       relevance-qualified pool), falling back to leftover relevance.

The R reserve is multi-probe (recent-Q + last-K interleaved) per the
ChatGPT-Pro revision: protect top evidence for multiple plausible query
directions, not one list's top-m. Lists are precomputed orderings — the grid
tests fixed compositions of candidate lists, not re-conditioned selection.

Spec encoding in router names: pf_m<spec>_h<spec>, spec in
{0, 1, f125, f250, f500} = {0, 1, ceil(.125K), ceil(.25K), ceil(.5K)}.
Combos with m+h > 0.75K are invalid (caller must skip).
"""

from __future__ import annotations

import math
import re

_PF_RE = re.compile(r"^pf_m(0|1|f\d{3})_h(0|1|f\d{3})$")


def parse_pf_name(name: str):
    m = _PF_RE.match(name)
    if not m:
        raise ValueError(f"bad portfolio router name: {name}")
    return m.group(1), m.group(2)


def spec_n(spec: str, K: int) -> int:
    if spec == "0":
        return 0
    if spec == "1":
        return 1
    return math.ceil(int(spec[1:]) / 1000.0 * K)


def pf_valid(name: str, K: int) -> bool:
    ms, hs = parse_pf_name(name)
    return spec_n(ms, K) + spec_n(hs, K) <= math.ceil(0.75 * K)


def compose(r1, r2, h, d, K: int, m: int, hq: int, pool=None):
    """Compose one head's kept-block list.

    r1, r2, h, d: python lists of block indices, best-first (r1=recent-Q
    relevance, r2=last-K relevance, h=mass, d=diversity). pool: set of
    qualified candidates for the D fill (None = no restriction). Returns a
    list of exactly min(K, available) block indices.
    """
    kept, seen = [], set()

    def take(seq, n, restrict=None):
        t = 0
        for b in seq:
            if t >= n:
                break
            if b in seen or (restrict is not None and b not in restrict):
                continue
            kept.append(b)
            seen.add(b)
            t += 1

    # R reserve: round-robin interleave of the two probe lists
    inter = []
    for i in range(max(len(r1), len(r2))):
        if i < len(r1):
            inter.append(r1[i])
        if i < len(r2):
            inter.append(r2[i])
    take(inter, m)
    take(h, hq)
    take(d, max(0, K - len(kept)), restrict=pool)
    if len(kept) < K:  # D pool exhausted -> fall back to leftover relevance
        take(inter, K - len(kept))
    return kept[:K]


def qualified_pool(r1, r2, K: int) -> set:
    """Top-2K of each relevance list (union <= 4K candidates)."""
    return set(r1[: 2 * K]) | set(r2[: 2 * K])
