# SPDX-License-Identifier: GPL-2.0
"""MVP 1 (manifest integrity) + MVP 2 (determinism / cache-key) tests.

Pure CPU, no torch. These pin the cache-contract semantics: that a partial
block is not intact, that a hole at block 1 truncates contiguous reuse, and
that a query-dependent artifact keyed only by prefix_hash is dangerous even
when its block survival looks fine.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from routing.prefix_integrity.datatypes import (
    BlockManifest,
    BlockStatus,
    Classification,
    Status,
)
from routing.prefix_integrity import metrics as M
from routing.prefix_integrity.invariants import InvariantInput, evaluate


# ---- manifest construction ------------------------------------------


def test_from_selected_whole_blocks():
    m = BlockManifest.from_selected(num_blocks=8, selected=[0, 1, 4])
    assert m.count(BlockStatus.INTACT) == 3
    assert m.count(BlockStatus.MISSING) == 5
    assert m.count(BlockStatus.PARTIAL) == 0
    assert m.selected == [0, 1, 4]


def test_from_token_mask_collapses_to_block_level():
    # block_size 4, 3 blocks; block0 full, block1 partial, block2 empty
    kept = [0, 1, 2, 3, 4, 5]  # all of block0, half of block1
    m = BlockManifest.from_token_mask(num_blocks=3, block_size=4, kept_tokens=kept)
    assert m.status[0] == BlockStatus.INTACT
    assert m.status[1] == BlockStatus.PARTIAL
    assert m.status[2] == BlockStatus.MISSING
    # a partial block still costs a read
    assert 1 in m.selected


# ---- block-survival metrics -----------------------------------------


def test_pre_partial_does_not_count_as_intact():
    m = BlockManifest(
        num_blocks=4,
        status=[
            BlockStatus.INTACT,
            BlockStatus.PARTIAL,
            BlockStatus.INTACT,
            BlockStatus.MISSING,
        ],
        selected=[0, 1, 2],
    )
    # 2 of 4 intact
    assert abs(M.prefix_reuse_efficiency(m) - 0.5) < 1e-9
    assert abs(M.partial_block_rate(m) - 0.25) < 1e-9


def test_contiguous_prefix_truncates_at_first_hole():
    m = BlockManifest(
        num_blocks=5,
        status=[
            BlockStatus.INTACT,
            BlockStatus.INTACT,
            BlockStatus.MISSING,
            BlockStatus.INTACT,
            BlockStatus.INTACT,
        ],
        selected=[0, 1, 3, 4],
    )
    # blocks 3,4 are individually intact but unreachable for prefix reuse
    assert M.contiguous_prefix_survival(m) == 2


def test_anchor_and_recent_survival():
    m = BlockManifest.from_selected(
        num_blocks=16, selected=[0, 14, 15] + list(range(3, 16))
    )
    assert M.anchor_survival(m, anchor_blocks=1) == 1.0
    assert M.recent_survival(m, recent_blocks=2) == 1.0
    m2 = BlockManifest.from_selected(num_blocks=16, selected=list(range(1, 16)))
    assert M.anchor_survival(m2, anchor_blocks=1) == 0.0  # dropped the anchor


def test_hot_weights_change_pre():
    m = BlockManifest(
        num_blocks=4,
        status=[
            BlockStatus.MISSING,
            BlockStatus.MISSING,
            BlockStatus.INTACT,
            BlockStatus.INTACT,
        ],
        selected=[2, 3],
    )
    # uniform PRE 0.5; but if the hot blocks are the dropped ones, HPRE collapses
    hot = [10.0, 10.0, 0.1, 0.1]
    assert M.hot_prefix_reuse_efficiency(m, hot) < 0.05


# ---- storage geometry -----------------------------------------------


def test_contiguous_read_ranges_counts_runs():
    assert M.contiguous_read_ranges([0, 1, 2]) == 1
    assert M.contiguous_read_ranges([0, 2, 4]) == 3
    assert M.contiguous_read_ranges([]) == 0
    assert M.contiguous_read_ranges([5, 6, 9, 10]) == 2


def test_compression_stats_cv():
    s = M.compression_stats([4.0, 4.0, 4.0])
    assert s["cr_cv"] == 0.0
    s2 = M.compression_stats([2.0, 8.0])
    assert s2["cr_cv"] > 0.4  # swings a lot


# ---- determinism / cache-key (MVP 2) --------------------------------


def test_manifest_stability_distinct_sets():
    sets = [[0, 1, 2], [0, 1, 2], [0, 1, 3]]
    assert M.manifest_stability(sets) == 2


def test_artifact_stability_distinct_digests():
    assert M.artifact_stability(["aa", "aa", "bb"]) == 2
    assert M.artifact_stability(["aa", "aa"]) == 1


# ---- invariant verdicts ---------------------------------------------


def test_deterministic_whole_block_prefix_only_is_safe():
    inp = InvariantInput(
        policy="prefix_cache",
        cache_key_fields=("prefix_hash",),
        artifact_stability_same_query=1,
        artifact_stability_across_queries=1,
        manifest_stability_across_queries=1,
        anchor_survival=1.0,
        partial_block_rate=0.0,
        pre=0.95,
    )
    r = evaluate(inp)
    assert r["status"] == Status.PASS.value
    assert r["classification"] == Classification.SAFE_FOR_PREFIX_OFFLOAD.value
    assert r["danger_score"] < 0.2


def test_query_dependent_prefix_keyed_is_dangerous():
    inp = InvariantInput(
        policy="prefix_cache",
        cache_key_fields=("prefix_hash",),
        artifact_stability_across_queries=17,  # selection changes per query
        manifest_stability_across_queries=17,
        anchor_survival=1.0,
        partial_block_rate=0.0,
        pre=0.71,
    )
    r = evaluate(inp)
    assert r["status"] == Status.FAIL.value
    assert r["classification"] == Classification.DANGEROUS_FOR_PREFIX_SHARING.value
    assert any("query-dependent" in v for v in r["violations"])
    assert r["danger_score"] > 0.5


def test_query_dependent_with_query_hash_is_extended_key():
    inp = InvariantInput(
        policy="prefix_cache",
        cache_key_fields=("prefix_hash", "query_hash"),
        artifact_stability_across_queries=17,
        manifest_stability_across_queries=17,
        anchor_survival=1.0,
        partial_block_rate=0.0,
        pre=0.71,
    )
    r = evaluate(inp)
    assert r["classification"] == Classification.SAFE_ONLY_WITH_EXTENDED_CACHE_KEY.value


def test_non_deterministic_same_query_fails():
    inp = InvariantInput(artifact_stability_same_query=3)
    r = evaluate(inp)
    assert r["status"] == Status.FAIL.value
    assert r["classification"] == Classification.DANGEROUS_FOR_PREFIX_SHARING.value


def test_partial_blocks_need_custom_connector():
    inp = InvariantInput(policy="prefix_cache", partial_block_rate=0.1)
    r = evaluate(inp)
    assert r["status"] == Status.FAIL.value  # vanilla path: partial is a violation
    inp2 = InvariantInput(
        policy="prefix_cache", partial_block_rate=0.1, has_custom_restore_path=True
    )
    r2 = evaluate(inp2)
    assert r2["classification"] == Classification.SAFE_ONLY_WITH_CUSTOM_CONNECTOR.value


def test_dropped_anchor_under_prefix_policy_fails():
    inp = InvariantInput(policy="prefix_cache", anchor_survival=0.0)
    r = evaluate(inp)
    assert r["status"] == Status.FAIL.value
    assert any("anchor" in v for v in r["violations"])


def test_routing_policy_query_dependent_is_routing_only():
    inp = InvariantInput(
        policy="routing",
        cache_key_fields=("prefix_hash",),
        artifact_stability_across_queries=9,
        manifest_stability_across_queries=9,
    )
    r = evaluate(inp)
    assert (
        r["classification"] == Classification.ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE.value
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
