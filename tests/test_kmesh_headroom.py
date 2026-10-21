# SPDX-License-Identifier: MIT
"""Check the optimistic residency bound against exact finite optima."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

from tools.reproduce.kmesh_residency.headroom import analyze, maximum_intervals


def brute_force(intervals, capacity):
    best = 0
    for mask in range(1 << len(intervals)):
        selected = [item for i, item in enumerate(intervals) if mask & (1 << i)]
        if all(sum(a <= t < b for a, b in selected) <= capacity for t in range(5)):
            best = max(best, len(selected))
    return best


class HeadroomTest(unittest.TestCase):
    def test_all_small_interval_sets_match_exhaustive_selection(self):
        universe = list(itertools.combinations(range(4), 2))
        for mask in range(1 << len(universe)):
            intervals = [item for i, item in enumerate(universe) if mask & (1 << i)]
            for capacity in (1, 2, 3):
                self.assertEqual(
                    maximum_intervals(intervals, capacity),
                    brute_force(intervals, capacity),
                    (intervals, capacity),
                )

    def test_nested_intervals_and_shared_endpoints(self):
        self.assertEqual(
            maximum_intervals([(0, 4), (0, 1), (1, 2), (2, 3), (3, 4)], 1), 4
        )
        self.assertEqual(maximum_intervals([(0, 2), (0, 2), (2, 4), (2, 4)], 2), 4)

    def test_invalid_budget_and_interval_are_rejected(self):
        for capacity in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                maximum_intervals([], capacity)
        with self.assertRaises(ValueError):
            maximum_intervals([(2, 1)], 1)


def invocation(name, tokens, output, computed, hits, start):
    blocks = [
        {
            "block_id": i + 1,
            "ref_count": 1,
            "hash": "hash-" + str(i),
            "hash_tokens": (i + 1) * 16,
        }
        for i in range(computed // 16)
    ]
    return {
        "schema": "knlp.kmesh.invocation.v1",
        "request_id": name,
        "family_id": "family",
        "parent_request_id": None,
        "engine_id": "engine",
        "clock_id": "clock",
        "start_ns": start,
        "end_ns": start + 10,
        "input_token_ids": tokens,
        "output_token_ids": output,
        "status": "complete",
        "purpose": "test",
        "ready": {
            "all_token_ids": tokens + output,
            "computed_tokens": computed,
            "cached_tokens": hits,
            "blocks": blocks,
        },
        "cache_observations": [
            {"action": "lookup", "hit_tokens": hits},
            {"action": "before-free", "computed_tokens": computed, "blocks": blocks},
        ],
    }


class DemandBoundTest(unittest.TestCase):
    def analyze_rows(self, rows, capacity):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "invocations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (p / "workload.json").write_text(
                json.dumps(
                    {
                        "successful_capture": True,
                        "identity": {"model": "test"},
                        "engine_cache": {
                            "physical_pool_blocks": capacity + 1,
                            "initial_free_blocks": capacity,
                            "final_nonnull_references": 0,
                            "completed_requests": len(rows),
                            "engine_id": "engine",
                            "clock_id": "clock",
                            "block_tokens": 16,
                        },
                    }
                )
            )
            return analyze(p)

    def test_compulsory_logit_boundary_is_not_an_avoidable_read(self):
        tokens = list(range(32))
        result = self.analyze_rows(
            [
                invocation("a", tokens, [99], 32, 0, 0),
                invocation("b", tokens, [99], 32, 16, 20),
            ],
            2,
        )
        self.assertEqual(result["observed_prefill_tokens"], 48)
        self.assertEqual(result["maximum_prefill_tokens_saved"], 0)

    def test_decode_writes_can_seed_later_reads_without_read_cost(self):
        tokens, output = list(range(17)), list(range(100, 132))
        result = self.analyze_rows(
            [
                invocation("a", tokens, output, 48, 0, 0),
                invocation("b", tokens + output, [999], 49, 48, 20),
            ],
            4,
        )
        self.assertEqual(result["observed_prefill_tokens"], 18)
        self.assertEqual(result["maximum_prefill_tokens_saved"], 0)
        self.assertEqual(result["observed_hit_blocks"], 3)

    def test_impossible_observed_hits_reject_the_bound(self):
        tokens = list(range(32))
        with self.assertRaisesRegex(ValueError, "lacks a prior computed prefix"):
            self.analyze_rows(
                [
                    invocation("a", tokens, [99], 32, 0, 0),
                    invocation("b", list(range(100, 132)), [99], 32, 16, 20),
                ],
                2,
            )

    def test_balanced_aggregate_cannot_hide_a_cold_start_hit(self):
        tokens = list(range(32))
        with self.assertRaisesRegex(ValueError, "lacks a prior computed prefix"):
            self.analyze_rows(
                [
                    invocation("a", tokens, [99], 32, 16, 0),
                    invocation("b", tokens, [99], 32, 0, 20),
                ],
                2,
            )

    def test_reported_hit_intervals_must_fit_the_observed_capacity(self):
        rows = [
            invocation(
                str(index),
                [ord(letter)] * 17,
                [99],
                17,
                16 if index in (3, 4, 5) else 0,
                index * 20,
            )
            for index, letter in enumerate("ABCABCDDEE")
        ]
        with self.assertRaisesRegex(ValueError, "hit intervals exceed"):
            self.analyze_rows(rows, 2)


if __name__ == "__main__":
    unittest.main()
