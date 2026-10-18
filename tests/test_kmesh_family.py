# SPDX-License-Identifier: MIT
"""Check real engine-loop release rules and reject false preload attribution."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
from family_worker import replay
from family_verify import check_family_receipt


class Clock:
    now = 0

    def sleep(self, seconds):
        self.now += max(1, int(seconds * 1e9))


class Engine:
    def __init__(self, clock):
        self.clock, self.pending, self.events = clock, {}, []

    def add_request(self, name, prompt, params):
        self.events.append(("add", name, self.clock.now))
        self.pending[name] = params.max_tokens
        return name + "-internal"

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        self.events.append(("step", self.clock.now))
        self.clock.now += 5_000_000
        if self.clock.now < 20_000_000:
            return []
        result = []
        for name, length in list(self.pending.items()):
            result.append(
                SimpleNamespace(
                    finished=True,
                    request_id=name,
                    num_cached_tokens=4096,
                    kv_transfer_params=None,
                    metrics=SimpleNamespace(first_token_latency=0.005),
                    outputs=[
                        SimpleNamespace(
                            finish_reason="length",
                            token_ids=[7] * length,
                            logprobs=[{7: SimpleNamespace(logprob=-1.0)}] * length,
                        )
                    ],
                )
            )
            del self.pending[name]
        return result


class FamilyReplayTest(unittest.TestCase):
    def run_case(self, policy, gap):
        clock, declarations = Clock(), []
        engine = Engine(clock)

        def declare(warm_id):
            self.assertFalse(any(e[0] == "step" for e in engine.events))
            declarations.append(warm_id)
            return "declared"

        with patch(
            "family_worker.time.perf_counter_ns", side_effect=lambda: clock.now
        ), patch("family_worker.time.sleep", side_effect=clock.sleep):
            result = replay(
                engine,
                [1] * 4096,
                [[1] * 4096 + [2]],
                SimpleNamespace(max_tokens=8),
                SimpleNamespace(max_tokens=1),
                policy,
                gap,
                "family",
                declare=declare if policy == "family" else None,
            )
        return result, declarations

    def test_child_really_arrives_before_preload_finishes(self):
        result, declarations = self.run_case("family", 10)
        self.assertEqual(result["actual_gap_ms"], 10)
        self.assertEqual(result["warm_complete_ms"], 20)
        self.assertFalse(result["warm_completed_before_release"])
        self.assertEqual(declarations, ["family-warm-internal"])
        self.assertEqual(result["declaration_identity"], "declared")

    def test_later_arrival_observes_completed_preload(self):
        result, _ = self.run_case("family", 30)
        self.assertGreaterEqual(result["actual_gap_ms"], 30)
        self.assertTrue(result["warm_completed_before_release"])

    def test_oracle_waits_for_observed_completion(self):
        result, _ = self.run_case("oracle", 0)
        self.assertTrue(result["warm_completed_before_release"])

    def test_reactive_has_no_warm_request(self):
        result, declarations = self.run_case("reactive", 0)
        self.assertEqual(result["warm_rows"], [])
        self.assertIsNone(result["warm_complete_ms"])
        self.assertEqual(declarations, [])


class FamilyReceiptTest(unittest.TestCase):
    def setUp(self):
        before = {
            "pending": 0,
            "family_ensure_total": 0,
            "family_prefetch_hit_total": 0,
            "family_prefetch_wasted_total": 0,
            "family_prefetch_bytes": 0,
        }
        self.row = {
            "policy": "family",
            "singleflight": {"inflight": 0, "joined": 0},
            "family_before": before,
            "family_after": before
            | {"family_ensure_total": 1, "family_prefetch_hit_total": 1},
            "actual_gap_ms": 10,
            "requested_gap_ms": 10,
            "warm_complete_ms": 20,
            "warm_completed_before_release": False,
            "warm_rows": [{}],
            "rows": [
                {
                    "kv_transfer_params": {
                        "cached_token_stats": {"num_vllm_cached_tokens": 4096}
                    }
                }
            ],
        }

    def test_actual_local_child_hit_is_accepted(self):
        check_family_receipt(self.row)

    def test_warm_request_cannot_add_a_child_credit(self):
        self.row["family_after"]["family_prefetch_hit_total"] = 2
        with self.assertRaisesRegex(ValueError, "attribution differs"):
            check_family_receipt(self.row)

    def test_release_before_requested_gap_is_rejected(self):
        self.row["actual_gap_ms"] = 5
        with self.assertRaisesRegex(ValueError, "before requested"):
            check_family_receipt(self.row)

    def test_false_warm_before_release_is_rejected(self):
        self.row["warm_completed_before_release"] = True
        with self.assertRaisesRegex(ValueError, "disagree"):
            check_family_receipt(self.row)
