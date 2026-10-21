# SPDX-License-Identifier: MIT
"""Exercise fixed arrivals, delayed preparation and child-only stopwatches."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
from prefetch_clock import replay_fixed


class Clock:
    def __init__(self):
        self.now = 0

    def perf_counter_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Preparation:
    def __init__(self, clock, ready):
        self.clock, self.ready = clock, ready
        self.results = 0

    def done(self):
        return self.clock.now >= self.ready

    def result(self, timeout=None):
        if not self.done():
            self.clock.now = self.ready
        self.results += 1
        return {"found_keys": 16, "total_keys": 16}


class Engine:
    def __init__(self, clock, first_step_ms=5):
        self.clock, self.pending, self.events = clock, {}, []
        self.first_step_ms = first_step_ms
        self.steps = 0

    def add_request(self, name, prompt, params):
        self.events.append((name, self.clock.now))
        self.pending[name] = [0, params.max_tokens]
        return name + "-internal"

    def has_unfinished_requests(self):
        return bool(self.pending)

    def abort_request(self, names):
        for name in names:
            self.pending.pop(name, None)

    def step(self):
        self.clock.now += round((self.first_step_ms if not self.steps else 5) * 1e6)
        self.steps += 1
        result = []
        for name, state in list(self.pending.items()):
            state[0] += 1
            done = state[0] == state[1]
            result.append(
                SimpleNamespace(
                    request_id=name,
                    finished=done,
                    num_cached_tokens=4096,
                    kv_transfer_params=None,
                    metrics=SimpleNamespace(first_token_latency=0.005),
                    outputs=[
                        SimpleNamespace(
                            finish_reason="length" if done else None,
                            token_ids=[7] * state[0],
                            logprobs=[{7: SimpleNamespace(logprob=-1.0)}] * state[0],
                        )
                    ],
                )
            )
            if done:
                del self.pending[name]
        return result


class FixedArrivalTest(unittest.TestCase):
    def test_final_only_delivery_refuses_a_false_first_token_clock(self):
        original = Engine.step

        def final_only(engine):
            return [row for row in original(engine) if row.finished]

        with patch.object(Engine, "step", final_only):
            with self.assertRaisesRegex(RuntimeError, "one streamed token"):
                self.run_case("demand")

    def run_case(self, arm, *, ready_ms=30, lead_ms=0, first_step_ms=5, unused=False):
        clock = Clock()
        engine = Engine(clock, first_step_ms)
        future = Preparation(clock, round(ready_ms * 1e6))
        declarations, released = [], []

        def declare(warm_id):
            declarations.append(warm_id)
            return "one-declaration"

        def prepare():
            if arm != "early":
                raise AssertionError("reference arm started a speculative fill")
            return future

        row = replay_fixed(
            engine,
            [1] * 4096,
            [] if unused else [[1] * 4096 + [2], [1] * 4096 + [3]],
            SimpleNamespace(max_tokens=8),
            SimpleNamespace(max_tokens=1),
            arm,
            lead_ms,
            "trial",
            start_prefetch=prepare,
            declare=declare,
            release_declaration=released.append,
            arrival_offset_ms=25,
            clock=clock,
        )
        return row, engine, future, declarations, released

    def test_slow_preparation_does_not_move_scheduled_or_actual_arrival(self):
        row, engine, future, _, _ = self.run_case("early", lead_ms=25)
        self.assertTrue(all(r["first_observed_token_count"] == 1 for r in row["rows"]))
        self.assertEqual(row["scheduled_arrival_ns"], 25_000_000)
        self.assertEqual(row["group_submit_ns"], 25_000_000)
        self.assertEqual(row["external_ready_ns"], 30_000_000)
        self.assertEqual([r["submit_ns"] for r in row["rows"]], [25_000_000] * 2)
        self.assertFalse(row["ready_by_scheduled_arrival"])
        self.assertFalse(engine.pending)
        self.assertEqual(row["warm_rows"], [])
        self.assertTrue(row["warm_skipped_due_to_arrival"])

    def test_blocking_engine_step_is_charged_to_deadline_latency(self):
        row, _, _, _, _ = self.run_case(
            "early", lead_ms=25, ready_ms=0, first_step_ms=40
        )
        self.assertEqual(row["group_submit_ns"], 40_000_000)
        self.assertEqual(row["rows"][0]["submission_delay_ms"], 15)
        self.assertEqual(row["rows"][0]["deadline_first_token_ms"], 20)
        self.assertEqual(row["rows"][0]["ttft_ms"], 5)

    def test_arrival_during_accepted_warm_admits_children_before_step(self):
        for slow_callback in ("warm_submit", "declaration"):
            with self.subTest(slow_callback=slow_callback):
                clock, steps, declarations = Clock(), [], []

                class SlowWarm(Engine):
                    def add_request(self, name, prompt, params):
                        internal = super().add_request(name, prompt, params)
                        if name.endswith("-warm") and slow_callback == "warm_submit":
                            clock.now += 30_000_000
                        return internal

                    def step(self):
                        steps.append(clock.now)
                        return super().step()

                def declare(warm_id):
                    declarations.append(warm_id)
                    if warm_id is not None and slow_callback == "declaration":
                        clock.now += 30_000_000
                    return "one-declaration"

                row = replay_fixed(
                    SlowWarm(clock),
                    [],
                    [[1]],
                    SimpleNamespace(max_tokens=8),
                    SimpleNamespace(max_tokens=1),
                    "early",
                    25,
                    "trial",
                    start_prefetch=lambda: Preparation(clock, 0),
                    declare=declare,
                    release_declaration=lambda _: None,
                    arrival_offset_ms=25,
                    clock=clock,
                )
                self.assertEqual(row["scheduled_arrival_ns"], 25_000_000)
                self.assertEqual(row["rows"][0]["submit_ns"], 30_000_000)
                self.assertEqual(steps[0], 30_000_000)
                self.assertFalse(row["warm_skipped_due_to_arrival"])
                self.assertEqual(len(row["warm_rows"]), 1)
                self.assertEqual(declarations, [None, "trial-warm-internal"])

    def test_arrival_during_ready_observation_skips_unsubmitted_warm(self):
        clock = Clock()

        class SlowReadyObservation(Preparation):
            def result(self, timeout=None):
                result = super().result(timeout)
                if self.results == 1:
                    clock.now += 30_000_000
                return result

        future = SlowReadyObservation(clock, 0)
        row = replay_fixed(
            Engine(clock),
            [],
            [[1]],
            SimpleNamespace(max_tokens=8),
            SimpleNamespace(max_tokens=1),
            "early",
            25,
            "trial",
            start_prefetch=lambda: future,
            arrival_offset_ms=25,
            clock=clock,
        )
        self.assertEqual(row["rows"][0]["submit_ns"], 30_000_000)
        self.assertTrue(row["warm_skipped_due_to_arrival"])
        self.assertIsNone(row["warm_submit_ns"])
        self.assertEqual(row["warm_rows"], [])
        self.assertEqual(future.results, 2)

    def test_child_completion_excludes_later_preparation_drain(self):
        row, _, _, _, _ = self.run_case("early", lead_ms=25, ready_ms=90)
        self.assertEqual(row["group_end_ns"], 65_000_000)
        self.assertEqual(row["children_wall_ms"], 40)
        self.assertEqual(row["deadline_group_completion_ms"], 40)
        self.assertEqual(row["drain_end_ns"], 90_000_000)

    def test_demand_and_placement_controls_do_not_issue_a_warm_request(self):
        for arm in ("demand", "external_resident", "engine_resident"):
            with self.subTest(arm=arm):
                row, engine, future, declarations, released = self.run_case(arm)
                self.assertEqual(row["warm_rows"], [])
                self.assertIsNone(row["prepare_start_ns"])
                self.assertEqual(len(engine.events), 2)
                self.assertEqual((future.results, declarations, released), (0, [], []))

    def test_declaration_only_has_bookkeeping_without_speculative_work(self):
        row, _, future, declarations, released = self.run_case("declaration")
        self.assertEqual(declarations, [None])
        self.assertEqual(released, ["one-declaration"])
        self.assertEqual(future.results, 0)
        self.assertEqual(row["warm_rows"], [])

    def test_early_hint_refreshes_one_identity_and_releases_once(self):
        _, _, _, declarations, released = self.run_case("early", lead_ms=25, ready_ms=0)
        self.assertEqual(declarations, [None, "trial-warm-internal"])
        self.assertEqual(released, ["one-declaration"])

    def test_unused_hint_accounts_warming_without_a_child_timer(self):
        row, engine, future, _, released = self.run_case(
            "early", lead_ms=25, unused=True
        )
        self.assertEqual(row["rows"], [])
        self.assertEqual(len(row["warm_rows"]), 1)
        self.assertIsNone(row["group_end_ns"])
        self.assertIsNone(row["children_wall_ms"])
        self.assertFalse(engine.pending)
        self.assertEqual(released, ["one-declaration"])

    def test_eighth_token_clock_precedes_later_terminal_acknowledgement(self):
        clock = Clock()

        class DelayedTerminal(Engine):
            delayed = False

            def step(self):
                outputs = super().step()
                if outputs[0].finished and not self.delayed:
                    self.delayed = True
                    outputs[0].finished = False
                    self.pending[outputs[0].request_id] = [7, 8]
                return outputs

        row = replay_fixed(
            DelayedTerminal(clock),
            [],
            [[1]],
            SimpleNamespace(max_tokens=8),
            None,
            "demand",
            0,
            "trial",
            arrival_offset_ms=25,
            clock=clock,
        )
        self.assertEqual(row["group_end_ns"], 65_000_000)
        self.assertEqual(row["rows"][0]["eighth_token_observed_ns"], 65_000_000)
        self.assertEqual(row["rows"][0]["terminal_observed_ns"], 70_000_000)
        self.assertEqual(row["drain_end_ns"], 70_000_000)

    def test_changed_refresh_identity_preserves_both_release_owners_and_ledger(self):
        clock, released, ledger = Clock(), [], []
        engine = Engine(clock)
        future = Preparation(clock, 0)
        identities = iter(("first", "second"))
        with self.assertRaisesRegex(RuntimeError, "changed identity"):
            replay_fixed(
                engine,
                [],
                [[1]],
                SimpleNamespace(max_tokens=8),
                SimpleNamespace(max_tokens=1),
                "early",
                25,
                "trial",
                start_prefetch=lambda: future,
                declare=lambda _: next(identities),
                release_declaration=released.append,
                on_partial=ledger.append,
                arrival_offset_ms=25,
                clock=clock,
            )
        self.assertEqual(set(released), {"first", "second"})
        self.assertFalse(engine.pending)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["scheduled_arrival_ns"], 25_000_000)

    def test_release_failure_preserves_completed_child_results(self):
        clock, ledger = Clock(), []

        def refuse_release(_):
            raise RuntimeError("release fixture failure")

        with self.assertRaisesRegex(RuntimeError, "declaration release failed"):
            replay_fixed(
                Engine(clock),
                [],
                [[1]],
                SimpleNamespace(max_tokens=8),
                None,
                "declaration",
                0,
                "trial",
                declare=lambda _: "owned",
                release_declaration=refuse_release,
                on_partial=ledger.append,
                arrival_offset_ms=25,
                clock=clock,
            )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(len(ledger[0]["rows"][0]["token_ids"]), 8)
        self.assertEqual(ledger[0]["group_end_ns"], 65_000_000)
        self.assertIn("release fixture failure", ledger[0]["cleanup_errors"][0])


if __name__ == "__main__":
    unittest.main()
