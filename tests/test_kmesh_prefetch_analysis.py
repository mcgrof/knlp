# SPDX-License-Identifier: MIT
"""Reject hidden arrival shifts and keep inference clustered by process."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
from prefetch_analysis import choose_lead, confirm, verify_clock


def trial(
    mode="staged",
    fanout=4,
    block=0,
    sample=0,
    arm="demand",
    lead=0,
    first_ms=20,
    group_ms=60,
    delay_ms=0,
    ready=True,
    phase="pilot",
):
    origin, arrival = 1_000_000_000, 1_250_000_000
    name = f"{mode}-{fanout}-{block}-{sample}-{arm}-{lead}"
    rows = []
    for i in range(fanout):
        submit = arrival + round((delay_ms + i * 0.01) * 1e6)
        first = arrival + round((first_ms + i * 0.05) * 1e6)
        eighth = arrival + round((group_ms - (fanout - i - 1) * 0.1) * 1e6)
        rows.append(
            {
                "request_id": f"{name}-child-{i}",
                "engine_request_id": f"{name}-internal-{i}",
                "token_ids": [7] * 8,
                "ttft_ms": 3,
                "submit_ns": submit,
                "submit_return_ns": submit + 1000,
                "first_token_observed_ns": first,
                "first_observed_token_count": 1,
                "eighth_token_observed_ns": eighth,
                "terminal_observed_ns": eighth + 2_000_000,
                "submission_delay_ms": (submit - arrival) / 1e6,
                "deadline_first_token_ms": (first - arrival) / 1e6,
                "deadline_completion_ms": (eighth - arrival) / 1e6,
            }
        )
    prepare = arrival - lead * 1_000_000 if arm == "early" else None
    external = prepare + 1_000_000 if prepare is not None else None
    warm_submit = prepare + 2_000_000 if prepare is not None else None
    warm_done = (
        (arrival - 1_000_000 if ready and lead else arrival + 5_000_000)
        if prepare is not None
        else None
    )
    skipped = arm == "early" and lead == 0
    if skipped:
        warm_submit = warm_done = None
    group_end = arrival + round(group_ms * 1e6)
    return {
        "mode": mode,
        "fanout": fanout,
        "block_id": f"{phase}-{block}-{mode}-n{fanout}",
        "sample": sample,
        "warmup": sample < 0,
        "arm": arm,
        "lead_ms": lead,
        "trace_origin_ns": origin,
        "scheduled_arrival_ns": arrival,
        "group_submit_ns": arrival + round(delay_ms * 1e6),
        "group_end_ns": group_end,
        "drain_end_ns": max(group_end + 3_000_000, warm_done or 0),
        "prepare_start_ns": prepare,
        "external_ready_ns": external,
        "warm_submit_ns": warm_submit,
        "warm_done_ns": warm_done,
        "warm_rows": (
            [{"request_id": name + "-warm", "token_ids": [7]}]
            if arm == "early" and not skipped
            else []
        ),
        "ready_by_scheduled_arrival": warm_done is not None and warm_done <= arrival,
        "warm_skipped_due_to_arrival": skipped,
        "rows": rows,
        "children_wall_ms": group_ms - delay_ms,
        "deadline_group_completion_ms": group_ms,
        "workflow_completion_ms": 250 + group_ms,
    }


def pilot_grid(ready_counts=None):
    ready_counts = ready_counts or {
        "staged": {25: 9, 100: 10, 250: 10},
        "direct": {25: 8, 100: 9, 250: 10},
    }
    cases = [
        ("demand", 0),
        ("declaration", 0),
        ("external_resident", 0),
        ("engine_resident", 0),
    ] + [("early", lead) for lead in (0, 25, 100, 250)]
    return [
        trial(
            mode=mode,
            block=block,
            sample=sample,
            arm=arm,
            lead=lead,
            ready=(block * 5 + sample < ready_counts[mode].get(lead, 0)),
        )
        for mode in ("staged", "direct")
        for block in range(2)
        for arm, lead in cases
        for sample in range(-3, 5)
    ]


def confirmation_grid(regression=False):
    rows = []
    for mode in ("staged", "direct"):
        for fanout in (4, 8):
            for block in range(10):
                effect = 3 * (block + 1)
                for sample in range(-3, 20):
                    for arm in ("demand", "early"):
                        first = 100 + 10 * block - (effect if arm == "early" else 0)
                        group = 400 + 10 * block
                        if arm == "early":
                            group += 50 if regression else -2 * effect
                        rows.append(
                            trial(
                                mode,
                                fanout,
                                block,
                                sample,
                                arm,
                                100 if arm == "early" else 0,
                                first_ms=first,
                                group_ms=group,
                                phase="confirm",
                            )
                        )
    return rows


class ClockEvidenceTest(unittest.TestCase):
    def test_final_response_cannot_stand_in_for_first_token_evidence(self):
        for fault in ("missing", "count", "clock"):
            with self.subTest(fault=fault):
                row = trial()
                child = row["rows"][0]
                if fault == "missing":
                    del child["first_observed_token_count"]
                elif fault == "count":
                    child["first_observed_token_count"] = 8
                else:
                    child["first_token_observed_ns"] = child["eighth_token_observed_ns"]
                with self.assertRaises(ValueError):
                    verify_clock(row)

    def test_submission_delay_cannot_be_hidden_in_engine_ttft(self):
        row = trial(delay_ms=15, first_ms=20)
        metrics = verify_clock(row)
        self.assertEqual(metrics["a_deadline_first_token_ms"], 20)
        self.assertEqual(metrics["a_submission_first_token_ms"], 5)
        self.assertEqual(row["rows"][0]["ttft_ms"], 3)
        row["rows"][0]["deadline_first_token_ms"] = 5
        with self.assertRaisesRegex(ValueError, "stored duration"):
            verify_clock(row)

    def test_shifted_arrival_and_terminal_as_eighth_are_rejected(self):
        row = trial()
        row["scheduled_arrival_ns"] += 1_000_000
        with self.assertRaisesRegex(ValueError, "fixed origin"):
            verify_clock(row)
        row = trial()
        self.assertGreater(row["rows"][-1]["terminal_observed_ns"], row["group_end_ns"])
        verify_clock(row)
        row["group_end_ns"] = row["rows"][-1]["terminal_observed_ns"]
        with self.assertRaisesRegex(ValueError, "last eighth"):
            verify_clock(row)

    def test_a_is_not_whichever_child_outputs_first(self):
        row = trial(first_ms=30)
        child = row["rows"][1]
        child["first_token_observed_ns"] = row["scheduled_arrival_ns"] + 5_000_000
        child["deadline_first_token_ms"] = 5
        result = verify_clock(row)
        self.assertEqual(result["a_deadline_first_token_ms"], 30)
        self.assertEqual(min(result["all_request_deadline_first_token_ms"]), 5)

    def test_readiness_uses_scheduled_not_delayed_actual_arrival(self):
        row = trial(arm="early", lead=25, delay_ms=15, first_ms=25, ready=False)
        self.assertFalse(verify_clock(row)["ready_by_scheduled_arrival"])
        self.assertLess(row["warm_done_ns"], row["rows"][0]["submit_ns"])
        row["ready_by_scheduled_arrival"] = True
        with self.assertRaisesRegex(ValueError, "readiness"):
            verify_clock(row)

    def test_explicit_late_warm_skip_keeps_external_work_and_not_readiness(self):
        for admission_delay_ms in (0, 7):
            with self.subTest(admission_delay_ms=admission_delay_ms):
                row = trial(arm="early", lead=25, delay_ms=admission_delay_ms)
                row.update(
                    warm_skipped_due_to_arrival=True,
                    warm_rows=[],
                    warm_submit_ns=None,
                    warm_done_ns=None,
                    external_ready_ns=row["scheduled_arrival_ns"] + 5_000_000,
                    ready_by_scheduled_arrival=False,
                )
                result = verify_clock(row)
                self.assertEqual(result["warm_request_count"], 0)
                self.assertTrue(result["warm_skipped_due_to_arrival"])
                row["external_ready_ns"] = row["scheduled_arrival_ns"] - 5_000_000
                with self.assertRaisesRegex(ValueError, "late-warm skip"):
                    verify_clock(row)

    def test_duplicate_missing_children_and_nonfinite_evidence_fail(self):
        for change in ("duplicate", "missing", "nonfinite", "wrong-order"):
            with self.subTest(change=change):
                row = trial()
                if change == "duplicate":
                    row["rows"][1]["request_id"] = row["rows"][0]["request_id"]
                elif change == "missing":
                    row["rows"].pop()
                elif change == "nonfinite":
                    row["rows"][0]["deadline_first_token_ms"] = float("nan")
                else:
                    row["rows"].reverse()
                with self.assertRaises(ValueError):
                    verify_clock(row)


class PilotSelectionTest(unittest.TestCase):
    def test_shortest_common_readiness_lead_not_best_single_path(self):
        result = choose_lead(pilot_grid())
        self.assertEqual(result["selected_lead_ms"], 100)
        self.assertTrue(result["readiness_rule_satisfied"])
        self.assertEqual(
            result["readiness_by_path_and_lead_ms"]["direct"]["25"]["ready"], 8
        )

    def test_failed_readiness_keeps_tested_limit_with_failure_flag(self):
        result = choose_lead(
            pilot_grid({mode: {25: 8, 100: 8, 250: 8} for mode in ("staged", "direct")})
        )
        self.assertEqual(result["selected_lead_ms"], 250)
        self.assertFalse(result["readiness_rule_satisfied"])
        self.assertTrue(result["fallback_to_tested_limit"])

    def test_missing_placement_warmup_duplicate_or_block_is_not_a_pilot(self):
        full = pilot_grid()
        mutations = (
            lambda: [row for row in full if row["arm"] != "external_resident"],
            lambda: [row for row in full if row["sample"] != -3],
            lambda: full + [deepcopy(full[0])],
            lambda: [
                row for row in full if not row["block_id"].startswith("pilot-1-staged")
            ],
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):
                    choose_lead(mutate())


class ConfirmationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = confirmation_grid()
        cls.result = confirm(cls.rows, selected_lead_ms=100)

    def test_paired_block_bootstrap_keeps_correlated_endpoints_together(self):
        cell = self.result["cells"]["staged-n4"]
        first = cell["endpoints"]["a_deadline_first_token_ms"]
        group = cell["endpoints"]["deadline_group_completion_ms"]
        self.assertAlmostEqual(first["improvement_ms"]["estimate"], 16.5)
        self.assertAlmostEqual(
            first["improvement_percent"]["estimate"], 100 * 16.5 / 145
        )
        self.assertGreater(
            first["improvement_ms"]["ci95"][1] - first["improvement_ms"]["ci95"][0], 8
        )
        for a, b in zip(
            first["improvement_ms"]["ci95"], group["improvement_ms"]["ci95"]
        ):
            self.assertAlmostEqual(b, 2 * a)
        self.assertEqual(cell["process_blocks"], 10)
        self.assertEqual(cell["measured_groups_per_arm"], 200)
        self.assertTrue(cell["engineering_rule_satisfied"])
        self.assertEqual(
            cell["all_request_deadline_first_token"]["demand"]["count"], 800
        )

    def test_a_gain_cannot_hide_a_large_group_regression(self):
        result = confirm(confirmation_grid(regression=True), selected_lead_ms=100)
        for cell in result["cells"].values():
            self.assertFalse(cell["engineering_rule_satisfied"])
            self.assertLess(
                cell["endpoints"]["deadline_group_completion_ms"][
                    "improvement_percent"
                ]["ci95"][1],
                -5,
            )

    def test_wrong_frozen_lead_missing_pairs_blocks_or_warmups_rejected(self):
        mutations = (
            lambda: self.rows[:-1],
            lambda: self.rows + [deepcopy(self.rows[0])],
            lambda: [
                r
                for r in self.rows
                if not r["block_id"].startswith("confirm-9-direct-n8")
            ],
            lambda: [r for r in self.rows if r["sample"] != -3],
        )
        with self.assertRaisesRegex(ValueError, "wrong lead"):
            confirm(self.rows, selected_lead_ms=25)
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):
                    confirm(mutate(), selected_lead_ms=100)


if __name__ == "__main__":
    unittest.main()
