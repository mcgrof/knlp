# SPDX-License-Identifier: MIT
"""CPU regressions for strict replay receipt validation."""

from __future__ import annotations

import copy
import unittest

from research.agent_baselines import replay_validation


def valid_receipts():
    manifest = {
        "tasks": 2,
        "steps": 3,
        "files": [
            {"instance_id": "task-a", "steps": 2},
            {"instance_id": "task-b", "steps": 1},
        ],
    }
    replay = {
        "requests": 3,
        "forced_match": 3,
        "forced_mismatch": 0,
        "errors": 0,
        "prompt_len_mismatch": 0,
        "tasks": 2,
        "all_outputs_forced_exactly": True,
        "interrupted": False,
    }
    requests = [
        {
            "instance_id": "task-a",
            "seq": 1,
            "forced_match": True,
            "prompt_ok": True,
        },
        {
            "instance_id": "task-a",
            "seq": 3,
            "forced_match": True,
            "prompt_ok": True,
        },
        {
            "instance_id": "task-b",
            "seq": 7,
            "forced_match": True,
            "prompt_ok": True,
        },
    ]
    tasks = [
        {"instance_id": "task-a", "steps": 2, "ok": True},
        {"instance_id": "task-b", "steps": 1, "ok": True},
    ]
    return {
        "return_code": 0,
        "run": {"status": "passed", "replay": copy.deepcopy(replay)},
        "replay": replay,
        "manifest": manifest,
        "requests": requests,
        "tasks": tasks,
    }


class ReplayValidationTests(unittest.TestCase):
    def test_complete_replay_is_accepted(self):
        result = replay_validation.validate_replay_records(**valid_receipts())
        self.assertEqual(result["status"], "complete-token-replay")
        self.assertTrue(all(result["checks"].values()))

    def test_bad_summary_fields_fail_closed(self):
        for field, value in (
            ("interrupted", True),
            ("prompt_len_mismatch", 1),
            ("errors", 1),
            ("forced_mismatch", 1),
            ("forced_match", 2),
            ("requests", 2),
            ("all_outputs_forced_exactly", False),
        ):
            with self.subTest(field=field):
                values = valid_receipts()
                values["replay"][field] = value
                values["run"]["replay"][field] = value
                with self.assertRaises(ValueError):
                    replay_validation.validate_replay_records(**values)

    def test_missing_summary_field_is_rejected(self):
        values = valid_receipts()
        del values["replay"]["prompt_len_mismatch"]
        del values["run"]["replay"]["prompt_len_mismatch"]
        with self.assertRaisesRegex(ValueError, "prompt_len_mismatch"):
            replay_validation.validate_replay_records(**values)

    def test_run_and_summary_must_match(self):
        values = valid_receipts()
        values["run"]["replay"]["workers"] = 1
        with self.assertRaisesRegex(ValueError, "summaries differ"):
            replay_validation.validate_replay_records(**values)

    def test_request_ids_counts_and_flags_are_exact(self):
        mutations = (
            (
                "unexpected ID",
                lambda values: values["requests"][0].update(instance_id="x"),
            ),
            ("duplicate sequence", lambda values: values["requests"][1].update(seq=1)),
            (
                "forced mismatch",
                lambda values: values["requests"][0].update(forced_match=False),
            ),
            (
                "prompt mismatch",
                lambda values: values["requests"][0].update(prompt_ok=False),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                values = valid_receipts()
                mutate(values)
                with self.assertRaises(ValueError):
                    replay_validation.validate_replay_records(**values)

    def test_task_ids_counts_and_flags_are_exact(self):
        mutations = (
            (
                "unexpected ID",
                lambda values: values["tasks"][0].update(instance_id="x"),
            ),
            (
                "duplicate ID",
                lambda values: values["tasks"][1].update(instance_id="task-a"),
            ),
            ("wrong steps", lambda values: values["tasks"][0].update(steps=1)),
            ("not okay", lambda values: values["tasks"][0].update(ok=False)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                values = valid_receipts()
                mutate(values)
                with self.assertRaises(ValueError):
                    replay_validation.validate_replay_records(**values)

    def test_boolean_counts_are_not_integers(self):
        values = valid_receipts()
        values["replay"]["errors"] = False
        values["run"]["replay"]["errors"] = False
        with self.assertRaisesRegex(ValueError, "expected an integer"):
            replay_validation.validate_replay_records(**values)
