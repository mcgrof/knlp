# SPDX-License-Identifier: MIT
"""Reject group aliases before native work and mislabeled model counters."""

import copy
from types import SimpleNamespace
import unittest

from tools.reproduce.kmesh_registered.group_bridge import prepare_members
from tools.reproduce.kmesh_registered.group_capture import (
    MODEL_COUNTS,
    recovery_contract,
)


def requests():
    class Slots:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class Spans:
        spans = tuple(
            SimpleNamespace(start=start, end=start + count)
            for start, count in zip((1, 140, 250, 360), (123, 95, 97, 101))
        )

        def validate_token_count(self, count):
            if count != 512:
                raise ValueError("changed token count")

    return tuple(
        SimpleNamespace(
            request_id=f"request-{i}",
            generation=1,
            tokens=list(range(512)),
            retrieve_kwargs={
                "explicit_spans": Spans(),
                "slot_mapping": Slots(list(range(i * 512, (i + 1) * 512))),
            },
        )
        for i in range(4)
    )


class GroupPreflight(unittest.TestCase):
    def worker(self):
        class Worker:
            checked = []

            def block_table(self, slots):
                self.checked.append(slots)
                return tuple(s // 32 for s in slots[::32])

            def restore(self, *args):
                raise AssertionError("preflight issued native work")

        return Worker()

    def test_preflight_snapshots_tokens_and_checks_all_private_tables(self):
        worker, members = self.worker(), requests()
        prepared = prepare_members(worker, members)
        self.assertEqual(len(worker.checked), 4)
        self.assertEqual(len({b for m in prepared for b in m.blocks}), 64)
        members[0].tokens[0] = 999
        self.assertEqual(prepared[0].tokens[0], 0)

    def test_alias_in_the_last_member_refuses_before_any_restore(self):
        worker, members = self.worker(), requests()
        members[-1].retrieve_kwargs["slot_mapping"] = members[0].retrieve_kwargs[
            "slot_mapping"
        ]
        with self.assertRaisesRegex(ValueError, "private complete"):
            prepare_members(worker, members)
        self.assertEqual(len(worker.checked), 4)

    def test_duplicate_or_changed_generation_refuses_without_native_work(self):
        for case in ("identity", "generation", "bool-generation"):
            worker, members = self.worker(), requests()
            if case == "identity":
                members[-1].request_id = members[0].request_id
            else:
                members[-1].generation = True if case == "bool-generation" else 2
            with self.subTest(case=case), self.assertRaises(ValueError):
                prepare_members(worker, members)


def group_capture():
    captured = {
        i: SimpleNamespace(
            engine_request_id=f"engine-{i}",
            binding=SimpleNamespace(trace_request_id=f"trace-{i}"),
        )
        for i in range(1, 5)
    }
    return captured, {
        "serial_members": [],
        "groups": [
            {
                "call_index": 0,
                "num_layers": 24,
                "plan_id": "synthetic-plan",
                "reuse_group_id": "synthetic-family",
                "engine_request_order": [f"engine-{i}" for i in range(1, 5)],
                "members": [
                    {
                        "engine_request_id": f"engine-{i}",
                        "trace_request_id": f"trace-{i}",
                        "generation": 1,
                        "outcome": "recovered",
                        "error": None,
                        "operation_counts": dict(MODEL_COUNTS),
                    }
                    for i in range(1, 5)
                ],
            }
        ],
    }


class GroupCapture(unittest.TestCase):
    def test_lockstep_receipt_keeps_model_operations_request_local(self):
        captured, recovery = group_capture()
        family, members = recovery_contract(recovery, captured)
        self.assertEqual(family["profile"], "collective_lockstep_v1")
        self.assertEqual(len(members), 4)

    def test_capture_never_invents_batched_operations_or_relabels_serial(self):
        for case in (
            "batched",
            "rotation",
            "failed",
            "generation",
            "bool-count",
            "serial",
            "order",
        ):
            captured, recovery = group_capture()
            member = recovery["groups"][0]["members"][0]
            if case == "batched":
                member["operation_counts"]["collective_rope_operations"] = 24
            elif case == "rotation":
                member["operation_counts"]["position_recovery_serial_operations"] = 24
            elif case == "failed":
                member["outcome"] = "failed"
            elif case == "generation":
                member["generation"] = 2
            elif case == "bool-count":
                member["operation_counts"]["serial_diff_operations"] = True
            elif case == "serial":
                recovery["serial_members"] = copy.deepcopy(
                    recovery["groups"][0]["members"]
                )
            else:
                recovery["groups"][0]["members"].reverse()
            with self.subTest(case=case), self.assertRaises(ValueError):
                recovery_contract(recovery, captured)
