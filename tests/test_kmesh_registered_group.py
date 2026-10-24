# SPDX-License-Identifier: MIT
"""Reject group aliases before native work and mislabeled model counters."""

import copy
from dataclasses import make_dataclass
from types import SimpleNamespace
import unittest

from tools.reproduce.kmesh_registered.group_bridge import prepare_members
from tools.reproduce.kmesh_registered.group_capture import (
    MODEL_COUNTS,
    recovery_contract,
)
from tools.reproduce.kmesh_registered.verify import recovery_members
from tools.reproduce.kmesh_registered.group_controls import (
    install_member_failure,
    serial_reference_outputs,
    verify_member_failure,
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
    def test_saved_group_masks_keep_their_actual_execution_identity(self):
        captured, recovery = group_capture()
        manifest = {
            "collective": {"profile": "collective_lockstep_v1"},
            "members": [
                {
                    "sequence_index": i,
                    "engine_request_id": c.engine_request_id,
                    "trace_request_id": c.binding.trace_request_id,
                }
                for i, c in captured.items()
            ],
        }
        value = {
            "kind": "registered_native_group_model",
            "model_execution": "collective-lockstep",
            "recovery_evidence": recovery,
        }
        self.assertEqual(
            recovery_members(value, manifest), recovery["groups"][0]["members"]
        )
        with self.assertRaises(ValueError):
            recovery_members(
                value | {"kind": "registered_native_serial_model"}, manifest
            )
        manifest["collective"]["profile"] = "collective_batched_qkv_v1"
        with self.assertRaises(ValueError):
            recovery_members(value, manifest)

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


def failure_control():
    captured, recovery = group_capture()
    members = recovery["groups"][0]["members"]
    bindings = [
        {"engine_request_id": f"engine-{i}", "trace_request_id": f"trace-{i}"}
        for i in range(1, 5)
    ]
    recovery["request_id_bindings"] = bindings
    for member in members:
        member.update(
            selected_spans={"synthetic-span": True},
            retrieval_mask={"synthetic-mask": True},
        )
    serial_members = copy.deepcopy(members)
    failed = members[1]
    failed.update(
        outcome="failed",
        retrieval_mask=None,
        error={"type": "RuntimeError"},
        operation_counts=MODEL_COUNTS
        | {"serial_rope_operations": 1, "serial_diff_operations": 0},
    )
    observations = [
        {"trace_request_id": f"trace-{i}", "output_token_ids": [i]} for i in range(5)
    ]
    serial_selection = [
        {"request_id": f"engine-{i}", "layer_id": layer, "selected_indices": [layer]}
        for i in range(1, 5)
        for layer in range(24)
    ]
    serial = {
        "status": "complete",
        "kind": "serial_model_reference",
        "model_execution": "serial",
        "observations": copy.deepcopy(observations),
        "recovery_evidence": {
            "serial_members": serial_members,
            "request_id_bindings": bindings,
        },
        "selection_evidence": serial_selection,
    }
    ordinary = {o["trace_request_id"]: [99] for o in observations}
    observations[2]["output_token_ids"] = [99]
    events = (
        [{"action": "registered.span.stored"} for _ in range(4)]
        + [
            {
                "action": "registered.span.restored",
                "source_start": 0,
                "destination_start": 1,
            }
            for _ in range(16)
        ]
        + [
            {
                "action": "gate.sibling_failure.injected",
                "request_id": "engine-2",
                "completed_layers": 1,
                "completed_serial_operations": MODEL_COUNTS
                | {"serial_rope_operations": 1, "serial_diff_operations": 0},
            },
            {
                "action": "registered.model.group.consumed",
                "span_counts": [4] * 4,
                "restored_tokens": [416] * 4,
                "request_order": [f"engine-{i}" for i in range(1, 5)],
                "source_closed": True,
                "pending": False,
            },
        ]
        + [
            {
                "action": "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin",
                "request_id": "engine-2",
                "invalidate_full_prompt": True,
            }
        ]
        + [
            {"action": "registered.model.selection", **row}
            for row in serial_selection
            if row["request_id"] != "engine-2" or row["layer_id"] == 0
        ]
    )
    injection = {"count": 1, "target_request_id": "engine-2"}
    return events, recovery, observations, ordinary, serial, injection


class MemberFailure(unittest.TestCase):
    def test_injection_follows_actual_model_completion_and_fires_once(self):
        rows, calls = [], []
        counts_type = make_dataclass("Counts", [(name, int) for name in MODEL_COUNTS])

        class Session:
            def __init__(self, request_id):
                self.request_id = request_id
                self.completed_layers = 0
                self.operation_counts = counts_type(**{k: 0 for k in MODEL_COUNTS})

            def step_model(self):
                calls.append(self.request_id)
                self.completed_layers += 1
                self.operation_counts.serial_rope_operations += 1

        class Blender:
            def blend_group(self, requests):
                return requests

        ledger = SimpleNamespace(
            emit=lambda action, **data: rows.append({"action": action, **data})
        )
        state = install_member_failure(ledger, Session, Blender)
        Blender().blend_group(
            [SimpleNamespace(request_id=f"engine-{i}") for i in range(4)]
        )
        Session("engine-0").step_model()
        target = Session("engine-1")
        with self.assertRaisesRegex(RuntimeError, "first repaired layer"):
            target.step_model()
        self.assertEqual(calls, ["engine-0", "engine-1"])
        self.assertEqual(target.completed_layers, 1)
        self.assertEqual(state["count"], 1)
        self.assertEqual(
            rows[0]["completed_serial_operations"]["serial_rope_operations"], 1
        )
        target.step_model()
        self.assertEqual(state["count"], 1)

    def test_completed_serial_reference_keeps_its_role(self):
        serial = failure_control()[4]
        self.assertEqual(serial_reference_outputs(serial)["trace-2"], [2])
        self.assertNotIn("arm", serial)

    def test_serial_reference_refuses_wrong_roles_and_malformed_outputs(self):
        for case in (
            "ordinary",
            "incomplete",
            "group",
            "short",
            "duplicate",
            "empty",
            "bool",
        ):
            serial = failure_control()[4]
            if case == "ordinary":
                serial["kind"] = "ordinary"
            elif case == "incomplete":
                serial["status"] = "running"
            elif case == "group":
                serial["model_execution"] = "collective-lockstep"
            elif case == "short":
                serial["observations"].pop()
            elif case == "duplicate":
                serial["observations"][-1] = serial["observations"][0]
            elif case == "empty":
                serial["observations"][-1]["output_token_ids"] = []
            else:
                serial["observations"][-1]["output_token_ids"] = [True]
            with self.subTest(case=case), self.assertRaises(ValueError):
                serial_reference_outputs(serial)

    def test_failure_requires_settled_native_inputs_and_survivor_selections(self):
        args = failure_control()
        self.assertTrue(verify_member_failure(*args)["passed"])
        for case in (
            "pending",
            "short-restore",
            "fallback",
            "survivor",
            "selection",
            "mask",
            "invalidation",
            "early-injection",
            "injection-count",
            "bool-count",
        ):
            events, recovery, observations, ordinary, serial, injection = (
                failure_control()
            )
            if case == "pending":
                next(
                    e
                    for e in events
                    if e["action"] == "registered.model.group.consumed"
                )["pending"] = True
            elif case == "short-restore":
                events.pop(4)
            elif case == "fallback":
                observations[2]["output_token_ids"] = [2]
            elif case == "survivor":
                observations[1]["output_token_ids"] = [99]
            elif case == "selection":
                events[-1]["selected_indices"] = [999]
            elif case == "mask":
                recovery["groups"][0]["members"][0]["retrieval_mask"] = {}
            elif case in {"early-injection", "injection-count"}:
                event = next(
                    e for e in events if e["action"] == "gate.sibling_failure.injected"
                )
                if case == "early-injection":
                    event["completed_layers"] = 0
                else:
                    event["completed_serial_operations"]["serial_rope_operations"] = 0
            elif case == "bool-count":
                recovery["groups"][0]["members"][1]["operation_counts"][
                    "serial_rope_operations"
                ] = True
            else:
                next(e for e in events if "invalidate_full_prompt" in e)[
                    "invalidate_full_prompt"
                ] = False
            with self.subTest(case=case), self.assertRaises(ValueError):
                verify_member_failure(
                    events, recovery, observations, ordinary, serial, injection
                )
