# SPDX-License-Identifier: MIT
"""Distinguish simultaneous destination ownership from later block reuse."""

import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "coldburst_verify", HERE / "coldburst_verify.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def event(tables, action="allocate"):
    return {
        "action": action,
        "tables": tables,
        "statuses": {key: "RUNNING" for key in tables},
        "computed_tokens": {key: 4096 for key in tables},
    }


class LiveOwnershipTest(unittest.TestCase):
    def evidence(self, events):
        return {
            "null_block_id": 0,
            "total_blocks": 4096,
            "events": events + [event({}, "free")],
        }

    def test_simultaneous_private_destinations(self):
        result = VERIFY.live_stats(
            self.evidence(
                [event({"a": list(range(1, 257)), "b": list(range(257, 513))})]
            )
        )
        self.assertEqual(result["max_live_distinct_ancestor_blocks"], 512)
        self.assertEqual(result["max_live_pairs_sharing_ancestor_blocks"], 0)

    def test_simultaneous_shared_destination(self):
        result = VERIFY.live_stats(
            self.evidence([event({"a": list(range(1, 257)), "b": list(range(1, 257))})])
        )
        self.assertEqual(result["max_live_distinct_ancestor_blocks"], 256)
        self.assertEqual(result["max_live_pairs_sharing_ancestor_blocks"], 1)

    def test_later_reuse_is_not_simultaneous_sharing(self):
        result = VERIFY.live_stats(
            self.evidence(
                [event({"a": list(range(1, 257))}), event({"b": list(range(1, 257))})]
            )
        )
        self.assertEqual(result["max_live_requests"], 1)
        self.assertEqual(result["max_live_pairs_sharing_ancestor_blocks"], 0)

    def test_waiting_remote_slots_are_not_ready_materializations(self):
        snapshot = event({"a": list(range(1, 257))})
        snapshot["statuses"]["a"] = "WAITING_FOR_REMOTE_KVS"
        result = VERIFY.live_stats(self.evidence([snapshot]))
        self.assertEqual(result["max_live_distinct_ancestor_blocks"], 256)
        self.assertEqual(result["max_live_ready_ancestor_blocks"], 0)

    def test_local_hits_are_distinct_from_external_loads(self):
        evidence = self.evidence([])
        evidence["lookups"] = [
            {"request_id": "leader", "local_tokens": 0, "external_new_tokens": None},
            {"request_id": "leader", "local_tokens": 0, "external_new_tokens": 4096},
            {"request_id": "follower", "local_tokens": 4096, "external_new_tokens": 0},
        ]
        result = VERIFY.live_stats(evidence)
        self.assertEqual(result["resolved_local_hit_requests"], 1)
        self.assertEqual(result["resolved_external_load_requests"], 1)

    def test_null_block_cannot_be_an_ancestor(self):
        with self.assertRaisesRegex(ValueError, "invalid ancestor"):
            VERIFY.live_stats(self.evidence([event({"a": list(range(256))})]))

    def test_unreleased_requests_are_rejected(self):
        evidence = self.evidence([event({"a": list(range(1, 257))})])
        evidence["events"].pop()
        with self.assertRaisesRegex(ValueError, "not drained"):
            VERIFY.live_stats(evidence)


if __name__ == "__main__":
    unittest.main()
