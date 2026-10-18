# SPDX-License-Identifier: MIT
"""Reject false load-coalescing claims independently of backend read counts."""

from pathlib import Path
import sys
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
from singleflight import check_admission


def family(copies=1):
    tables = {str(i): list(range(1 + i * 256, 257 + i * 256)) for i in range(copies)}
    return {
        "fanout": 2,
        "singleflight": {"inflight": 0, "joined": 0},
        "live_prefix": {
            "null_block_id": 0,
            "total_blocks": 4096,
            "events": [
                {
                    "action": "allocate",
                    "tables": tables,
                    "statuses": {key: "RUNNING" for key in tables},
                    "computed_tokens": {key: 4096 for key in tables},
                },
                {"action": "free", "tables": {}, "statuses": {}, "computed_tokens": {}},
            ],
            "lookups": [
                {
                    "request_id": "leader",
                    "local_tokens": 0,
                    "external_new_tokens": 4096,
                },
                {
                    "request_id": "follower",
                    "local_tokens": 4096,
                    "external_new_tokens": 0,
                },
            ],
        },
    }


class SingleflightTest(unittest.TestCase):
    def test_one_shared_ancestor(self):
        check_admission(family(), True)

    def test_duplicate_ready_ancestor_rejected(self):
        with self.assertRaisesRegex(ValueError, "one shared ancestor"):
            check_admission(family(2), True)

    def test_disabled_control_can_have_multiple_copies(self):
        check_admission(family(2), False)

    def test_follower_state_must_drain_in_both_arms(self):
        row = family()
        row["singleflight"]["joined"] = 1
        for enabled in (False, True):
            with self.assertRaisesRegex(ValueError, "state remains"):
                check_admission(row, enabled)

    def test_extra_external_load_rejected_even_with_one_ready_copy(self):
        row = family()
        row["live_prefix"]["lookups"][1].update(
            local_tokens=0, external_new_tokens=4096
        )
        with self.assertRaisesRegex(ValueError, "one shared ancestor"):
            check_admission(row, True)


if __name__ == "__main__":
    unittest.main()
