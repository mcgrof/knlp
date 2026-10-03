# SPDX-License-Identifier: MIT
"""Tests for bounded PrefixPlan layout selection."""

import copy
import unittest

from research.agent_baselines import prefix_plan


class PrefixPlanTests(unittest.TestCase):
    def families(self):
        return [
            {
                "family_id": "repo-cluster",
                "candidates": {
                    "repo-first": {
                        "prefix_tokens": 1024,
                        "prefixes": ["repo-a"] * 5,
                    },
                    "tool-first": {
                        "prefix_tokens": 1024,
                        "prefixes": [f"tool-{index}" for index in range(5)],
                    },
                },
            },
            {
                "family_id": "tool-cluster",
                "candidates": {
                    "repo-first": {
                        "prefix_tokens": 1024,
                        "prefixes": [f"repo-{index}" for index in range(5)],
                    },
                    "tool-first": {
                        "prefix_tokens": 1024,
                        "prefixes": ["tool-z"] * 5,
                    },
                },
            },
        ]

    def test_joint_plan_beats_each_fixed_layout(self):
        plan = prefix_plan.compile_prefix_plan(
            self.families(),
            host_capacity_bytes=4 * 1024 * 1024,
            kv_bytes_per_token=1024,
        )
        self.assertEqual(plan["assignments_enumerated"], 4)
        self.assertEqual(
            plan["selected"]["assignment"],
            {"repo-cluster": "repo-first", "tool-cluster": "tool-first"},
        )
        self.assertEqual(plan["strongest_fixed"]["estimated_reuse_tokens"], 4096)
        self.assertEqual(plan["selected"]["estimated_reuse_tokens"], 8192)
        self.assertEqual(plan["estimated_incremental_reuse_tokens"], 4096)
        self.assertEqual(prefix_plan.validate_prefix_plan(plan), plan)

    def test_capacity_applies_to_joint_residency(self):
        plan = prefix_plan.compile_prefix_plan(
            self.families(),
            host_capacity_bytes=1024 * 1024,
            kv_bytes_per_token=1024,
        )
        self.assertEqual(plan["selected"]["estimated_reuse_tokens"], 4096)
        self.assertLessEqual(
            plan["selected"]["resident_bytes"], plan["host_capacity_bytes"]
        )

    def test_mixed_plan_can_fit_when_uniform_layouts_do_not(self):
        families = [
            {
                "family_id": "A",
                "candidates": {
                    "x": {"prefix_tokens": 1, "prefixes": ["a", "a"]},
                    "y": {"prefix_tokens": 10, "prefixes": ["b", "b"]},
                },
            },
            {
                "family_id": "B",
                "candidates": {
                    "x": {"prefix_tokens": 10, "prefixes": ["c", "c"]},
                    "y": {"prefix_tokens": 1, "prefixes": ["d", "d"]},
                },
            },
        ]
        plan = prefix_plan.compile_prefix_plan(
            families, host_capacity_bytes=2, kv_bytes_per_token=1
        )
        self.assertIsNone(plan["strongest_fixed"])
        self.assertIsNotNone(plan["strongest_fixed_unavailable_reason"])
        self.assertIsNone(plan["estimated_incremental_reuse_tokens"])
        self.assertEqual(plan["selected"]["assignment"], {"A": "x", "B": "y"})
        self.assertEqual(plan["selected"]["resident_bytes"], 2)
        self.assertEqual(prefix_plan.validate_prefix_plan(plan), plan)

    def test_global_prefix_identity_is_allocated_once(self):
        families = []
        for index in range(2):
            families.append(
                {
                    "family_id": f"family-{index}",
                    "candidates": {
                        "shared": {
                            "prefix_tokens": 4,
                            "prefixes": ["same-prefix", "same-prefix"],
                        },
                        "unique": {
                            "prefix_tokens": 4,
                            "prefixes": [
                                f"unique-{index}-a",
                                f"unique-{index}-b",
                            ],
                        },
                    },
                }
            )
        plan = prefix_plan.compile_prefix_plan(
            families, host_capacity_bytes=4, kv_bytes_per_token=1
        )
        score = next(
            candidate
            for candidate in plan["fixed_candidates"]
            if set(candidate["assignment"].values()) == {"shared"}
        )
        self.assertTrue(score["feasible"])
        self.assertEqual(score["resident_bytes"], 4)
        self.assertEqual(score["estimated_reuse_tokens"], 12)
        self.assertEqual(score["global_reusable_prefixes"], 1)

    def test_global_identity_requires_one_token_length(self):
        families = self.families()
        families[1]["candidates"]["tool-first"] = {
            "prefix_tokens": 2048,
            "prefixes": ["repo-a"] * 5,
        }
        with self.assertRaisesRegex(ValueError, "one prefix_tokens value"):
            prefix_plan.compile_prefix_plan(
                families, host_capacity_bytes=1, kv_bytes_per_token=1
            )

    def test_malformed_or_unbounded_search_is_rejected(self):
        many = []
        for index in range(4):
            for family in self.families():
                item = copy.deepcopy(family)
                item["family_id"] += f"-{index}"
                many.append(item)
        with self.assertRaisesRegex(ValueError, "above max_assignments"):
            prefix_plan.compile_prefix_plan(
                many,
                host_capacity_bytes=1,
                kv_bytes_per_token=1,
                max_assignments=4,
            )
        malformed = self.families()
        malformed[1]["candidates"].pop("tool-first")
        with self.assertRaises(ValueError):
            prefix_plan.compile_prefix_plan(
                malformed, host_capacity_bytes=1, kv_bytes_per_token=1
            )

    def test_validation_rejects_tampered_receipt(self):
        plan = prefix_plan.compile_prefix_plan(
            self.families(),
            host_capacity_bytes=4 * 1024 * 1024,
            kv_bytes_per_token=1024,
        )
        changes = [
            ("estimated_incremental_reuse_tokens", 1),
            ("families_sha256", "forged"),
            ("host_capacity_bytes", 1),
            ("selected.assignment", {"missing-family": "missing-layout"}),
            ("selected.resident_bytes", 10**20),
            ("selected.estimated_reuse_tokens", 10**12),
        ]
        for name, replacement in changes:
            with self.subTest(name=name):
                changed = copy.deepcopy(plan)
                target = changed
                parts = name.split(".")
                for part in parts[:-1]:
                    target = target[part]
                if name == "estimated_incremental_reuse_tokens":
                    replacement = changed[name] + replacement
                target[parts[-1]] = replacement
                with self.assertRaises(ValueError):
                    prefix_plan.validate_prefix_plan(changed)


if __name__ == "__main__":
    unittest.main()
