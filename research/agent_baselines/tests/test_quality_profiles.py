# SPDX-License-Identifier: MIT
"""Regressions for the frozen PrefixPlan quality-profile matrix."""

import unittest

from research.agent_baselines import quality_profiles

TEMPLATE = """<pr_description>
Task: {{task}}
</pr_description>

<instructions>
Current instructions.
</instructions>
"""


class QualityProfileTests(unittest.TestCase):
    def test_two_by_two_arm_map_is_explicit(self):
        self.assertEqual(
            {
                (
                    spec["instruction_profile"],
                    spec["layout"],
                )
                for spec in quality_profiles.ARM_SPECS.values()
            },
            {
                ("current", "current"),
                ("current", "stable-prefix"),
                ("clear-technical-v1", "current"),
                ("clear-technical-v1", "stable-prefix"),
            },
        )

    def test_current_profile_preserves_both_blocks(self):
        self.assertEqual(
            quality_profiles.render_instance_template(TEMPLATE, "stock"), TEMPLATE
        )
        stable = quality_profiles.render_instance_template(TEMPLATE, "stable-prefix")
        self.assertLess(stable.index("<instructions>"), stable.index("{{task}}"))
        self.assertEqual(stable.count("{{task}}"), 1)
        self.assertIn("Current instructions.", stable)

    def test_clear_arms_differ_only_by_block_order(self):
        current = quality_profiles.render_instance_template(TEMPLATE, "clear")
        stable = quality_profiles.render_instance_template(
            TEMPLATE, "clear-stable-prefix"
        )
        instruction = quality_profiles.CLEAR_TECHNICAL_INSTRUCTIONS
        task = "<pr_description>\nTask: {{task}}\n</pr_description>"
        self.assertEqual(current, task + "\n\n" + instruction + "\n")
        self.assertEqual(stable, instruction + "\n\n" + task + "\n")

    def test_clear_profile_preserves_required_operational_contracts(self):
        rendered = quality_profiles.render_instance_template(TEMPLATE, "clear")
        for required in (
            "non-test source files",
            "Do not modify tests or configuration files",
            "Make at least one `bash` tool call",
            "new subshell",
            "Do not commit",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
            "Do not read, edit, test, or",
        ):
            with self.subTest(required=required):
                self.assertIn(required, rendered)

    def test_receipt_binds_profile_layout_and_content(self):
        receipts = {}
        for arm in quality_profiles.ARM_SPECS:
            rendered = quality_profiles.render_instance_template(TEMPLATE, arm)
            receipts[arm] = quality_profiles.quality_profile_receipt(rendered, arm)
        self.assertEqual(len({row["template_sha256"] for row in receipts.values()}), 4)
        self.assertEqual(
            receipts["clear"]["instructions_sha256"],
            receipts["clear-stable-prefix"]["instructions_sha256"],
        )
        self.assertNotEqual(
            receipts["stock"]["instructions_sha256"],
            receipts["clear"]["instructions_sha256"],
        )

    def test_malformed_template_and_receipt_drift_fail_closed(self):
        with self.assertRaises(ValueError):
            quality_profiles.render_instance_template("{{task}}", "clear")
        rendered = quality_profiles.render_instance_template(TEMPLATE, "clear")
        with self.assertRaisesRegex(ValueError, "declared layout"):
            quality_profiles.quality_profile_receipt(rendered, "clear-stable-prefix")


if __name__ == "__main__":
    unittest.main()
