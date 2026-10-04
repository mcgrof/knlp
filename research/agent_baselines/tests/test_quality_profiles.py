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
    def test_two_by_two_arm_map_and_finish_protocol_are_explicit(self):
        self.assertEqual(
            {
                (
                    spec["instruction_profile"],
                    spec["layout"],
                    spec["execution_protocol"],
                )
                for spec in quality_profiles.ARM_SPECS.values()
            },
            {
                ("current", "current", "current"),
                ("current", "stable-prefix", "current"),
                ("clear-technical-v1", "current", "current"),
                ("clear-technical-v1", "stable-prefix", "current"),
                ("current", "current", "finish-protocol-v1"),
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
            observation = quality_profiles.apply_observation_protocol("result", arm)
            receipts[arm] = quality_profiles.quality_profile_receipt(
                rendered, arm, observation
            )
        self.assertEqual(len({row["template_sha256"] for row in receipts.values()}), 5)
        self.assertEqual(
            receipts["clear"]["instructions_sha256"],
            receipts["clear-stable-prefix"]["instructions_sha256"],
        )
        self.assertNotEqual(
            receipts["stock"]["instructions_sha256"],
            receipts["clear"]["instructions_sha256"],
        )

    def test_finish_protocol_preserves_current_text_and_reports_live_budget(self):
        from jinja2 import Template

        stock = quality_profiles.render_instance_template(TEMPLATE, "stock")
        finish = quality_profiles.render_instance_template(TEMPLATE, "finish-protocol")
        self.assertIn("Current instructions.", finish)
        self.assertIn("Observed facts", finish)
        self.assertIn("Reserve the final three", finish)
        self.assertIn("calls, in order", finish)
        self.assertNotEqual(stock, finish)
        observation = quality_profiles.apply_observation_protocol(
            "<output>result</output>", "finish-protocol"
        )
        rendered = Template(observation).render(step_limit=30, n_model_calls=25)
        self.assertIn("model_calls_remaining=5", rendered)
        self.assertIn("checkpoint=decide_essential_verification", rendered)
        rendered = Template(observation).render(step_limit=30, n_model_calls=27)
        self.assertIn("required=Create patch.txt in this call.", rendered)
        with self.assertRaisesRegex(ValueError, "live observation state"):
            quality_profiles.quality_profile_receipt(finish, "finish-protocol")

    def test_malformed_template_and_receipt_drift_fail_closed(self):
        with self.assertRaises(ValueError):
            quality_profiles.render_instance_template("{{task}}", "clear")
        rendered = quality_profiles.render_instance_template(TEMPLATE, "clear")
        with self.assertRaisesRegex(ValueError, "declared layout"):
            quality_profiles.quality_profile_receipt(rendered, "clear-stable-prefix")


if __name__ == "__main__":
    unittest.main()
