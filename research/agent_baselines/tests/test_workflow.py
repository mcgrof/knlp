# SPDX-License-Identifier: MIT
"""Tests for PrefixPlan's backend-neutral workflow and event contracts."""

import copy
import unittest

from research.agent_baselines import workflow


class WorkflowTests(unittest.TestCase):
    def segments(self):
        return [
            workflow.prompt_segment(
                "instructions",
                role="developer",
                origin="mini.instance_template.instructions",
                content="Use the tools and fix the issue.",
                token_ids=[1, 2, 3],
                movable=True,
            ),
            workflow.prompt_segment(
                "task",
                role="user",
                origin="swebench.problem_statement",
                content="Fix this bug.",
                token_ids=[4, 5],
                inputs=["problem_statement"],
            ),
        ]

    def request(self, segments, request_id="request-1", depends_on=()):
        return workflow.workflow_request(
            request_id,
            workflow_id="workflow-1",
            branch_id="branch-0",
            segment_ids=["instructions", "task"],
            segments=segments,
            prompt_token_ids=[1, 2, 3, 4, 5],
            engine_settings={"temperature": 0, "max_tokens": 64},
            depends_on=depends_on,
            ready_ns=10,
        )

    def manifest(self):
        segments = self.segments()
        return workflow.workflow_manifest(
            "workflow-1",
            model_id="test/model",
            model_revision="a" * 40,
            tokenizer_revision="b" * 40,
            chat_template_sha256="c" * 64,
            segments=segments,
            requests=[self.request(segments)],
        )

    def test_receipts_hash_content_without_retaining_it(self):
        segments = self.segments()
        self.assertNotIn("content", segments[0])
        self.assertEqual(segments[0]["token_count"], 3)
        self.assertEqual(
            segments[0]["content_sha256"],
            workflow.text_sha256("Use the tools and fix the issue."),
        )
        manifest = self.manifest()
        self.assertEqual(manifest["schema"], workflow.WORKFLOW_SCHEMA)
        self.assertEqual(manifest["requests"][0]["prompt"]["token_count"], 5)

    def test_layout_hash_changes_with_order_role_or_content(self):
        segments = self.segments()
        original = workflow.prompt_layout_sha256(["instructions", "task"], segments)
        self.assertNotEqual(
            original,
            workflow.prompt_layout_sha256(["task", "instructions"], segments),
        )
        changed = copy.deepcopy(segments)
        changed[0]["role"] = "user"
        self.assertNotEqual(
            original,
            workflow.prompt_layout_sha256(["instructions", "task"], changed),
        )
        changed = copy.deepcopy(segments)
        changed[0]["content_sha256"] = "d" * 64
        self.assertNotEqual(
            original,
            workflow.prompt_layout_sha256(["instructions", "task"], changed),
        )

    def test_unknown_segments_and_tampered_layout_fail(self):
        segments = self.segments()
        with self.assertRaises(ValueError):
            workflow.workflow_request(
                "request-1",
                workflow_id="workflow-1",
                branch_id="branch-0",
                segment_ids=["missing"],
                segments=segments,
                prompt_token_ids=[1],
                engine_settings={},
            )
        manifest = self.manifest()
        manifest["requests"][0]["prompt"]["layout_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            workflow.validate_workflow_manifest(manifest)

    def test_malformed_graph_identifiers_fail_as_schema_errors(self):
        for field, value in (
            ("segment_ids", [{}]),
            ("depends_on", [{}]),
        ):
            manifest = self.manifest()
            manifest["requests"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                workflow.validate_workflow_manifest(manifest)
        manifest = self.manifest()
        manifest["segments"][0]["inputs"] = [{}]
        with self.assertRaises(ValueError):
            workflow.validate_workflow_manifest(manifest)

    def test_dependency_graph_rejects_unknown_and_cycles(self):
        segments = self.segments()
        first = self.request(segments, "first", depends_on=["second"])
        second = self.request(segments, "second", depends_on=["first"])
        base = {
            "schema": workflow.WORKFLOW_SCHEMA,
            "workflow_id": "workflow-1",
            "model": {
                "model_id": "test/model",
                "model_revision": "a" * 40,
                "tokenizer_revision": "b" * 40,
                "chat_template_sha256": "c" * 64,
            },
            "segments": segments,
            "requests": [first, second],
            "tool_effects": [],
        }
        with self.assertRaisesRegex(ValueError, "cycle"):
            workflow.validate_workflow_manifest(base)
        first["depends_on"] = ["missing"]
        second["depends_on"] = []
        with self.assertRaisesRegex(ValueError, "unknown"):
            workflow.validate_workflow_manifest(base)

    def test_tool_effects_bind_to_requests_and_versions(self):
        manifest = self.manifest()
        manifest["tool_effects"] = [
            {
                "effect_id": "edit-1",
                "request_id": "request-1",
                "kind": "write",
                "resource": "repo-worktree",
                "before_version": "tree-a",
                "after_version": "tree-b",
                "isolated": True,
            }
        ]
        self.assertEqual(
            workflow.validate_workflow_manifest(manifest)["tool_effects"][0][
                "after_version"
            ],
            "tree-b",
        )
        manifest["tool_effects"][0]["request_id"] = "missing"
        with self.assertRaises(ValueError):
            workflow.validate_workflow_manifest(manifest)


class CacheEventTests(unittest.TestCase):
    def event(self, **changes):
        value = {
            "schema": workflow.CACHE_EVENT_SCHEMA,
            "event_id": "event-1",
            "workflow_id": "workflow-1",
            "request_id": "request-1",
            "backend": "sglang",
            "phase": "observed",
            "source": "engine",
            "action": "retrieve",
            "tier": "host",
            "timestamp_ns": 100,
            "token_count": 1024,
            "byte_count": 4096,
            "pressure": True,
            "enforced": True,
            "provenance": {"field": "cached_tokens_details.host"},
        }
        value.update(changes)
        return value

    def test_physical_event_requires_engine_observation(self):
        event = workflow.validate_cache_event(self.event())
        self.assertEqual(event["tier"], "host")
        for changes in (
            {"source": "offline"},
            {"enforced": None},
            {"token_count": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                workflow.validate_cache_event(self.event(**changes))

    def test_offline_reuse_cannot_claim_physical_enforcement(self):
        analysis = self.event(
            phase="analysis",
            source="offline",
            action="potential_reuse",
            tier="none",
            enforced=None,
            pressure=None,
        )
        workflow.validate_cache_event(analysis)
        analysis["enforced"] = True
        with self.assertRaises(ValueError):
            workflow.validate_cache_event(analysis)
        analysis["enforced"] = None
        analysis["tier"] = "host"
        with self.assertRaises(ValueError):
            workflow.validate_cache_event(analysis)

    def test_requested_action_keeps_enforcement_unknown(self):
        requested = self.event(
            phase="requested",
            source="client",
            action="prefetch",
            enforced=None,
        )
        workflow.validate_cache_event(requested)
        requested["enforced"] = True
        with self.assertRaises(ValueError):
            workflow.validate_cache_event(requested)
        for phase, source, enforced in (
            ("requested", "client", None),
            ("observed", "engine", True),
        ):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                workflow.validate_cache_event(
                    self.event(
                        phase=phase,
                        source=source,
                        action="potential_reuse",
                        tier="none",
                        enforced=enforced,
                    )
                )


if __name__ == "__main__":
    unittest.main()
