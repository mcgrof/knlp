# SPDX-License-Identifier: MIT
"""Tests for source-pinned SGLang capability receipts."""

import unittest

from research.agent_baselines import sglang_adapter


class SGLangAdapterTests(unittest.TestCase):
    def receipt(self):
        return sglang_adapter.capability_receipt(
            ["openai_chat", "sglext_token_ids", "cache_source_report"]
        )

    def test_receipt_keeps_availability_and_enablement_separate(self):
        receipt = self.receipt()
        self.assertTrue(receipt["features"]["frontend_ir"]["available"])
        self.assertFalse(receipt["features"]["frontend_ir"]["enabled"])
        self.assertFalse(receipt["features"]["explicit_request_residency"]["available"])
        self.assertEqual(
            sglang_adapter.validate_capability_receipt(receipt, require_live=True),
            receipt,
        )

    def test_unavailable_or_unknown_features_cannot_be_enabled(self):
        for feature in ("explicit_request_residency", "not-a-feature"):
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                sglang_adapter.capability_receipt([feature])
        for enabled in (
            "openai_chat",
            ["openai_chat", "openai_chat"],
            ["openai_chat", 1],
        ):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                sglang_adapter.capability_receipt(enabled)

    def test_receipt_rejects_revision_and_evidence_drift(self):
        for mutate in (
            lambda value: value.update(source_revision="a" * 40),
            lambda value: value["features"]["openai_chat"].update(evidence="different"),
            lambda value: value["features"]["prompt_reordering_compiler"].update(
                enabled=True
            ),
        ):
            receipt = self.receipt()
            mutate(receipt)
            with self.assertRaises(ValueError):
                sglang_adapter.validate_capability_receipt(receipt)

    def test_live_request_requires_exact_token_capture_features(self):
        receipt = sglang_adapter.capability_receipt(["openai_chat"])
        with self.assertRaisesRegex(ValueError, "sglext_token_ids"):
            sglang_adapter.request_extensions(
                receipt, request_id="request-1", cache_scope="experiment-1"
            )

    def test_request_extensions_are_bounded_to_audited_fields(self):
        extensions = sglang_adapter.request_extensions(
            self.receipt(), request_id="request-1", cache_scope="experiment-1"
        )
        self.assertEqual(extensions["rid"], "request-1")
        self.assertEqual(extensions["extra_key"], "experiment-1")
        self.assertTrue(extensions["return_input_ids_in_sglext"])
        self.assertTrue(extensions["return_output_ids_in_sglext"])
        self.assertTrue(extensions["return_cached_tokens_details"])
        self.assertNotIn("retain", extensions)
        self.assertNotIn("prefetch", extensions)

    def test_cache_scope_is_explicit_and_separate_from_request_identity(self):
        first = sglang_adapter.request_extensions(
            self.receipt(), request_id="task-a:1", cache_scope="tenant:model:adapter-a"
        )
        second = sglang_adapter.request_extensions(
            self.receipt(), request_id="task-b:1", cache_scope="tenant:model:adapter-a"
        )
        isolated = sglang_adapter.request_extensions(
            self.receipt(), request_id="task-c:1", cache_scope="tenant:model:adapter-b"
        )
        self.assertNotEqual(first["rid"], second["rid"])
        self.assertEqual(first["extra_key"], second["extra_key"])
        self.assertNotEqual(first["extra_key"], isolated["extra_key"])

    def test_invalid_cache_scopes_are_rejected(self):
        for value in (None, "", "has spaces", "../escape", "x" * 129):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sglang_adapter.request_extensions(
                    self.receipt(), request_id="request-1", cache_scope=value
                )

    def test_validation_returns_a_copy(self):
        receipt = self.receipt()
        checked = sglang_adapter.validate_capability_receipt(receipt)
        checked["features"]["openai_chat"]["enabled"] = False
        self.assertTrue(receipt["features"]["openai_chat"]["enabled"])


if __name__ == "__main__":
    unittest.main()
