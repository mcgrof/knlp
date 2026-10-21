# SPDX-License-Identifier: MIT
"""Check prompt token identity and bounded arithmetic grading."""

import unittest
from unittest.mock import Mock, patch
import sys
from types import SimpleNamespace

from tools.reproduce.kmesh_residency.tot_workload import (
    grade,
    LocalProvider,
    prompt_tokens,
    WORKLOAD_SECONDS,
)


class PromptTokensTest(unittest.TestCase):
    def test_flat_ids_preserve_chat_template_output(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [151644, 8948, 198]
        self.assertEqual(prompt_tokens(tokenizer, "1 1 4 6"), [151644, 8948, 198])
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["return_dict"])

    def test_bad_shapes_and_oversized_prompts_are_rejected(self):
        for result in (
            {"input_ids": [1, 2], "attention_mask": [1, 1]},
            ["input_ids", "attention_mask"],
            [[1, 2]],
            [True],
            [-1],
            [],
            [1] * 1857,
        ):
            with self.subTest(result_type=type(result).__name__):
                tokenizer = Mock()
                tokenizer.apply_chat_template.return_value = result
                with self.assertRaises(ValueError):
                    prompt_tokens(tokenizer, "1 1 4 6")


class ArithmeticTest(unittest.TestCase):
    def test_exact_rationals_and_original_number_multiset(self):
        self.assertTrue(grade([1, 1, 4, 6], "Answer: 4 * 6 + 1 - 1 = 24"))
        self.assertTrue(grade([3, 3, 8, 8], "8 / (3 - 8 / 3) = 24"))
        self.assertFalse(grade([1, 1, 4, 6], "4 * 6 = 24"))
        self.assertFalse(grade([1, 1, 4, 6], "4 * 6 + 1 + 1 = 24"))

    def test_calls_names_powers_and_large_expressions_are_rejected(self):
        for expression in (
            '__import__("os").system("false")',
            'open("anything")',
            "24",
            "2 ** 10000000",
            "(1 / 0)",
            "True",
            "1+" * 300 + "1",
        ):
            self.assertFalse(grade([1, 1, 4, 6], expression))


class WorkloadBudgetTest(unittest.TestCase):
    def provider(self):
        llm, capture = Mock(), Mock()
        llm.get_tokenizer.return_value.apply_chat_template.return_value = [1, 2]
        capture.invoke.return_value = ("reply", None)
        with patch(
            "tools.reproduce.kmesh_residency.tot_workload.time.perf_counter_ns",
            return_value=0,
        ):
            provider = LocalProvider(llm, capture)
        provider.family, provider.purpose = "case", "propose"
        return provider, capture

    def test_wall_clock_and_request_limits_refuse_before_model_submission(self):
        for calls, stamp, error in (
            (384, 0, RuntimeError),
            (0, WORKLOAD_SECONDS * 1_000_000_000 + 1, TimeoutError),
        ):
            with self.subTest(calls=calls, stamp=stamp):
                provider, capture = self.provider()
                provider.calls = calls
                with patch(
                    "tools.reproduce.kmesh_residency.tot_workload.time.perf_counter_ns",
                    return_value=stamp,
                ):
                    with self.assertRaises(error):
                        provider("prompt")
                capture.invoke.assert_not_called()
                self.assertEqual(provider.calls, calls)

    def test_longer_allowance_preserves_submission_settings(self):
        provider, capture = self.provider()
        sampling = Mock(return_value="sampling-parameters")
        with patch.dict(
            sys.modules, {"vllm": SimpleNamespace(SamplingParams=sampling)}
        ):
            with patch(
                "tools.reproduce.kmesh_residency.tot_workload.time.perf_counter_ns",
                return_value=901_000_000_000,
            ):
                self.assertEqual(provider("prompt"), ["reply"])
        sampling.assert_called_once_with(temperature=0.7, max_tokens=192, seed=123)
        self.assertEqual(capture.invoke.call_args.args[3], [1, 2])
        self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
