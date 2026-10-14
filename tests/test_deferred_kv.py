"""Check query separation and content interventions before using real models."""

import unittest
import re

from tools.kv.prequantized.deferred import fixture, query_variant


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def __call__(self, text, **kwargs):
        pieces = list(re.finditer(r" [.]|.", text, re.DOTALL))
        return {
            "input_ids": [
                ord(m.group()) if len(m.group()) == 1 else 0xE000 for m in pieces
            ],
            "offset_mapping": [m.span() for m in pieces],
        }

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"


class DateTokenizer(CharacterTokenizer):
    date = "2001-01-01"

    def apply_chat_template(self, messages, **kwargs):
        return self.date + super().apply_chat_template(messages, **kwargs)


class DeferredTests(unittest.TestCase):
    def test_query_interventions_preserve_the_frozen_chat_frame(self):
        tokenizer = DateTokenizer()
        sample = fixture(tokenizer, "example/model", 2048, 0, 20261016, True)
        tokenizer.date = "2001-01-02"
        variant = query_variant(tokenizer, "example/model", sample, 0)
        self.assertIn("2001-01-01<user>", variant["prompt"])
        self.assertEqual(
            sample["token_ids"][: sample["catalogue_tokens"]],
            variant["token_ids"][: variant["catalogue_tokens"]],
        )
        changed = query_variant(
            tokenizer, "example/model", sample, sample["target_index"], "one two three"
        )
        self.assertIn("2001-01-01<user>", changed["prompt"])
        self.assertEqual(
            sample["token_ids"][sample["catalogue_tokens"] :],
            changed["token_ids"][changed["catalogue_tokens"] :],
        )

    def test_sensitivity_cannot_pass_a_ceiling_workload_that_ignores_cache(self):
        from tools.kv.prequantized.sensitivity import CRITERION, sensitivity_verdict

        rows = []
        for length in (2048, 8192):
            for condition, count in (
                ("query", 15),
                ("zero", 5),
                ("changed", 5),
                ("concatenated", 5),
            ):
                for index in range(count):
                    rows.append(
                        {
                            "length": length,
                            "condition": condition,
                            "original_target": index < 5,
                            "result": {
                                "correct": condition != "zero",
                                "malformed": False,
                                "truncated": False,
                            },
                        }
                    )
        self.assertEqual(sensitivity_verdict(rows, CRITERION)["status"], "passed")
        for row in rows:
            row["result"]["correct"] = True
        self.assertEqual(sensitivity_verdict(rows, CRITERION)["status"], "failed")
        self.assertRaises(ValueError, sensitivity_verdict, rows[:-1], CRITERION)

    def test_query_changes_share_one_catalogue(self):
        tokenizer = CharacterTokenizer()
        sample = fixture(tokenizer, "Qwen/example", 2048, 0, 20261016, True)
        self.assertEqual(len(sample["token_ids"]), 2048)
        prefix = sample["token_ids"][: sample["catalogue_tokens"]]
        self.assertNotIn("Question:", "".join(map(chr, prefix)))
        keys = [r["key"] for r in sample["records"]]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(k.startswith("REC-") for k in keys))
        for index in (0, len(keys) // 2, len(keys) - 1):
            variant = query_variant(tokenizer, "Qwen/example", sample, index)
            self.assertEqual(
                prefix, variant["token_ids"][: variant["catalogue_tokens"]]
            )
            self.assertEqual(variant["answer"], sample["records"][index]["value"])

    def test_changed_value_changes_only_catalogue_content(self):
        tokenizer = CharacterTokenizer()
        sample = fixture(tokenizer, "Qwen/example", 2048, 0, 20261016, True)
        changed = query_variant(
            tokenizer, "Qwen/example", sample, sample["target_index"], "one two three"
        )
        self.assertNotEqual(
            sample["token_ids"][: sample["catalogue_tokens"]],
            changed["token_ids"][: changed["catalogue_tokens"]],
        )
        self.assertEqual(
            sample["token_ids"][sample["catalogue_tokens"] :],
            changed["token_ids"][changed["catalogue_tokens"] :],
        )
        self.assertEqual(changed["answer"], "one two three")


if __name__ == "__main__":
    unittest.main()
