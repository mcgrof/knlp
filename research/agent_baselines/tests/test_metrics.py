"""Verify counter semantics without a running inference server."""

import unittest
from unittest.mock import patch

from research.agent_baselines.metrics import compare_snapshots, snapshot


def sample(text, time_ns=1):
    return {
        "ok": True,
        "url": "http://localhost/metrics",
        "raw_text": text,
        "time_ns": time_ns,
    }


def metrics(queries, hits, labels='model_name="m",engine="0"', prompts=None):
    text = (
        f"vllm:prefix_cache_queries_total{{{labels}}} {queries}\n"
        f"vllm:prefix_cache_hits_total{{{labels}}} {hits}\n"
    )
    if prompts is not None:
        text += f"vllm:prompt_tokens_total{{{labels}}} {prompts}\n"
    return text


class MetricsTests(unittest.TestCase):
    def compare(self, before, after):
        return compare_snapshots(sample(before), sample(after, 2))

    def test_weighted_ratio_and_label_order(self):
        before = metrics(10, 5) + metrics(200, 100, 'engine="1",model_name="m"')
        after = metrics(20, 15, 'engine="0",model_name="m"') + metrics(
            290, 100, 'engine="1",model_name="m"'
        )
        result = self.compare(before, after)
        self.assertTrue(result["valid"])
        self.assertEqual(result["queried_tokens"], 100)
        self.assertEqual(result["token_hit_ratio"], 0.1)
        self.assertEqual(len(result["series"]), 2)

    def test_zero_hits_is_measured_zero(self):
        result = self.compare(metrics(0, 0), metrics(100, 0))
        self.assertTrue(result["valid"])
        self.assertEqual(result["token_hit_ratio"], 0)

    def test_no_queries_is_not_zero_hit_ratio(self):
        result = self.compare(metrics(1, 0), metrics(1, 0))
        self.assertTrue(result["valid"])
        self.assertIsNone(result["token_hit_ratio"])
        self.assertEqual(result["reason"], "no_queried_tokens")

    def test_reset_cannot_be_hidden_by_another_engine(self):
        before = metrics(100, 10) + metrics(1, 0, 'engine="1"')
        after = metrics(10, 1) + metrics(1000, 500, 'engine="1"')
        result = self.compare(before, after)
        self.assertEqual(result["reason"], "counter_reset")
        self.assertIsNone(result["token_hit_ratio"])

    def test_missing_counter(self):
        result = self.compare(metrics(1, 0), "vllm:prefix_cache_queries_total 10")
        self.assertFalse(result["valid"])
        self.assertIsNone(result["token_hit_ratio"])

    def test_changed_model_series(self):
        result = self.compare(
            metrics(1, 0), metrics(2, 0, 'model_name="other",engine="0"')
        )
        self.assertEqual(result["reason"], "changed_series")

    def test_unpaired_hit_query_labels(self):
        raw = 'vllm:prefix_cache_queries_total{engine="0"} 10\nvllm:prefix_cache_hits_total{engine="1"} 2'
        result = self.compare(raw, raw)
        self.assertEqual(result["reason"], "unpaired_hit_query_series")

    def test_optional_prompt_counter_not_ratio_denominator(self):
        result = self.compare(metrics(0, 0, prompts=0), metrics(100, 40, prompts=60))
        self.assertEqual(result["token_hit_ratio"], 0.4)
        self.assertEqual(result["prompt_tokens_delta"], 60)

    def test_optional_prompt_reset_is_observable(self):
        result = self.compare(metrics(0, 0, prompts=100), metrics(100, 40, prompts=60))
        self.assertTrue(result["valid"])
        self.assertIsNone(result["prompt_tokens_delta"])
        self.assertEqual(result["prompt_tokens_reason"], "counter_reset")

    def test_invalid_samples_never_become_zero(self):
        for raw in (
            metrics("NaN", 0),
            metrics(1, 0) * 2,
            metrics(-1, 0),
            metrics(1, 0, 'engine="0",engine="1"'),
        ):
            with self.subTest(raw=raw):
                result = self.compare(raw, metrics(10, 0))
                self.assertEqual(result["reason"], "invalid_exposition")

    def test_impossible_hit_delta(self):
        result = self.compare(metrics(0, 0), metrics(1, 2))
        self.assertEqual(result["reason"], "hits_exceed_queries")

    def test_url_validation_precedes_network(self):
        with patch("urllib.request.urlopen") as request:
            for url in (
                "http://user:secret@localhost/metrics",
                "http://localhost/metrics?token=secret",
                "file:///etc/passwd",
            ):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    snapshot(url)
            request.assert_not_called()

    def test_fetch_failure_is_observable_and_sanitized(self):
        with patch("urllib.request.urlopen", side_effect=OSError("secret")):
            result = snapshot("http://localhost/metrics")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "OSError")
        self.assertIsNone(result["raw_text"])
        comparison = compare_snapshots(result, result)
        self.assertEqual(comparison["reason"], "snapshot_unavailable")

    def test_endpoint_and_clock_changes(self):
        before = sample(metrics(0, 0))
        after = sample(metrics(1, 0), 2)
        after["url"] = "http://other/metrics"
        self.assertEqual(compare_snapshots(before, after)["reason"], "changed_endpoint")
        self.assertEqual(
            compare_snapshots(before, before)["reason"], "unordered_snapshots"
        )


if __name__ == "__main__":
    unittest.main()
