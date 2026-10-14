# SPDX-License-Identifier: MIT
"""Check seed gates, input denominators and immutable preparation inputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_results import analyze, summarize
from prepare_cache import parse_args


def row(seed, fidelity=0.96):
    return {
        "seed": seed,
        "exact_top10_candidate_recall": fidelity,
        "document_count": 10,
        "query_count": 3,
        "repetitions": 20,
        "k_sim": 4,
        "projection_dim": 8,
        "dimension": 2560,
        "record_bytes": 10240,
    }


class FidelityGateTests(unittest.TestCase):
    def test_strong_mean_does_not_hide_a_failing_seed(self):
        rows = [row(1, 0.94), row(1001, 0.99), row(2001, 0.99)]
        result = summarize(rows, [1, 1001, 2001], 0.95, 10, 3)[0]
        self.assertGreater(result["mean_fidelity"], 0.95)
        self.assertFalse(result["passes_every_seed"])

    def test_missing_and_duplicate_seed_results_are_refused(self):
        for rows in ([row(1)], [row(1), row(1), row(2001)]):
            with self.assertRaisesRegex(ValueError, "exactly every expected seed"):
                summarize(rows, [1, 1001, 2001], 0.95, 10, 3)

    def test_denominators_must_match_loaded_inputs(self):
        with self.assertRaisesRegex(ValueError, "denominators"):
            summarize([row(1)], [1], 0.95, 10, 4)

    def test_overlapping_repetition_ranges_are_refused(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            summarize([row(1), row(2)], [1, 2], 0.95, 10, 3)

    def test_dimensions_must_match_across_seeds(self):
        rows = [row(1), row(1001)]
        rows[1]["dimension"] = 1280
        with self.assertRaisesRegex(ValueError, "dimensions"):
            summarize(rows, [1, 1001], 0.95, 10, 3)

    def test_nonfinite_fidelity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize([row(1, np.nan)], [1], 0.95, 10, 3)

    def test_existing_analysis_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError, "new directory"):
                analyze(directory / "missing-input", directory)


class PinnedPreparationTests(unittest.TestCase):
    def arguments(self, directory):
        snapshot = directory / ("a" * 40)
        snapshot.mkdir()
        return [
            "--output-dir",
            str(directory / "output"),
            "--colbert-snapshot",
            str(snapshot),
            "--bge-snapshot",
            str(snapshot),
            "--dataset-revision",
            "b" * 40,
            "--qrels-revision",
            "c" * 40,
        ]

    def test_local_immutable_snapshots_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(self.arguments(Path(tmp)))
            self.assertEqual(args.colbert_snapshot.name, "a" * 40)
            self.assertEqual(args.colbert_batch_size, 32)
            self.assertEqual(args.bge_batch_size, 128)

    def test_mutable_dataset_revision_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.arguments(Path(tmp))
            args[args.index("--dataset-revision") + 1] = "main"
            with self.assertRaises(SystemExit):
                parse_args(args)


if __name__ == "__main__":
    unittest.main()
