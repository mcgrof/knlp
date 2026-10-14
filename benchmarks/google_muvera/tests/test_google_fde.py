# SPDX-License-Identifier: MIT
"""Correctness fixtures against the real compiled Google implementation."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_google_fde import verify_sources
from google_fde import FDEConfig, GoogleFDE


class GoogleFDETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A missing native binary is a failure, never a skipped correctness gate.
        cls.encoder = GoogleFDE(binary=os.environ.get("GOOGLE_FDE_BINARY"))

    def test_official_sources_match_pin(self):
        self.assertEqual(
            verify_sources()["commit"], "621ddb4e4c3b3ba3dc7d15ad34f1e0bb39ef25d5"
        )

    def test_zero_hash_query_sum_document_mean(self):
        points = np.array([[1, 2], [3, 4]], dtype=np.float32)
        config = FDEConfig(repetitions=3, k_sim=0)
        query = self.encoder.encode([points], "query", config)
        document = self.encoder.encode([points], "document", config)
        np.testing.assert_array_equal(query, [[4, 6, 4, 6, 4, 6]])
        np.testing.assert_array_equal(document, [[2, 3, 2, 3, 2, 3]])

    def test_document_duplication_invariant_query_duplication_additive(self):
        points = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        duplicated = np.concatenate([points, points])
        config = FDEConfig(repetitions=2, k_sim=2)
        docs = self.encoder.encode([points, duplicated], "document", config)
        queries = self.encoder.encode([points, duplicated], "query", config)
        np.testing.assert_array_equal(docs[0], docs[1])
        np.testing.assert_array_equal(queries[0] * 2, queries[1])

    def test_document_empty_buckets_fill_and_query_stays_sparse(self):
        points = np.array([[1, 0]], dtype=np.float32)
        filled = FDEConfig(repetitions=2, k_sim=3, fill_empty=True)
        sparse = FDEConfig(repetitions=2, k_sim=3, fill_empty=False)
        doc_filled = self.encoder.encode([points], "document", filled).reshape(2, 8, 2)
        doc_sparse = self.encoder.encode([points], "document", sparse).reshape(2, 8, 2)
        np.testing.assert_array_equal(doc_filled, np.tile(points, (2, 8, 1)))
        self.assertEqual(np.count_nonzero(doc_sparse), 2)
        np.testing.assert_array_equal(
            self.encoder.encode([points], "query", filled),
            self.encoder.encode([points], "query", sparse),
        )

    def test_singleton_shared_query_document_partitions(self):
        points = np.array([[0.6, -0.8]], dtype=np.float32)
        for seed in (1, 19):
            config = FDEConfig(repetitions=4, k_sim=3, seed=seed)
            query = self.encoder.encode([points], "query", config)
            doc = self.encoder.encode([points], "document", config)
            self.assertAlmostEqual(float((query @ doc.T)[0, 0] / 4), 1.0, places=6)

    def test_seed_determinism_and_batch_consistency(self):
        rng = np.random.default_rng(28)
        clouds = [rng.normal(size=(7, 8)), rng.normal(size=(3, 8))]
        config = FDEConfig(repetitions=3, k_sim=3, seed=9)
        batch = self.encoder.encode(clouds, "query", config)
        again = self.encoder.encode(clouds, "query", config)
        separate = np.vstack(
            [self.encoder.encode([cloud], "query", config) for cloud in clouds]
        )
        np.testing.assert_array_equal(batch, again)
        np.testing.assert_array_equal(batch, separate)
        other = self.encoder.encode(
            clouds, "query", FDEConfig(repetitions=3, k_sim=3, seed=10)
        )
        self.assertFalse(np.array_equal(batch, other))

    def test_official_inner_and_final_projections(self):
        clouds = [np.eye(8, dtype=np.float32)]
        for projection, final, expected in ((None, None, 64), (3, None, 24), (3, 7, 7)):
            config = FDEConfig(
                repetitions=2, k_sim=2, projection_dim=projection, final_dim=final
            )
            for role in ("query", "document"):
                output = self.encoder.encode(clouds, role, config)
                self.assertEqual(output.shape, (1, expected))
                self.assertTrue(np.isfinite(output).all())
                np.testing.assert_array_equal(
                    output, self.encoder.encode(clouds, role, config)
                )

    def test_unprojected_filled_fde_never_exceeds_chamfer(self):
        # Each filled document bucket is a point or convex average of points.
        # Therefore its dot product cannot exceed the per-query-vector maximum.
        rng = np.random.default_rng(410)
        queries = [rng.normal(size=(5, 8)).astype(np.float32) for _ in range(4)]
        docs = [rng.normal(size=(9, 8)).astype(np.float32) for _ in range(6)]
        for cloud in queries + docs:
            cloud /= np.linalg.norm(cloud, axis=1, keepdims=True)
        config = FDEConfig(repetitions=3, k_sim=3, seed=6, fill_empty=True)
        q_fde = self.encoder.encode(queries, "query", config)
        d_fde = self.encoder.encode(docs, "document", config)
        approximation = q_fde @ d_fde.T / config.repetitions
        oracle = np.array(
            [[(query @ doc.T).max(axis=1).sum() for doc in docs] for query in queries]
        )
        self.assertTrue(np.all(approximation <= oracle + 1e-5))

    def test_reject_invalid_clouds_and_configs(self):
        for clouds in (
            [],
            [np.zeros((0, 3))],
            [np.zeros(3)],
            [np.array([[np.nan]])],
            [np.zeros((1, 2)), np.zeros((1, 3))],
        ):
            with self.subTest(clouds=clouds), self.assertRaises(ValueError):
                self.encoder.encode(clouds, "document")
        for config in (
            {"repetitions": 0},
            {"k_sim": 21},
            {"seed": -1},
            {"projection_dim": -1},
            {"fill_empty": "yes"},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                FDEConfig(**config)
        with self.assertRaises(ValueError):
            self.encoder.encode([np.eye(2)], "unknown")


if __name__ == "__main__":
    unittest.main()
