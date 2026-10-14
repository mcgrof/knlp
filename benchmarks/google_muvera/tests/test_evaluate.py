# SPDX-License-Identifier: MIT
"""Small deterministic checks for metric semantics and cache/storage integrity."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("google_muvera_evaluate", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def write_fixture(path):
    np.savez(
        path,
        doc_values=np.array([[1, 0], [0, 0], [0, 1], [-1, 0]], dtype=np.float32),
        doc_offsets=np.array([0, 2, 3, 4], dtype=np.int64),
        query_values=np.array([[1, 0], [0, 1]], dtype=np.float32),
        query_offsets=np.array([0, 1, 2], dtype=np.int64),
        doc_ids=np.array(["a", "b", "c"]),
        query_ids=np.array(["q1", "q2"]),
        qrel_query_ids=np.array(["q1", "q1", "q2"]),
        qrel_doc_ids=np.array(["a", "b", "b"]),
        qrel_scores=np.array([1, 1, 1]),
        model=np.array("synthetic-unit-vectors"),
    )


class EvaluationTest(unittest.TestCase):
    def test_recall_and_hit_differ_for_multiple_relevant_documents(self):
        metrics = evaluation.relevance_metrics(
            [[0, 2, 3], [1, 0, 2]], [{0, 1}, {1}], cutoffs=(1, 3)
        )
        self.assertEqual(metrics["hit_at_1"], 1.0)
        self.assertEqual(metrics["recall_at_1"], 0.75)
        self.assertEqual(metrics["recall_at_3"], 0.75)

    def test_npz_cache_keeps_ids_and_drops_only_zero_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp) / "embeddings.npz")
            data = evaluation.load_embeddings(tmp)
        self.assertEqual(data["doc_ids"], ["a", "b", "c"])
        self.assertEqual(data["relevant"], [{0, 1}, {1}])
        self.assertEqual([len(doc) for doc in data["docs"]], [1, 1, 1])
        self.assertEqual(data["provenance"]["document_zero_padding_dropped"], 1)

    def test_invalid_norm_is_not_silently_changed(self):
        with self.assertRaisesRegex(ValueError, "not token-normalized"):
            evaluation._clean_clouds([np.array([[2.0, 0.0]])], "document")

    def test_missing_relevant_document_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "absent from corpus"):
            evaluation._relevance(
                [{"query-id": "q", "corpus-id": "missing", "score": 1}],
                ["a"],
                ["q"],
            )

    def test_bge_order_and_judgments_must_match(self):
        data = {"doc_ids": ["a", "b"], "query_ids": ["q"], "relevant": [{0}]}
        src = {
            "cid_list": ["b", "a"],
            "test_qids": ["q"],
            "qrels_test": [{"query-id": "q", "corpus-id": "a", "score": 1}],
        }
        with self.assertRaisesRegex(ValueError, "document IDs/order"):
            evaluation.validate_alignment(src, data)
        src["cid_list"] = ["a", "b"]
        evaluation.validate_alignment(src, data)
        src["qrels_test"][0]["corpus-id"] = "b"
        with self.assertRaisesRegex(ValueError, "judgments differ"):
            evaluation.validate_alignment(src, data)

    def test_chamfer_oracle_and_rerank_known_geometry(self):
        query = np.eye(2, dtype=np.float32)
        docs = [query.copy(), query[:1], -query]
        scores = evaluation.chamfer_scores(query, docs, token_block=1)
        np.testing.assert_array_equal(scores, [2, 1, 0])
        # Approximate search puts the weaker match first. Exact reranking
        # restores the oracle's order but cannot recover excluded documents.
        np.testing.assert_array_equal(
            evaluation.rerank_candidates([1, 0], scores), [0, 1]
        )
        np.testing.assert_array_equal(
            evaluation.rerank_candidates([2, 1], scores), [1, 2]
        )

    def test_ties_use_corpus_order_after_reranking(self):
        scores = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(evaluation.ranked_indices(scores, 2), [0, 1])
        np.testing.assert_array_equal(
            evaluation.rerank_candidates([2, 0, 1], scores), [0, 1, 2]
        )

    def test_quantization_files_count_payload_scales_and_metadata(self):
        matrix = np.array([[1.0, -0.5, 0.25], [0, 0, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for fmt, payload_bytes, scale_bytes in [
                ("fp32", 24, 0),
                ("fp16", 12, 0),
                ("int8", 6, 8),
            ]:
                prefix = Path(tmp) / fmt
                restored, stats = evaluation.write_representation(matrix, fmt, prefix)
                self.assertEqual(stats["payload_bytes"], payload_bytes)
                self.assertEqual(stats["scale_bytes"], scale_bytes)
                self.assertEqual(
                    stats["record_bytes"], (payload_bytes + scale_bytes) / 2
                )
                self.assertEqual(
                    stats["serialized_bytes"],
                    payload_bytes
                    + scale_bytes
                    + prefix.with_suffix(".json").stat().st_size,
                )
                header = json.loads(prefix.with_suffix(".json").read_text())
                self.assertEqual(header["shape"], [2, 3])
                np.testing.assert_allclose(restored, matrix, atol=1 / 127)
                self.assertEqual(restored.dtype, np.float32)

    def test_length_groups_do_not_change_candidate_search(self):
        exact = np.array([[0, 1, 2], [2, 0, 1]])
        selected = [np.array([1, 2]), np.array([0, 1])]
        rows = evaluation._bucket_rows(
            {}, selected, exact, np.array([2, 5, 20]), [4, 8]
        )
        self.assertEqual([row["query_count"] for row in rows], [1, 1])
        self.assertEqual([row["length_min"] for row in rows], [1, 9])
        self.assertTrue(
            all(row["exact_top10_candidate_recall"] == 2 / 3 for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
