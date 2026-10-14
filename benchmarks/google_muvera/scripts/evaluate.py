#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluate official Google FDEs against the same ColBERT token vectors.

The NPZ input uses numeric ragged arrays (values and offsets) and Unicode
IDs. It never requires pickle. Existing locally produced PyTorch caches
remain supported; loading those caches requires trusting their producer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(values, name):
    result = [str(value) for value in values]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _ragged(source, prefix):
    values = np.asarray(source[f"{prefix}_values"], dtype=np.float32)
    offsets = source[f"{prefix}_offsets"]
    if (
        values.ndim != 2
        or offsets.ndim != 1
        or offsets.dtype.kind not in "iu"
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(values)
        or np.any(np.diff(offsets.astype(np.int64)) <= 0)
    ):
        raise ValueError(f"invalid {prefix} ragged arrays")
    return [values[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:])]


def _clean_clouds(clouds, name):
    result, dropped = [], 0
    for index, cloud in enumerate(clouds):
        cloud = _array(cloud)
        if cloud.ndim != 2 or cloud.shape[1] == 0 or not np.isfinite(cloud).all():
            raise ValueError(f"invalid {name} token vectors at row {index}")
        norms = np.linalg.norm(cloud, axis=1)
        keep = norms > 0
        dropped += int((~keep).sum())
        cloud, norms = cloud[keep], norms[keep]
        if not len(cloud):
            raise ValueError(f"{name} row {index} has no nonzero token vectors")
        if not np.allclose(norms, 1.0, atol=2e-3, rtol=2e-3):
            raise ValueError(
                f"{name} row {index} is not token-normalized; "
                "the evaluator does not silently normalize input"
            )
        result.append(np.ascontiguousarray(cloud))
    return result, dropped


def _relevance(rows, doc_ids, query_ids):
    doc_index = {value: index for index, value in enumerate(doc_ids)}
    relevant = {value: set() for value in query_ids}
    for row in rows:
        query_id, doc_id = str(row["query-id"]), str(row["corpus-id"])
        score = float(row.get("score", 1))
        if not np.isfinite(score):
            raise ValueError("relevance scores must be finite")
        if query_id not in relevant or score <= 0:
            continue
        if doc_id not in doc_index:
            raise ValueError(f"positive qrel document {doc_id!r} is absent from corpus")
        relevant[query_id].add(doc_index[doc_id])
    if any(not value for value in relevant.values()):
        raise ValueError("every evaluated query needs a positive qrel in this corpus")
    return [relevant[value] for value in query_ids]


def load_embeddings(cache_dir):
    """Load and validate either embeddings.npz or the legacy ColBERT cache."""
    cache_dir = Path(cache_dir)
    path = cache_dir / "embeddings.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as src:
            docs, queries = _ragged(src, "doc"), _ragged(src, "query")
            doc_ids = _ids(src["doc_ids"], "document IDs")
            query_ids = _ids(src["query_ids"], "query IDs")
            qids, cids, scores = (
                src["qrel_query_ids"],
                src["qrel_doc_ids"],
                src["qrel_scores"],
            )
            if not (len(qids) == len(cids) == len(scores)):
                raise ValueError("qrel arrays have different lengths")
            rows = [
                {"query-id": q, "corpus-id": d, "score": s}
                for q, d, s in zip(qids, cids, scores)
            ]
            model = str(src["model"].item()) if "model" in src else "unspecified"
    else:
        path = cache_dir / "colbertv2_embeddings.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing embedding cache in {cache_dir}")
        import torch

        src = torch.load(path, map_location="cpu", weights_only=False)
        docs, queries = src["doc_embs"], src["q_embs"]
        doc_ids = _ids(src["cid_list"], "document IDs")
        query_ids = _ids(src["test_qids"], "query IDs")
        rows, model = src["qrels_test"], src.get("model", "unspecified")
    if len(docs) != len(doc_ids) or len(queries) != len(query_ids):
        raise ValueError("embedding row counts do not match IDs")
    docs, doc_dropped = _clean_clouds(docs, "document")
    queries, query_dropped = _clean_clouds(queries, "query")
    dimensions = {cloud.shape[1] for cloud in docs + queries}
    if len(dimensions) != 1:
        raise ValueError("query and document token dimensions must match")
    provenance = {
        "cache_file": str(path.resolve()),
        "cache_sha256": _sha256(path),
        "model": model,
        "document_count": len(docs),
        "query_count": len(queries),
        "token_dimension": dimensions.pop(),
        "document_zero_padding_dropped": doc_dropped,
        "query_zero_padding_dropped": query_dropped,
        "doc_ids_sha256": hashlib.sha256(json.dumps(doc_ids).encode()).hexdigest(),
        "query_ids_sha256": hashlib.sha256(json.dumps(query_ids).encode()).hexdigest(),
        "input_normalization": "validate unit token vectors; drop zero padding only",
    }
    return {
        "docs": docs,
        "queries": queries,
        "doc_ids": doc_ids,
        "query_ids": query_ids,
        "relevant": _relevance(rows, doc_ids, query_ids),
        "provenance": provenance,
    }


def validate_alignment(src, data):
    """Reject comparing BGE rows against a different corpus/query order."""
    if [str(value) for value in src["cid_list"]] != data["doc_ids"]:
        raise ValueError("BGE document IDs/order differ from the ColBERT cache")
    if [str(value) for value in src["test_qids"]] != data["query_ids"]:
        raise ValueError("BGE query IDs/order differ from the ColBERT cache")
    relevant = _relevance(src["qrels_test"], data["doc_ids"], data["query_ids"])
    if relevant != data["relevant"]:
        raise ValueError("BGE relevance judgments differ from the ColBERT cache")


def ranked_indices(scores, limit):
    """Descending score, then corpus row index; deterministic even at ties."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("ranking requires finite one-dimensional scores")
    return np.lexsort((np.arange(len(scores)), -scores))[:limit]


def relevance_metrics(rankings, relevant, cutoffs=(10, 100)):
    """Macro recall counts every relevant document; hit counts any match."""
    if len(rankings) != len(relevant) or not len(relevant):
        raise ValueError("rankings and relevance need equal nonzero query counts")
    result = {}
    for cutoff in cutoffs:
        recalls, hits = [], []
        for ranking, truth in zip(rankings, relevant):
            if not truth:
                raise ValueError("recall is undefined for an unjudged query")
            found = len(set(map(int, ranking[:cutoff])) & truth)
            recalls.append(found / len(truth))
            hits.append(float(found > 0))
        result[f"recall_at_{cutoff}"] = float(np.mean(recalls))
        result[f"hit_at_{cutoff}"] = float(np.mean(hits))
    return result


def chamfer_scores(query, docs, token_block=256):
    """ColBERT sum-of-token-max scores with a bounded token similarity tile."""
    output = np.empty(len(docs), dtype=np.float32)
    for index, doc in enumerate(docs):
        maxima = np.full(len(query), -np.inf, dtype=np.float32)
        for start in range(0, len(doc), token_block):
            tile = query @ doc[start : start + token_block].T
            maxima = np.maximum(maxima, tile.max(axis=1))
        output[index] = maxima.sum()
    return output


def rerank_candidates(candidates, exact_scores):
    candidates = np.asarray(candidates, dtype=np.int64)
    # Tie-breaking must use global corpus IDs, not candidate positions.
    return candidates[np.lexsort((candidates, -exact_scores[candidates]))]


def write_representation(matrix, fmt, prefix):
    """Write raw packed document payload and return its FP32 reconstruction."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("representation must be a finite matrix")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scale = None
    if fmt == "fp32":
        packed = matrix.astype("<f4", copy=False)
        reconstructed = packed
    elif fmt == "fp16":
        packed = matrix.astype("<f2")
        if not np.isfinite(packed).all():
            raise ValueError("FP16 packing overflowed; change configuration")
        reconstructed = packed.astype(np.float32)
    elif fmt == "int8":
        scale = np.max(np.abs(matrix), axis=1) / 127.0
        scale = np.where(scale == 0, 1.0, scale).astype("<f4")
        packed = np.clip(np.rint(matrix / scale[:, None]), -127, 127).astype("i1")
        reconstructed = packed.astype(np.float32) * scale[:, None]
    else:
        raise ValueError(f"unsupported format {fmt!r}; use fp32, fp16, or int8")
    payload = prefix.with_suffix(".bin")
    packed.tofile(payload)
    scale_path = prefix.with_suffix(".scales.bin")
    if scale is not None:
        scale.tofile(scale_path)
    metadata = {
        "shape": list(matrix.shape),
        "format": fmt,
        "dtype": packed.dtype.str,
        "order": "C",
        "payload_file": payload.name,
        "scale_file": scale_path.name if scale is not None else None,
        "scale_dtype": "<f4" if scale is not None else None,
        "scale_rule": (
            "one max-absolute scale per document" if scale is not None else None
        ),
        "query_format": "fp32",
        "search": "exhaustive FP32 dot products after reconstruction",
        "interpretation": "quantization quality simulation, not packed-format latency",
    }
    metadata_path = prefix.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    stats = {
        "dimension": matrix.shape[1],
        "payload_bytes": payload.stat().st_size,
        "scale_bytes": scale_path.stat().st_size if scale is not None else 0,
        "metadata_bytes": metadata_path.stat().st_size,
        "codebook_bytes": 0,
    }
    stats["record_bytes"] = (stats["payload_bytes"] + stats["scale_bytes"]) / len(
        matrix
    )
    stats["serialized_bytes"] = sum(
        stats[key] for key in ("payload_bytes", "scale_bytes", "metadata_bytes")
    )
    return reconstructed, stats


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _get(cfg, key, default):
    value = cfg.get(f"CONFIG_GOOGLE_MUVERA_{key}", default)
    return value.strip('"') if isinstance(value, str) else value


def _integers(cfg, key, default):
    values = [int(value) for value in str(_get(cfg, key, default)).split()]
    if not values or min(values) < 1:
        raise ValueError(f"{key} requires positive integers")
    return list(dict.fromkeys(values))


def _fidelity(candidates, exact, cutoff=10):
    return len(set(map(int, candidates)) & set(map(int, exact[:cutoff]))) / min(
        cutoff, len(exact)
    )


def _bucket_rows(base, candidate_rankings, oracle, lengths, bins):
    # Search uses the full corpus. Only aggregation uses the length of each
    # query's exact top-1 document, so this is not a length-restricted task.
    groups = np.searchsorted(bins, lengths[oracle[:, 0]], side="left")
    rows = []
    for index in range(len(bins) + 1):
        selected = np.flatnonzero(groups == index)
        if not len(selected):
            continue
        values = [_fidelity(candidate_rankings[i], oracle[i], 10) for i in selected]
        rows.append(
            {
                **base,
                "length_min": 1 if index == 0 else bins[index - 1] + 1,
                "length_max": bins[index] if index < len(bins) else "unbounded",
                "query_count": len(selected),
                "exact_top10_candidate_recall": float(np.mean(values)),
                "group_rule": "exact top1 document nonzero token count",
            }
        )
    return rows


def evaluate(cache_dir: Path, run_dir: Path, cfg: dict, binary: Path) -> dict:
    """Run the official FDE sweep; all quality scores use exhaustive CPU search."""
    if __package__:
        from .google_fde import FDEConfig, GoogleFDE
    else:
        from google_fde import FDEConfig, GoogleFDE

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    formats = str(_get(cfg, "FORMATS", "fp32 fp16 int8")).split()
    if not formats or set(formats) - {"fp32", "fp16", "int8"}:
        raise ValueError("FORMATS supports fp32 fp16 int8; PQ is not implemented")
    candidates = _integers(cfg, "CANDIDATES", "10 100")
    seeds = _integers(cfg, "SEEDS", "1")
    bins = _integers(cfg, "LENGTH_BINS", "32 64 128 256 512 1024")
    if bins != sorted(bins):
        raise ValueError("LENGTH_BINS must be increasing")
    triples = []
    for item in str(_get(cfg, "FDE_CONFIGS", "5:4:16 20:4:8 20:5:16")).split():
        parts = [int(value) for value in item.split(":")]
        if len(parts) != 3 or min(parts) < 1:
            raise ValueError("FDE_CONFIGS requires positive R:k:projection triples")
        triples.append(parts)
    if not triples:
        raise ValueError("FDE_CONFIGS cannot be empty")
    data = load_embeddings(cache_dir)
    docs, queries = data["docs"], data["queries"]
    ndoc, nquery = len(docs), len(queries)
    lengths = np.array([len(doc) for doc in docs], dtype=np.int64)
    retained_bytes = sum(doc.nbytes for doc in docs)
    provenance = data["provenance"]
    provenance.update(
        {
            "config": cfg,
            "fde_binary": str(Path(binary).resolve()),
            "fde_binary_sha256": _sha256(binary),
            "original_multivector_fp32_bytes": retained_bytes,
            "original_multivector_lengths": lengths.tolist(),
            "original_multivectors_required_for_reranking": True,
            "retrieval_scope": "entire loaded corpus",
            "metric": "macro relevant-document recall and query hit rate separately",
            "oracle": "ColBERT sum of maxima on validated unit token vectors",
            "fidelity": "overlap with exact ColBERT top10, independent of qrels",
            "length_buckets": bins,
            "length_bucket_rule": "query grouped by exact top1 document token count",
            "nvme_measured": False,
            "physical_memory_traffic_measured": False,
            "index_overhead_measured": False,
        }
    )
    # The score table is evaluation scratch, not a serving index. A memmap
    # bounds RAM while making repeated reranking inexpensive and exact.
    score_table = np.lib.format.open_memmap(
        run_dir / "exact_chamfer_scores.npy",
        mode="w+",
        dtype=np.float32,
        shape=(nquery, ndoc),
    )
    top_limit = min(ndoc, max(100, max(candidates)))
    oracle = np.empty((nquery, top_limit), dtype=np.int64)
    started = time.perf_counter()
    for index, query in enumerate(queries):
        score_table[index] = chamfer_scores(query, docs)
        oracle[index] = ranked_indices(score_table[index], top_limit)
    score_table.flush()
    rows = [
        {
            "method": "exact-colbert",
            "format": "fp32",
            "stage": "full-corpus",
            "document_count": ndoc,
            "query_count": nquery,
            "payload_bytes": retained_bytes,
            "record_bytes": retained_bytes / ndoc,
            "score_seconds": time.perf_counter() - started,
            "search_execution": "exhaustive CPU FP32 Chamfer",
            **relevance_metrics(oracle, data["relevant"]),
        }
    ]
    bge_path = Path(cache_dir) / "bge_embeddings.pt"
    if bge_path.exists():
        import torch

        bge = torch.load(bge_path, map_location="cpu", weights_only=False)
        validate_alignment(bge, data)
        bdocs, bqueries = _array(bge["doc_emb"]), _array(bge["q_emb"])
        if (
            bdocs.ndim != 2
            or bqueries.ndim != 2
            or len(bdocs) != ndoc
            or len(bqueries) != nquery
            or bdocs.shape[1] != bqueries.shape[1]
            or not np.isfinite(bdocs).all()
            or not np.isfinite(bqueries).all()
        ):
            raise ValueError("BGE shapes/values do not match the shared evaluation")
        branks = [ranked_indices(bdocs @ query, top_limit) for query in bqueries]
        rows.append(
            {
                "method": "bge-pooled",
                "model": bge.get("model", "unspecified"),
                "format": "fp32",
                "stage": "full-corpus",
                "dimension": bdocs.shape[1],
                "payload_bytes": bdocs.nbytes,
                "record_bytes": bdocs.nbytes / ndoc,
                "query_count": nquery,
                "document_count": ndoc,
                "search_execution": "exhaustive CPU FP32 dot products",
                **relevance_metrics(branks, data["relevant"]),
            }
        )
        provenance["bge_cache_sha256"] = _sha256(bge_path)
        provenance["bge_comparison"] = "different encoder; not compression of ColBERT"
    else:
        provenance["bge_comparison"] = "not run; optional cache absent"

    adapter = GoogleFDE(binary=binary)
    provenance["upstream_build"] = adapter.provenance
    bucket_rows = []
    for repetitions, k_sim, projection_dim in triples:
        for seed in seeds:
            config = FDEConfig(
                repetitions=repetitions,
                k_sim=k_sim,
                projection_dim=projection_dim,
                seed=seed,
                fill_empty=_get(cfg, "FILL_EMPTY", "y") in ("y", True, 1),
                final_dim=int(_get(cfg, "FINAL_DIM", 0)) or None,
            )
            name = f"r{repetitions}-k{k_sim}-p{projection_dim}-s{seed}"
            started = time.perf_counter()
            dfde = adapter.encode(docs, role="document", config=config)
            qfde = adapter.encode(queries, role="query", config=config)
            encode_seconds = time.perf_counter() - started
            if dfde.shape[0] != ndoc or qfde.shape != (nquery, dfde.shape[1]):
                raise ValueError("official FDE output has unexpected dimensions")
            qfde.astype("<f4").tofile(run_dir / f"{name}-queries.f32.bin")
            for fmt in formats:
                reconstructed, storage = write_representation(
                    dfde, fmt, run_dir / "representations" / f"{name}-{fmt}"
                )
                started = time.perf_counter()
                rankings = [
                    ranked_indices(reconstructed @ query, top_limit) for query in qfde
                ]
                base = {
                    "method": "google-muvera",
                    "repetitions": repetitions,
                    "k_sim": k_sim,
                    "projection_dim": projection_dim,
                    "final_dim": config.final_dim or 0,
                    "seed": seed,
                    "format": fmt,
                    "document_count": ndoc,
                    "query_count": nquery,
                    "encode_seconds": encode_seconds,
                    "original_multivector_fp32_bytes": retained_bytes,
                    "search_execution": "exhaustive CPU FP32 after reconstruction",
                    "quantization_quality_only": fmt != "fp32",
                    **storage,
                }
                rows.append(
                    {
                        **base,
                        "stage": "fde-ranking",
                        "score_seconds": time.perf_counter() - started,
                        **relevance_metrics(rankings, data["relevant"]),
                    }
                )
                for budget in candidates:
                    selected = [ranking[:budget] for ranking in rankings]
                    candidate_quality = relevance_metrics(
                        selected, data["relevant"], cutoffs=(budget,)
                    )
                    reranked = [
                        rerank_candidates(candidate, score_table[index])
                        for index, candidate in enumerate(selected)
                    ]
                    fidelity = float(
                        np.mean([_fidelity(c, o) for c, o in zip(selected, oracle)])
                    )
                    rows.append(
                        {
                            **base,
                            "stage": "exact-rerank",
                            "candidate_budget": budget,
                            "effective_candidate_budget": min(budget, ndoc),
                            "candidate_relevant_recall": candidate_quality[
                                f"recall_at_{budget}"
                            ],
                            "candidate_hit": candidate_quality[f"hit_at_{budget}"],
                            "exact_top10_candidate_recall": fidelity,
                            "exact_top100_candidate_recall": float(
                                np.mean(
                                    [
                                        _fidelity(c, o, 100)
                                        for c, o in zip(selected, oracle)
                                    ]
                                )
                            ),
                            "rerank_execution": "cached exact oracle scores; no rerank latency claim",
                            **relevance_metrics(reranked, data["relevant"]),
                        }
                    )
                    bucket_rows.extend(
                        _bucket_rows(
                            {
                                "config": name,
                                "format": fmt,
                                "dimension": storage["dimension"],
                                "record_bytes": storage["record_bytes"],
                                "candidate_budget": budget,
                                "effective_candidate_budget": min(budget, ndoc),
                            },
                            selected,
                            oracle,
                            lengths,
                            bins,
                        )
                    )
                del reconstructed
            del dfde, qfde
    report = {
        "schema_version": 1,
        "provenance": provenance,
        "metrics": rows,
        "length_bucket_fidelity": bucket_rows,
        "limitations": [
            "No DiskANN, packed-format search timing, NVMe I/O, or physical RAM traffic measurements.",
            "Length groups describe this corpus; they do not establish required dimension by document size.",
            "Original token vectors remain necessary for exact reranking.",
            "INT8 scales and JSON headers are counted; index structures are not built or counted.",
            "PQ is not implemented. The paper's PQ compression ratio is not claimed.",
        ],
    }
    _write_csv(run_dir / "metrics.csv", rows)
    _write_csv(run_dir / "length-bucket_fidelity.csv", bucket_rows)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return report
