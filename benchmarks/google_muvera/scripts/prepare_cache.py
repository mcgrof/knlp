#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Encode pinned SciFact snapshots on CUDA/ROCm and record input provenance."""

import argparse
import hashlib
import re
import json
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colbert-snapshot", type=Path, required=True)
    parser.add_argument("--bge-snapshot", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--qrels-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--colbert-batch-size", type=int, default=32)
    parser.add_argument("--bge-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.colbert_batch_size < 1 or args.bge_batch_size < 1:
        parser.error("batch sizes must be positive")
    for key in ("dataset_revision", "qrels_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, key)):
            parser.error(f"{key} must be a full immutable revision")
    for key in ("colbert_snapshot", "bge_snapshot"):
        path = getattr(args, key).expanduser().resolve()
        if not path.is_dir() or not re.fullmatch(r"[0-9a-f]{40}", path.name):
            parser.error(
                f"{key} must be a local snapshot directory named by its revision"
            )
        setattr(args, key, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(output_dir, name, value):
    (output_dir / name).write_text(json.dumps(value, indent=2) + "\n")


def main(argv=None):
    args = parse_args(argv)
    import torch
    from datasets import load_dataset
    from pylate import models
    from sentence_transformers import SentenceTransformer

    OUT = args.output_dir
    COLBERT = args.colbert_snapshot
    BGE = args.bge_snapshot
    DATA_REV = args.dataset_revision
    QRELS_REV = args.qrels_revision
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA/ROCm encoding is required; CPU fallback is refused")
    assert not OUT.exists(), "fresh input directory required"
    OUT.mkdir(parents=True)
    assert torch.cuda.is_available()
    torch.cuda.set_device(device)
    x = torch.ones((64, 64), device=device, dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize(device)
    assert y.is_cuda and torch.isfinite(y).all() and y[0, 0].item() == 64
    preflight = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(device),
        "free_total_vram_bytes": list(torch.cuda.mem_get_info(device)),
        "matmul_value": y[0, 0].item(),
    }
    save(OUT, "gpu-preflight.json", preflight)
    print(json.dumps(preflight), flush=True)

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus", revision=DATA_REV)
    queries = load_dataset(
        "BeIR/scifact", "queries", split="queries", revision=DATA_REV
    )
    qrels = load_dataset("BeIR/scifact-qrels", split="test", revision=QRELS_REV)
    raw = {
        "corpus": [
            {"id": r["_id"], "title": r["title"], "text": r["text"]} for r in corpus
        ],
        "queries": [{"id": r["_id"], "text": r["text"]} for r in queries],
        "qrels_test": [
            {
                "query-id": str(r["query-id"]),
                "corpus-id": str(r["corpus-id"]),
                "score": int(r["score"]),
            }
            for r in qrels
        ],
    }
    save(OUT, "scifact_raw.json", raw)
    texts = [
        (r["title"] + " " if r["title"] else "") + r["text"] for r in raw["corpus"]
    ]
    cids = [r["id"] for r in raw["corpus"]]
    lookup = {r["id"]: r["text"] for r in raw["queries"]}
    qids = sorted(set(r["query-id"] for r in raw["qrels_test"]))
    assert all(q in lookup for q in qids)
    qtexts = [lookup[q] for q in qids]
    assert len(set(cids)) == len(cids)
    metadata = {
        "dataset": "BeIR/scifact",
        "dataset_revision": DATA_REV,
        "qrels_dataset": "BeIR/scifact-qrels",
        "qrels_revision": QRELS_REV,
        "corpus_count": len(cids),
        "query_count": len(qids),
        "qrels_count": len(qrels),
        "evaluation": "descriptive SciFact test pilot",
        "models": {},
    }

    model = models.ColBERT(model_name_or_path=str(COLBERT), device=device)
    assert next(model.parameters()).is_cuda
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    docs = model.encode(
        texts,
        batch_size=args.colbert_batch_size,
        is_query=False,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    qs = model.encode(
        qtexts,
        batch_size=args.colbert_batch_size,
        is_query=True,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    torch.cuda.synchronize(device)
    assert all(t.is_cuda and torch.isfinite(t).all() for t in docs + qs)
    elapsed = time.perf_counter() - t0
    retained_d = [int(t.shape[0]) for t in docs]
    retained_q = [int(t.shape[0]) for t in qs]
    untruncated = [
        len(v) + int(model.document_prefix_id is not None)
        for v in model.tokenizer(texts, truncation=False, padding=False)["input_ids"]
    ]
    tokenized = []
    for i in range(0, len(texts), args.colbert_batch_size):
        tokenized.extend(
            model.tokenize(texts[i : i + args.colbert_batch_size], is_query=False)[
                "attention_mask"
            ]
            .sum(1)
            .tolist()
        )
    assert all(r <= t for r, t in zip(retained_d, tokenized))
    metadata["models"]["colbert"] = {
        "id": "lightonai/colbertv2.0",
        "revision": COLBERT.name,
        "snapshot": str(COLBERT),
        "device": str(next(model.parameters()).device),
        "batch_size": args.colbert_batch_size,
        "document_length": model.document_length,
        "query_length": model.query_length,
        "query_expansion": model.do_query_expansion,
        "tokenizer_class": type(model.tokenizer).__name__,
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "retained_document_vectors": retained_d,
        "retained_query_vectors": retained_q,
        "source_document_token_counts_including_prefix": untruncated,
        "tokenized_document_nonpadding_counts": tokenized,
        "truncated_documents": sum(n > model.document_length for n in untruncated),
        "document_padding_removed_by_encoder": True,
        "document_skiplist_token_count": len(model.skiplist),
        "nonpadding_tokens_removed_by_skiplist": sum(
            t - r for r, t in zip(retained_d, tokenized)
        ),
    }
    torch.save(
        {
            "doc_embs": [t.cpu() for t in docs],
            "q_embs": [t.cpu() for t in qs],
            "cid_list": cids,
            "cid_to_idx": {c: i for i, c in enumerate(cids)},
            "test_qids": qids,
            "qrels_test": raw["qrels_test"],
            "model": str(COLBERT),
        },
        OUT / "colbertv2_embeddings.pt",
    )
    print("ColBERT encoded", len(docs), len(qs), "seconds", elapsed, flush=True)
    del model, docs, qs
    torch.cuda.empty_cache()
    model = SentenceTransformer(str(BGE), device=device)
    assert next(model.parameters()).is_cuda
    t0 = time.perf_counter()
    d = model.encode(
        texts,
        batch_size=args.bge_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q = model.encode(
        qtexts,
        batch_size=args.bge_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    torch.cuda.synchronize(device)
    assert (
        d.is_cuda and q.is_cuda and torch.isfinite(d).all() and torch.isfinite(q).all()
    )
    metadata["models"]["bge"] = {
        "id": "BAAI/bge-small-en-v1.5",
        "revision": BGE.name,
        "snapshot": str(BGE),
        "device": str(next(model.parameters()).device),
        "batch_size": args.bge_batch_size,
        "max_seq_length": model.max_seq_length,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    torch.save(
        {
            "doc_emb": d.cpu().numpy(),
            "q_emb": q.cpu().numpy(),
            "cid_list": cids,
            "test_qids": qids,
            "qrels_test": raw["qrels_test"],
            "model": str(BGE),
        },
        OUT / "bge_embeddings.pt",
    )
    for info, snapshot in (
        (metadata["models"]["colbert"], COLBERT),
        (metadata["models"]["bge"], BGE),
    ):
        info["snapshot_files"] = {
            str(p.relative_to(snapshot)): {
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        }
    metadata["cache_files"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in OUT.iterdir()
        if p.is_file()
    }
    save(OUT, "embedding-provenance.json", metadata)
    print("Frozen cache", json.dumps(metadata["cache_files"]), flush=True)


if __name__ == "__main__":
    main()
