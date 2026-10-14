"""Pinned local Parquet data, exact duplicate removal, and packed targets."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

DATA_REV = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKEN_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SEEDS = (75201, 75211, 75212, 75213, 75290)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(paths):
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=["text"]
        ):
            for text in batch.column(0).to_pylist():
                yield f"{path.name}:{offset}", text
                offset += 1


def prepare(root, hub=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    hub = Path(hub) if hub is not None else Path.home() / ".cache/huggingface/hub"
    source = hub / "datasets--roneneldan--TinyStories/snapshots" / DATA_REV / "data"
    tokenizer_dir = hub / "models--gpt2/snapshots" / TOKEN_REV
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == 50257
    train_paths = sorted(source.glob("train-*.parquet"))
    val_paths = sorted(source.glob("validation-*.parquet"))
    assert len(train_paths) == 4 and len(val_paths) == 1
    train_hashes = set()
    train_docs = []
    train = []
    pending = []
    for docid, text in rows(train_paths):
        digest = hashlib.sha256(text.encode()).hexdigest()
        train_hashes.add(digest)
        if len(train) < 40_000_000:
            pending.append((docid, digest, text))
            if len(pending) == 256:
                for (did, h, _), encoded in zip(
                    pending, tokenizer.encode_batch([x[2] for x in pending])
                ):
                    if len(train) >= 40_000_000:
                        break
                    train.extend(encoded.ids + [50256])
                    train_docs.append({"id": did, "text_sha256": h})
                pending.clear()
    if len(train) < 40_000_000:
        for (did, h, _), encoded in zip(
            pending, tokenizer.encode_batch([x[2] for x in pending])
        ):
            train.extend(encoded.ids + [50256])
            train_docs.append({"id": did, "text_sha256": h})
    assert len(train) >= 40_000_000
    np.save(root / "train.npy", np.asarray(train[:40_000_000], dtype=np.uint16))
    del train
    eval_seen = set()
    excluded = []
    partitions = {"curve": [], "final": []}
    for docid, text in rows(val_paths):
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in train_hashes or digest in eval_seen:
            excluded.append(
                {
                    "id": docid,
                    "text_sha256": digest,
                    "reason": (
                        "train_overlap" if digest in train_hashes else "eval_duplicate"
                    ),
                }
            )
            continue
        eval_seen.add(digest)
        group = "curve" if int(digest[:2], 16) % 2 == 0 else "final"
        partitions[group].append((docid, digest, text))
    eval_ids = {}
    for group, count in (("curve", 32768), ("final", 131072)):
        tokens, used = [], []
        for docid, digest, text in partitions[group]:
            tokens.extend(tokenizer.encode(text).ids + [50256])
            used.append({"id": docid, "text_sha256": digest})
            if len(tokens) >= count + 1:
                break
        assert len(tokens) >= count + 1
        np.save(root / f"{group}.npy", np.asarray(tokens[: count + 1], dtype=np.uint16))
        eval_ids[group] = used
    n = (40_000_000 - 1) // 256
    for seed in SEEDS:
        np.save(
            root / f"order-{seed}.npy",
            np.random.default_rng(seed).permutation(n).astype(np.int32),
        )
    (root / "DOCUMENTS.json").write_text(
        json.dumps(
            {
                "train_used": train_docs,
                "eval_used": eval_ids,
                "excluded_eval": excluded,
                "partition_rule": "first SHA256 byte modulo 2, zero=curve; first exact-text duplicate retained",
            }
        )
        + "\n"
    )
    manifest = {
        "dataset": "roneneldan/TinyStories",
        "dataset_revision": DATA_REV,
        "tokenizer": "gpt2",
        "tokenizer_revision": TOKEN_REV,
        "vocabulary": 50257,
        "eos": 50256,
        "training_tokens": 40_000_000,
        "curve_predicted_tokens": 32768,
        "final_predicted_tokens": 131072,
        "source_files": {p.name: sha(p) for p in train_paths + val_paths},
        "tokenizer_files": {
            p.name: sha(p)
            for p in tokenizer_dir.iterdir()
            if p.name
            in (
                "tokenizer.json",
                "merges.txt",
                "vocab.json",
                "tokenizer_config.json",
                "config.json",
            )
        },
        "train_unique_texts_scanned": len(train_hashes),
        "excluded_eval_count": len(excluded),
        "files": {p.name: sha(p) for p in root.iterdir()},
        "elapsed_cpu_seconds": time.perf_counter() - started,
        "packing": "stride 256; 257 source tokens; inputs[:-1], targets[1:]; no internal loss",
    }
    (root / "DATA.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1])
