"""Reserve untouched confirmation documents from the pinned validation source."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from research.tinystories_depth.data import sha
from research.tinystories_depth.model import generator


def prepare(root):
    root = Path(root)
    data, inputs = root / "data", root / "inputs"
    out = data / "confirmation"
    if out.exists():
        raise FileExistsError("Never overwrite a previously reserved partition")
    started = time.perf_counter()
    expected = {
        data
        / "DOCUMENTS.json": json.loads((data / "DATA.json").read_text())["files"][
            "DOCUMENTS.json"
        ],
        inputs
        / "validation.parquet": "33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab",
        inputs
        / "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    }
    for path, digest in expected.items():
        assert sha(path) == digest, path
    old = json.loads((data / "DOCUMENTS.json").read_text())
    # The verified inherited scan examined all official training texts, not
    # just the 40M-token training prefix. Its validation exclusion receipt
    # covers every exact-text training overlap and repeated validation text.
    excluded = {d["text_sha256"] for d in old["excluded_eval"]}
    prior_eval = {d["text_sha256"] for docs in old["eval_used"].values() for d in docs}
    train_used = {d["text_sha256"] for d in old["train_used"]}
    forbidden = excluded | prior_eval | train_used
    tok = Tokenizer.from_file(str(inputs / "tokenizer.json"))
    assert tok.get_vocab_size() == 50257
    pools = {"curve": [], "final": []}
    seen, offset = set(), 0
    for batch in pq.ParquetFile(inputs / "validation.parquet").iter_batches(
        batch_size=4096, columns=["text"]
    ):
        for text in batch.column(0).to_pylist():
            digest = hashlib.sha256(text.encode()).hexdigest()
            docid = f"validation-00000-of-00001-869c898b519ad725.parquet:{offset}"
            offset += 1
            if digest in forbidden or digest in seen:
                continue
            seen.add(digest)
            # A new frozen namespace, independent of model and data-order RNG.
            bucket = (
                hashlib.sha256(
                    ("mo-confirmation-20261011:" + digest).encode()
                ).digest()[0]
                % 2
            )
            pools[("curve", "final")[bucket]].append((docid, digest, text))
    reserved, arrays = {}, {}
    for name, count in (("curve", 32768), ("final", 131072)):
        tokens, used = [], []
        for docid, digest, text in pools[name]:
            tokens.extend(tok.encode(text).ids + [50256])
            used.append({"id": docid, "text_sha256": digest})
            if len(tokens) >= count + 1:
                break
        assert len(tokens) >= count + 1, "Insufficient untouched validation documents"
        reserved[name] = used
        arrays[name] = np.asarray(tokens[: count + 1], dtype=np.uint16)
    a, b = ({d["text_sha256"] for d in reserved[n]} for n in ("curve", "final"))
    assert not a & b and not (a | b) & forbidden
    out.mkdir(mode=0o700)
    for name, tokens in arrays.items():
        np.save(out / f"{name}.npy", tokens)
    (out / "DOCUMENTS.json").write_text(
        json.dumps(
            {
                "eval_used": reserved,
                "prior_eval_hash_count": len(prior_eval),
                "inherited_exclusion_hash_count": len(excluded),
                "partition_rule": "SHA256('mo-confirmation-20261011:'+text_sha256)[0] mod 2; 0 curve, 1 final",
                "exclusion_provenance": "verified parent full-training-scan validation exclusion receipt plus all parent eval_used documents",
                "overlap_counts": {
                    "between_partitions": 0,
                    "prior_evaluation": 0,
                    "training_used": 0,
                    "inherited_training_overlap_and_eval_duplicate": 0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    orders = []
    n = (40_000_000 - 1) // 256
    for seed in (
        76390,
        76301,
        76311,
        76312,
        76313,
        *range(76411, 76419),
        76501,
        76511,
        76512,
        76513,
    ):
        path = data / f"order-{seed}.npy"
        assert not path.exists(), path
        # Same named seed order is reused for every row and LR. A dedicated
        # namespace keeps it independent of initialization and code RNG.
        order_seed = generator(seed, "mo-training-order").initial_seed()
        np.save(path, np.random.default_rng(order_seed).permutation(n).astype(np.int32))
        orders.append(
            {
                "seed": seed,
                "rng_seed": order_seed,
                "sha256": sha(path),
                "path": str(path),
            }
        )
    record = {
        "dataset_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "tokenizer_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "verified_source_hashes": {str(p): d for p, d in expected.items()},
        "confirmation_files": {p.name: sha(p) for p in out.iterdir()},
        "confirmation_predicted_tokens": {"curve": 32768, "final": 131072},
        "reserved_document_counts": {k: len(v) for k, v in reserved.items()},
        "eligible_document_pool_counts": {k: len(v) for k, v in pools.items()},
        "scope_of_prior_evaluation_exclusion": "all discovery evaluation documents and the official full-training-text exclusion scan",
        "data_orders": orders,
        "elapsed_cpu_seconds": time.perf_counter() - started,
        "model_evaluation": "none; no confirmation loss inspected",
    }
    (root / "records/CONFIRMATION-DATA.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    prepare(p.parse_args().root)
