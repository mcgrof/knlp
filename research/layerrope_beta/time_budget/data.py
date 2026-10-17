# SPDX-License-Identifier: MIT
"""Prepare document-disjoint holdouts, keeping final tokens outside training inputs."""

import argparse
import hashlib
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.data import DATA_REV, TOKEN_REV


def document_hashes(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "text_sha256":
                if not isinstance(item, str) or len(item) != 64:
                    raise ValueError("Invalid document hash")
                yield item
            else:
                yield from document_hashes(item)
    elif isinstance(value, list):
        for item in value:
            yield from document_hashes(item)


def prepare(input_manifest, out):
    inputs = read(input_manifest)
    out = Path(out)
    required = ("training", "tokenizer", "validation_parquet")
    if (
        inputs.get("dataset_revision") != DATA_REV
        or inputs.get("tokenizer_revision") != TOKEN_REV
    ):
        raise ValueError("Pinned dataset/tokenizer revision mismatch")
    if set(inputs.get("predecessor_scopes", [])) != {
        "training-and-original-evaluation",
        "full-gain-beta",
        "depth-code-confirmation",
    }:
        raise ValueError(
            "All predecessor scopes, including confirmation, must be covered"
        )
    receipts = [inputs[k] for k in required] + inputs["documents"]
    for receipt in receipts:
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Input provenance hash mismatch: " + receipt["path"])
    forbidden = set()
    provenance = []
    for receipt in inputs["documents"]:
        hashes = set(document_hashes(read(receipt["path"])))
        if not hashes:
            raise ValueError("Empty predecessor document manifest")
        forbidden.update(hashes)
        provenance.append(dict(**receipt, document_hashes=len(hashes)))
    array = np.load(inputs["training"]["path"], mmap_mode="r")
    if array.dtype != np.uint16 or len(array) != 40000000:
        raise ValueError("Preserved training array geometry mismatch")
    tok = Tokenizer.from_file(inputs["tokenizer"]["path"])
    if tok.get_vocab_size() != 50257:
        raise ValueError("Tokenizer vocabulary mismatch")
    seen = set()
    pools = {"validation": [], "final": []}
    index = 0
    excluded = 0
    for batch in pq.ParquetFile(inputs["validation_parquet"]["path"]).iter_batches(
        batch_size=4096, columns=["text"]
    ):
        for text in batch.column(0).to_pylist():
            digest = hashlib.sha256(text.encode()).hexdigest()
            docid = f"validation:{index}"
            index += 1
            if digest in forbidden or digest in seen:
                excluded += 1
                continue
            seen.add(digest)
            bucket = (
                hashlib.sha256(("equal-time-holdout-v1:" + digest).encode()).digest()[0]
                % 2
            )
            pools[("validation", "final")[bucket]].append((docid, digest, text))
    out.mkdir(parents=True, exist_ok=False)
    training = out / "training"
    sealed = out / "sealed"
    training.mkdir()
    sealed.mkdir(mode=0o700)
    # Copy bytes; never overwrite inherited arrays or grant a mutable alias.
    shutil.copyfile(inputs["training"]["path"], training / "train.npy")
    used = {}
    for name, count in [("validation", 32768), ("final", 131072)]:
        tokens = []
        docs = []
        for docid, digest, text in pools[name]:
            tokens.extend(tok.encode(text).ids + [50256])
            docs.append(dict(id=docid, text_sha256=digest))
            if len(tokens) >= count + 1:
                break
        if len(tokens) < count + 1:
            raise ValueError("Insufficient untouched documents")
        dest = training if name == "validation" else sealed
        np.save(
            dest / (name + ".npy"), np.asarray(tokens[: count + 1], dtype=np.uint16)
        )
        used[name] = docs
    sets = [{d["text_sha256"] for d in used[n]} for n in ("validation", "final")]
    if sets[0] & sets[1] or (sets[0] | sets[1]) & forbidden:
        raise RuntimeError("Holdout overlap")
    write(
        out / "DOCUMENTS.json",
        dict(
            eval_used=used,
            predecessors=provenance,
            excluded_hash_count=len(forbidden),
            excluded_or_duplicate_rows=excluded,
            partition_rule="SHA256(equal-time-holdout-v1:+exact-text-sha)[0] modulo 2",
            scope="Exact-text document exclusion from the declared pinned predecessor manifests; no paraphrase guarantee",
        ),
    )
    write(
        training / "TRAINING.json",
        dict(
            schema="equal-time-training-capability.v1",
            files={n: sha(training / n) for n in ("train.npy", "validation.npy")},
            dataset_revision=DATA_REV,
            tokenizer_revision=TOKEN_REV,
            provenance_input_sha256=sha(input_manifest),
            documents_sha256=sha(out / "DOCUMENTS.json"),
        ),
    )
    write(
        sealed / "SEALED.json",
        dict(
            schema="equal-time-final-capability.v1",
            file="final.npy",
            sha256=sha(sealed / "final.npy"),
            predicted_tokens=131072,
            documents_sha256=sha(out / "DOCUMENTS.json"),
            scope="Protocol seal, not a security boundary between same-UID processes",
        ),
    )
    (sealed / "final.npy").chmod(0o400)
    write(
        out / "PREPARED.json",
        dict(
            inputs=inputs,
            source_manifest_sha256=sha(input_manifest),
            training_manifest_sha256=sha(training / "TRAINING.json"),
            sealed_manifest_sha256=sha(sealed / "SEALED.json"),
            validation_documents=len(used["validation"]),
            final_documents=len(used["final"]),
            observed_overlaps=0,
            model_evaluations=0,
        ),
    )
    return read(out / "PREPARED.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prepare(a.inputs, a.out)
