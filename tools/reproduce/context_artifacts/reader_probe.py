# SPDX-License-Identifier: MIT
"""CPU readout parity for a released Memorilla reader and cached PubMedQA vectors.

The author source stays unchanged. This probe checks the reader stage and exact
file bytes. It does not execute a decoder or qualify inference cache identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

from .artifacts import (
    ConsumerIdentity,
    ContextArtifactManifest,
    ContextArtifactRegistry,
    DocumentMetadata,
    InputMetadata,
    canonical,
    digest_file,
    inspect_payload,
)
from .run import persistent


def tensor_digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(__import__("torch").uint8).numpy().tobytes()
    ).hexdigest()


def worker(args):
    import pyarrow.parquet as pq
    import torch
    from safetensors.torch import load, save_file

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(args.source.resolve()))
    from memorilla import MemoryModule
    from memorilla.embeddings import EmbeddingStore
    from memorilla.data import build_messages, apply_chat_template
    from memorilla.utils import prepare_tokenizer
    from memorilla.benchmarks import BENCHMARKS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    memory_token_id = prepare_tokenizer(tokenizer)
    protocol = BENCHMARKS["pubmedqa"]

    reader = MemoryModule.from_pretrained(args.checkpoint).eval()
    checkpoint_digest = digest_file(args.checkpoint / "memory.pt")
    consumer = ConsumerIdentity("memorilla/Memorilla-Qwen3-8B", args.reader_revision)
    store = ContextArtifactRegistry(args.output / "store")
    bank = EmbeddingStore("pubmedqa", "test", root=args.data)
    table = pq.read_table(args.data / "data/pubmedqa/test-00000-of-00001.parquet")
    artifacts = []
    if args.phase == "restore":
        artifacts = json.loads((args.output / "artifacts.json").read_text())
    cases = []
    with torch.inference_mode():
        for row in range(args.rows):
            collection = table["collection_id"][row].as_py()
            original_docs = bank.documents(collection)
            question = bank.question(row)
            messages = build_messages(
                table["question"][row].as_py(),
                protocol.system_prompt,
                reader.num_memories,
                protocol.suffix,
            )
            prompt = apply_chat_template(
                tokenizer, messages, add_generation_prompt=True
            )
            token_ids = tokenizer(prompt)["input_ids"]
            positions = tuple(
                i for i, token in enumerate(token_ids) if token == memory_token_id
            )
            if len(positions) != reader.num_memories:
                raise ValueError("author prompt has an unexpected memory placement")
            template_digest = hashlib.sha256(
                canonical(
                    {
                        "template": tokenizer.chat_template,
                        "messages": messages,
                        "token_ids": token_ids,
                    }
                )
            ).hexdigest()
            preceding_digest = hashlib.sha256(
                canonical(token_ids[: positions[0]])
            ).hexdigest()
            baseline = reader(
                original_docs.unsqueeze(0).to(torch.bfloat16),
                None,
                question.unsqueeze(0).to(torch.bfloat16),
            ).squeeze(0)
            if args.phase == "import":
                path = args.output / f"documents-{row}.safetensors"
                save_file({"embeddings": original_docs.contiguous()}, str(path))
                manifest = ContextArtifactManifest(
                    1,
                    "memorilla-reader-probe",
                    "document_embeddings",
                    "Memorilla",
                    args.source_revision,
                    checkpoint_digest,
                    (consumer,),
                    digest_file(path),
                    inspect_payload(path),
                    DocumentMetadata(
                        ConsumerIdentity(
                            "Qwen/Qwen3-Embedding-4B", args.encoder_revision
                        ),
                        hashlib.sha256(
                            canonical(
                                {
                                    "data_revision": args.data_revision,
                                    "config": "pubmedqa",
                                    "split": "test",
                                }
                            )
                        ).hexdigest(),
                        hashlib.sha256(
                            canonical(table.slice(row, 1).to_pylist())
                        ).hexdigest(),
                        original_docs.shape[0],
                        original_docs.shape[1],
                    ),
                )
                bank_id = store.register(manifest, path)
                soft_path = args.output / f"soft-tokens-{row}.safetensors"
                save_file({"embeddings": baseline.contiguous()}, str(soft_path))
                soft = ContextArtifactManifest(
                    1,
                    "memorilla-reader-probe",
                    "input_embeddings",
                    "Memorilla",
                    args.source_revision,
                    checkpoint_digest,
                    (ConsumerIdentity("Qwen/Qwen3-8B", args.decoder_revision),),
                    digest_file(soft_path),
                    inspect_payload(soft_path),
                    InputMetadata(
                        checkpoint_digest,
                        bank_id,
                        tensor_digest(question),
                        template_digest,
                        preceding_digest,
                        "system_message",
                        positions,
                        baseline.shape[1],
                    ),
                )
                soft_id = store.register(soft, soft_path)
                artifacts.append(
                    {
                        "bank_id": bank_id,
                        "soft_id": soft_id,
                        "collection": collection,
                        "question_digest": tensor_digest(question),
                        "output_digest": tensor_digest(baseline),
                    }
                )
            else:
                bank_id = artifacts[row]["bank_id"]
                soft_id = artifacts[row]["soft_id"]
            restored_docs = load(store.load(bank_id, consumer))["embeddings"]
            assert torch.equal(original_docs, restored_docs)
            restored_output = reader(
                restored_docs.unsqueeze(0).to(torch.bfloat16),
                None,
                question.unsqueeze(0).to(torch.bfloat16),
            ).squeeze(0)
            assert torch.equal(baseline, restored_output)
            assert torch.isfinite(restored_output).all()
            decoder = ConsumerIdentity("Qwen/Qwen3-8B", args.decoder_revision)
            persisted_soft = load(store.load(soft_id, decoder))["embeddings"]
            assert torch.equal(baseline, persisted_soft)
            assert tensor_digest(baseline) == artifacts[row]["output_digest"]
            store.evict(bank_id)
            assert bank_id not in store.resident
            assert torch.equal(
                load(store.restore(bank_id, consumer))["embeddings"], original_docs
            )
            cases.append(
                {
                    "row": row,
                    "collection": collection,
                    "document_rows": original_docs.shape[0],
                    "bank_id": bank_id,
                    "soft_id": soft_id,
                    "output_shape": list(baseline.shape),
                    "exact_docs": True,
                    "exact_reader": True,
                    "exact_soft_tokens": True,
                    "output_digest": tensor_digest(baseline),
                }
            )
    store.close()
    (args.output / "artifacts.json").write_bytes(canonical(artifacts))
    report = {
        "status": "CPU_READER_STAGE_PASS",
        "phase": args.phase,
        "cases": cases,
        "decoder_executed": False,
        "gpu_executed": False,
        "interface_pass": False,
        "reader_checkpoint_digest": checkpoint_digest,
        "device": "cpu",
        "reader_dtype": "bfloat16",
        "stored_document_dtype": "float16",
        "torch": torch.__version__,
        "memorilla_module": str(
            Path(importlib.import_module("memorilla").__file__).resolve()
        ),
        "page_cache_state": "unknown",
    }
    (args.output / (args.phase + "-result.json")).write_bytes(canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "data", "output", "tokenizer"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "source-revision",
        "reader-revision",
        "encoder-revision",
        "decoder-revision",
        "data-revision",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--phase", choices=("import", "restore"))
    args = parser.parse_args()
    if args.rows < 1 or args.rows > 8:
        parser.error("select one through eight fixed rows")
    for path in (args.source, args.checkpoint, args.data, args.output, args.tokenizer):
        persistent(path)
    actual = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != args.source_revision:
        parser.error("author source revision mismatch")
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(args.source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip():
        parser.error("author source has tracked modifications")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase:
        print(json.dumps(worker(args), indent=2))
        return
    # The writer exits before the restoring reader/registry process starts.
    for phase in ("import", "restore"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.reproduce.context_artifacts.reader_probe",
                *sys.argv[1:],
                "--phase",
                phase,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
