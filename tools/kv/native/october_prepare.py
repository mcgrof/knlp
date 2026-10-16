# SPDX-License-Identifier: MIT
"""Prepare public, document-disjoint inputs for a new October replication."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .october import VARIANTS


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from . import gsm8k
    from tools.kv.prequantized.fixtures import prepare_samples

    for pin in (args.revision, args.dataset_revision, args.gsm_revision):
        if re.fullmatch(r"[a-f0-9]{40}", pin) is None:
            raise ValueError("all revisions must be immutable full commit IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    samples, inventory, calibration, evaluation = prepare_samples(
        tokenizer,
        {"quality": {"input_tokens": args.lengths}},
        args.model,
        args.dataset_revision,
        "wikimedia/wikipedia",
    )
    manifest = {
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer_revision": args.revision,
        "samples": samples,
        "inventory": inventory,
        "calibration_article_ids": calibration,
        "evaluation_article_ids": evaluation,
        "max_output_tokens": 128,
        "dataset": "wikimedia/wikipedia",
        "dataset_revision": args.dataset_revision,
    }
    (args.output / "qwen.json.gz").write_bytes(
        gzip.compress(json.dumps(manifest, allow_nan=False).encode(), mtime=0)
    )
    train = list(
        load_dataset("openai/gsm8k", "main", split="train", revision=args.gsm_revision)
    )
    test = list(
        load_dataset("openai/gsm8k", "main", split="test", revision=args.gsm_revision)
    )
    write(
        args.output / "gsm8k-frozen.json",
        gsm8k.build(
            tokenizer,
            train,
            test,
            {"model_revision": args.revision, "dataset_revision": args.gsm_revision},
            final_instruction=True,
        ),
    )


def freeze(args):
    root = args.output.resolve()
    for name in ("qwen.json.gz", "gsm8k-frozen.json", "scales.json"):
        if not (root / name).is_file():
            raise ValueError("missing prepared input: " + name)
    source = Path(__file__).resolve().parents[3]
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--", "tools/kv"], text=True
    ):
        raise ValueError("commit runner source before freezing")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    inputs = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("qwen.json.gz", "gsm8k-frozen.json", "scales.json")
    }
    task = {"task_file_sha256": inputs["gsm8k-frozen.json"], "primary_cases": 1319}
    write(root / "task-freeze.json", task)
    for name, variant in VARIANTS.items():
        target = root / name
        target.mkdir(exist_ok=False)
        for filename in (*inputs, "task-freeze.json"):
            (target / filename).symlink_to(root / filename)
        for backend in ("vllm", "flashinfer"):
            (target / backend).symlink_to(args.install.resolve() / backend)
        engine = {
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "max_model_len": 10240,
            "max_num_seqs": 96,
            "max_num_batched_tokens": 65536,
            "block_size": 16,
            "kv_cache_memory_bytes": args.cache_bytes,
            "enforce_eager": False,
            "disable_log_stats": False,
            "enable_chunked_prefill": False,
            "calculate_kv_scales": False,
            "attention_config": {
                "backend": "FLASHINFER",
                "use_trtllm_attention": False,
            },
            "distributed_executor_backend": "uni",
            "worker_cls": "tools.kv.native.memory_worker.ObservedWorker",
            "seed": 20261018,
            "compilation_config": {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 8, 32, 44, 45, 59, 60, 96],
            },
        }
        packed = (
            {"format": "fp16", "linear_modules": 144, "embedding_modules": 2}
            if args.format == "fp16"
            else {
                "method": "AutoAWQMarlinLinearMethod",
                "kernel": "MacheteLinearKernel",
                "weight_bits": 4,
                "dispatch_marker": "machete_mm",
            }
        )
        if args.format == "awq":
            engine["quantization"] = "awq"
        write(
            target / "protocol.json",
            {
                "schema": 1,
                "source_sha": sha,
                "torch": "2.11.0+cu129",
                "backend_sources": dict(variant.sources),
                "runner_files_sha256": hashes,
                "inputs_sha256": inputs,
                "engine": engine,
                "torch_allocator_reserved_cap_bytes": 68 * 2**30,
                "total_driver_envelope_bytes": 72 * 2**30,
                "packed_execution": packed,
            },
        )
    print(
        "Commit the prepared protocol and task freeze in your result repository before creating SCORING-UNLOCKED.json with that protocol_commit."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument(
        "--model", choices=("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-AWQ"), required=True
    )
    for name in ("revision", "dataset-revision", "gsm-revision"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--install", type=Path, required=True)
    p.add_argument("--format", choices=("awq", "fp16"), required=True)
    p.add_argument("--cache-bytes", type=int, required=True)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else freeze)(args)


if __name__ == "__main__":
    main()
