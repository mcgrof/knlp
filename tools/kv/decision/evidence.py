# SPDX-License-Identifier: MIT
"""Independently audit retained token/text and physical-storage evidence on CPU.

This does not change scores or certify accuracy. It validates complete fixed
panels and distinguishes requested backing bytes from allocator high-water.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile


ARMS = ("N", "AF", "RF", "SF", "AD", "RD", "SD")
ELEMENT_BYTES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float8_e4m3fn": 1,
    "torch.float32": 4,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def storage(audit, arm, config, dtype, length):
    layers, heads = config["num_hidden_layers"], config["num_key_value_heads"]
    dimension = (
        config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    )
    if (
        config.get("layer_types", ["full_attention"] * layers)
        != ["full_attention"] * layers
    ):
        raise ValueError(
            "This accounting audit covers the two frozen full-attention checkpoints"
        )
    expected = {}
    for layer in range(layers):
        for component, field in [("K", "keys"), ("V", "values")]:
            compressed = arm.startswith("S") or arm.startswith(
                "R" if component == "K" else "A"
            )
            expected[(layer, field)] = (
                "torch.float8_e4m3fn" if compressed else "torch." + dtype,
                [1, heads, length + 128, dimension],
            )
            if compressed:
                expected[(layer, component.lower() + "_scale")] = (
                    "torch.float32",
                    [] if arm.endswith("F") else [1, heads, length + 128, 1],
                )
    rows = audit["tensors"]
    if len(rows) != len(expected) or {(t["layer"], t["field"]) for t in rows} != set(
        expected
    ):
        raise ValueError("Persistent tensor inventory differs from frozen geometry")
    allocations = {}
    logical = scale_bytes = 0
    for row in rows:
        kind, shape = expected[(row["layer"], row["field"])]
        if row["dtype"] != kind or row["shape"] != shape:
            raise ValueError("Physical dtype or shape differs")
        size = math.prod(shape) * ELEMENT_BYTES[kind]
        if row["logical_bytes"] != size or row["backing_bytes"] != size:
            raise ValueError("Backing allocation differs from stored tensor")
        stride = row["stride"]
        if len(stride) != len(shape) or any(
            type(s) is not int or s < 0 for s in stride
        ):
            raise ValueError("Invalid storage strides")
        extent = (1 + sum((n - 1) * s for n, s in zip(shape, stride))) * ELEMENT_BYTES[
            kind
        ]
        if extent > size:
            raise ValueError("Stride extends outside backing storage")
        packed_stride = 1
        for step, width in sorted((s, n) for s, n in zip(stride, shape) if n > 1):
            if step != packed_stride:
                raise ValueError(
                    "Cache tensor is not a dense nonoverlapping allocation"
                )
            packed_stride *= width
        if row["storage_id"] in allocations:
            raise ValueError("Independent reference tensors unexpectedly alias")
        allocations[row["storage_id"]] = size
        logical += size
        if row["field"].endswith("scale"):
            scale_bytes += size
    if (
        audit["logical_bytes"] != logical
        or audit["unique_backing_bytes"] != sum(allocations.values())
        or audit["scale_logical_bytes"] != scale_bytes
    ):
        raise ValueError("Recorded allocation totals do not reconcile")
    return {
        "persistent_requested_backing_bytes": logical,
        "scale_backing_bytes": scale_bytes,
        "allocation_count": len(allocations),
        "allocation_size_multiset": sorted(allocations.values()),
        "allocator_scope": "Backing bytes requested by tensor storages. CUDA allocator rounding/fragmentation and whole-process reserved peaks are separate; equal A/R size multisets establish the same allocation size/count requirements.",
    }


def generation(row, tokenizer, cap, eos):
    ids = row["tokens"]
    if any(type(x) is not int or x < 0 or x in eos for x in ids):
        raise ValueError("Invalid retained generated tokens")
    if len(ids) > cap or row["output_tokens"] != len(ids):
        raise ValueError("Generated length differs")
    if tokenizer.decode(ids, skip_special_tokens=True) != row["text"]:
        raise ValueError("Retained text differs from independently decoded tokens")
    if type(row["eos"]) is not bool or row["truncated"] != (
        not row["eos"] and len(ids) == cap
    ):
        raise ValueError("Retained EOS/truncation flags are inconsistent")
    if not row["eos"] and len(ids) != cap:
        raise ValueError("Unexplained early generation termination")
    if row["error"] is not None:
        raise ValueError("Retained task error")


def audit_model(study, name):
    from transformers import AutoTokenizer

    plan = json.loads((study / "PLAN.json").read_text())
    expected_model = plan["models"][name]
    inputs_dir = study / "inputs" / name
    input_path = inputs_dir / "inputs.json.gz"
    data = json.loads(gzip.decompress(input_path.read_bytes()))
    execution = plan.get("executions", {}).get(name, name)
    mirror = study / execution / "artifact-mirror"
    root = mirror / "confirmation"
    start = json.loads((root / "START.json").read_text())
    done = json.loads((root / "DONE.json").read_text())
    if (
        start["input_sha256"] != sha(input_path)
        or sha(input_path) != expected_model["inputs_sha256"]
    ):
        raise ValueError("Frozen input identity differs")
    if any(start[k] != expected_model[k] for k in ["model", "revision", "dtype"]):
        raise ValueError("Model identity differs")
    if start["source"]["commit"] != plan.get("source_commit_by_model", {}).get(
        name, plan["source_commit"]
    ):
        raise ValueError("Measured source identity differs from the plan")
    archive_name = plan.get("source_archive_by_model", {}).get(name, "source.tar.gz")
    if sha(study / archive_name) != plan["local_files"][archive_name]:
        raise ValueError("Frozen source archive changed")
    with tarfile.open(study / archive_name) as archive:
        if json.load(archive.extractfile("SOURCE.json")) != start["source"]:
            raise ValueError("Runtime source-byte map differs from frozen archive")
    if not done["native_parity_pass"] or start["args"]["stage"] != "confirmation":
        raise ValueError("Complete native-qualified confirmation required")
    official = json.loads((inputs_dir / "official-weights.json").read_text())
    if start["weight_sha256"] != {k: v["sha256"] for k, v in official["files"].items()}:
        raise ValueError("Official checkpoint hashes differ")
    tokenizer_dir = inputs_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    eos = start["generation_config"]["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    hashes, memory, counts = {}, {}, {}
    for arm in ARMS:
        memory[arm] = {}
        counts[arm] = {}
        for kind in ["nll", "retrieval", "gsm8k"]:
            path = root / arm / (kind + ".jsonl")
            hashes[str(path.relative_to(mirror))] = sha(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            expected = {
                r["id"]: r for r in data["samples"]["confirmation"] if r["kind"] == kind
            }
            if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
                raise ValueError("Missing, repeated or replaced fixed panel item")
            counts[arm][kind] = len(rows)
            for row in rows:
                item = expected[row["id"]]
                if row["length"] != (None if kind == "gsm8k" else item["length"]):
                    raise ValueError("Recorded context differs from frozen item")
                if kind != "nll":
                    generation(row, tokenizer, 2048 if kind == "gsm8k" else 128, eos)
                else:
                    if (
                        row["article_id"] != item["article_id"]
                        or len(row["nll"]) != 128
                        or not all(math.isfinite(x) for x in row["nll"])
                    ):
                        raise ValueError("NLL document/target identity differs")
                    value = storage(
                        row["cache"], arm, start["config"], data["dtype"], row["length"]
                    )
                    length = str(row["length"])
                    if length in memory[arm] and memory[arm][length] != value:
                        raise ValueError("Fixed-shape backing allocations changed")
                    memory[arm][length] = value
    for policy in ["F", "D"]:
        for length in ["1024", "8192"]:
            if memory["A" + policy][length] != memory["R" + policy][length]:
                raise ValueError("A/R allocation size/count requirements differ")
    return {
        "model": start["model"],
        "revision": start["revision"],
        "source": start["source"],
        "input_sha256": sha(input_path),
        "tokenizer_files_sha256": {
            str(p.relative_to(tokenizer_dir)): sha(p)
            for p in sorted(tokenizer_dir.rglob("*"))
            if p.is_file()
        },
        "panel_files_sha256": hashes,
        "counts": counts,
        "memory": memory,
        "retained_tokens_decode_exactly": True,
        "scope": "Audit of retained serialized evidence; does not rescore or substitute for endpoint confidence bounds.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", choices=["qwen3", "qwen25", "all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ["qwen3", "qwen25"] if args.model == "all" else [args.model]
    result = {name: audit_model(args.study, name) for name in names}
    args.output.write_text(
        json.dumps(
            {"reader_sha256": sha(Path(__file__)), "models": result},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
