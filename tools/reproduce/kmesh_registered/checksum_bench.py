#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify exact CUDA SHA-256 and time four whole cache objects against CPU hashing."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

ROOT = Path(__file__).resolve().parent


def qualify(hasher, torch):
    """Check padding boundaries, independent bytes, corruption and stream ordering."""
    generator = random.Random(24)
    cases = []
    for size in (
        0,
        1,
        3,
        31,
        32,
        55,
        56,
        63,
        64,
        65,
        119,
        120,
        127,
        128,
        129,
        255,
        256,
        257,
        4095,
        4096,
        4097,
        3 << 20,
    ):
        for pattern in ("zero", "ones", "random"):
            payload = (
                bytes(size)
                if pattern == "zero"
                else (
                    bytes([255]) * size
                    if pattern == "ones"
                    else generator.randbytes(size)
                )
            )
            # Exercise an unaligned source and the empty message on real CUDA storage.
            backing = torch.empty(size + 8, dtype=torch.uint8, device="cuda")
            source = backing[1 : size + 1]
            if size:
                source.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
            if hasher.hexdigest(source) != hashlib.sha256(payload).hexdigest():
                raise RuntimeError(f"GPU SHA-256 disagrees for {size} {pattern} bytes")
            cases.append({"bytes": size, "pattern": pattern})
    abc = torch.frombuffer(bytearray(b"abc"), dtype=torch.uint8).to("cuda")
    if (
        hasher.hexdigest(abc)
        != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    ):
        raise RuntimeError("published SHA-256 abc vector differs")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()
    if not 3 <= args.trials <= 100:
        parser.error("choose three through one hundred matched trials")
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT.parents[2]):
        parser.error("keep captures outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    import torch

    if __package__:
        from .gpu_integrity import CudaSha256
    else:
        from gpu_integrity import CudaSha256

    hasher = CudaSha256(args.library)
    cases = qualify(hasher, torch)
    size = 3 << 20
    generator = random.Random(24)
    payloads = [generator.randbytes(size) for _ in range(4)]
    host = torch.frombuffer(bytearray(b"".join(payloads)), dtype=torch.uint8).reshape(
        4, size
    )
    source = host.to("cuda")
    expected = tuple(hashlib.sha256(p).hexdigest() for p in payloads)
    if hasher.hexdigests(source) != expected:
        raise RuntimeError("four GPU object digests differ from the CPU oracle")
    corruptions = []
    for offset in (0, 123, 2 << 20, size - 1):
        altered = source.clone()
        altered[2, offset] ^= 1
        actual = hasher.hexdigests(altered)
        bad = bytearray(payloads[2])
        bad[offset] ^= 1
        if (
            any((a != b) != (i == 2) for i, (a, b) in enumerate(zip(actual, expected)))
            or actual[2] != hashlib.sha256(bad).hexdigest()
        ):
            raise RuntimeError("GPU checksum misses corruption or changes a sibling")
        corruptions.append(offset)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    ready = torch.cuda.Event()
    with torch.cuda.stream(producer):
        copied = source.clone()
        copied[1, -1] ^= 1
        ready.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        actual = hasher.hexdigests(copied)
    altered_host = bytearray(payloads[1])
    altered_host[-1] ^= 1
    ordered = tuple(
        hashlib.sha256(altered_host).hexdigest() if i == 1 else value
        for i, value in enumerate(expected)
    )
    if actual != ordered:
        raise RuntimeError("GPU checksum differs after the explicit producer handoff")
    # Reinterpret the same raw bytes as BF16, including arbitrary bit patterns.
    if hasher.hexdigest(source[0].view(torch.bfloat16)) != expected[0]:
        raise RuntimeError("BF16 byte interpretation changes the checksum")

    def cpu():
        return tuple(
            hashlib.sha256(row.cpu().numpy().tobytes()).hexdigest() for row in source
        )

    def gpu_serial():
        return tuple(hasher.hexdigest(row) for row in source)

    def gpu_batch():
        return hasher.hexdigests(source)

    methods = {
        "cpu-copy-hash": cpu,
        "gpu-serial": gpu_serial,
        "gpu-batch-four": gpu_batch,
    }
    for _ in range(3):
        for method in methods.values():
            if method() != expected:
                raise RuntimeError("warmup digests differ")
    rows = []
    for trial in range(args.trials):
        names = list(methods)
        names = names[trial % 3 :] + names[: trial % 3]
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            actual = methods[name]()
            elapsed = time.perf_counter_ns() - start
            if actual != expected:
                raise RuntimeError("timed checksum differs from the CPU oracle")
            rows.append({"trial": trial, "method": name, "elapsed_ns": elapsed})
    result = {
        "passed": True,
        "qualification": {
            "byte_cases": cases,
            "abc_vector": True,
            "bf16_bytes": True,
            "corruption_offsets": corruptions,
            "cross_stream_handoff": True,
        },
        "bytes_per_object": size,
        "objects": 4,
        "rows": rows,
        "median_ms": {
            name: statistics.median(
                r["elapsed_ns"] for r in rows if r["method"] == name
            )
            / 1e6
            for name in methods
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "compute_capability": torch.cuda.get_device_capability(0),
            "omp_threads": torch.get_num_threads(),
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                ROOT / "gpu_sha256.cu",
                ROOT / "gpu_integrity.py",
                Path(__file__),
                args.library,
            )
        },
        "limits": "Four independent full 3 MiB SHA-256 messages, not a tree hash or a serving benchmark. This CUDA kernel uses one thread per message and is not evidence about every GPU hash implementation.",
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "median_ms": result["median_ms"]}, indent=2))


if __name__ == "__main__":
    main()
