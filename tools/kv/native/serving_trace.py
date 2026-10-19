# SPDX-License-Identifier: MIT
"""Generate explicit, reusable request arrivals from frozen token prompts."""

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


def trace(
    manifest,
    *,
    length,
    count,
    concurrency=None,
    rate=None,
    seed=20261019,
    max_tokens=128,
):
    samples = [
        s
        for s in manifest["samples"]
        if s["kind"] == "nll" and s["length"] == length and not s.get("reserve")
    ]
    if len(samples) != 32 or length not in (1024, 8192):
        raise ValueError("expected the frozen 32-document panel and 1K/8K length")
    if any(len(sample["token_ids"]) < length for sample in samples):
        raise ValueError("frozen prompt is shorter than its declared length")
    if (concurrency is None) == (rate is None) or count < 1:
        raise ValueError("choose closed concurrency or an open arrival rate")
    rng, offset, requests = random.Random(seed), 0.0, []
    for index in range(count):
        if rate is not None and index:
            offset += rng.expovariate(rate)
        requests.append(
            {
                "id": f"trace-{seed}-{index:05}",
                "offset_ns": round(offset * 1e9),
                "prompt_token_ids": samples[index % 32]["token_ids"][:length],
                "max_tokens": max_tokens,
            }
        )
    return {
        "schema": 1,
        "model": "study",
        "seed": seed,
        "mode": "closed" if concurrency is not None else "open",
        "concurrency": concurrency,
        "offered_rate_per_s": rate,
        "arrival_duration_ns": requests[-1]["offset_ns"],
        "request_timeout_s": 300,
        "drain_timeout_s": 300,
        "requests": requests,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, choices=(1024, 8192), required=True)
    p.add_argument("--count", type=int, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--concurrency", type=int)
    group.add_argument("--rate", type=float)
    p.add_argument("--seed", type=int, default=20261019)
    args = p.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(
        gzip.decompress(raw) if args.manifest.suffix == ".gz" else raw
    )
    result = trace(
        manifest,
        length=args.length,
        count=args.count,
        concurrency=args.concurrency,
        rate=args.rate,
        seed=args.seed,
    )
    result["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
