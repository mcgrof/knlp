#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export retained-cache evidence without machine paths or request identities."""

import argparse
import json
from pathlib import Path

from restore_common import (
    ANCESTOR,
    CHUNK,
    CHUNK_BYTES,
    CHUNKS,
    FAMILY_PREFETCH_COMMIT,
    MODEL,
    REVISION,
    VLLM_COMMIT,
    write_json,
)
from retained_verify import verify_retained


def export(root: Path):
    """Verify the complete capture and retain only public metadata and samples."""
    verified = verify_retained(root)
    provenance = json.loads((root / "provenance.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    for component, pin in (("vllm", VLLM_COMMIT), ("lmcache", FAMILY_PREFETCH_COMMIT)):
        if provenance["sources"][component] != {"commit": pin, "clean": True}:
            raise ValueError("component source differs from the pinned profile")
    native = {Path(name).name: value for name, value in provenance["native"].items()}
    if len(native) != len(provenance["native"]):
        raise ValueError("native basenames are ambiguous")
    return verified | {
        "model": MODEL,
        "model_revision": REVISION,
        "configuration": {
            "ancestor_tokens": ANCESTOR,
            "chunk_tokens": CHUNK,
            "chunk_bytes": CHUNK_BYTES,
            "ancestor_bytes": CHUNK_BYTES * CHUNKS,
            "tail_tokens": 512,
            "child_decode_tokens": 8,
            "warm_decode_tokens": 1,
            "retained_families_per_arm": plan["serving_samples"],
            "warmups_per_arm": 3,
            "engine_kv_pool_bytes": 512 << 20,
            "external_arena_bytes": 512 << 20,
            "external_memory_tier": {"staged": "host DRAM", "direct": "GPU memory"},
            "max_sequences": 9,
            "engine_processes_per_path_and_fanout": 1,
            "pair_order": "engine/retained then retained/engine, alternating per sample",
            "timing_definition": "HTTP submission and completion polling precede the engine warm request inside the fork clock. First-child TTFT starts at actual engine arrival.",
            "storage": "NVMe O_DIRECT regular file, io_uring QD8; staged host_fixed or experimental vulkan_rm dmabuf_fixed",
        },
        "runtime": {
            key: runtime[key]
            for key in (
                "torch",
                "cuda",
                "gpu",
                "capability",
                "threads",
                "batch_invariant",
            )
        },
        "python": provenance["python"].split()[0],
        "sources": provenance["sources"],
        "harness_sha256": provenance["harness"],
        "model_files_sha256": provenance["model_files"],
        "native_sha256_by_basename": native,
        "external_engine_loads_per_family_including_warm_request": 1,
        "children_loading_external_ancestor": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, export(args.root))
