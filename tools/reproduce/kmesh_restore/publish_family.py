#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export verified preload results without private paths or request identities."""

import argparse
import json
from pathlib import Path

from family_verify import verify_family
from restore_common import (
    ANCESTOR,
    CHUNK,
    CHUNK_BYTES,
    CHUNKS,
    FAMILY_PREFETCH_COMMIT,
    VLLM_COMMIT,
    MODEL,
    REVISION,
    write_json,
)


def export(root: Path) -> dict:
    """Recheck raw evidence and retain bounded public observations and hashes."""
    verified = verify_family(root)
    plan = json.loads((root / "plan.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    for component, pin in (("vllm", VLLM_COMMIT), ("lmcache", FAMILY_PREFETCH_COMMIT)):
        if provenance["sources"][component] != {"commit": pin, "clean": True}:
            raise ValueError("component source differs from the pinned profile")
    native = {Path(name).name: value for name, value in provenance["native"].items()}
    if len(native) != len(provenance["native"]):
        raise ValueError("native basenames are ambiguous")
    cases = verified["cases"]
    wasted = {}
    for mode in ("staged", "direct"):
        for fanout in (4, 8):
            cell = root / f"{mode}-n{fanout}"
            rows = json.loads((cell / "consumer.json").read_text())
            for record in rows:
                all_requests = record["rows"] + record["warm_rows"]
                loads = [
                    r["kv_transfer_params"]["cached_token_stats"][
                        "num_lmcache_extra_cached_tokens"
                    ]
                    for r in all_requests
                ]
                if sum(value > 0 for value in loads) != 1 or sum(loads) != ANCESTOR:
                    raise ValueError(
                        "family did not use one complete external engine load"
                    )
            for policy, gap in [("reactive", 0), ("oracle", 0)] + [
                ("family", gap) for gap in (0, 10, 25, 100)
            ]:
                selected = [
                    r
                    for r in rows
                    if r["policy"] == policy
                    and r["requested_gap_ms"] == gap
                    and not r["warmup"]
                ]
                cases[f"{mode}-n{fanout}-{policy}-g{gap}"]["retained_samples"] = [
                    {
                        "first_submitted_child_ttft_ms": r["rows"][0]["ttft_ms"],
                        "actual_fork_gap_ms": r["actual_gap_ms"],
                        "observed_warm_request_completion_ms": r["warm_complete_ms"],
                        "warm_completed_before_release": r[
                            "warm_completed_before_release"
                        ],
                        "children_with_local_ancestor": sum(
                            child["kv_transfer_params"]["cached_token_stats"][
                                "num_vllm_cached_tokens"
                            ]
                            >= ANCESTOR
                            for child in r["rows"]
                        ),
                        "children_loading_external_ancestor": sum(
                            child["kv_transfer_params"]["cached_token_stats"][
                                "num_lmcache_extra_cached_tokens"
                            ]
                            > 0
                            for child in r["rows"]
                        ),
                        "children_wall_ms": r["children_wall_ms"],
                        "declaration_to_completion_ms": r[
                            "declaration_to_completion_ms"
                        ],
                    }
                    for r in selected
                ]
            unused = json.loads((cell / "wasted.json").read_text())
            before, after = (
                unused["before"]["l2_adapters"][0]["core"],
                unused["after"]["l2_adapters"][0]["core"],
            )
            wasted[f"{mode}-n{fanout}"] = {
                "backend_chunk_reads": after["payload_reads"]["logical_requests"]
                - before["payload_reads"]["logical_requests"],
                "unused_declarations_charged": unused["family_after"][
                    "family_prefetch_wasted_total"
                ]
                - unused["family_before"]["family_prefetch_wasted_total"],
            }
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
            "retained_families_per_case": plan["serving_samples"],
            "warmups_per_case": 3,
            "engine_kv_pool_bytes": 512 << 20,
            "server_arena_bytes_per_path": 512 << 20,
            "max_sequences": 8 + 1,
            "engine_processes_per_path_and_fanout": 1,
            "timing_definition": "First submitted child's engine time to first token from actual arrival; warm completion includes its one-token request.",
            "requested_fork_gaps_ms": [0, 10, 25, 100],
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
        "unused_preload_controls": wasted,
        "external_engine_loads_per_family_including_warm_request": 1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, export(args.root))
