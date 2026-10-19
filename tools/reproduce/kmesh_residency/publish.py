# SPDX-License-Identifier: MIT
"""Verify a controlled capture and export counts without request identities."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .audit import lineage
from .cpu import VLLM_COMMIT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def export(root):
    root = Path(root)
    summary_path = root / "serving.json"
    trace_path = root / "invocations.jsonl"
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    require(summary["schema"] == "knlp.kmesh.residency-serving.v1", "unknown capture")
    require(
        summary["controlled_synthetic"] is True
        and summary["storage_connector"] is None,
        "expected a controlled engine-only capture",
    )
    require(summary["source_revision"] == VLLM_COMMIT, "unexpected source")
    require(
        summary["pool_blocks"] == 129
        and summary["ancestor_tokens"] == 512
        and summary["generated_tokens"] == 8,
        "unexpected workload geometry",
    )
    samples = summary["samples_per_arm"]
    require(type(samples) is int and samples >= 2, "insufficient paired capture")
    require(len(rows) == samples * 20, "missing invocation records")
    edges = lineage(rows)
    require(
        edges["observed_edges"] == samples * 2
        and edges["unknown_parents"]
        == edges["cross_clock_edges"]
        == edges["cross_engine_edges"]
        == edges["non_exact_parent_prefix_edges"]
        == 0,
        "incomplete exact lineage",
    )
    pairs = {}
    for row in rows:
        require(
            type(row["sample"]) is int
            and type(row["lease"]) is bool
            and type(row["cached_tokens"]) is int
            and 0 <= row["cached_tokens"] <= len(row["input_token_ids"]),
            "invalid request counters or identity",
        )
        require(
            row["status"] == "complete" and len(row["output_token_ids"]) == 8,
            "incomplete output",
        )
        require(
            len(row["chosen_logprobs"]) == 8
            and all(math.isfinite(x) for x in row["chosen_logprobs"]),
            "invalid chosen logprob",
        )
        key = (row["sample"], row["role"], int(row["request_id"].rsplit("-", 1)[1]))
        pairs.setdefault(key, []).append(row)
    require(len(pairs) == samples * 10, "unexpected request pairing")
    for pair in pairs.values():
        require(
            len(pair) == 2 and {r["lease"] for r in pair} == {True, False},
            "unpaired arms",
        )
        for key in ("input_token_ids", "output_token_ids", "chosen_logprobs"):
            require(pair[0][key] == pair[1][key], "paired continuation mismatch")
    cells = []
    require(len(summary["cells"]) == samples * 2, "missing cells")
    seen = set()
    for cell in summary["cells"]:
        sample, lease = cell["sample"], cell["lease"]
        require(
            type(sample) is int and 0 <= sample < samples and type(lease) is bool,
            "invalid cell identity",
        )
        require((sample, lease) not in seen, "duplicate cell")
        require(
            cell["pool_blocks"] == 129 and cell["ancestor_blocks"] == 32,
            "cell geometry disagrees",
        )
        seen.add((sample, lease))
        members = [r for r in rows if (r["sample"], r["lease"]) == (sample, lease)]
        parents = [r for r in members if r["role"] == "parent"]
        children = [r for r in members if r["role"] == "child"]
        competing = [r for r in members if r["role"] == "competitor"]
        require(
            len(parents) == len(children) == 1 and len(competing) == 8,
            "unexpected cell members",
        )
        parent, child = parents[0], children[0]
        require(
            len(parent["input_token_ids"]) == 513
            and len(child["input_token_ids"]) == 585
            and all(len(r["input_token_ids"]) == 497 for r in competing),
            "unexpected prompt geometry",
        )
        require(child["parent_request_id"] == parent["request_id"], "wrong parent")
        require(
            cell["child_cached_tokens"] == child["cached_tokens"]
            and cell["competitor_cached_tokens"]
            == [r["cached_tokens"] for r in competing],
            "cache counters disagree",
        )
        require(
            cell["final_free_blocks"] == 128 and cell["final_nonnull_references"] == 0,
            "allocator references did not drain",
        )
        require(
            all(
                cell["lease_state"][key] == 0
                for key in ("families", "unique_held_blocks", "family_block_references")
            ),
            "family references did not drain",
        )
        events = cell["events"]
        require(
            cell["evictions"] == sum(e["action"] == "evicted" for e in events),
            "eviction count disagrees",
        )
        require(
            all(e["prior_ref_count"] == 0 for e in events if e["action"] == "evicted"),
            "eviction of a referenced block",
        )
        if lease:
            acquisitions = [
                e
                for e in parent["cache_observations"]
                if e["action"] == "lease-acquired"
            ]
            allocations = [
                e for e in child["cache_observations"] if e["action"] == "allocated"
            ]
            require(len(acquisitions) == 1 and allocations, "missing alias observation")
            a, b = acquisitions[0], allocations[0]
            require(
                a["block_ids"][:32] == b["block_ids"][:32]
                and len(set(a["block_ids"][:32])) == 32,
                "physical ancestor alias mismatch",
            )
            require(
                len(a["ref_counts"]) >= 32
                and len(b["ref_counts"]) >= 32
                and all(
                    refs == 2 for refs in a["ref_counts"][:32] + b["ref_counts"][:32]
                ),
                "missing independent family/request refs",
            )
            require(
                a["lease_state"]["unique_held_blocks"] == 32
                and cell["lease_state"]["released"]["complete"] == 1,
                "incomplete hold lifetime",
            )
            require(child["cached_tokens"] >= 512, "protected ancestor was not hit")
        cells.append(
            {
                key: cell[key]
                for key in (
                    "sample",
                    "lease",
                    "pool_blocks",
                    "ancestor_blocks",
                    "child_cached_tokens",
                    "competitor_cached_tokens",
                    "evictions",
                    "lease_state",
                    "final_free_blocks",
                    "final_nonnull_references",
                )
            }
        )
    return {
        "schema": "knlp.kmesh.residency-public.v1",
        "controlled_mechanism_qualified": True,
        "production_workload_qualified": False,
        "storage_tested": False,
        "latency_benchmark": False,
        "vllm_commit": VLLM_COMMIT,
        "runtime": summary["runtime"],
        "model_config_sha256": summary["model_config_sha256"],
        "configuration": {
            "physical_pool_blocks": 129,
            "reserved_null_blocks": 1,
            "block_tokens": 16,
            "ancestor_tokens": 512,
            "lease_block_cap": 32,
            "lease_family_cap": 2,
            "lease_ttl_seconds": 60,
            "competitor_prompt_tokens": 497,
            "competitors_per_pass": 4,
            "competitor_passes": 2,
            "child_tail_tokens": 64,
            "decode_tokens": 8,
            "samples_per_arm": samples,
            "active_requests": 1,
            "synchronous": True,
            "eager": True,
            "batch_invariant": True,
        },
        "completed_invocations": len(rows),
        "paired_output_tokens_and_logprobs": len(pairs) * 8,
        "physical_aliases_checked": samples,
        "signed_fork_gap_ms": edges["signed_fork_gap_ms"],
        "capture_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (summary_path, trace_path)
        },
        "cells": cells,
        "limits": summary["limits"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    result = export(args.root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
