# SPDX-License-Identifier: MIT
"""Verify the seven serving cases from their allocation-boundary observations."""

import hashlib
import json
from pathlib import Path

SCENARIOS = (
    "children_after_parent_cached",
    "children_after_parent_uncached",
    "burst_with_parent",
    "cold_sibling_burst",
    "churn_050",
    "churn_100",
    "fork_from_decode",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_case(name, result):
    require(result.get("measurement") == "serving-fork-replay", "wrong measurement")
    require(
        result.get("seed") == 20260805 and result.get("block_size") == 16,
        "workload drift",
    )
    require(result.get("tree") == "a4096-b5-t512", "wrong tree")
    value = result[name]
    rows = value["rows"]
    children = [row for row in rows if row["label"].startswith("d0.")]
    require(
        len(children) == (3 if name == "fork_from_decode" else 5), "incomplete children"
    )
    by_digest = {row["prompt_sha256"]: row for row in rows}
    require(len(by_digest) == len(rows), "ambiguous prompt ownership")
    require(
        set(value["resolved_scheduler"])
        == {
            "max_num_batched_tokens",
            "max_num_seqs",
            "enable_chunked_prefill",
            "async_scheduling",
        },
        "missing scheduler provenance",
    )
    require(
        bool(value["attention_backends"])
        and all(isinstance(name, str) for name in value["attention_backends"]),
        "missing attention backend provenance",
    )
    for row in rows:
        hit = row["num_cached_tokens"]
        require(
            type(hit) is int and 0 <= hit < row["prompt_tokens"] and hit % 16 == 0,
            "invalid cached-token count",
        )
        require(
            row["output_tokens"] == len(row["output_token_ids"]) > 0,
            "missing generated output",
        )
    if name == "children_after_parent_cached":
        require(
            [row["num_cached_tokens"] for row in children] == [4096] * 5,
            "warm ancestor was not reused",
        )
    if name == "children_after_parent_uncached":
        require(
            all(row["num_cached_tokens"] == 0 for row in rows),
            "uncached control reused a prefix",
        )
    if name.startswith("churn_"):
        require(value["total_blocks"] == 1024, "churn pool changed")
        require(
            value["churn_tokens"] == (8192 if name == "churn_050" else 16384),
            "churn workload changed",
        )
    if name == "fork_from_decode":
        require(
            value["fork_point"] == 4096 + value["parent_generated"],
            "decode fork boundary changed",
        )
        require(
            all(row["num_cached_tokens"] >= 4096 for row in children),
            "decode fork lost its ready ancestor",
        )
    events = value["allocation_events"]
    require(
        events and events[-1]["action"] == "free" and events[-1]["tables"] == {},
        "live allocation ownership remains",
    )
    peak = 0
    shared_pairs = set()
    uncached_disjoint_pairs = set()
    observed = set()
    for event in events:
        used = event["used_blocks"]
        require(
            type(used) is int and 1 <= used <= value["total_blocks"],
            "invalid physical pool use",
        )
        peak = max(peak, used)
        require(
            set(event["tables"]) == set(event["prompt_sha256"]),
            "allocation identity differs",
        )
        active = []
        for request_id, groups in event["tables"].items():
            require(len(groups) == 1, "expected one full-attention group")
            ids = groups[0]
            require(
                len(ids) == len(set(ids))
                and all(
                    type(i) is int
                    and 0 <= i < value["total_blocks"]
                    and i != value["null_block_id"]
                    for i in ids
                ),
                "invalid request block table",
            )
            digest = event["prompt_sha256"][request_id]
            # Churn rows are intentionally not part of the fork family.
            row = by_digest.get(digest)
            if row is not None:
                observed.add(digest)
                if row in children:
                    active.append((row, ids))
        for index, (left, left_ids) in enumerate(active):
            for right, right_ids in active[index + 1 :]:
                pair = tuple(sorted((left["label"], right["label"])))
                boundary = value.get("fork_point", 4096) // 16
                require(
                    not (set(left_ids[boundary:]) & set(right_ids))
                    and not (set(right_ids[boundary:]) & set(left_ids)),
                    "private sibling tails alias",
                )
                hits = min(left["num_cached_tokens"], right["num_cached_tokens"]) // 16
                if hits and len(left_ids) >= hits and len(right_ids) >= hits:
                    require(
                        left_ids[:hits] == right_ids[:hits],
                        "cached prefixes are copied instead of aliased",
                    )
                    shared_pairs.add(pair)
                if name == "children_after_parent_uncached":
                    require(
                        not set(left_ids) & set(right_ids),
                        "uncached requests share physical blocks",
                    )
                    uncached_disjoint_pairs.add(pair)
    require(observed == set(by_digest), "a fork request has no allocation observation")
    require(
        peak == value["allocation_boundary_peak_used_blocks"],
        "allocation peak differs from event ledger",
    )
    if name == "children_after_parent_cached":
        require(
            len(shared_pairs) == 10,
            "not all ten live sibling pairs demonstrate sharing",
        )
    if name == "children_after_parent_uncached":
        require(
            len(uncached_disjoint_pairs) == 10,
            "uncached siblings were not simultaneously observed",
        )
    return {
        "child_hit_tokens": [row["num_cached_tokens"] for row in children],
        "allocation_boundary_peak_used_blocks": peak,
        "sampled_peak_used_blocks": value["sampled_peak_used_blocks"],
        "observed_shared_sibling_pairs": len(shared_pairs),
        "resolved_scheduler": value["resolved_scheduler"],
        "attention_backends": value["attention_backends"],
    }


def verify_bundle(directory):
    directory = Path(directory)
    root = Path(__file__).resolve().parent
    profile = json.loads((root.parent / "tokendance/profile.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    source = provenance["source"]
    require(
        not source["status_porcelain"]
        and all(
            source[key] == value
            for key, value in json.loads((root / "profile-current.json").read_text())[
                "vllm"
            ].items()
        ),
        "source provenance differs",
    )
    require(
        provenance["image"] == profile["current_cuda"]["image"],
        "image provenance differs",
    )
    require(
        provenance["inputs_sha256"]
        == {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("serving.py", "serving_run.py", "serving_verify.py")
        },
        "serving inputs differ",
    )
    results, cases, hashes = {}, {}, {}
    for name in SCENARIOS:
        path = directory / (name + ".json")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        results[name] = json.loads(path.read_text())
        cases[name] = verify_case(name, results[name])
    identity_fields = (
        "vllm_version",
        "torch_version",
        "python_version",
        "cuda_version",
        "model_revision",
        "gpu_name",
        "seed",
        "block_size",
    )
    identity = {key: results[SCENARIOS[0]][key] for key in identity_fields}
    require(
        all(
            {key: result[key] for key in identity_fields} == identity
            for result in results.values()
        ),
        "serving runtimes differ",
    )
    for field, key in (
        ("python_version", "python"),
        ("torch_version", "torch"),
        ("vllm_version", "overlay_vllm_version"),
        ("cuda_version", "cuda"),
    ):
        require(
            identity[field] == profile["current_cuda"]["runtime"][key],
            "serving runtime pin differs",
        )
    require(
        identity["model_revision"] == profile["model_revision"],
        "model revision differs",
    )
    warm = results[SCENARIOS[0]][SCENARIOS[0]]["rows"]
    cold = results[SCENARIOS[1]][SCENARIOS[1]]["rows"]
    parity = {
        a["label"]: a["output_token_ids"] == b["output_token_ids"]
        for a, b in zip(warm, cold, strict=True)
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "cases": cases,
        "cached_uncached_greedy_output_parity": parity,
        "result_sha256": hashes,
        "performance_claim": False,
    }
