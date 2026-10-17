#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the complete fork-accounting matrix against the frozen experiment."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES = {"historical": "profile.json", "current": "profile-current.json"}


def load_profile(name):
    return json.loads((ROOT / PROFILES[name]).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(result, profile_name="historical"):
    reference = json.loads((ROOT / "reference.json").read_text())
    profile = load_profile(profile_name)
    require(result.get("schema_version") == 1, "missing result schema")
    require(
        result.get("measurement") == "cpu-allocator-metadata",
        "result is not the allocator experiment",
    )
    for key in ("seed", "block_size", "bytes_per_token"):
        require(result.get(key) == reference[key], f"wrong {key}")
    for key, field in (
        ("python", "python"),
        ("torch", "torch_version"),
        ("vllm", "vllm_version"),
    ):
        require(result.get(field) == profile["runtime_versions"][key], f"wrong {key}")
    scenarios = result.get("scenarios", {})
    require(set(scenarios) == set(reference["scenarios"]), "incomplete tree matrix")
    for name, expected_cases in reference["scenarios"].items():
        cases = scenarios[name]
        require(
            set(cases) == set(expected_cases) | {"members"},
            f"incomplete scenarios: {name}",
        )
        for case, expected in expected_cases.items():
            value = cases[case]
            summary, rows = value["summary"], value["rows"]
            require(
                summary == expected["summary"], f"changed accounting: {name}/{case}"
            )
            require(len(rows) == cases["members"], "missing tree member")
            for key, field in (("hits", "hit_tokens"), ("allocated", "allocated")):
                require(
                    [row[field] for row in rows] == expected[key],
                    f"changed {key}: {name}/{case}",
                )
            require(
                sum(r["written_full_blocks"] + r["partial_blocks"] for r in rows)
                == summary["written_blocks"],
                "row writes disagree with summary",
            )
            require(
                sum(r["hit_tokens"] for r in rows) == summary["hit_tokens"],
                "row hits disagree with summary",
            )
            require(
                sum(r["duplicated_boundary_tokens"] for r in rows)
                == summary["duplicated_boundary_tokens"],
                "row boundary tokens disagree with summary",
            )
            for row in rows:
                if row["allocated"]:
                    ids = row["block_ids"]
                    require(
                        len(ids) == (row["tokens"] + 15) // 16
                        and len(set(ids)) == len(ids)
                        and all(type(i) is int and i > 0 for i in ids),
                        "invalid allocator block table",
                    )
            if case == "concurrent_cached":
                by_label = {r["label"]: r for r in rows}
                for row in rows[1:]:
                    parent = by_label[row["label"].rsplit(".", 1)[0]]
                    full = row["divergence"] // 16
                    require(
                        row["block_ids"][:full] == parent["block_ids"][:full],
                        "ancestor blocks are not aliased",
                    )
                    siblings = [
                        other
                        for other in rows[1:]
                        if other["label"].rsplit(".", 1)[0]
                        == row["label"].rsplit(".", 1)[0]
                        and other is not row
                    ]
                    for other in siblings:
                        require(
                            set(row["block_ids"][full:]).isdisjoint(
                                other["block_ids"][full:]
                            ),
                            "siblings share a writable tail",
                        )
    require(
        result.get("loss_window_sweep") == reference["loss_window_sweep"],
        "changed or incomplete eviction sweep",
    )
    return {
        "status": "pass",
        "profile": profile_name,
        "measurement": "cpu-allocator-metadata",
        "tree_scenarios": sum(len(cases) for cases in reference["scenarios"].values()),
        "eviction_points": len(reference["loss_window_sweep"]["points"]),
        "gpu_inference_measured": False,
        "throughput_measured": False,
        "hbm_read_traffic_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="historical")
    args = parser.parse_args()
    report = verify(json.loads(args.result.read_text()), args.profile)
    report["result_sha256"] = hashlib.sha256(args.result.read_bytes()).hexdigest()
    report["reference_sha256"] = hashlib.sha256(
        (ROOT / "reference.json").read_bytes()
    ).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
