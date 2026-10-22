# SPDX-License-Identifier: MIT
"""Report equal-slot bytes separately from equal-budget physical capacity."""

import argparse
import json
from pathlib import Path

from .production import read_inputs, sha
from .production_analysis import resources


def analyze(root, quality, protocol, digest):
    if (
        quality["protocol_sha256"] != digest
        or quality["source_sha"] != protocol["source_sha"]
    ):
        raise ValueError("Memory comparison names a different quality invocation")
    arms = list(quality["resource_records"])
    if not {"native_fi", "native_fa3", "AF"} <= set(arms):
        raise ValueError("Both native controls and asymmetric execution are required")
    equal = {
        arm: resources(root / arm, protocol, digest, arm, phase="equal_slots")
        for arm in arms
    }
    for arm, value in equal.items():
        observed = quality["resource_records"][arm]
        if (
            value["startup"]["uuid"] != observed["startup"]["uuid"]
            or value["cpu_placement"] != observed["cpu_placement"]
        ):
            raise ValueError("Equal-slot probe changed GPU or CPU resources")
    native = equal["native_fi"]["storage"]
    return {
        "source_sha": protocol["source_sha"],
        "protocol_sha256": digest,
        "equal_slot_blocks": protocol["equal_slot_blocks"],
        "persistent_budget_bytes": protocol["persistent_budget_bytes"],
        "arms": {
            arm: {
                "equal_slots": record["storage"],
                "equal_budget": quality["resource_records"][arm]["storage"],
                "equal_slot_payload_saving_fraction_vs_native_fi": 1
                - (
                    record["storage"]["K_payload_bytes"]
                    + record["storage"]["V_payload_bytes"]
                )
                / (native["K_payload_bytes"] + native["V_payload_bytes"]),
                "equal_slot_persistent_allocation_saving_fraction_vs_native_fi": 1
                - record["storage"]["persistent_allocated_bytes"]
                / native["persistent_allocated_bytes"],
                "quality_eligibility": quality[
                    "eligibility_against_both_native_controls"
                ].get(arm),
            }
            for arm, record in equal.items()
        },
        "scope": "Allocator block bytes include KV, inline dynamic scales counted once, and separate scalar constants. Triton softmax scratch is reported outside the persistent-cache budget. More physical token slots do not establish more successful requests per second. Weights, graphs, staging and inactive pools remain separate whole-process observations.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--equal-slot-root", type=Path, required=True)
    parser.add_argument("--quality-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, _ = read_inputs(args.inputs, args.protocol_sha256)
    quality = json.loads(args.quality_analysis.read_text())
    result = analyze(args.equal_slot_root, quality, protocol, args.protocol_sha256)
    result["quality_analysis_sha256"] = sha(args.quality_analysis)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
