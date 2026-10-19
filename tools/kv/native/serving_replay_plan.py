# SPDX-License-Identifier: MIT
"""Bind an exact HTTP design to a replication's own qualification evidence.

This CPU utility never launches a server. It restores the original traces,
reruns both independent audits, and emits plans accepted by serving_trial.
The measured source must remain checked out when those plans run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .serving_trace_bundle import read_json, restore_bundle
from .serving_trial import CHECKS, IMPLEMENTATION, read_frozen, validate

ARMS = ("candidate", "native_fi", "native_fa3")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def qualified(design, production, quality):
    """Reject mixed identities and partial or failed qualification."""
    if (
        design["format"] != "fp16"
        or design["length"] not in (1024, 8192)
        or design["blocks"] != 10
        or design["arms"] != list(ARMS)
        or design["output_tokens"] != 128
    ):
        raise ValueError("expected the fixed dense Qwen3 HTTP study")
    if (
        quality["format"] != design["format"]
        or quality["length"] != design["length"]
        or quality["quality_passed"] is not True
        or quality["controlled_parity_passed"] is not True
    ):
        raise ValueError("replication quality and parity must both pass")
    if set(production) != set(ARMS) or set(quality["audit"]) != set(ARMS):
        raise ValueError("all three arms need their own qualification")
    for arm in ARMS:
        p = production[arm]
        if (
            p["production_passed"] is not True
            or p["source_sha"] != design["source_sha"]
            or p["format"] != design["format"]
            or p["arm"] != arm
            or p["cache_bytes"] != design["persistent_cache_bytes"]
            or p["protocol_sha256"] != quality["audit"][arm]["protocol_sha256"]
        ):
            raise ValueError("production and quality identities differ: " + arm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--production-tag", default="pilot")
    p.add_argument("--source-tree", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    design, catalog = read_json(a.design), read_json(a.catalog)
    if (
        sha(a.design) != catalog["measured_design_sha256"]
        or design["source_sha"] != catalog["measured_harness_sha"]
        or design["length"] != catalog["input_tokens"]
    ):
        raise ValueError("design differs from the exact public trace catalog")
    implementation = {}
    for name in IMPLEMENTATION:
        raw = subprocess.check_output(
            [
                "git",
                "-C",
                str(a.source_tree),
                "show",
                design["source_sha"] + ":tools/kv/native/" + name,
            ]
        )
        implementation[name] = hashlib.sha256(raw).hexdigest()
    restore_bundle(catalog, a.manifest, a.output)
    (a.output / "DESIGN.json").write_bytes(a.design.read_bytes())
    for module, extra, filename in (
        (
            "serving_production_audit",
            ["--tag", a.production_tag],
            "production-audit.json",
        ),
        (
            "serving_quality_audit",
            [
                "--length",
                str(design["length"]),
                "--cache-bytes",
                str(design["persistent_cache_bytes"]),
            ],
            "quality-audit.json",
        ),
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.kv.native." + module,
                "--root",
                str(a.captures),
                "--format",
                "fp16",
                "--prefix",
                a.prefix,
                "--output",
                str(a.output / filename),
                *extra,
            ],
            check=True,
        )
    production = read_json(a.output / "production-audit.json")
    quality = read_json(a.output / "quality-audit.json")
    qualified(design, production, quality)
    for arm in ARMS:
        protocol = read_json(
            a.captures
            / f"{a.prefix}-{arm}-{a.production_tag}"
            / "server/inputs/protocol.json"
        )
        if any(
            protocol["runner_files_sha256"][name] != digest
            for name, digest in implementation.items()
        ):
            raise ValueError("captured implementation differs from measured source")
    plans = []
    for arm in ARMS:
        q = {
            "passed": True,
            "checks": dict.fromkeys(CHECKS, True),
            "implementation_sha256": implementation,
            "format": "fp16",
            "arm": arm,
            "length": design["length"],
            "cache_bytes": design["persistent_cache_bytes"],
            "backend_sources": production[arm]["backend_sources"],
            "evidence": {
                name: sha(a.output / name)
                for name in ("production-audit.json", "quality-audit.json")
            },
            "scope": "This replication's raw evidence, independently audited.",
        }
        qname = f"qualification-{arm}.json"
        save(a.output / qname, q)
        for block in list(range(design["blocks"])) + ["descriptive"]:
            descriptive = block == "descriptive"
            labels = (
                design["counts"]["descriptive_cells"]
                if descriptive
                else design["cell_orders"][block]
            )
            cells = []
            for label in labels:
                name = (
                    label + ".json" if descriptive else f"block-{block:02}-{label}.json"
                )
                entry = design["traces"][name]
                item = {k: entry[k] for k in ("file", "sha256")}
                read_frozen(a.output, item)
                cells.append({"label": label, "trace": item})
            warm = design["traces"]["warmup.json"]
            plan = {
                "stage": "descriptive" if descriptive else "confirmation",
                "block": block,
                "source_sha": design["source_sha"],
                **{
                    k: q[k]
                    for k in (
                        "format",
                        "arm",
                        "length",
                        "cache_bytes",
                        "backend_sources",
                    )
                },
                "output_tokens": 128,
                "retry_count": 0,
                "qualification": {"file": qname, "sha256": sha(a.output / qname)},
                "warmup_trace": {k: warm[k] for k in ("file", "sha256")},
                **{
                    k: design["runtime_policy"][k]
                    for k in (
                        "warmup_repetitions",
                        "settle_seconds",
                        "startup_timeout_seconds",
                    )
                },
                "cells": cells,
                "frozen_design_sha256": sha(a.design),
            }
            validate(plan, q, implementation)
            name = f"plan-{block}-{arm}.json"
            save(a.output / name, plan)
            plans.append({"file": name, "sha256": sha(a.output / name)})
    save(
        a.output / "REPLICATION.json",
        {
            "qualified": True,
            "plans": plans,
            "raw_evidence_reaudited": True,
            "scope": "Exact-design replication with newly audited qualification. "
            "Follow DESIGN.json arm and cell orders; retain every failure. "
            "This CPU command performs no performance measurements.",
        },
    )
    print("Prepared 33 plans; preserve their hashes and fixed order before execution.")


if __name__ == "__main__":
    main()
