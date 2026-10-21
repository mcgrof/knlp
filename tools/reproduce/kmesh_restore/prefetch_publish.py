#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export verified fixed-arrival aggregates without private capture metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from prefetch_analysis import LEADS, verify_clock
from prefetch_campaign import verify_capture
from prefetch_verify import ARENA, PAYLOAD, core
from restore_common import (
    ANCESTOR,
    CHUNK,
    CHUNKS,
    MODEL,
    RETAINED_DEMAND_COMMIT,
    REVISION,
    VLLM_COMMIT,
    write_json,
)

HARNESS = "tools/reproduce/kmesh_restore/"
PHASES = ("qualification", "pilot", "confirmation")
MEMORY_COUNTS = (
    "engine_configured_bytes",
    "engine_block_pool_bytes",
    "external_configured_bytes",
    "engine_valid_full_block_bytes",
    "engine_live_read_blocks",
    "external_used_bytes",
    "external_backed_bytes",
    "registered_host_arena_bytes",
    "external_gpu_arena_bytes",
)
BOUNDS = (
    "external_payload_byte_seconds_lower_bound",
    "external_payload_byte_seconds_upper_bound",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    require(
        type(value) in (int, float) and math.isfinite(value), "invalid public scalar"
    )
    return value


def count(value):
    require(type(value) is int and value >= 0, "invalid public count")
    return value


def flag(value):
    require(type(value) is bool, "invalid public flag")
    return value


def load(path):
    return json.loads(path.read_text())


def git_bytes(repository, *args):
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL
    )


def digest_map(values, *, native=False):
    require(isinstance(values, dict) and values, "missing hash manifest")
    result = {}
    for name, value in values.items():
        require(isinstance(name, str), "invalid hash label")
        label = Path(name).name if native else name
        pattern = r"[A-Za-z0-9_+.-]+\.so" if native else r"[A-Za-z0-9_+.-]+"
        require(
            re.fullmatch(pattern, label) is not None
            and label not in (".", "..")
            and "\\" not in name,
            "invalid public hash label",
        )
        require(label not in result, "native basenames are ambiguous")
        require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid SHA256 receipt",
        )
        result[label] = value
    return result


def authenticate(repository, commit, provenance):
    require(
        isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
        "harness commit must be a full lowercase commit ID",
    )
    require(
        git_bytes(repository, "cat-file", "-t", commit).strip() == b"commit",
        "not a commit",
    )
    names = git_bytes(repository, "ls-tree", "-r", "--name-only", commit, "--", HARNESS)
    paths = [
        name
        for name in names.decode().splitlines()
        if name.startswith(HARNESS)
        and Path(name).parent.as_posix() == HARNESS.rstrip("/")
        and name.endswith(".py")
    ]
    expected = {}
    for path in paths:
        source = git_bytes(repository, "show", f"{commit}:{path}")
        expected[Path(path).name] = hashlib.sha256(source).hexdigest()
    require(
        expected and provenance["harness"] == expected,
        "captured harness differs from pinned Git tree",
    )
    digest_map(expected)
    model_files = json.loads(
        git_bytes(repository, "show", f"{commit}:{HARNESS}model-files.json")
    )
    require(
        provenance["model_files"] == model_files,
        "captured model hashes differ from pinned manifest",
    )
    return expected, digest_map(model_files)


def public_runtime(runtime):
    result = {}
    patterns = {
        "torch": (
            r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:(?:a|b|rc)[0-9]+)?"
            r"(?:\.dev[0-9]+)?(?:\+(?:cpu|cu[0-9]+|rocm[0-9.]+|(?:git|g)?[0-9a-f]{7,40}))?"
        ),
        "cuda": r"[0-9]+\.[0-9]+(?:\.[0-9]+)?",
    }
    for key in ("torch", "cuda"):
        value = runtime[key]
        require(
            isinstance(value, str) and re.fullmatch(patterns[key], value) is not None,
            "invalid runtime version",
        )
        result[key] = value
    capability = runtime["capability"]
    require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(type(x) is int and x >= 0 for x in capability),
        "invalid compute capability",
    )
    result["compute_capability"] = capability[:]
    return result


def public_selection(selection):
    lead = count(selection["selected_lead_ms"])
    require(lead in LEADS, "invalid selected lead")
    readiness = {}
    for mode in ("staged", "direct"):
        readiness[mode] = {
            str(lead): {
                key: count(
                    selection["readiness_by_path_and_lead_ms"][mode][str(lead)][key]
                )
                for key in (
                    "ready",
                    "groups",
                    "warm_requests",
                    "warm_skipped_due_to_arrival",
                )
            }
            for lead in LEADS
        }
    return {
        "selected_lead_ms": lead,
        "readiness_rule_satisfied": flag(selection["readiness_rule_satisfied"]),
        "fallback_to_tested_limit": flag(selection["fallback_to_tested_limit"]),
        "readiness_by_path_and_lead_ms": readiness,
        "process_blocks_per_path": count(selection["process_blocks_per_path"]),
    }


def distribution(value):
    return {
        "count": count(value["count"]),
        **{key: number(value[key]) for key in ("mean_ms", "p50_ms", "p95_ms")},
    }


def interval(value):
    bounds = value["ci95"]
    require(
        isinstance(bounds, list) and len(bounds) == 2, "invalid confidence interval"
    )
    return {"estimate": number(value["estimate"]), "ci95": [number(x) for x in bounds]}


def public_confirmation(analysis):
    endpoints = ("a_deadline_first_token_ms", "deadline_group_completion_ms")
    cells = {}
    for mode in ("staged", "direct"):
        for fanout in (4, 8):
            name = f"{mode}-n{fanout}"
            cell = analysis["cells"][name]
            metrics = {}
            for endpoint in endpoints:
                raw = cell["endpoints"][endpoint]
                metrics[endpoint] = {
                    "demand": distribution(raw["demand"]),
                    "early": distribution(raw["early"]),
                    **{
                        key: interval(raw[key])
                        for key in ("improvement_ms", "improvement_percent")
                    },
                }
            qualifying = cell["qualifying_endpoints"]
            require(
                isinstance(qualifying, list)
                and all(x in endpoints for x in qualifying),
                "invalid endpoint label",
            )
            cells[name] = {
                **{
                    key: count(cell[key])
                    for key in (
                        "process_blocks",
                        "measured_groups_per_arm",
                        "warmup_groups_per_arm",
                        "early_ready_by_scheduled_arrival",
                        "early_warm_skipped_due_to_arrival",
                    )
                },
                "warm_request_counts": {
                    arm: count(cell["warm_request_counts"][arm])
                    for arm in ("demand", "early")
                },
                "endpoints": metrics,
                **{
                    key: {
                        arm: distribution(cell[key][arm]) for arm in ("demand", "early")
                    }
                    for key in ("all_request_deadline_first_token", "submission_delay")
                },
                "engineering_rule_satisfied": flag(cell["engineering_rule_satisfied"]),
                "qualifying_endpoints": qualifying[:],
            }
    return {
        "selected_lead_ms": count(analysis["selected_lead_ms"]),
        "bootstrap_draws": count(analysis["bootstrap_draws"]),
        "bootstrap_seed": count(analysis["bootstrap_seed"]),
        "cells": cells,
    }


def read_counts(before, after):
    old, new = before["payload_reads"], after["payload_reads"]
    return {
        "backend_chunk_reads": count(new["logical_requests"] - old["logical_requests"]),
        "backend_payload_bytes": count(new["logical_bytes"] - old["logical_bytes"]),
    }


def public_memory(memory):
    result = {key: count(memory[key]) for key in MEMORY_COUNTS}
    result.update({key: number(memory[key]) for key in BOUNDS})
    for key in (
        "diagnostic_engine_peak_occupied_bytes",
        "diagnostic_external_observed_peak_bytes",
    ):
        result[key] = None if memory[key] is None else count(memory[key])
    return result


def elapsed(row, start, end):
    if row[start] is None and row[end] is None:
        return None
    value = number((row[end] - row[start]) / 1e6)
    require(value >= 0, "negative preparation duration")
    return value


def public_sample(row, phase, index, *, unused=False):
    require(
        row["mode"] in ("staged", "direct") and row["fanout"] in (4, 8),
        "invalid sample stratum",
    )
    require(
        row["arm"]
        in ("demand", "early", "declaration", "external_resident", "engine_resident"),
        "invalid sample arm",
    )
    require(type(row["sample"]) is int and row["sample"] >= -3, "invalid sample index")
    lead = count(row["lead_ms"])
    require(lead in LEADS, "invalid sample lead")
    placement = row["placement"]
    sample = {
        "phase": phase,
        "mode": row["mode"],
        "fanout": count(row["fanout"]),
        "block_index": index,
        "arm": row["arm"],
        "lead_ms": lead,
        "sample": row["sample"],
        "warmup": flag(row["warmup"]),
        **read_counts(core(row["before"]), core(row["after"])),
        "placement": (
            None
            if placement is None
            else {
                **read_counts(
                    placement["native"]["before"], placement["native"]["after"]
                ),
                "duration_ms": number(
                    (placement["ready_ns"] - placement["start_ns"]) / 1e6
                ),
                "warm_request_count": len(placement.get("warm_rows", [])),
            }
        ),
        "preparation": {
            "external_completion_ms": elapsed(
                row, "prepare_start_ns", "external_ready_ns"
            ),
            "engine_warm_ms": elapsed(row, "warm_submit_ns", "warm_done_ns"),
            "warm_request_count": len(row["warm_rows"]),
            "warm_decode_tokens": sum(len(w["token_ids"]) for w in row["warm_rows"]),
            "ready_by_scheduled_arrival": flag(row["ready_by_scheduled_arrival"]),
            "warm_skipped_due_to_arrival": flag(row["warm_skipped_due_to_arrival"]),
        },
        "memory": public_memory(row["memory"]),
        "ownership_drained": True,
    }
    if unused:
        require(not row["rows"], "unused hint has children")
        sample["child_count"] = 0
        sample["unused_hints_charged"] = 1
    else:
        metrics = verify_clock(row)
        sample["clock"] = {
            **{
                key: number(metrics[key])
                for key in (
                    "a_deadline_first_token_ms",
                    "deadline_group_completion_ms",
                    "a_submission_first_token_ms",
                    "actual_group_completion_ms",
                    "workflow_completion_ms",
                )
            },
            **{
                key: [number(value) for value in metrics[key]]
                for key in (
                    "submission_delays_ms",
                    "all_request_deadline_first_token_ms",
                    "all_request_submission_first_token_ms",
                )
            },
        }
        sample["child_count"] = len(row["rows"])
    return sample


def _build(root, *, harness_commit, repository):
    captures = {}
    baseline = baseline_runtime = None
    for phase in PHASES:
        path = root / phase
        result, rows = verify_capture(path)
        require(
            result == load(path / "RESULT.json"),
            "claimed result differs from complete raw verification",
        )
        require(
            result["phase"] == phase and result["accepted"] is True,
            "missing accepted campaign phase",
        )
        provenance, runtime = load(path / "provenance.json"), load(
            path / "runtime.json"
        )
        if baseline is None:
            baseline, baseline_runtime = provenance, runtime
        require(provenance == baseline, "raw provenance changed across phases")
        require(runtime == baseline_runtime, "raw runtime changed across phases")
        config = load(path / "CONFIG.json")
        require(
            config["phase"] == phase
            and config["arena_bytes"] == ARENA
            and config["decode_tokens"] == 8
            and config["max_num_seqs"] == 9
            and all(
                config[k] is True
                for k in ("external_singleflight", "family_prefetch", "retained_fill")
            ),
            "configuration differs from fixed profile",
        )
        captures[phase] = (path, result, rows, config)
    pins = {"vllm": VLLM_COMMIT, "lmcache": RETAINED_DEMAND_COMMIT}
    for component, commit in pins.items():
        require(
            baseline["sources"][component] == {"commit": commit, "clean": True},
            "component source differs from pinned profile",
        )
    harness, model_files = authenticate(repository, harness_commit, baseline)
    native = digest_map(baseline["native"], native=True)
    selection = public_selection(captures["pilot"][1]["selection"])
    require(
        captures["confirmation"][3]["selected_lead_ms"]
        == selection["selected_lead_ms"],
        "confirmation changed frozen pilot lead",
    )
    samples, unused, validation = [], [], {}
    for phase, (path, result, rows, _) in captures.items():
        indexes = {}
        for block in result["verified_blocks"]:
            name = block["block_id"]
            match = re.fullmatch(rf"{phase}-(staged|direct)-n(4|8)-b([0-9]{{2}})", name)
            require(
                match is not None and name not in indexes,
                "invalid verified block label",
            )
            indexes[name] = int(match[3])
            if phase == "qualification":
                unused.append(
                    public_sample(
                        load(path / name / "unused.json"),
                        phase,
                        indexes[name],
                        unused=True,
                    )
                )
        samples.extend(
            public_sample(row, phase, indexes[row["block_id"]]) for row in rows
        )
        validation[phase] = {
            "accepted": True,
            "engine_process_blocks": count(result["blocks"]),
            "trial_groups_including_warmups": len(rows),
            "measured_trial_groups": sum(not row["warmup"] for row in rows),
            **{
                key: sum(count(block[key]) for block in result["verified_blocks"])
                for key in (
                    "child_continuations_checked",
                    "warm_continuations_checked",
                    "chosen_logprob_comparisons",
                )
            },
            "maximum_logprob_delta": max(
                number(block["maximum_logprob_delta"])
                for block in result["verified_blocks"]
            ),
            "overlap_joins_by_path": {
                mode: count(result["overlap_joins_by_path"][mode])
                for mode in ("staged", "direct")
            },
        }
    return {
        "schema": "kmesh-fixed-arrival-prefetch-v1",
        "sources": {"knlp_harness_commit": harness_commit, **pins},
        "model": {"id": MODEL, "revision": REVISION, "files_sha256": model_files},
        "harness_sha256": harness,
        "native_sha256_by_basename": native,
        "runtime": public_runtime(baseline_runtime),
        "configuration": {
            "ancestor_tokens": ANCESTOR,
            "chunk_tokens": CHUNK,
            "ancestor_chunks": CHUNKS,
            "ancestor_bytes": PAYLOAD,
            "engine_configured_bytes": ARENA,
            "external_configured_bytes": ARENA,
            "child_decode_tokens": 8,
            "warm_decode_tokens": 1,
            "arrival_offset_ms": 250,
        },
        "pilot": {"selection": selection},
        "confirmation": {
            "confirmation": public_confirmation(
                captures["confirmation"][1]["confirmation"]
            )
        },
        "samples": samples,
        "unused_hint_controls": unused,
        "validation": validation,
        "limits": [
            "Offline consistency checks cannot independently authenticate hardware execution.",
            "Backend payload operations are not physical NVMe command traces.",
            "Diagnostic external occupancy is a sampled lower bound on the peak; byte-seconds are bounded estimates.",
            "Confidence intervals are nominal per endpoint; ten process clusters limit tail evidence.",
            "Observed pilot readiness counts do not certify a readiness probability or service-level objective.",
            "Preparation outputs are reported separately and are excluded from child timing endpoints.",
        ],
    }


def build(root: Path, *, harness_commit: str, repository: Path) -> dict:
    """Recompute all saved phases and export only declared scientific fields."""
    try:
        result = _build(
            Path(root), harness_commit=harness_commit, repository=Path(repository)
        )
        json.dumps(result, allow_nan=False)
        return result
    except (
        KeyError,
        TypeError,
        IndexError,
        OSError,
        OverflowError,
        subprocess.CalledProcessError,
    ) as error:
        raise ValueError("incomplete or unauthenticated publication capture") from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output,
        build(
            args.capture, harness_commit=args.harness_commit, repository=args.repository
        ),
    )
