#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independently reject wrong routes, incomplete reads and incorrect outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from restore_common import ANCESTOR, CHUNK_BYTES, CHUNKS, distribution, write_json


def check_core(core: dict, mode: str, *, terminal: bool = False) -> None:
    """Require the requested registered direct-I/O route with no poisoned state."""
    expected = "dmabuf" if mode == "direct" else "classic"
    if core["buffer_registration_mode"] != expected:
        raise ValueError("registered route differs from the requested arm")
    if not core["is_healthy"] or core["poisoned"] or core["worker_error"]:
        raise ValueError("storage core is unhealthy")
    if (
        not core["use_odirect"]
        or core["use_uring_cmd"]
        or core["iouring_queue_depth"] != 8
    ):
        raise ValueError("unexpected I/O engine geometry")
    if core["inflight_io_count"] or core["quarantined_slot_count"]:
        raise ValueError("native operations or quarantine remain")
    if terminal and (core["locked_key_count"] or core["inflight_key_count"]):
        raise ValueError("terminal key ownership is not drained")
    native = core["io_by_request"]
    if (
        native["unanswered"]
        or native["evidence_failures"]
        or any(
            native[key]
            for key in (
                "untagged_operations",
                "dropped_rows",
                "evicted_requests",
                "malformed_rows",
                "duplicate_rows",
                "context_conflicts",
                "collection_failures",
            )
        )
    ):
        raise ValueError("native operation evidence is incomplete")


def read_delta(before: dict, after: dict, mode: str) -> dict:
    """Cross-check completed bytes and enforce the payload route independently."""
    check_core(before, mode)
    check_core(after, mode)
    delta = {
        k: v - before["payload_reads"][k] for k, v in after["payload_reads"].items()
    }
    if any(v < 0 for v in delta.values()):
        raise ValueError("payload counters moved backwards")
    if delta["submitted_padded_bytes"] != delta["completed_padded_bytes"]:
        raise ValueError("payload submission did not fully complete")
    if delta["submitted_operations"] != delta["completed_operations"]:
        raise ValueError("payload operations remain unanswered")
    wanted = "dmabuf_fixed" if mode == "direct" else "host_fixed"
    old = {
        (op["device_instance_id"], op["operation_id"], op["batch_id"])
        for op in before["io_by_request"]["operations"]
    }
    operations = [
        op
        for op in after["io_by_request"]["operations"]
        if op["direction"] == "read"
        and (op["device_instance_id"], op["operation_id"], op["batch_id"]) not in old
    ]
    for op in operations:
        if op["path"] != wanted:
            raise ValueError(f"unexpected payload route: {op['path']}")
        if (
            not op["complete"]
            or op["requested_bytes"] != CHUNK_BYTES
            or op["completed_bytes"] != CHUNK_BYTES
        ):
            raise ValueError(
                "native payload operation did not complete its exact chunk"
            )
        attempts = op["attempts"]
        if len(attempts) != 1 or attempts[0] != {
            "attempt": 0,
            "submitted_bytes": CHUNK_BYTES,
            "outcome": "completed",
            "completed_bytes": CHUNK_BYTES,
        }:
            raise ValueError("native payload submission and completion do not agree")
    route_count = len(operations)
    if route_count != delta["logical_requests"] or route_count == 0:
        raise ValueError("payload route counts do not account for every read")
    if delta["logical_bytes"] != route_count * CHUNK_BYTES:
        raise ValueError("restored bytes do not match the model chunk geometry")
    if delta["completed_padded_bytes"] != route_count * CHUNK_BYTES:
        raise ValueError("completed payload bytes differ from requested geometry")
    return delta


def expected_payload() -> tuple[str, int]:
    """Derive the entire synthetic byte fixture without the GPU implementation."""
    digest = hashlib.sha256()
    total = 0
    for i in range(CHUNKS):
        period = bytes((j * 17 + i * 29) % 251 for j in range(251))
        payload = (period * (CHUNK_BYTES // 251 + 1))[:CHUNK_BYTES]
        digest.update(payload)
        total += sum(payload)
    return digest.hexdigest(), total


def verify(root: Path, stage: str, section: str = "all") -> dict:
    """Verify every raw cell before computing latency summaries."""
    golden, checksum = expected_payload()
    plan = json.loads((root / "plan.json").read_text())
    rounds = 1 if stage == "smoke" else plan["rounds"]
    adapter_samples = 2 if stage == "smoke" else plan["adapter_samples"]
    serving_samples = 1 if stage == "smoke" else plan["serving_samples"]
    for mode in ("direct", "staged"):
        control = json.loads(
            (root / ("corruption-" + mode + "-rejected.json")).read_text()
        )
        if control["returncode"] == 0 or not control["expected_payload_rejection"]:
            raise ValueError("missing corruption rejection")
        if (
            "restored GPU consumer bytes differ"
            not in (root / ("corruption-" + mode + ".log")).read_text()
        ):
            raise ValueError(
                "corruption log does not show the payload comparison failure"
            )
    references = {}
    if section != "adapter":
        oracle = json.loads((root / "oracle/oracle.json").read_text())
        references = {
            fanout: next(row["rows"] for row in oracle if row["fanout"] == fanout)
            for fanout in (1, 5)
        }
    summary = {
        "stage": stage,
        "section": section,
        "adapter": {},
        "serving": {},
        "checks": {
            "routes_proven": True,
        },
    }
    if section != "serving":
        summary["checks"]["payloads_exact"] = True
    if section != "adapter":
        summary["checks"]["output_tokens_exact"] = True
        summary["checks"]["model_kv_digests_match"] = True
    for mode in ("direct", "staged"):
        adapters = sorted(root.glob(f"{stage}-*-{mode}-adapter"))
        serves = sorted(root.glob(f"{stage}-*-{mode}-serving"))
        if section != "serving" and len(adapters) != rounds:
            raise ValueError("missing paired adapter cells")
        if section != "adapter" and len(serves) != rounds:
            raise ValueError("missing paired serving cells")
        if section != "serving":
            ready, loads = [], []
            for path in adapters:
                result = json.loads((path / "result.json").read_text())
                check_core(result["terminal"]["core"], mode, terminal=True)
                records = json.loads((path / "samples.json").read_text())
                measured = [r for r in records if not r["warmup"]]
                if result["samples"] != adapter_samples or [
                    r["sample"] for r in measured
                ] != list(range(adapter_samples)):
                    raise ValueError("adapter sample count differs")
                for row in records:
                    delta = read_delta(
                        row["before"]["core"], row["after"]["core"], mode
                    )
                    if delta["logical_requests"] != CHUNKS:
                        raise ValueError("adapter did not restore exactly one ancestor")
                    if (
                        row["payload_sha256"] != golden
                        or row["gpu_checksum"] != checksum
                    ):
                        raise ValueError("GPU consumer payload is incorrect")
                    if not row["warmup"]:
                        loads.append(row["load_ms"])
                        ready.append(row["ready_ms"])
            summary["adapter"][mode] = {
                "load": distribution(loads),
                "ready": distribution(ready),
                "rounds": len(adapters),
            }
        if section == "adapter":
            continue
        for fanout in (1, 5):
            times, walls, reads, byte_counts, deltas = [], [], [], [], []
            for path in serves:
                producer = json.loads((path / "producer-prefix.json").read_text())
                control = json.loads(
                    (path / f"consumer-prefix-n{fanout}.json").read_text()
                )
                if len(producer["members"]) != 1 or len(control["members"]) != fanout:
                    raise ValueError("model KV integrity control is incomplete")
                expected_kv = producer["members"][0]["checksums"]["chunk_checksums"]
                if set(expected_kv) != {f"layer_{i}" for i in range(24)} or any(
                    len(hashes) != CHUNKS for hashes in expected_kv.values()
                ):
                    raise ValueError("producer model KV digest geometry is incomplete")
                for member in control["members"]:
                    if (
                        len(member["block_ids"]) != ANCESTOR // 16
                        or member["checksums"]["chunk_checksums"] != expected_kv
                    ):
                        raise ValueError(
                            "consumer model KV differs from the stored ancestor"
                        )
                records = json.loads((path / "consumer.json").read_text())
                if [
                    r["sample"]
                    for r in records
                    if r["fanout"] == fanout and not r["warmup"]
                ] != list(range(serving_samples)):
                    raise ValueError("serving sample sequence is incomplete")
                for row in records:
                    if row["fanout"] != fanout:
                        continue
                    delta = read_delta(
                        row["before"]["l2_adapters"][0]["core"],
                        row["after"]["l2_adapters"][0]["core"],
                        mode,
                    )
                    if not CHUNKS <= delta["logical_requests"] <= fanout * CHUNKS:
                        raise ValueError(
                            "serving ancestor read count is outside the bounded workload"
                        )
                    if len(row["rows"]) != fanout:
                        raise ValueError("serving family lost a member")
                    for actual, expected in zip(
                        row["rows"], references[fanout], strict=True
                    ):
                        if actual["token_ids"] != expected["token_ids"]:
                            raise ValueError(
                                "storage-backed output tokens differ from recomputation"
                            )
                        if not all(
                            math.isfinite(value["logprob"])
                            for value in (actual, expected)
                        ):
                            raise ValueError("chosen-token likelihood is not finite")
                        error = abs(actual["logprob"] - expected["logprob"])
                        if error > 0.02:
                            raise ValueError(
                                "chosen-token likelihood exceeds the fixed 0.02 bound"
                            )
                        if actual["num_cached_tokens"] < ANCESTOR:
                            raise ValueError(
                                "ancestor was recomputed instead of restored/reused"
                            )
                        if not row["warmup"]:
                            times.append(actual["ttft_ms"])
                            deltas.append(error)
                    if not row["warmup"]:
                        walls.append(row["batch_wall_ms"])
                        reads.append(delta["logical_requests"])
                        byte_counts.append(delta["completed_padded_bytes"])
                terminal = json.loads((path / "terminal.json").read_text())[
                    "storage_manager"
                ]
                check_core(terminal["l2_adapters"][0]["core"], mode, terminal=True)
                if terminal["l1_manager"]["external_io_lease_count"]:
                    raise ValueError("external L1 leases remain")
            if not times:
                raise ValueError("no serving latency samples")
            summary["serving"][f"{mode}-n{fanout}"] = {
                "ttft": distribution(times),
                "batch_wall": distribution(walls),
                "logical_reads_per_family": reads,
                "payload_bytes_per_family": byte_counts,
                "maximum_logprob_delta": max(deltas),
            }
    suffix = "" if section == "all" else "-" + section
    write_json(root / (stage + suffix + "-verified.json"), summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=("smoke", "measured"), default="measured")
    parser.add_argument(
        "--section",
        choices=("all", "adapter", "serving", "coldburst", "family", "retained"),
        default="all",
    )
    args = parser.parse_args()
    if args.section == "retained":
        from retained_verify import verify_retained

        result = verify_retained(args.root)
        write_json(args.root / "retained-verified.json", result)
    elif args.section == "family":
        from family_verify import verify_family

        result = verify_family(args.root)
        write_json(args.root / "family-verified.json", result)
    elif args.section == "coldburst":
        from coldburst_verify import verify_coldburst

        result = verify_coldburst(args.root)
        write_json(args.root / "coldburst-verified.json", result)
    else:
        result = verify(args.root, args.stage, args.section)
    print(json.dumps(result, indent=2))
