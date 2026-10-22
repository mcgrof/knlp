# SPDX-License-Identifier: MIT
"""Replay the separately frozen actual-worker quality certificates on CPU."""

import argparse
import json
from pathlib import Path
import re

from .production import ARMS, read_inputs, sha
from .paged_memory import audit_record
from .production_quality import body, score, teacher_rows
from .production_qualify import coverage
from .production_resources import audit_window, placement
from .statistics import loss_bounds, paired_binary, paired_nll


def feasibility(alpha):
    result = {
        "gsm8k_zero_loss_upper": loss_bounds(0, 1319, alpha)[1],
        "retrieval_zero_loss_upper": loss_bounds(0, 200, alpha)[1],
    }
    if (
        result["gsm8k_zero_loss_upper"] > 0.02
        or result["retrieval_zero_loss_upper"] > 0.05
    ):
        raise ValueError("Declared panel cannot meet its own zero-loss certificate")
    return result


def decisions(endpoints):
    common = all(
        endpoints[name]["native_correct"] / endpoints[name]["n"] >= 0.5
        for name in ("gsm8k", "retrieval-1024", "retrieval-8192")
    ) and all(
        endpoints[f"retrieval-{length}"]["gross_loss_one_sided_bounds"][1] <= 0.05
        and endpoints[f"nll-{length}"]["relative_one_sided_bounds"][1] <= 0.01
        for length in (1024, 8192)
    )
    return {
        "preservation": common
        and endpoints["gsm8k"]["gross_loss_one_sided_bounds"][1] <= 0.02,
        "aggregate_math": common
        and endpoints["gsm8k"]["net_one_sided_bounds"][0] >= -0.01,
    }


def resources(root, protocol, digest, arm, *, phase="numerical_quality"):
    """Recompute all sampled resource gates from the retained raw observations."""

    def read(name):
        return json.loads((root / name).read_text())

    done = read("DONE.json")
    identity = read("server/worker-identity.json")
    for key, value in {
        "arm": arm,
        "protocol_sha256": digest,
        "source_sha": protocol["source_sha"],
    }.items():
        if done.get(key) != value or identity.get(key) != value:
            raise ValueError("Collected worker identity differs")
    if (
        done["phase"] != phase
        or identity["allocation"]
        != ("equal_slots" if phase == "equal_slots" else "capacity")
        or done["worker_identity_sha256"] != sha(root / "server/worker-identity.json")
    ):
        raise ValueError("Worker collection phase or identity differs")
    shutdown = read("server-exit.json")
    if shutdown.get("error") is not None or shutdown["returncode"] not in (0, -15):
        raise ValueError("Worker invocation failed before orderly shutdown")
    cpu = read("cpu-contract.json")
    if cpu["placement"] != placement(cpu["allowed_cpus"], cpu["cpu_quota"]):
        raise ValueError("Recorded CPU placement differs from the common policy")
    if identity["cpu_affinity"] != cpu["placement"]["server"]:
        raise ValueError("Actual worker CPU placement differs")
    storage = audit_record(
        read("server/initial-storage.json"),
        arm,
        protocol["geometry"],
        protocol["persistent_budget_bytes"],
        equal_blocks=protocol["equal_slot_blocks"] if phase == "equal_slots" else None,
    )
    if phase == "numerical_quality":
        boundaries = (
            "numerical/QUALIFIED.json",
            "quality/START.json",
            "quality/DONE.json",
        )
        collectors = (("numerical", False), ("quality", True))
    elif phase == "timing":
        boundaries = ("timing/START.json", "timing/DONE.json")
        collectors = (("timing", True),)
    elif phase == "equal_slots":
        boundaries, collectors = (), ()
    else:
        raise ValueError("Unknown collected resource phase")
    for name in boundaries:
        if read(name)["client_cpu_affinity"] != cpu["placement"]["client"]:
            raise ValueError("Actual collector CPU placement differs")
    rows = [
        json.loads(line)
        for line in (root / "driver-samples.jsonl").read_text().splitlines()
    ]
    startup = read("startup-resources.json")
    computed = audit_window(
        rows, startup["start_ns"], startup["end_ns"], policy=protocol["resource_policy"]
    )
    if computed != startup or not computed["eligible"]:
        raise ValueError("Startup resource evidence is incomplete or outside limits")
    result = {
        "startup": computed,
        "cpu_placement": cpu["placement"],
        "storage": storage,
    }
    for name, enforce in collectors:
        command = read(name + "-command.json")
        if command.get("error") is not None or command["returncode"] != 0:
            raise ValueError("Fixed worker collector failed")
        computed = audit_window(
            rows,
            command["started_ns"],
            command["ended_ns"],
            policy=protocol["resource_policy"],
            enforce=enforce,
        )
        if computed != command["resources"] or not computed["eligible"]:
            raise ValueError(
                "Collector resource evidence is incomplete or outside limits"
            )
        result[name] = computed
    return result


def cohort_receipt(server, quality, label, protocol):
    """Require matching RPC/worker receipts for every cohort, not just NLL."""
    rpc = json.loads((quality / (label + "-worker.json")).read_text())
    worker = json.loads((server / label / "receipt.json").read_text())
    steps = json.loads((server / label / "steps.json").read_text())
    if rpc["results"] != [worker]:
        raise ValueError("RPC and retained worker cohort receipts differ")
    if (
        worker["label"] != label
        or worker["mode"] != "timing"
        or worker["oracle_checks"] != 0
        or worker["decision_numerical_checks"] != 0
        or worker["preemption_events"] != 0
        or worker["steps"] != len(steps)
        or not steps
        or not worker["scheduler_events"]
        or any(r["label"] != label or r["preempted_request_ids"] for r in steps)
    ):
        raise ValueError("Quality cohort execution differs from the frozen policy")
    memory = worker["memory"]
    if (
        memory["peak_reserved_since_reset"]
        > protocol["torch_allocator_reserved_cap_bytes"]
        or memory["cuda_device_used_bytes"]
        > protocol["resource_policy"]["driver_envelope_bytes"]
    ):
        raise ValueError("Worker cohort exceeded the common memory envelope")
    return worker


def build_identity(server, receipt, label):
    if len(receipt["results"]) != 1:
        raise ValueError("Build inventory must come from one actual worker")
    row = receipt["results"][0]
    if row["file"] != "build-identity-" + label + ".json":
        raise ValueError("Build identity boundary differs")
    path = server / row["file"]
    if sha(path) != row["sha256"]:
        raise ValueError("Retained build inventory changed")
    result = json.loads(path.read_text())
    if (
        not result["loaded_shared_objects_sha256"]
        or not result["generated_cache_files_sha256"]
    ):
        raise ValueError("Actual compiled-kernel identities are missing")
    if row["loaded_shared_objects"] != len(
        result["loaded_shared_objects_sha256"]
    ) or row["generated_files"] != len(result["generated_cache_files_sha256"]):
        raise ValueError("RPC and retained build inventory counts differ")
    for name in ("loaded_shared_objects_sha256", "generated_cache_files_sha256"):
        if any(
            not path or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for path, digest in result[name].items()
        ):
            raise ValueError("Invalid compiled-kernel digest")
    return result


def support_gap(root, declaration, protocol, digest):
    """Retain an unfilled SD certificate slot without reallocating its alpha."""
    if declaration["arm"] != "SD" or declaration["protocol_sha256"] != digest:
        raise ValueError("Only the prospectively declared SD support gap is allowed")
    if declaration["source_sha"] != protocol["source_sha"] or declaration[
        "phase"
    ] not in ("startup", "numerical"):
        raise ValueError("Support-gap source or failure phase differs")
    if (
        declaration["classification"]
        not in (
            "unsupported_interface",
            "numerical_qualification_failure",
            "runtime_engineering_failure",
        )
        or not declaration["reason"].strip()
    ):
        raise ValueError(
            "An explicit evidence-based support classification is required"
        )
    base = root / "SD"
    if (base / "quality").exists():
        raise ValueError("A started quality panel cannot be relabelled a support gap")
    if (base / "DONE.json").exists():
        raise ValueError("A completed SD invocation is not a support gap")
    required = {"server/launch.json", "server.log", "server-exit.json"}
    if declaration["phase"] == "numerical":
        required.add("numerical-command.json")
    if set(declaration["evidence_sha256"]) != required:
        raise ValueError(
            "Support failure requires its exact raw launch/log/exit evidence"
        )
    for name, expected in declaration["evidence_sha256"].items():
        if sha(base / name) != expected:
            raise ValueError("Retained support-failure evidence changed")
    launch = json.loads((base / "server/launch.json").read_text())
    if any(
        launch[k] != value
        for k, value in {
            "arm": "SD",
            "protocol_sha256": digest,
            "source_sha": protocol["source_sha"],
            "allocation": "capacity",
        }.items()
    ):
        raise ValueError("Unsupported worker launch identity differs")
    failed = json.loads((base / "server-exit.json").read_text())
    if not failed.get("error"):
        raise ValueError("No failed worker invocation supports this declaration")
    if declaration["phase"] == "numerical":
        failed = json.loads((base / "numerical-command.json").read_text())
        if not failed.get("error") or (base / "numerical/QUALIFIED.json").exists():
            raise ValueError("Numerical support failure was not retained")
    return {
        **declaration,
        "quality_evaluated": False,
        "alpha_reallocated": False,
        "scope": "Unqualified optimized SD implementation; no task-quality degradation or reference-quality verdict follows. Serving conclusions must be restricted to the qualified available controls.",
    }


def load_arm(root, arm, data, protocol, digest, tokenizer):
    server, quality, numerical = (
        root / arm / name for name in ("server", "quality", "numerical")
    )
    start = json.loads((quality / "START.json").read_text())
    done = json.loads((quality / "DONE.json").read_text())
    qualified_path = numerical / "QUALIFIED.json"
    qualified = json.loads(qualified_path.read_text())
    identity = sha(server / "worker-identity.json")
    resources(root / arm, protocol, digest, arm)
    if any(
        start.get(key) != value or done.get(key) != value
        for key, value in {
            "arm": arm,
            "protocol_sha256": digest,
            "source_sha": protocol["source_sha"],
            "worker_identity_sha256": identity,
            "numerical_qualification_sha256": sha(qualified_path),
        }.items()
    ):
        raise ValueError("Worker quality source/protocol/numerical identity differs")
    if (
        not qualified["passed"]
        or qualified["worker_identity_sha256"] != identity
        or qualified["protocol_sha256"] != digest
        or qualified["arm"] != arm
    ):
        raise ValueError("Actual worker was not numerically qualified")
    numerical_build = build_identity(server, qualified["build_identity"], "numerical")
    quality_build = build_identity(server, done["build_identity"], "quality")
    for name, before in numerical_build["loaded_shared_objects_sha256"].items():
        if quality_build["loaded_shared_objects_sha256"].get(name, before) != before:
            raise ValueError("A loaded kernel binary changed during quality collection")
    numerical_rows = []
    for case in qualified["cases"]:
        path = server / case["label"] / "numerics/checks.jsonl"
        if sha(path) != case["numerical_checks_sha256"]:
            raise ValueError("Retained numerical evidence changed")
        numerical_rows.extend(
            json.loads(line) for line in path.read_text().splitlines()
        )
    if coverage(numerical_rows, arm) != qualified["coverage"]:
        raise ValueError("Retained worker numerical coverage differs")
    attempts = [
        json.loads(line)
        for line in (quality / "http-attempts.jsonl").read_text().splitlines()
    ]
    expected = {s["id"]: s for s in data["samples"]["confirmation"]}
    if (
        len(attempts) != len(expected)
        or {r["id"] for r in attempts} != set(expected)
        or any(r["error"] is not None for r in attempts)
    ):
        raise ValueError("Every fixed HTTP quality item must complete once")
    raw = {r["id"]: r for r in attempts}
    for item_id, attempt in raw.items():
        if (
            attempt["request"] != body(expected[item_id], protocol["eos_token_ids"])
            or attempt["status"] != 200
            or json.loads(attempt["raw_response"]) != attempt["response"]
        ):
            raise ValueError(
                "Retained request/response bytes differ from the frozen workload"
            )
    result = {}
    for kind in ("nll", "retrieval", "gsm8k"):
        samples = [s for s in data["samples"]["confirmation"] if s["kind"] == kind]
        stored = [
            json.loads(line)
            for line in (quality / (kind + ".jsonl")).read_text().splitlines()
        ]
        if len(stored) != len(samples) or {r["id"] for r in stored} != {
            s["id"] for s in samples
        }:
            raise ValueError("Missing or repeated scored worker item")
        by_id = {r["id"]: r for r in stored}
        for first in range(0, len(samples), protocol["quality_concurrency"]):
            cohort_receipt(server, quality, f"quality-{kind}-{first:04d}", protocol)
        if kind == "nll":
            for first in range(0, len(samples), protocol["quality_concurrency"]):
                group = samples[first : first + protocol["quality_concurrency"]]
                label = f"quality-nll-{first:04d}"
                path = server / (label + "-teacher.jsonl")
                for row in teacher_rows(group, path):
                    if row != by_id[row["id"]]:
                        raise ValueError(
                            "Saved NLL differs from original teacher logits"
                        )
                for sample in group:
                    choices = raw[sample["id"]]["response"]["choices"]
                    expected_tokens = sample["token_ids"][
                        sample["length"] : sample["length"] + 129
                    ]
                    if (
                        len(choices) != 1
                        or choices[0]["token_ids"] != expected_tokens
                        or choices[0]["finish_reason"] != "length"
                    ):
                        raise ValueError(
                            "Retained HTTP teacher output differs from targets"
                        )
        else:
            for sample in samples:
                row = score(
                    sample,
                    raw[sample["id"]]["response"],
                    data,
                    tokenizer,
                    protocol["eos_token_ids"],
                )
                if row != by_id[sample["id"]]:
                    raise ValueError("Saved worker score differs from raw tokens/text")
        result[kind] = by_id
    return result


def compare(base, candidate, alpha):
    result = {}
    for kind in ("gsm8k", "retrieval"):
        for length in [None] if kind == "gsm8k" else [1024, 8192]:
            ids = [
                i
                for i, r in base[kind].items()
                if length is None or r["length"] == length
            ]
            name = kind if length is None else f"{kind}-{length}"
            result[name] = paired_binary(
                {i: base[kind][i]["correct"] for i in ids},
                {i: candidate[kind][i]["correct"] for i in ids},
                alpha,
            )
    for length in (1024, 8192):
        ids = sorted(i for i, r in base["nll"].items() if r["length"] == length)
        result[f"nll-{length}"] = paired_nll(
            [base["nll"][i]["mean_nll"] for i in ids],
            [candidate["nll"][i]["mean_nll"] for i in ids],
            alpha=alpha,
        )
    return {"endpoints": result, "objectives": decisions(result)}


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-gap", type=Path)
    args = parser.parse_args()
    protocol, data = read_inputs(args.inputs, args.protocol_sha256)
    alpha = (
        protocol["certificates"]["family_alpha"]
        / protocol["certificates"]["certificates_per_objective"]
    )
    feasible = feasibility(alpha)
    tokenizer = AutoTokenizer.from_pretrained(
        args.inputs / "tokenizer", local_files_only=True
    )
    gap = (
        support_gap(
            args.root,
            json.loads(args.support_gap.read_text()),
            protocol,
            args.protocol_sha256,
        )
        if args.support_gap is not None
        else None
    )
    measured = [arm for arm in ARMS if not (gap is not None and arm == "SD")]
    panels = {
        arm: load_arm(args.root, arm, data, protocol, args.protocol_sha256, tokenizer)
        for arm in measured
    }
    resource_records = {
        arm: resources(args.root / arm, protocol, args.protocol_sha256, arm)
        for arm in measured
    }
    if any(
        row["cpu_placement"] != resource_records[ARMS[0]]["cpu_placement"]
        or row["startup"]["uuid"] != resource_records[ARMS[0]]["startup"]["uuid"]
        for row in resource_records.values()
    ):
        raise ValueError("Worker arms did not use identical CPU placement and device")
    certificates = {
        candidate: {
            control: compare(panels[control], panels[candidate], alpha)
            for control in ("native_fi", "native_fa3")
        }
        for candidate in ("AF", "SD")
        if candidate in panels
    }
    eligibility = {
        candidate: {
            objective: all(row["objectives"][objective] for row in controls.values())
            for objective in ("preservation", "aggregate_math")
        }
        for candidate, controls in certificates.items()
    }
    route = protocol["serving_qualification_route"]
    result = {
        "protocol_sha256": args.protocol_sha256,
        "source_sha": protocol["source_sha"],
        "alpha_per_certificate": alpha,
        "declared_certificates_per_objective": 4,
        "executed_certificates_per_objective": 2 * len(certificates),
        "symmetric_support_gap": gap,
        "zero_loss_feasibility": feasible,
        "resource_records": resource_records,
        "qualified_builds": {
            arm: json.loads(
                (args.root / arm / "server/build-identity-quality.json").read_text()
            )
            for arm in measured
        },
        "certificates": certificates,
        "eligibility_against_both_native_controls": eligibility,
        "preselected_serving_route": route,
        "serving_eligible_candidates": [a for a, e in eligibility.items() if e[route]],
        "scope": "New actual-worker certificates on the exposed fixed reference panel; no useful-serving performance conclusion",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
