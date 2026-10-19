# SPDX-License-Identifier: MIT
"""Replay frozen complete-cohort captures; blocks are statistical units."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from .cohort_audit import verify
from .serving_statistics import paired_effect, holm

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_capture(base, planpath, arm):
    plan = read(planpath / "PLAN.json")
    assert read(base / "plan.json") == plan
    done = read(base / "DONE.json")
    assert done["capture_complete"] and done["stage"] == plan["stage"]
    assert done["excluded_from_confirmation"] == (plan["stage"] == "pilot")
    assert done["recorder_public_commit"] == plan["recorder_public_commit"]
    assert sha(base / "qualification.json") == plan["qualification_sha256"]
    qual = read(base / "qualification.json")
    assert all(qual[k] for k in ["passed", "production", "nll", "retrieval", "gsm8k"])
    assert sha(planpath / "run.py") == plan["wrapper_sha256"]
    for name, digest in plan["recorder_sha256"].items():
        assert sha(planpath / name) == digest
    inputs = base / "inputs"
    p = read(inputs / "protocol.json")
    prov = read(base / "provenance.json")
    assert sha(inputs / "protocol.json") in qual["protocol_sha256"]
    assert {"input_tokens": plan["input_tokens"], "batch": plan["batch"]} in qual[
        "workloads"
    ]
    for name, digest in p["inputs_sha256"].items():
        assert sha(inputs / name) == digest
    assert prov["source_sha"] == p["source_sha"] == plan["engine_source_sha"]
    assert prov["protocol_sha256"] == sha(inputs / "protocol.json")
    assert prov["runner_files_sha256"] == p["runner_files_sha256"]
    assert prov["backend_sources"] == p["backend_sources"]
    assert prov["scale_file_sha256"] == sha(inputs / "scales.json")
    assert prov["manifest_file_sha256"] == sha(inputs / "qwen.json.gz")
    assert prov["multiprocessing"] == "0" and prov["model_runner"] == "V2"
    assert prov["resolved_async_scheduling"] and prov["resolved_batch_queue_size"] == 2
    engine = prov["engine_config"]
    assert all(engine[k] == v for k, v in p["engine"].items())
    assert engine["kv_cache_memory_bytes"] == plan["cache_bytes"]
    assert engine["dtype"] == "float16" and engine["tensor_parallel_size"] == 1
    assert not engine["enable_prefix_caching"] and not engine["enable_chunked_prefill"]
    assert engine["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert prov["arm"] == ("k16_v8" if arm == "candidate" else "k16_v16")
    manifest = json.loads(gzip.decompress((inputs / "qwen.json.gz").read_bytes()))
    samples = [
        s
        for s in manifest["samples"]
        if s["kind"] == "nll"
        and s["length"] == plan["input_tokens"]
        and not s.get("reserve")
    ]
    # The frozen 8K transfer uses the first sixteen disjoint panel documents.
    expected = [
        s["token_ids"][: plan["input_tokens"]] for s in samples[: plan["batch"]]
    ]
    prompts = [r["prompt_token_ids"] for r in read(base / "prompts.json")]
    assert len(prompts) == plan["batch"] and prompts == expected
    measure = base / "measure"
    assert read(measure / "DONE.json") == {
        "cohorts": plan["cohorts"],
        "warmup_cohorts": plan["warmup_cohorts"],
        "output_tokens": plan["output_tokens"],
    }
    files = sorted(measure.glob("cohort-*.json"))
    assert [f.name for f in files] == [
        f"cohort-{i:03}.json" for i in range(plan["cohorts"])
    ]
    previous = 0
    rows, memories, hashes = [], read(measure / "warm-memory.json"), {}
    for i, file in enumerate(files):
        raw = read(file)
        assert raw["index"] == i and raw["start_monotonic_ns"] > previous
        assert raw["heavy_oracles_or_profiler_during_measurement"] is False
        previous = raw["end_monotonic_ns"]
        rows.append(verify(raw, prompts))
        memories.extend(raw["after_memory"])
        hashes[file.name] = sha(file)
    peak_reserved = max(m["peak_reserved_since_reset"] for m in memories)
    sampled_driver = max(m["cuda_device_used_bytes"] for m in memories)
    telemetry = list(csv.reader((base / "gpu-telemetry.csv").read_text().splitlines()))
    assert len(telemetry) >= 2 and all(len(r) == 6 for r in telemetry)
    driver_telemetry = max(float(r[1]) * 2**20 for r in telemetry)
    assert peak_reserved <= p["torch_allocator_reserved_cap_bytes"]
    assert max(sampled_driver, driver_telemetry) <= p["total_driver_envelope_bytes"]
    return {
        "valid": True,
        "cohorts": rows,
        "output_tokens_per_second": sum(r["output_tokens"] for r in rows)
        / sum(r["seconds"] for r in rows),
        "peak_reserved_bytes": peak_reserved,
        "sampled_driver_peak_bytes": max(sampled_driver, driver_telemetry),
        "driver_telemetry_samples": len(telemetry),
        "source_sha": prov["source_sha"],
        "backend_sources": prov["backend_sources"],
        "protocol_sha256": sha(inputs / "protocol.json"),
        "raw_sha256": hashes,
        "start_monotonic_ns": read(files[0])["start_monotonic_ns"],
        "end_monotonic_ns": read(files[-1])["end_monotonic_ns"],
    }


def run(root, planpath, prefix):
    plan = read(planpath / "PLAN.json")
    assert (
        plan["stage"] == "confirmation" and plan["blocks"] == len(plan["orders"]) == 10
    )
    assert all(
        sorted(order) == ["candidate", "native_fa3", "native_fi"]
        for order in plan["orders"]
    )
    assert {h["baseline"] for h in plan["primary_family"]} == {
        "native_fi",
        "native_fa3",
    }
    assert len(plan["primary_family"]) == 2
    rows, failed, pending, previous = [], [], [], 0
    for block, order in enumerate(plan["orders"]):
        arms = {}
        for arm in order:
            name = f"{prefix}-b{block:02}-{arm}"
            base = root / name
            if not (base / "DONE.json").exists():
                failures = list((root / "jobs").glob(f"*-{name}-result.json"))
                if failures and not read(failures[0])["passed"]:
                    failed.append(
                        {"block": block, "arm": arm, "reason": read(failures[0])}
                    )
                else:
                    pending.append({"block": block, "arm": arm})
                continue
            try:
                row = arm_capture(base, planpath, arm)
                assert row["start_monotonic_ns"] > previous
                previous = row["end_monotonic_ns"]
                commands = [read(f) for f in (root / "commands").glob(f"*-{name}.json")]
                assert len(commands) == 1 and commands[0]["returncode"] == 0
                jobs = list((planpath / "jobs").glob(f"*-{name}.json"))
                assert len(jobs) == 1 and read(root / "jobs" / jobs[0].name) == read(
                    jobs[0]
                )
                assert commands[0]["argv"] == read(jobs[0])["argv"]
                arms[arm] = row
            except (AssertionError, ValueError, KeyError, OSError) as exc:
                failed.append(
                    {
                        "block": block,
                        "arm": arm,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        rows.append({"block": block, "order": order, "arms": arms})
    allvalid = not failed and not pending and all(len(r["arms"]) == 3 for r in rows)
    family = []
    # Never silently analyze a reduced set of successful blocks.
    if allvalid:
        for hypothesis in plan["primary_family"]:
            baseline = hypothesis["baseline"]
            effect = paired_effect(
                [r["arms"]["candidate"]["output_tokens_per_second"] for r in rows],
                [r["arms"][baseline]["output_tokens_per_second"] for r in rows],
                higher_is_better=True,
                screen_percent=100 * hypothesis["median_gain_screen"],
            )
            effect.update(
                id=baseline, baseline=baseline, all_blocks_valid_and_complete=True
            )
            family.append(effect)
        family = holm(family)
    return {
        "complete": not pending,
        "all_blocks_valid_and_complete": allvalid,
        "failed": failed,
        "pending": pending,
        "blocks": rows,
        "primary_family": family,
        "pilot_pooled": False,
        "plan_sha256": sha(planpath / "PLAN.json"),
        "reader_sha256": sha(Path(__file__)),
        "scope": plan["performance_scope"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["root", "plan", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--prefix", required=True)
    a = p.parse_args()
    result = run(a.root, a.plan, a.prefix)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "blocks"}, indent=2))
