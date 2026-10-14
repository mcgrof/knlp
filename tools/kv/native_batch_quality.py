# SPDX-License-Identifier: MIT
"""Qualify actual native decode concurrency with request-aware fixed targets."""

import argparse
import hashlib
import json
import math
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm import SamplingParams

from tools.kv.native import runner as core
from tools.kv.native import instrument, schedule
from tools.kv.prequantized.contract import paired_nll


def install(worker):
    runner = worker.model_runner
    if hasattr(runner, "_batch_quality_original"):
        raise RuntimeError("batch teacher already installed")
    original = runner.sample
    runner._batch_quality_original = original
    runner._batch_quality_state = None

    def sample(hidden_states, input_batch, grammar_output):
        state = runner._batch_quality_state
        if state is None or grammar_output is not None or input_batch.num_draft_tokens:
            raise RuntimeError("batch teacher requires staged ordinary requests")
        ids = input_batch.req_ids
        if not set(ids) <= set(state["requests"]):
            raise RuntimeError("batch contains an unregistered teacher request")
        compute = runner.model.compute_logits

        def measured_logits(*args, **kwargs):
            logits = compute(*args, **kwargs)
            if logits.shape[0] != len(ids):
                raise RuntimeError("logit rows do not match the actual V2 request map")
            values = logits.float()
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError("nonfinite batched native logits")
            steps = [state["counts"][rid] for rid in ids]
            targets = [
                state["requests"][rid]["teacher"][step] for rid, step in zip(ids, steps)
            ]
            target_tensor = torch.tensor(
                targets, device=logits.device, dtype=torch.long
            )
            nll = -values.log_softmax(-1).gather(1, target_tensor[:, None]).squeeze(1)
            losses, top1 = nll.cpu().tolist(), values.argmax(-1).cpu().tolist()
            active = sum(
                int(n) == 1 and not bool(p)
                for n, p in zip(
                    input_batch.num_scheduled_tokens, input_batch.is_prefilling_np
                )
            )
            with open(state["path"], "a") as stream:
                for rid, step, target, loss, prediction in zip(
                    ids, steps, targets, losses, top1
                ):
                    request = state["requests"][rid]
                    row = {
                        "request_id": rid,
                        "sample_id": request["sample_id"],
                        "replica": request["replica"],
                        "step": step,
                        "target": target,
                        "nll": loss,
                        "top1": prediction,
                        "active_decode_rows": active,
                        "sample_rows": len(ids),
                        "provenance_sha256": state["provenance_sha256"],
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    state["counts"][rid] += 1
            logits.fill_(-torch.inf)
            logits[torch.arange(len(ids), device=logits.device), target_tensor] = 0
            return logits

        runner.model.compute_logits = measured_logits
        try:
            return original(hidden_states, input_batch, grammar_output)
        finally:
            runner.model.compute_logits = compute

    runner.sample = sample


def stage(worker, requests, path, provenance_sha256):
    runner = worker.model_runner
    if not hasattr(runner, "_batch_quality_original"):
        raise RuntimeError("batch teacher is not installed")
    runner._batch_quality_state = {
        "requests": requests,
        "counts": {key: 0 for key in requests},
        "path": path,
        "provenance_sha256": provenance_sha256,
    }


def finish(worker):
    runner = worker.model_runner
    runner.sample = runner._batch_quality_original
    del runner._batch_quality_original
    del runner._batch_quality_state


def cohort(engine, samples, output, label):
    params = SamplingParams(temperature=0, max_tokens=129, ignore_eos=True)
    prompts = [
        {"prompt_token_ids": sample["token_ids"][: sample["length"]]}
        for sample in samples
    ]
    ids = engine.enqueue(prompts, params, use_tqdm=False)
    requests = {
        rid: {
            "sample_id": s["id"],
            "replica": index,
            "teacher": s["token_ids"][s["length"] : s["length"] + 129],
        }
        for index, (rid, s) in enumerate(zip(ids, samples))
    }
    receipt = {
        "core": engine._study_provenance,
        "requests": requests,
        "teacher_module_sha256": core.file_hash(__file__),
        "recipe": "raw native logits before forcing fixed target IDs; V2 request-map observer; APC disabled",
    }
    instrument.save(output / (label + "-provenance.json"), receipt)
    digest = core.file_hash(output / (label + "-provenance.json"))
    path = output / (label + ".jsonl")
    engine.collective_rpc(
        partial(stage, requests=requests, path=str(path), provenance_sha256=digest)
    )
    schedule.label(label)
    try:
        values = engine.wait_for_completion(use_tqdm=False)
    finally:
        schedule.label(None)
    if len(values) != len(samples):
        raise RuntimeError("teacher request coverage is incomplete")
    for sample, result in zip(samples, values):
        if (
            list(result.prompt_token_ids) != sample["token_ids"][: sample["length"]]
            or list(result.outputs[0].token_ids)
            != sample["token_ids"][sample["length"] : sample["length"] + 129]
        ):
            raise RuntimeError("batched generated targets differ from frozen IDs")
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    for rid, request in requests.items():
        group = [r for r in raw if r["request_id"] == rid]
        if (
            len(group) != 129
            or [r["step"] for r in group] != list(range(129))
            or [r["target"] for r in group] != request["teacher"]
        ):
            raise RuntimeError("batched teacher target/step coverage mismatch")
    events = [
        json.loads(line)
        for line in (core.OUT / f"schedule-batch-quality-{core.OPTIONS.arm}-0.jsonl")
        .read_text()
        .splitlines()
    ]
    events = [e for e in events if e["label"] == label]
    full = sum(e["active_decode_sequences"] == len(samples) for e in events)
    preemptions = sum(len(e["preempted"]) for e in events)
    if len(samples) > 1 and full < 32 or preemptions:
        raise RuntimeError("teacher cohort did not sustain its claimed active batch")
    instrument.save(
        output / (label + "-schedule.json"),
        {
            "batch": len(samples),
            "full_active_decode_steps": full,
            "preemptions": preemptions,
            "resident_computed_token_peak": max(
                e["resident_computed_tokens_before"] for e in events
            ),
        },
    )
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output", "quality"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("k16_v16", "k16_v8", "k8_v8"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--length", type=int, choices=(2048, 8192))
    parser.add_argument("--utilization", type=float)
    parser.add_argument("--cache-bytes", type=int, default=48 * 1024**3)
    args = parser.parse_args()
    if (
        os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0"
        or os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1"
    ):
        parser.error("use the controlled V2 in-process engine explicitly")
    if not 1 <= args.batch <= 96:
        parser.error("batch must be between one and ninety-six")
    core.ROOT = args.inputs.resolve()
    core.OUT = args.output.resolve()
    core.OUT.mkdir(parents=True, exist_ok=True)
    core.PROTOCOL = core.read_json(core.ROOT / "protocol.json")
    variant = core.read_json(core.ROOT / "batch-quality-protocol.json")
    if variant["module_sha256"] != core.file_hash(__file__) or variant[
        "core_protocol_sha256"
    ] != core.file_hash(core.ROOT / "protocol.json"):
        raise RuntimeError("batch-quality source/protocol mismatch")
    qualified = core.read_json(core.ROOT / "qualification.json")
    if (
        qualified["protocol_sha256"] != core.file_hash(core.ROOT / "protocol.json")
        or args.arm not in qualified["qualified_modes"]
    ):
        raise RuntimeError("complete native/GSM8K qualification is required")
    core.OPTIONS = SimpleNamespace(
        arm=args.arm,
        stage="batch-quality",
        repeat=0,
        legacy=False,
        blocks=None,
        cache_bytes=args.cache_bytes if args.utilization is None else None,
        utilization=args.utilization,
    )
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        core.PROTOCOL["torch_allocator_reserved_cap_bytes"] / total
    )
    schedule.install(core.OUT / f"schedule-batch-quality-{args.arm}-0.jsonl")
    manifest = core.read_json(core.ROOT / "qwen.json.gz")
    scales = core.read_json(core.ROOT / "scales.json")["scales"]
    engine = core.make_engine(args.arm, manifest, performance=True)
    core.start_instrument(engine, scales, args.arm, core.OUT / "dispatch")
    engine.generate(
        [{"prompt_token_ids": [1, 2, 3]}],
        SamplingParams(temperature=0, max_tokens=3),
        use_tqdm=False,
    )
    core.stop_instrument(engine, core.OUT / "dispatch")
    instrument.save(
        core.OUT / "provenance.json",
        {
            "core": engine._study_provenance,
            "variant": variant,
            "variant_protocol_sha256": core.file_hash(
                core.ROOT / "batch-quality-protocol.json"
            ),
        },
    )
    engine.collective_rpc(install)
    cells = []
    try:
        for length in ([args.length] if args.length else [2048, 8192]):
            examples = sorted(
                [
                    s
                    for s in manifest["samples"]
                    if s["kind"] == "nll"
                    and s["length"] == length
                    and not s.get("reserve")
                ],
                key=lambda s: s["id"],
            )
            original_rows = [
                json.loads(line)
                for line in (args.quality / args.arm / "nll.jsonl")
                .read_text()
                .splitlines()
            ]
            if len(original_rows) != 64 * 129 or any(
                r["provenance"]["protocol_sha256"]
                != core.file_hash(core.ROOT / "protocol.json")
                for r in original_rows
            ):
                raise RuntimeError(
                    "primary native reference rows are incomplete or from another protocol"
                )
            by_key = {(r["sample_id"], r["step"]): r for r in original_rows}
            bridge = cohort(
                engine,
                [examples[0]],
                core.OUT,
                f"bridge-{length}",
            )
            if any(
                (r["target"], r["nll"], r["top1"])
                != (
                    by_key[(r["sample_id"], r["step"])]["target"],
                    by_key[(r["sample_id"], r["step"])]["nll"],
                    by_key[(r["sample_id"], r["step"])]["top1"],
                )
                for r in bridge
            ):
                raise RuntimeError("request-aware teacher B1 bridge is not bit-for-bit")
            count = math.ceil(32 / args.batch) * args.batch
            selected = [examples[i % 32] for i in range(count)]
            raw = []
            for index in range(0, count, args.batch):
                raw.extend(
                    cohort(
                        engine,
                        selected[index : index + args.batch],
                        core.OUT,
                        f"nll-{length}-{index // args.batch}",
                    )
                )
            means = {
                sample["id"]: sum(
                    r["nll"]
                    for r in raw
                    if r["sample_id"] == sample["id"] and r["step"] > 0
                )
                / sum(r["sample_id"] == sample["id"] and r["step"] > 0 for r in raw)
                for sample in examples
            }
            baseline = core.read_json(args.quality / "k16_v16/native-quality.json")
            own = core.read_json(args.quality / args.arm / "native-quality.json")
            against = lambda source: {
                r["sample_id"]: r["native_mean_nll"]
                for r in source["nll_windows"]
                if r["length"] == length
            }
            base_stats, own_stats = paired_nll(means, against(baseline)), paired_nll(
                means, against(own)
            )
            cells.append(
                {
                    "length": length,
                    "batch": args.batch,
                    "distinct_document_units": 32,
                    "requests": count,
                    "duplicate_requests_do_not_increase_sample_size": True,
                    "paired_to_baseline_B1": base_stats,
                    "paired_to_own_B1": own_stats,
                    "passed": base_stats["supported_margin"]
                    and own_stats["supported_margin"],
                    "mean_nll_by_document": means,
                }
            )
    finally:
        engine.collective_rpc(finish)
    result = {
        "cells": cells,
        "passed": all(c["passed"] for c in cells),
        "memory": engine.collective_rpc(instrument.native_memory_state),
        "qualification_scope": "fixed-target native cached NLL at verified full active concurrency; no arithmetic or retrieval task equivalence is inferred",
    }
    instrument.save(core.OUT / "decision.json", result)
    if not result["passed"]:
        raise RuntimeError("active-batch native NLL preservation gate failed")


if __name__ == "__main__":
    main()
