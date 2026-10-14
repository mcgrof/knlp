# SPDX-License-Identifier: MIT
"""Native packed-cache qualification with explicit caller-owned inputs."""

import argparse, gzip, hashlib, json, os, subprocess, time
from pathlib import Path
from functools import partial
import torch
from vllm import LLM, SamplingParams
from . import instrument as callbacks
from . import schedule, gsm8k
from .instrument import save

ROOT = None
OUT = None
OPTIONS = None
PROTOCOL = None


def read_json(path):
    path = Path(path)
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    return json.loads(raw)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_engine(arm, manifest, performance=False):
    mode = {"k16_v16": "auto", "k16_v8": ("auto", "fp8_e4m3"), "k8_v8": "fp8_e4m3"}[arm]
    engine_config = dict(PROTOCOL["engine"])
    engine_config.update(enable_prefix_caching=not performance, kv_cache_dtype=mode)
    if OPTIONS.utilization is not None:
        engine_config["gpu_memory_utilization"] = OPTIONS.utilization
    if OPTIONS.legacy:
        engine_config["max_num_seqs"] = 8
    if OPTIONS.blocks is not None:
        engine_config["num_gpu_blocks_override"] = OPTIONS.blocks
    if OPTIONS.cache_bytes is not None:
        engine_config["kv_cache_memory_bytes"] = OPTIONS.cache_bytes
    import vllm
    import flashinfer

    versions = {
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "flashinfer": flashinfer.__version__,
    }
    sources = {
        name: subprocess.check_output(
            ["git", "-C", str(ROOT / name), "rev-parse", "HEAD"], text=True
        ).strip()
        for name in ("vllm", "flashinfer")
    }
    if sources != PROTOCOL["backend_sources"]:
        raise RuntimeError("backend source pin mismatch")
    if torch.__version__ != PROTOCOL["torch"]:
        raise RuntimeError("Torch pin mismatch")
    runner_files = {
        p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    if runner_files != PROTOCOL["runner_files_sha256"]:
        raise RuntimeError("runner code changed after protocol freeze")
    if not Path(vllm.__file__).resolve().is_relative_to((ROOT / "vllm").resolve()):
        raise RuntimeError("loaded vLLM package is outside the pinned source checkout")
    if (
        not Path(flashinfer.__file__)
        .resolve()
        .is_relative_to((ROOT / "flashinfer").resolve())
    ):
        raise RuntimeError(
            "loaded FlashInfer package is outside the pinned source checkout"
        )
    for name, expected_hash in PROTOCOL["inputs_sha256"].items():
        if file_hash(ROOT / name) != expected_hash:
            raise RuntimeError("frozen input bytes changed: " + name)
    geometry = read_json(ROOT / "geometry.json")
    if geometry["passed"] is not True:
        raise RuntimeError("allocation/writer/read gate is closed")
    os.environ["KNLP_NATIVE_TORCH_LIMIT_BYTES"] = str(
        PROTOCOL["torch_allocator_reserved_cap_bytes"]
    )
    os.environ["KNLP_NATIVE_ALLOC_TRACE"] = str(
        OUT / f"allocator-{arm}-{OPTIONS.stage}-{OPTIONS.repeat}.jsonl"
    )
    engine = LLM(
        model=manifest["model"],
        revision=manifest["model_revision"],
        tokenizer_revision=manifest["tokenizer_revision"],
        **engine_config,
    )
    engine.apply_model(
        partial(
            callbacks.ensure_schedule_observer,
            path=str(
                OUT / f"schedule-{OPTIONS.stage}-{OPTIONS.arm}-{OPTIONS.repeat}.jsonl"
            ),
        )
    )
    engine._study_provenance = {
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "manifest_file_sha256": file_hash(ROOT / "qwen.json.gz"),
        "scale_file_sha256": file_hash(ROOT / "scales.json"),
        "protocol_sha256": file_hash(ROOT / "protocol.json"),
        "source_sha": PROTOCOL["source_sha"],
        "runner_files_sha256": runner_files,
        "backend_sources": sources,
        "versions": versions,
        "arm": arm,
        "engine_config": engine_config,
        "loaded_module_paths": {
            "vllm": str(Path(vllm.__file__).resolve()),
            "flashinfer": str(Path(flashinfer.__file__).resolve()),
        },
        "geometry_sha256": file_hash(ROOT / "geometry.json"),
        "model_runner": "V2",
        "multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
    }
    # JSON rows must compare exactly after a fresh-process resume, including
    # mixed dtype tuples which JSON represents as lists.
    engine._study_provenance = json.loads(
        json.dumps(engine._study_provenance, allow_nan=False)
    )
    return engine


def start_instrument(llm, scales, arm, directory):
    directory.mkdir(parents=True, exist_ok=True)
    llm.apply_model(
        partial(
            callbacks.install,
            scales=scales,
            arm=arm,
            trace_dir=str(directory),
            packed_expectation=PROTOCOL.get("packed_execution"),
        )
    )


def stop_instrument(llm, directory):
    llm.apply_model(partial(callbacks.finish_profile, trace_dir=str(directory)))


def native_nll(llm, sample, path, repeat=False):
    llm.reset_prefix_cache()
    length = sample["length"]
    count = sample.get("diagnostic_tokens", 128) + 1
    teacher = sample["token_ids"][length : length + count]
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text().splitlines()]
    chosen = [r for r in existing if r["sample_id"] == sample["id"]]
    if chosen:
        if (
            len(chosen) != count
            or [r["step"] for r in chosen] != list(range(count))
            or [r["target"] for r in chosen] != teacher
            or any(r.get("provenance") != llm._study_provenance for r in chosen)
        ):
            raise RuntimeError("incomplete or mismatched native resume rows")
        return chosen[1:]
    params = SamplingParams(
        temperature=0,
        max_tokens=count,
        ignore_eos=True,
        extra_args={
            "kv_study": {
                "teacher": teacher,
                "sample_id": sample["id"],
                "path": str(path),
                "provenance": llm._study_provenance,
            }
        },
    )
    study = {
        "teacher": teacher,
        "sample_id": sample["id"],
        "path": str(path),
        "provenance": llm._study_provenance,
    }
    llm.apply_model(partial(callbacks.stage_teacher, study=study))
    try:
        output = llm.generate(
            [{"prompt_token_ids": sample["token_ids"][:length]}], params, use_tqdm=False
        )[0]
    finally:
        llm.apply_model(callbacks.clear_teacher)
    if list(output.outputs[0].token_ids) != teacher:
        raise RuntimeError("native teacher forcing produced different tokens")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    chosen = [r for r in rows if r["sample_id"] == sample["id"]]
    if len(chosen) != count or [r["step"] for r in chosen] != list(range(count)):
        raise RuntimeError("native cached-decode token coverage incomplete")
    return chosen[1:]


def parse_native(text):
    import re

    text = re.split(r"<\|(?:im_end|end)\|>", text)[0]
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    return text.strip(" \n\t.\"'`")


def bridge(args, manifest, scales):
    arm = args.arm
    directory = OUT / arm
    directory.mkdir(parents=True, exist_ok=True)
    trace_dir = directory / "full" if args.full else directory
    trace_dir.mkdir(parents=True, exist_ok=True)
    llm = make_engine(arm, manifest)
    save(trace_dir / "provenance.json", llm._study_provenance)
    llm.apply_model(
        partial(
            callbacks.install,
            scales=scales,
            arm=arm,
            trace_dir=str(trace_dir),
            packed_expectation=PROTOCOL.get("packed_execution"),
        )
    )
    llm.apply_model(callbacks.require_finite)
    sample = next(s for s in manifest["samples"] if s["kind"] == "nll")
    short = {**sample, "id": "dispatch-smoke", "length": 32, "diagnostic_tokens": 3}
    native_nll(llm, short, trace_dir / "smoke.jsonl")
    llm.apply_model(partial(callbacks.finish_profile, trace_dir=str(trace_dir)))
    expected = {}
    source_rows = ROOT / "reference" / f"{arm}.jsonl"
    for line in source_rows.read_text().splitlines():
        row = json.loads(line)
        expected[row["sample_id"]] = row
    historical_rows = {}
    historical_path = ROOT / "historical-native" / arm / "nll.jsonl"
    if historical_path.exists():
        for line in historical_path.read_text().splitlines():
            row = json.loads(line)
            if row["step"] > 0:
                historical_rows.setdefault(row["sample_id"], []).append(row)
    comparisons = []
    retrieval = []
    tokenizer = llm.get_tokenizer()
    for length in (2048, 8192):
        windows = [
            s
            for s in manifest["samples"]
            if s["kind"] == "nll" and s["length"] == length and not s.get("reserve")
        ][: 32 if args.full else 16]
        for index, sample in enumerate(windows):
            rows = native_nll(llm, sample, directory / "nll.jsonl")
            if index == 0:
                repeated = native_nll(
                    llm, sample, trace_dir / f"repeat-{length}.jsonl", True
                )
                if rows != repeated:
                    raise RuntimeError("native cached-decode repeat is not exact")
            control = expected[sample["id"]]
            errors = [abs(a["nll"] - b) for a, b in zip(rows, control["token_nll"])]
            agreement = (
                sum(a["top1"] == b for a, b in zip(rows, control["top1_ids"])) / 128
            )
            item = {
                "sample_id": sample["id"],
                "length": length,
                "mean_absolute_target_nll_error": sum(errors) / 128,
                "max_target_nll_error": max(errors),
                "top1_agreement": agreement,
                "native_mean_nll": sum(r["nll"] for r in rows) / 128,
                "qdq_mean_nll": control["mean_nll"],
            }
            if sample["id"] in historical_rows:
                previous = historical_rows[sample["id"]]
                if len(previous) != 128:
                    raise RuntimeError("historical native token coverage incomplete")
                historical_errors = [
                    abs(a["nll"] - b["nll"]) for a, b in zip(rows, previous)
                ]
                item["historical_native_bridge"] = {
                    "mean_absolute_target_nll_error": sum(historical_errors) / 128,
                    "max_target_nll_error": max(historical_errors),
                    "top1_agreement": sum(
                        a["top1"] == b["top1"] for a, b in zip(rows, previous)
                    )
                    / 128,
                    "tolerance": {
                        "mean_absolute_target_nll_error": 0.01,
                        "max_target_nll_error": 0.1,
                        "top1_agreement": 0.99,
                    },
                }
            comparisons.append(item)
            print(json.dumps({"arm": arm, "bridge_nll": item}), flush=True)
        cases = [
            s
            for s in manifest["samples"]
            if s["kind"] == "retrieval" and s["length"] == length
        ][: 100 if args.full else 25]
        for sample in cases:
            target_file = directory / f'retrieval-{sample["id"]}.json'
            if target_file.exists():
                saved = json.loads(target_file.read_text())
                if (
                    saved["sample_id"] != sample["id"]
                    or saved["answer"] != sample["answer"]
                    or saved["length"] != length
                    or saved.get("provenance") != llm._study_provenance
                ):
                    raise RuntimeError("native retrieval resume mismatch")
                retrieval.append(saved)
                continue
            llm.reset_prefix_cache()
            split = sample["catalogue_tokens"]
            llm.generate(
                [{"prompt_token_ids": sample["token_ids"][:split]}],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
            output = llm.generate(
                [{"prompt_token_ids": sample["token_ids"]}],
                SamplingParams(temperature=0, max_tokens=manifest["max_output_tokens"]),
                use_tqdm=False,
            )[0]
            completion = output.outputs[0]
            final = parse_native(completion.text)
            fact = f"Record {sample['target_key']}: {sample['answer']}."
            fact_end = sample["prompt"].index(fact) + len(fact)
            encoded = tokenizer(
                sample["prompt"], add_special_tokens=False, return_offsets_mapping=True
            )
            if encoded["input_ids"] != sample["token_ids"]:
                raise RuntimeError(
                    "native tokenizer differs from the frozen combined prompt"
                )
            fact_token_end = next(
                index + 1
                for index, (_, end) in enumerate(encoded["offset_mapping"])
                if end >= fact_end
            )
            if (
                output.num_cached_tokens is None
                or output.num_cached_tokens < fact_token_end
            ):
                raise RuntimeError(
                    "native query did not reuse stored KV covering the requested fact"
                )
            control = expected[sample["id"]]
            retrieval.append(
                {
                    "sample_id": sample["id"],
                    "length": length,
                    "final": final,
                    "answer": sample["answer"],
                    "correct": final == sample["answer"]
                    and completion.finish_reason != "length",
                    "qdq_final": control["final"],
                    "agree": final == control["final"],
                    "finish_reason": completion.finish_reason,
                    "malformed": not bool(final),
                    "truncated": completion.finish_reason == "length",
                    "num_cached_tokens": getattr(output, "num_cached_tokens", None),
                    "target_fact_token_end": fact_token_end,
                    "cached_target_fact_verified": True,
                    "catalogue_tokens": split,
                    "token_ids": list(completion.token_ids),
                    "provenance": llm._study_provenance,
                }
            )
            save(target_file, retrieval[-1])
    good = all(
        r["mean_absolute_target_nll_error"] <= 0.01
        and r["max_target_nll_error"] <= 0.10
        and r["top1_agreement"] >= 0.99
        for r in comparisons
    )
    good &= sum(r["agree"] for r in retrieval) / len(retrieval) >= 0.99
    save(
        directory / ("native-quality.json" if args.full else "bridge.json"),
        {
            "arm": arm,
            "status": "measured" if args.full else "qualified" if good else "failed",
            "nll_windows": comparisons,
            "retrieval_cases": retrieval,
            "recipe": "native read-after-write; prefix-cache block reuse with padding-tail replay",
            "qdq_recipe": manifest["cache_recipe"],
            "backend_semantics_not_assumed_identical": True,
            "provenance": llm._study_provenance,
        },
    )


def run_gsm(args, manifest, scales):
    task = gsm8k.read(args.task)
    if args.subset != "development":
        freeze = read_json(ROOT / "task-freeze.json")
        unlock = read_json(ROOT / "SCORING-UNLOCKED.json")
        if (
            freeze["task_file_sha256"] != file_hash(args.task)
            or freeze["primary_cases"] != 1319
            or len(unlock["protocol_commit"]) != 40
        ):
            raise RuntimeError(
                "GSM8K protocol must be committed and unlocked before test scoring"
            )
    directory = OUT / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gsm8k.jsonl"
    llm = make_engine(args.arm, manifest, performance=True)
    start_instrument(llm, scales, args.arm, directory / "dispatch")
    llm.generate(
        [{"prompt_token_ids": [1, 2, 3]}],
        SamplingParams(temperature=0, max_tokens=2),
        use_tqdm=False,
    )
    stop_instrument(llm, directory / "dispatch")
    llm.apply_model(callbacks.require_finite)
    llm.apply_model(callbacks.ensure_no_teacher)
    provenance = {
        **llm._study_provenance,
        "task_file_sha256": file_hash(args.task),
        "task_sample_hash": task["sample_hash"],
        "batch": args.batch,
    }
    seen = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in seen or row["provenance"] != provenance:
                raise RuntimeError(
                    "incomplete, duplicate or mixed-provenance GSM8K resume"
                )
            seen[row["id"]] = row
    cases = task["development"] if args.subset == "development" else task["samples"]
    if args.subset == "pilot":
        lookup = {r["id"]: r for r in cases}
        cases = [lookup[key] for key in task["pilot_ids"]]
    expected = {
        r["id"]: r
        for r in (
            task["development"] if args.subset == "development" else task["samples"]
        )
    }
    if not set(seen) <= set(expected):
        raise RuntimeError("unexpected existing GSM8K question")
    pending = [r for r in cases if r["id"] not in seen]
    for start in range(0, len(pending), args.batch):
        batch = pending[start : start + args.batch]
        params = SamplingParams(
            temperature=0, max_tokens=task["max_output_tokens"], stop=task["stop"]
        )
        schedule.label(f"gsm8k-{args.subset}-{start}")
        begun = time.monotonic()
        try:
            outputs = llm.generate(
                [{"prompt_token_ids": r["token_ids"]} for r in batch],
                params,
                use_tqdm=False,
            )
            if len(outputs) != len(batch):
                raise RuntimeError("missing GSM8K engine outputs")
            rows = []
            for sample, output in zip(batch, outputs):
                if list(output.prompt_token_ids) != sample["token_ids"]:
                    raise RuntimeError("GSM8K prompt IDs changed")
                completion = output.outputs[0]
                strict = gsm8k.extract(completion.text)
                flexible = gsm8k.extract(completion.text, flexible=True)
                truncated = completion.finish_reason == "length"
                finish_error = (
                    None
                    if completion.finish_reason in ("stop", "length")
                    else "unexpected native finish: " + str(completion.finish_reason)
                )
                rows.append(
                    {
                        "id": sample["id"],
                        "prompt_hash": sample["prompt_hash"],
                        "gold": sample["gold"],
                        "text": completion.text,
                        "generated_token_ids": list(completion.token_ids),
                        "input_tokens": len(sample["token_ids"]),
                        "output_tokens": len(completion.token_ids),
                        "strict_answer": strict,
                        "flexible_answer": flexible,
                        "correct": strict == sample["gold"]
                        and not truncated
                        and finish_error is None,
                        "flexible_correct": flexible == sample["gold"]
                        and not truncated
                        and finish_error is None,
                        "parser_failure": strict is None,
                        "truncated": truncated,
                        "error": finish_error,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "eos_stop": completion.finish_reason == "stop"
                        and completion.stop_reason is None,
                        "provenance": provenance,
                    }
                )
        except Exception as exc:
            rows = [
                {
                    "id": r["id"],
                    "prompt_hash": r["prompt_hash"],
                    "gold": r["gold"],
                    "text": "",
                    "generated_token_ids": [],
                    "input_tokens": len(r["token_ids"]),
                    "output_tokens": 0,
                    "strict_answer": None,
                    "flexible_answer": None,
                    "correct": False,
                    "flexible_correct": False,
                    "parser_failure": True,
                    "truncated": False,
                    "error": type(exc).__name__ + ": " + str(exc),
                    "provenance": provenance,
                }
                for r in batch
            ]
            with path.open("a") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            raise
        finally:
            schedule.label(None)
        with path.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                seen[row["id"]] = row
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "subset": args.subset,
                    "completed": len(seen),
                    "batch_seconds": time.monotonic() - begun,
                    "correct": sum(r["correct"] for r in seen.values()),
                    "output_tokens": sum(r["output_tokens"] for r in seen.values()),
                }
            ),
            flush=True,
        )
    selected = [seen[r["id"]] for r in cases]
    save(
        directory / f"{args.subset}-summary.json",
        {
            "arm": args.arm,
            "subset": args.subset,
            "cases": len(selected),
            "correct": sum(r["correct"] for r in selected),
            "parsed": sum(not r["parser_failure"] for r in selected),
            "truncated": sum(r["truncated"] for r in selected),
            "errors": sum(r["error"] is not None for r in selected),
            "provenance": provenance,
        },
    )


def timing_record(llm, prompts, batch, label, out, provenance):
    import numpy as np

    params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    # Prefix caching is disabled. All requests are enqueued before stepping
    # by the in-process engine; the observer still verifies actual concurrency.
    schedule.label(None)
    llm.generate(prompts, params, use_tqdm=False)
    warm_memory = llm.collective_rpc(callbacks.reset_warm_peak)
    schedule.label(label)
    started = time.monotonic()
    completion_wall = {}
    original_step = llm.llm_engine.step

    def observed_step(*args, **kwargs):
        values = original_step(*args, **kwargs)
        if any(value.finished for value in values):
            stamp = time.time()
            for value in values:
                if value.finished:
                    completion_wall[value.request_id] = stamp
        return values

    llm.llm_engine.step = observed_step
    try:
        outputs = llm.generate(prompts, params, use_tqdm=False)
    finally:
        llm.llm_engine.step = original_step
    ended = time.monotonic()
    schedule.label(None)
    metrics = []
    for output in outputs:
        completion = output.outputs[0]
        if len(completion.token_ids) != 128 or output.num_cached_tokens:
            raise RuntimeError("synthetic timing output/cache coverage mismatch")
        m = dict(vars(output.metrics))
        m["observed_frontend_completion_wall_time"] = completion_wall[output.request_id]
        if not m.get("first_token_ts") or m["last_token_ts"] <= m["first_token_ts"]:
            raise RuntimeError("native request timestamps missing")
        metrics.append(
            {
                k: v
                for k, v in m.items()
                if v is None or isinstance(v, (str, int, float, bool))
            }
        )
    events = [
        json.loads(line)
        for line in (
            OUT / f"schedule-{OPTIONS.stage}-{OPTIONS.arm}-{OPTIONS.repeat}.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    events = [e for e in events if e["label"] == label]
    full = [i for i, e in enumerate(events) if e["active_decode_sequences"] == batch]
    intervals = [
        (events[i + 1]["monotonic"] - events[i]["monotonic"])
        for i in full
        if i + 1 < len(events)
    ]
    complete = batch == 1 or len(intervals) >= 32
    # vLLM arrival_time is wall-clock; core token timestamps are monotonic.
    # Observe each finished result at the frontend instead of mixing clocks.
    latencies = [
        m["observed_frontend_completion_wall_time"] - m["arrival_time"] for m in metrics
    ]
    ttft = [m["first_token_latency"] for m in metrics]
    if any(not 0 <= first <= elapsed for first, elapsed in zip(ttft, latencies)):
        raise RuntimeError("inconsistent frontend request latency timestamps")
    prefill_span = max(
        m["arrival_time"] + m["first_token_latency"] for m in metrics
    ) - min(m["arrival_time"] for m in metrics)
    itl = [(m["last_token_ts"] - m["first_token_ts"]) / 127 for m in metrics]
    memory = llm.collective_rpc(callbacks.native_memory_state)
    driver_peak = max(x["cuda_device_used_bytes"] for x in warm_memory + memory)
    row = {
        "batch": batch,
        "seconds": ended - started,
        "output_tokens": batch * 128,
        "input_tokens_per_request": len(prompts[0]["prompt_token_ids"]),
        "output_tokens_per_request": 128,
        "ignore_eos": True,
        "warmup_excluded": True,
        "end_to_end_output_tokens_per_second": batch * 128 / (ended - started),
        "full_active_decode_tokens_per_second": (
            batch * len(intervals) / sum(intervals) if intervals else None
        ),
        "full_active_decode_steps": len(intervals),
        "active_concurrency_qualified": complete,
        "max_active_decode_sequences": max(
            (e["active_decode_sequences"] for e in events), default=0
        ),
        "resident_computed_token_peak": max(
            (e["resident_computed_tokens_before"] for e in events), default=0
        ),
        "preemptions": sum(len(e["preempted"]) for e in events),
        "request_metrics": metrics,
        "clock_notes": "Frontend arrival/completion use wall-clock; TTFT uses vLLM first_token_latency; mean ITL uses core monotonic first/last token timestamps.",
        "effective_prefill_input_tokens_per_second": batch
        * len(prompts[0]["prompt_token_ids"])
        / prefill_span,
        "prefill_rate_note": "All input tokens divided by the interval from first arrival to last first token; includes scheduler admissions and host overhead.",
        "itl_percentile_note": "Percentiles across per-request mean decode intervals, not individual-token tail latency.",
        "request_latency_p50_seconds": float(np.quantile(latencies, 0.5)),
        "request_latency_p95_seconds": float(np.quantile(latencies, 0.95)),
        "ttft_p50_seconds": float(np.quantile(ttft, 0.5)),
        "ttft_p95_seconds": float(np.quantile(ttft, 0.95)),
        "itl_p50_seconds": float(np.quantile(itl, 0.5)),
        "itl_p95_seconds": float(np.quantile(itl, 0.95)),
        "warm_peak_reset_state": warm_memory,
        "after_workload_memory": memory,
        "driver_used_observed_peak_bytes": driver_peak,
        "driver_peak_note": "Stage snapshots; continuous external telemetry must be joined for the total-HBM envelope.",
        "provenance": provenance,
    }
    save(out, row)
    print(
        json.dumps(
            {
                "measurement": label,
                "batch": batch,
                "seconds": ended - started,
                "active_qualified": complete,
                "e2e_tokens_per_second": row["end_to_end_output_tokens_per_second"],
            }
        ),
        flush=True,
    )
    return row


def run_measurement(args, manifest, scales):
    if args.stage == "writer":
        from .geometry import qualify

        qualify(ROOT, OUT)
        return
    qualification = (
        read_json(
            ROOT
            / ("legacy-qualification.json" if args.legacy else "qualification.json")
        )
        if args.stage in ("timing", "profile", "retrieval-batch")
        else None
    )
    if qualification is not None:
        if (
            qualification["protocol_sha256"] != file_hash(ROOT / "protocol.json")
            or args.arm not in qualification["qualified_modes"]
        ):
            raise RuntimeError("matched native/GSM8K quality gate is closed")
    directory = OUT / args.arm / f"{args.stage}-{args.repeat}"
    directory.mkdir(parents=True, exist_ok=True)
    llm = make_engine(args.arm, manifest, performance=True)
    cold_memory = llm.collective_rpc(callbacks.native_memory_state)
    start_instrument(llm, scales, args.arm, directory)
    llm.generate(
        [{"prompt_token_ids": [1, 2, 3]}],
        SamplingParams(temperature=0, max_tokens=3),
        use_tqdm=False,
    )
    stop_instrument(llm, directory)
    save(
        directory / "cold-memory.json",
        {"memory": cold_memory, "provenance": llm._study_provenance},
    )
    if args.stage == "storage":
        return
    examples = {
        length: [
            s
            for s in manifest["samples"]
            if s["kind"] == "nll" and s["length"] == length and not s.get("reserve")
        ]
        for length in (2048, 8192)
    }
    if args.stage == "retrieval-batch":
        llm.apply_model(callbacks.require_finite)
        cases = [
            s
            for s in manifest["samples"]
            if s["kind"] == "retrieval" and s["length"] == 8192
        ]
        rows = []
        for start in range(0, len(cases), args.batch):
            selected = cases[start : start + args.batch]
            schedule.label(f"retrieval-8k-{args.batch}-{start}")
            outputs = llm.generate(
                [{"prompt_token_ids": s["token_ids"]} for s in selected],
                SamplingParams(temperature=0, max_tokens=manifest["max_output_tokens"]),
                use_tqdm=False,
            )
            schedule.label(None)
            for sample, output in zip(selected, outputs):
                completion = output.outputs[0]
                final = parse_native(completion.text)
                rows.append(
                    {
                        "sample_id": sample["id"],
                        "answer": sample["answer"],
                        "final": final,
                        "correct": final == sample["answer"]
                        and completion.finish_reason != "length",
                        "truncated": completion.finish_reason == "length",
                        "text": completion.text,
                        "token_ids": list(completion.token_ids),
                        "batch": args.batch,
                        "provenance": llm._study_provenance,
                    }
                )
        save(
            directory / f"retrieval-batch-{args.batch}.json",
            {
                "rows": rows,
                "recipe": "native full-prompt prefill, APC disabled; extra schedule-specific regression",
                "provenance": llm._study_provenance,
            },
        )
        return
    shapes = (
        [(8192, args.batch)]
        if args.single
        else [(length, batch) for length in (2048, 8192) for batch in (1, 8, 32)]
    )
    if args.legacy:
        shapes = [(8192, 1), (8192, 8)]
    import random

    random.Random(20261022 + args.repeat).shuffle(shapes)
    for length, batch in shapes:
        prompts = [
            {
                "prompt_token_ids": examples[length][i % len(examples[length])][
                    "token_ids"
                ][:length]
            }
            for i in range(batch)
        ]
        label = f"{args.stage}-{length}-{batch}"
        if args.stage == "profile":
            schedule.label(None)
            llm.generate(
                prompts,
                SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
                use_tqdm=False,
            )
            schedule.label(label)
            llm.apply_model(
                partial(
                    callbacks.profile_start,
                    trace_dir=str(directory / f"{length}-{batch}"),
                )
            )
            llm.generate(
                prompts,
                SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
                use_tqdm=False,
            )
            llm.apply_model(callbacks.profile_end)
            schedule.label(None)
        else:
            timing_record(
                llm,
                prompts,
                batch,
                label,
                directory / f"{length}-{batch}.json",
                llm._study_provenance,
            )


def main():
    global ROOT, OUT, OPTIONS, PROTOCOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("k16_v16", "k16_v8", "k8_v8"), required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "bridge",
            "quality",
            "gsm8k",
            "timing",
            "profile",
            "storage",
            "writer",
            "retrieval-batch",
        ),
        required=True,
    )
    parser.add_argument("--task", type=Path)
    parser.add_argument(
        "--subset", choices=("development", "pilot", "full"), default="pilot"
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--cache-bytes", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--utilization", type=float)
    args = parser.parse_args()
    ROOT = args.inputs.resolve()
    OUT = args.output.resolve()
    OPTIONS = args
    if args.blocks is not None and args.cache_bytes is not None:
        parser.error("use blocks OR cache bytes")
    if args.batch < 1:
        parser.error("batch must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL = read_json(ROOT / "protocol.json")
    manifest = read_json(ROOT / "qwen.json.gz")
    scales = read_json(ROOT / "scales.json")["scales"]
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        PROTOCOL["torch_allocator_reserved_cap_bytes"] / total
    )
    schedule.install(OUT / f"schedule-{args.stage}-{args.arm}-{args.repeat}.jsonl")
    try:
        if args.stage in ("bridge", "quality"):
            args.full = args.stage == "quality"
            bridge(args, manifest, scales)
        elif args.stage == "gsm8k":
            run_gsm(args, manifest, scales)
        else:
            run_measurement(args, manifest, scales)
    except Exception as exc:
        path = OUT / f"{args.stage}-{args.arm}-blocker-{args.repeat}.json"
        save(
            path,
            {
                "exception": type(exc).__name__,
                "message": str(exc),
                "at": time.time(),
                "qualified": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
