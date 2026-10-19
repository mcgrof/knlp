# SPDX-License-Identifier: MIT
"""Validate the installed serving recipe; these profiles are not benchmarks."""

import argparse, collections, gzip, hashlib, json, pathlib, shutil, sys
import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from tools.kv.native import instrument, schedule
from tools.kv.native.october import build_engine
from tools.kv.native.compact import observe_routes, worker_workspace
from tools.kv.native.routing_oracle import validate_routing_layers
from tools.kv.native.profile_ranges import trace_attention_routes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--variant", choices=("mixed_prefill",), default="mixed_prefill")
    p.add_argument("--format", choices=("awq", "fp16"), required=True)
    p.add_argument("--cache-bytes", type=int, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from vllm.v1.attention.backends.compact_prefill import CompactPrefillWorkspace
    from .workspace_tracking import InstanceTracker

    workspace_tracker = InstanceTracker(CompactPrefillWorkspace)
    workspace_tracker.start()

    llm, m, scales = build_engine(
        a.inputs, a.output, a.variant, "k16_v8", a.cache_bytes, "profile"
    )
    # Validation workspace is additional accounted instrumentation only.
    torch.cuda.set_per_process_memory_fraction(1.0)
    samples = [
        s
        for s in m["samples"]
        if s["kind"] == "nll" and s["length"] == 8192 and not s.get("reserve")
    ]
    assert len(samples) == 32
    prompts = [{"prompt_token_ids": s["token_ids"][:8192]} for s in samples]
    params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner_stack = []
    old_execute = GPUModelRunner.execute_model

    def execute(runner, scheduler_output, *args, **kwargs):
        runner_stack.append((runner, getattr(scheduler_output, "_knlp_batch_id", None)))
        try:
            return old_execute(runner, scheduler_output, *args, **kwargs)
        finally:
            runner_stack.pop()

    GPUModelRunner.execute_model = execute
    graph_inputs = []
    replays = collections.Counter()
    old = CudaGraphManager.run_fullgraph

    def replay(manager, desc):
        replays[(desc.num_reqs, desc.num_tokens)] += 1
        if (
            schedule._label is not None
            and schedule._label.startswith("production-check-")
            and desc.num_reqs == 32
            and desc.num_tokens == 32
        ):
            runner, bid = runner_stack[-1]
            values = {
                name: getattr(runner.input_buffers, name)[:32].cpu().tolist()
                for name in ["input_ids", "positions", "seq_lens"]
            }
            graph_inputs.append(
                dict(
                    label=schedule._label,
                    batch_id=bid,
                    values=values,
                    sha256=hashlib.sha256(
                        json.dumps(values, sort_keys=True).encode()
                    ).hexdigest(),
                )
            )
        return old(manager, desc)

    CudaGraphManager.run_fullgraph = replay
    with observe_routes(a.output / "route.jsonl"):
        for attempt, selected in enumerate([prompts, prompts[1:] + prompts[:1]]):
            schedule.label(f"production-check-{attempt}")
            with validate_routing_layers(a.output / f"layers-{attempt}") as checks:
                outputs = llm.generate(selected, params, use_tqdm=False)
            assert len(outputs) == 32 and all(
                len(x.outputs[0].token_ids) == 128 for x in outputs
            )
            assert len(checks) == 180 and len({x["layer"] for x in checks}) == 36
            (a.output / f"generation-{attempt}.json").write_text(
                json.dumps(
                    [
                        dict(
                            request_id=x.request_id,
                            prompt_sha256=hashlib.sha256(
                                json.dumps(x.prompt_token_ids).encode()
                            ).hexdigest(),
                            output_token_ids=x.outputs[0].token_ids,
                        )
                        for x in outputs
                    ],
                    indent=2,
                )
                + "\n"
            )
        import gc, subprocess, time
        from vllm.v1.attention.backends.compact_prefill import CompactPrefillWorkspace

        validation_memory = llm.collective_rpc(instrument.native_memory_state)
        (a.output / "VALIDATION-MEMORY.json").write_text(
            json.dumps(validation_memory, indent=2) + "\n"
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(
            68 * 2**30 / torch.cuda.get_device_properties(0).total_memory
        )
        objects = list(workspace_tracker.live.values())
        assert len(objects) == 1, len(objects)
        persistent_workspace = objects[0]
        del objects
        workspace_tracker.stop()
        torch.cuda.reset_peak_memory_stats()
        clean_start = time.time()
        sampler_log = (a.output / "clean-memory-samples.csv").open("w")
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used",
                "--format=csv,noheader,nounits",
                "--loop-ms=500",
            ],
            stdout=sampler_log,
            stderr=subprocess.STDOUT,
        )
        schedule.label("profile-warmup")
        llm.generate(prompts, params, use_tqdm=False)
        schedule.label(None)
        torch.cuda.synchronize()
        clean_states = llm.collective_rpc(instrument.native_memory_state)
        clean_end = time.time()
        sampler.terminate()
        sampler.wait(timeout=10)
        sampler_log.close()
        schedule.label("production-profile")
        with trace_attention_routes(
            a.output / "PROFILE-OPERATIONS.json"
        ), torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
        ) as trace:
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
        schedule.label(None)
    torch.cuda.synchronize()
    trace.export_chrome_trace(str(a.output / "trace.json"))
    with (a.output / "trace.json").open("rb") as src, gzip.open(
        a.output / "trace.json.gz", "wb"
    ) as dst:
        shutil.copyfileobj(src, dst)
    (a.output / "trace.json").unlink()
    assert replays.get((32, 32), 0) >= 32
    for label in ["production-check-0", "production-check-1"]:
        selected = [x for x in graph_inputs if x["label"] == label]
        assert len(selected) >= 32 and len({x["sha256"] for x in selected}) >= 3
    (a.output / "CHANGING-GRAPH-INPUTS.json").write_text(
        json.dumps(graph_inputs, indent=2) + "\n"
    )
    rows = [json.loads(x) for x in (a.output / "route.jsonl").read_text().splitlines()]
    schedule_rows = [
        json.loads(x)
        for x in next(a.output.glob("schedule-*.jsonl")).read_text().splitlines()
    ]
    events = {x["batch_id"]: x for x in schedule_rows}
    coverage = {}
    for label in ["production-check-0", "production-check-1", "production-profile"]:
        grouped = collections.defaultdict(list)
        for r in rows:
            if r["label"] == label:
                grouped[r["batch_id"]].append(r)
        joined = []
        for bid, rs in sorted(grouped.items()):
            assert len(rs) == 36
            r = rs[0]
            e = events[bid]
            counts = e["scheduled_tokens"]
            fresh = [rid for rid in r["request_ids"] if counts[rid] > 1]
            assert len(fresh) == len(r["lengths"]) and all(
                counts[rid] == 8192 for rid in fresh
            )
            assert all(
                x["request_ids"] == r["request_ids"]
                and x["lengths"] == r["lengths"]
                and x["workspace_ptr"] == r["workspace_ptr"]
                for x in rs
            )
            joined.append(
                dict(
                    batch_id=bid,
                    fresh=len(fresh),
                    decode=len(r["request_ids"]) - len(fresh),
                    pages=r["pages"],
                    request_ids=r["request_ids"],
                    fresh_request_ids=fresh,
                )
            )
        coverage[label] = joined
        assert [x["fresh"] for x in joined] == [8, 7, 7, 7, 3], coverage
        assert [x["decode"] for x in joined] == [0, 8, 15, 22, 29]
    workspace = llm.collective_rpc(worker_workspace)
    assert workspace[0]["payload_capacity_bytes"] == 256 * 2**20
    installed = json.loads((a.output / "before-capture/installed.json").read_text())
    assert len(installed["layers"]) == 36
    if a.format == "fp16":
        assert not installed["packed"]
        weights = installed["weights"]
        assert weights["format"] == "fp16"
        assert len(weights["linear_modules"]) == 144
        assert len(weights["embedding_modules"]) == 2
        assert all(
            r["method"] == "UnquantizedLinearMethod"
            and r["weight"][0]["dtype"] == "torch.float16"
            for r in weights["linear_modules"]
        )
        assert all(r["dtype"] == "torch.float16" for r in weights["parameters"])
        assert weights["unique_parameter_bytes"] > 14 * 2**30
    else:
        assert len(installed["packed"]) == 144
        assert all(
            x["method"] == "AutoAWQMarlinLinearMethod"
            and x["kernel"] == "MacheteLinearKernel"
            and x["weight_bits"] == 4
            for x in installed["packed"]
        )
    for layer in installed["layers"]:
        k, v = layer["cache"]
        assert (
            k["dtype"] == "torch.float16"
            and v["dtype"] == "torch.float8_e4m3fn"
            and k["storage_pointer"] == v["storage_pointer"]
        )
        assert k["logical_bytes"] + v["logical_bytes"] == k["storage_bytes"]
    from tools.kv.native.weights import fp16_dispatch_from_trace
    from .storage_receipts import inventory

    trace_data = json.loads(gzip.decompress((a.output / "trace.json.gz").read_bytes()))
    if a.format == "fp16":
        dispatch = fp16_dispatch_from_trace(trace_data, minimum_calls=144)
        assert dispatch["passed"]
    else:
        names = [
            x.get("name", "") for x in trace_data["traceEvents"] if x.get("dur", 0) > 0
        ]
        dispatch = {
            "passed": any("machete" in name.lower() for name in names),
            "weight_format": "packed INT4 AWQ",
        }
        assert dispatch["passed"], "missing actual packed-weight dispatch"
    (a.output / "WEIGHT-DISPATCH.json").write_text(
        json.dumps(dispatch, indent=2) + "\n"
    )
    (a.output / "STORAGE.json").write_text(
        json.dumps(llm.collective_rpc(inventory, args=("k16_v8",)), indent=2) + "\n"
    )
    del trace_data
    files = {}
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
        name = line.split()[-1]
        if (
            name.startswith("/")
            and ".so" in name
            and (
                any(k in name for k in ["vllm", "flashinfer"]) or "/flashinfer/" in name
            )
        ):
            path = pathlib.Path(name)
            if path.is_file():
                files[name] = dict(
                    sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                    bytes=path.stat().st_size,
                )
    values = [
        float(line.rsplit(",", 1)[1])
        for line in (a.output / "clean-memory-samples.csv").read_text().splitlines()
        if "," in line
    ]
    assert values
    clean = dict(
        oracle_buffers_present=False,
        pool_bytes=a.cache_bytes,
        payload_bytes=workspace[0]["payload_capacity_bytes"],
        payload_allocations=1,
        persistent_v_shadow_bytes=0,
        allocator_peak_bytes=max(x["peak_reserved_since_reset"] for x in clean_states),
        driver_observed_max_bytes=int(max(values) * 2**20),
        driver_samples=len(values),
        driver_cadence_seconds=0.5,
        window_start_epoch=clean_start,
        window_end_epoch=clean_end,
        workspace_count_after_gc=1,
        workspace_count_method="constructor weak references; includes GC-frozen instances",
        gc_frozen_objects=gc.get_freeze_count(),
        persistent_workspace_tensor_names=[
            k
            for k, v in vars(persistent_workspace).items()
            if isinstance(v, torch.Tensor)
        ],
        states=clean_states,
    )
    assert (
        clean["allocator_peak_bytes"] <= 68 * 2**30
        and clean["driver_observed_max_bytes"] <= 72 * 2**30
    )
    (a.output / "CLEAN-MEMORY.json").write_text(json.dumps(clean, indent=2) + "\n")
    result = dict(
        clean_memory=clean,
        passed=True,
        coverage=coverage,
        graph_replays=[
            dict(requests=k[0], tokens=k[1], calls=v) for k, v in replays.items()
        ],
        workspace=workspace,
        memory=llm.collective_rpc(instrument.native_memory_state),
        loaded_binaries=files,
        provenance=llm._study_provenance,
        profile_is_not_timing_data=True,
    )
    (a.output / "production.json").write_text(json.dumps(result, indent=2) + "\n")

    from tools.kv.native.qualification import snapshot_attention_libraries

    retained = snapshot_attention_libraries(
        [dict(path=k, **v) for k, v in files.items()], a.output / "build-objects"
    )
    (a.output / "BINARY-RECEIPTS.json").write_text(
        json.dumps(
            dict(
                libraries=[dict(path=k, **v) for k, v in files.items()],
                retained=retained,
            ),
            indent=2,
        )
        + "\n"
    )

    # A short ordinary prompt uses the same engine and scales. No timing claim.
    llm.apply_model(instrument.require_finite)
    tokenizer = llm.get_tokenizer()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    short = []
    for _ in range(2):
        value = llm.generate(
            [text], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False
        )[0].outputs[0]
        assert value.token_ids and value.text.strip()
        short.append(
            {
                "text": value.text,
                "token_ids": list(value.token_ids),
                "finish_reason": value.finish_reason,
            }
        )
    assert short[0] == short[1], "deterministic generation changed"
    instrument.save(
        a.output / "generation-demo.json",
        {"prompt": text, "outputs": short, "finite_logits": True, "exact_repeat": True},
    )
    print(
        "PASS: physical K16/V8; calibrated scales; mixed fresh-prefill route; changing decode graphs; clean memory; finite deterministic generation"
    )
    print(short[0]["text"])


if __name__ == "__main__":
    main()
