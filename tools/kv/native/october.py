# SPDX-License-Identifier: MIT
"""Captured October serving entrypoint using frozen native qualification inputs.

One invocation owns one engine. Checkout selection belongs to the caller and
must happen between invocations. This module never changes dependency source.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .graph import ExecutionVariant, resolve_variant

FLASHINFER = "800a550f8b736e49f8cfa136b82bfaf6d9d32711"
VARIANTS = {
    "native": ExecutionVariant(
        {"vllm": "c21c0878210695db7b108cdc285aaf0cf890190a", "flashinfer": FLASHINFER},
        False,
        True,
    ),
    "mixed": ExecutionVariant(
        {"vllm": "1a4dcc01ab09cd337499c0138c732eb741218b57", "flashinfer": FLASHINFER},
        True,
        True,
    ),
    "compact": ExecutionVariant(
        {"vllm": "26cd429bac01d07909ed91e353f564d7aaa8d38f", "flashinfer": FLASHINFER},
        True,
        True,
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(protocol, variant, environment):
    definition = resolve_variant(variant, VARIANTS)
    definition.verify_environment(environment)
    if environment.get("VLLM_COMPACT_PREFILL") != str(int(variant == "compact")):
        raise ValueError("compact flag differs from the selected variant")
    if protocol["backend_sources"] != dict(definition.sources):
        raise ValueError("protocol source differs from the selected variant")
    engine = protocol["engine"]
    capture = engine.get("compilation_config", {})
    if (
        engine.get("enforce_eager") is not False
        or capture.get("cudagraph_mode") != "FULL_DECODE_ONLY"
        or engine.get("enable_chunked_prefill") is not False
        or engine.get("tensor_parallel_size") != 1
    ):
        raise ValueError("October entrypoint requires the frozen captured TP1 recipe")
    return definition


def check_qualification(path, protocol_path, *, length, batch):
    value = json.loads(Path(path).read_text())
    if value.get("passed") is not True or digest(protocol_path) not in value.get(
        "protocol_sha256", []
    ):
        raise ValueError(
            "timing requires a passing qualification for this exact protocol"
        )
    if not all(
        value.get(key) is True for key in ("production", "nll", "retrieval", "gsm8k")
    ):
        raise ValueError("qualification is incomplete")
    if {"input_tokens": length, "batch": batch} not in value.get("workloads", []):
        raise ValueError("quality does not cover the requested workload")
    return value


def build_engine(inputs, output, variant, arm, cache_bytes, stage):
    import os
    from types import SimpleNamespace
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from . import instrument, runner as core, schedule
    from .graph import install_capture_hooks
    from .memory_worker import ObservedWorker

    protocol = core.read_json(inputs / "protocol.json")
    definition = check_protocol(protocol, variant, os.environ)
    core.ROOT, core.OUT, core.PROTOCOL = inputs, output, protocol
    stage_name = "batch-quality" if stage == "nll" else "october"
    core.OPTIONS = SimpleNamespace(
        arm=arm,
        stage=stage_name,
        repeat=0,
        legacy=False,
        development=False,
        utilization=None,
        blocks=None,
        cache_bytes=cache_bytes,
    )
    schedule._mode = "buffered"
    schedule.install(output / f"schedule-{stage_name}-{arm}-0.jsonl")
    manifest = core.read_json(inputs / "qwen.json.gz")
    scales = core.read_json(inputs / "scales.json")["scales"]
    capture = output / "before-capture"
    capture.mkdir()

    def configure(worker):
        instrument.configure(
            worker.model_runner.get_model(),
            scales,
            arm,
            str(capture),
            protocol["packed_execution"],
        )

    def check(runner):
        import vllm
        import flashinfer

        definition.verify_sources(
            {name: inputs / name for name in definition.sources},
            {"vllm": vllm.__file__, "flashinfer": flashinfer.__file__},
        )
        actual = []
        for name, module in runner.get_model().named_modules():
            if hasattr(module, "_v_scale") and hasattr(module, "kv_cache"):
                index = int(name.split("layers.")[1].split(".")[0])
                expected = scales[f"{index}.v"] if arm == "k16_v8" else 1.0
                if module._v_scale_float != expected or float(module._v_scale) != float(
                    torch.tensor(expected, dtype=torch.float32)
                ):
                    raise RuntimeError("value scale differs before graph capture")
                actual.append(
                    {
                        "layer": name,
                        "host": module._v_scale_float,
                        "gpu": float(module._v_scale),
                    }
                )
        if len(actual) != protocol.get("geometry", {}).get("layers", 36):
            raise RuntimeError(
                "capture attention inventory differs from frozen geometry"
            )
        instrument.save(output / "capture-time-scales.json", actual)

    restore = install_capture_hooks(
        ObservedWorker, GPUModelRunner, configure, check, definition
    )
    try:
        engine = core.make_engine(arm, manifest, performance=True)
    finally:
        restore()
    instrument.save(
        output / "cold-start-memory.json",
        engine.collective_rpc(instrument.native_memory_state),
    )
    return engine, manifest, scales


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("geometry", "nll", "retrieval", "gsm8k", "timing")
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--batch", type=int, choices=(8, 32), default=8)
    parser.add_argument("--length", type=int, choices=(1024, 2048, 8192), default=8192)
    parser.add_argument("--cache-bytes", type=int, required=True)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    inputs, output = args.inputs.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("use a fresh output directory")
    protocol = json.loads((inputs / "protocol.json").read_text())
    check_protocol(protocol, args.variant, os.environ)
    if protocol["engine"].get("kv_cache_memory_bytes") != args.cache_bytes:
        raise ValueError("cache pool differs from the frozen protocol")
    if args.stage == "timing":
        if args.qualification is None:
            raise ValueError("timing requires --qualification")
        check_qualification(
            args.qualification,
            inputs / "protocol.json",
            length=args.length,
            batch=args.batch,
        )
    output.mkdir(parents=True)
    from . import instrument, runner as core, schedule

    arm = "k16_v16" if args.variant == "native" else "k16_v8"
    if args.stage == "geometry":
        from .geometry import qualify

        qualify(inputs, output, geometry=protocol.get("geometry"))
        return
    engine, manifest, scales = build_engine(
        inputs, output, args.variant, arm, args.cache_bytes, args.stage
    )
    if args.stage in ("nll", "retrieval"):
        engine.apply_model(instrument.require_finite)
    instrument.save(output / "provenance.json", engine._study_provenance)
    if args.stage == "gsm8k":
        from types import SimpleNamespace

        core.make_engine = lambda *a, **kw: engine
        # Frozen scales were installed before warmup and capture. Do not wrap
        # compact's native temporary workspace as persistent mixed KV.
        core.start_instrument = lambda *a, **kw: None
        core.stop_instrument = lambda *a, **kw: None
        core.run_gsm(
            SimpleNamespace(
                arm=arm,
                task=inputs / "gsm8k-frozen.json",
                subset="full",
                batch=args.batch,
            ),
            manifest,
            scales,
        )
    elif args.stage == "nll":
        from tools.kv import native_batch_quality as teacher

        samples = [
            s
            for s in manifest["samples"]
            if s["kind"] == "nll"
            and s["length"] == args.length
            and not s.get("reserve")
        ]
        if len(samples) != 32:
            raise ValueError("quality requires exactly32 disjoint document windows")
        engine.collective_rpc(teacher.install)
        try:
            for start in range(0, 32, args.batch):
                teacher.cohort(
                    engine, samples[start : start + args.batch], output, f"nll-{start}"
                )
        finally:
            engine.collective_rpc(teacher.finish)
    elif args.stage == "retrieval":
        from vllm import SamplingParams

        samples = [
            s
            for s in manifest["samples"]
            if s["kind"] == "retrieval"
            and s["length"] == args.length
            and not s.get("reserve")
        ]
        if len(samples) != 100:
            raise ValueError("quality requires exactly100 frozen retrieval cases")
        for start in range(0, 100, args.batch):
            batch = samples[start : start + args.batch]
            values = engine.generate(
                [{"prompt_token_ids": s["token_ids"]} for s in batch],
                SamplingParams(temperature=0, max_tokens=128),
                use_tqdm=False,
            )
            rows = []
            for sample, value in zip(batch, values):
                completion = value.outputs[0]
                correct = (
                    core.parse_native(completion.text) == sample["answer"]
                    and completion.finish_reason != "length"
                )
                rows.append(
                    {
                        "id": sample["id"],
                        "answer": sample["answer"],
                        "text": completion.text,
                        "token_ids": list(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "correct": correct,
                    }
                )
            instrument.save(output / f"retrieval-{start}.json", rows)
    else:
        samples = [
            s
            for s in manifest["samples"]
            if s["kind"] == "nll"
            and s["length"] == args.length
            and not s.get("reserve")
        ][: args.batch]
        prompts = [{"prompt_token_ids": s["token_ids"][: args.length]} for s in samples]
        if (
            len(prompts) != args.batch
            or len({tuple(p["prompt_token_ids"]) for p in prompts}) != args.batch
        ):
            raise ValueError("timing requires a unique full prompt cohort")
        # The public recorder excludes its fixed warmup and reports complete
        # requests, completion-linked decode, TTFT, active rows and preemption.
        row = core.timing_record(
            engine,
            prompts,
            args.batch,
            "timing",
            output / "timing.json",
            engine._study_provenance,
        )
        if not row["active_concurrency_qualified"] or row["preemptions"]:
            raise RuntimeError("actual active batch did not pass")
    schedule.label(None)


if __name__ == "__main__":
    main()
