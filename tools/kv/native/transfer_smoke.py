# SPDX-License-Identifier: MIT
"""Untimed conventional-model qualification for the experimental geometry port.

Native controls and the transfer candidate keep independent source identities.
Passing this smoke does not satisfy scored quality or performance gates.
"""

import argparse
from contextlib import ExitStack
from functools import partial
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

from . import instrument, schedule
from .compact import observe_routes
from .october import build_engine
from .profile_ranges import trace_attention_routes
from .routing_oracle import validate_routing_fixture, validate_routing_layers
from .storage_receipts import inventory
from .weights import fp16_dispatch_from_trace
from .workspace_tracking import InstanceTracker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("native", "mixed_transfer"), required=True)
    p.add_argument("--length", type=int, choices=(1024, 8192), required=True)
    p.add_argument("--batch", type=int, choices=(16, 32), required=True)
    p.add_argument("--cache-bytes", type=int, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.inputs / "protocol.json").read_text())
    shape = protocol["geometry"]
    if "attention_layer_indices" in shape:
        raise ValueError("this smoke requires conventional attention at every layer")
    if protocol["packed_execution"]["format"] != "fp16":
        raise ValueError("transfer smoke requires separately audited FP16 weights")
    arm = "k16_v16" if args.variant == "native" else "k16_v8"
    tracker, geometry = None, None
    if args.variant == "mixed_transfer":
        from vllm.v1.attention.backends.compact_prefill import CompactPrefillWorkspace
        from vllm.v1.attention.backends.compact_prefill_metadata import select_geometry

        geometry = select_geometry(
            shape["query_heads"], shape["kv_heads"], shape["head_dim"]
        )
        validate_routing_fixture(args.output / "synthetic-routing", geometry=geometry)
        tracker = InstanceTracker(CompactPrefillWorkspace)
        tracker.start()
    llm, manifest, _ = build_engine(
        args.inputs, args.output, args.variant, arm, args.cache_bytes, "profile"
    )
    instrument.save(args.output / "provenance.json", llm._study_provenance)
    runner = llm.collective_rpc(lambda worker: worker.model_runner)[0]
    samples = [
        s
        for s in manifest["samples"]
        if s["kind"] == "nll" and s["length"] == args.length and not s.get("reserve")
    ][: args.batch]
    if len(samples) != args.batch:
        raise ValueError("incomplete frozen prompt cohort")
    prompts = [{"prompt_token_ids": s["token_ids"][: args.length]} for s in samples]
    params = SamplingParams(
        temperature=0, seed=20261019, max_tokens=128, ignore_eos=True
    )
    llm.apply_model(instrument.require_finite)
    graphs, generations, oracle_summaries = [], [], []
    old = CudaGraphManager.run_fullgraph

    def replay(manager, desc):
        if schedule._label in ("transfer-0", "transfer-1"):
            values = {
                name: getattr(runner.input_buffers, name)[: args.batch].cpu().tolist()
                for name in ("input_ids", "positions", "seq_lens")
            }
            graphs.append(
                {
                    "label": schedule._label,
                    "padded_requests": desc.num_reqs,
                    "padded_tokens": desc.num_tokens,
                    "values": values,
                    "sha256": hashlib.sha256(
                        json.dumps(values, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        return old(manager, desc)

    CudaGraphManager.run_fullgraph = replay
    torch.cuda.set_per_process_memory_fraction(1.0)
    try:
        with ExitStack() as outer:
            if geometry is not None:
                outer.enter_context(observe_routes(args.output / "routes.jsonl"))
            for index, selected in enumerate((prompts, prompts[1:] + prompts[:1])):
                schedule.label(f"transfer-{index}")
                with ExitStack() as checks:
                    if geometry is not None:
                        rows = checks.enter_context(
                            validate_routing_layers(
                                args.output / f"oracle-{index}", geometry=geometry
                            )
                        )
                    outputs = llm.generate(selected, params, use_tqdm=False)
                if geometry is not None:
                    oracle_summaries.append(
                        {
                            "calls": len(rows),
                            "layers": sorted({r["layer"] for r in rows}),
                            "passed": bool(rows)
                            and all(
                                r["routing_equivalence_passed"]
                                and r["forward_placement_verified"]
                                for r in rows
                            ),
                        }
                    )
                generation = [
                    {
                        "prompt_token_ids": x.prompt_token_ids,
                        "output_token_ids": list(x.outputs[0].token_ids),
                        "finish_reason": x.outputs[0].finish_reason,
                    }
                    for x in outputs
                ]
                instrument.save(args.output / f"generation-{index}.json", generation)
                generations.append(
                    {
                        tuple(x["prompt_token_ids"]): x["output_token_ids"]
                        for x in generation
                    }
                )
            order_exact = generations[0] == generations[1]
            differences = []
            for prompt, first in generations[0].items():
                second = generations[1][prompt]
                if first != second:
                    position = next(
                        (
                            i
                            for i, pair in enumerate(zip(first, second))
                            if pair[0] != pair[1]
                        ),
                        min(len(first), len(second)),
                    )
                    differences.append(
                        {
                            "prompt_sha256": hashlib.sha256(
                                json.dumps(prompt).encode()
                            ).hexdigest(),
                            "first_different_position": position,
                            "first_order_token": (
                                first[position] if position < len(first) else None
                            ),
                            "second_order_token": (
                                second[position] if position < len(second) else None
                            ),
                        }
                    )
            instrument.save(
                args.output / "order-comparison.json",
                {
                    "exact": order_exact,
                    "cases": len(generations[0]),
                    "differences": differences,
                },
            )
            for label in ("transfer-0", "transfer-1"):
                selected = [r for r in graphs if r["label"] == label]
                if len(selected) < 32 or len({r["sha256"] for r in selected}) < 3:
                    raise RuntimeError("changing graph replay coverage missing")
            if geometry is not None and not all(
                r["passed"] and len(r["layers"]) == shape["layers"]
                for r in oracle_summaries
            ):
                raise RuntimeError("independent routing qualification incomplete")
            instrument.save(args.output / "graphs.json", graphs)
            instrument.save(args.output / "oracle-coverage.json", oracle_summaries)
            gc.collect()
            torch.cuda.empty_cache()
            if tracker is not None:
                if len(tracker.live) != 1:
                    raise RuntimeError(
                        "validation left additional compact workspaces alive"
                    )
                tracker.stop()
            torch.cuda.set_per_process_memory_fraction(
                protocol["torch_allocator_reserved_cap_bytes"]
                / torch.cuda.get_device_properties(0).total_memory
            )
            torch.cuda.reset_peak_memory_stats()
            schedule.label("profile-warmup")
            llm.generate(prompts, params, use_tqdm=False)
            schedule.label("transfer-profile")
            with trace_attention_routes(
                args.output / "operations.json"
            ), torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
            path = args.output / "trace.json"
            profile.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            instrument.save(
                args.output / "weight-dispatch.json",
                fp16_dispatch_from_trace(
                    trace, minimum_calls=protocol["packed_execution"]["linear_modules"]
                ),
            )
            with path.open("rb") as src, gzip.open(str(path) + ".gz", "wb") as dest:
                shutil.copyfileobj(src, dest)
            path.unlink()
            instrument.save(
                args.output / "storage.json",
                llm.collective_rpc(partial(inventory, arm=arm, geometry=shape)),
            )
            instrument.save(
                args.output / "DONE.json",
                {
                    "production_smoke_passed": order_exact,
                    "evidence_capture_complete": True,
                    "greedy_order_exact": order_exact,
                    "scored_quality_complete": False,
                    "performance_complete": False,
                    "geometry": shape,
                    "length": args.length,
                    "batch": args.batch,
                },
            )
            # Keep graph, allocation and dispatch evidence when a native
            # control is itself sensitive to prefill cohort order. This does
            # not turn an order mismatch into a passing qualification.
            if not order_exact:
                raise RuntimeError("greedy output changed with request order")
    finally:
        CudaGraphManager.run_fullgraph = old
        schedule.label(None)
        if tracker is not None:
            tracker.stop()


if __name__ == "__main__":
    main()
