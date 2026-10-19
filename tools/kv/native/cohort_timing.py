# SPDX-License-Identifier: MIT
"""Record complete in-process cohorts on one synchronized host clock.

The caller must qualify the exact engine, inputs and workload before calling.
These records measure complete-request throughput, not HTTP or kernel latency.
"""

import time
from pathlib import Path


def measure(engine, prompts, output, *, cohorts, warmup_cohorts=2):
    import torch
    from vllm import SamplingParams
    from tools.kv.native import instrument, schedule

    if cohorts < 1 or warmup_cohorts < 1 or not prompts:
        raise ValueError("positive measurement and warmup counts are required")
    if len({tuple(p["prompt_token_ids"]) for p in prompts}) != len(prompts):
        raise ValueError("cohort prompts must be distinct")
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    schedule.label(None)
    for _ in range(warmup_cohorts):
        engine.generate(prompts, params, use_tqdm=False)
    instrument.save(
        output / "warm-memory.json",
        engine.collective_rpc(instrument.reset_warm_peak),
    )
    try:
        for index in range(cohorts):
            schedule._completed.clear()
            schedule.label(f"cohort-{index:03}")
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            values = engine.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            end = time.monotonic_ns()
            schedule.label(None)
            outputs = [
                {
                    "request_id": value.request_id,
                    "internal_request_id": schedule.internal_ids([value.request_id])[0],
                    "prompt_token_ids": list(value.prompt_token_ids),
                    "output_token_ids": list(value.outputs[0].token_ids),
                    "finish_reason": value.outputs[0].finish_reason,
                    "num_cached_tokens": value.num_cached_tokens,
                }
                for value in values
            ]
            row = {
                "index": index,
                "start_monotonic_ns": start,
                "end_monotonic_ns": end,
                "clock": "one process time.monotonic_ns",
                "endpoint": "CUDA-synchronized complete generate call",
                "outputs": outputs,
                "completions": list(schedule._completed),
                "after_memory": engine.collective_rpc(instrument.native_memory_state),
                "heavy_oracles_or_profiler_during_measurement": False,
            }
            instrument.save(output / f"cohort-{index:03}.json", row)
            if (
                end <= start
                or len(outputs) != len(prompts)
                or any(
                    r["prompt_token_ids"] != prompt["prompt_token_ids"]
                    or len(r["output_token_ids"]) != 128
                    or r["num_cached_tokens"]
                    for r, prompt in zip(outputs, prompts)
                )
            ):
                raise RuntimeError(
                    "incomplete or changed measured cohort; retained raw"
                )
            if any(event["preempted"] for event in row["completions"]):
                raise RuntimeError("measured cohort preempted; retained raw")
    finally:
        schedule.label(None)
    instrument.save(
        output / "DONE.json",
        {"cohorts": cohorts, "warmup_cohorts": warmup_cohorts, "output_tokens": 128},
    )
