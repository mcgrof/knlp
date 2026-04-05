#!/usr/bin/env python3
"""Stage 1: In-process torch.profiler kernel comparison.

Uses VLLM_ENABLE_V1_MULTIPROCESSING=0 to run GPU work in the same process,
enabling torch.profiler to see CUDA kernels.
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time, json, sys, gc

RUN_ROOT = "/tmp/acp-runs/20260404T2339Z-residgap-h100-cldopus-a1"
ARTIFACTS = f"{RUN_ROOT}/artifacts"
MODEL = "marin-community/marin-8b-base"

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{RUN_ROOT}/progress.log", "a") as f:
        f.write(line + "\n")

def append_event(etype, phase, note):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ev = {"ts": ts, "run_id": "20260404T2339Z-residgap-h100-cldopus-a1",
          "attempt": 1, "actor": "worker", "host": "lewwsrdy631hun",
          "type": etype, "phase": phase, "note": note}
    with open(f"{RUN_ROOT}/events.jsonl", "a") as f:
        f.write(json.dumps(ev) + "\n")

def profile_arm(arm_name, kv_cache_dtype):
    import torch
    import torch.profiler
    from vllm import LLM, SamplingParams

    log(f"--- ARM: {arm_name} (kv_cache_dtype={kv_cache_dtype}, in-process) ---")
    log(f"  creating LLM...")

    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=False,
    )

    prompt = " ".join(["token"] * 512)
    sp = SamplingParams(max_tokens=256, temperature=0)

    # Warmup
    log(f"  warmup...")
    for i in range(3):
        llm.generate([prompt], sp)
        log(f"  warmup {i+1}/3")

    # Timing runs
    log(f"  timing (5 trials)...")
    timing = []
    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n = len(out[0].outputs[0].token_ids)
        itl = (elapsed * 1000) / max(n, 1)
        timing.append({"trial": trial, "elapsed_s": round(elapsed, 4),
                       "tokens": n, "itl_ms": round(itl, 3)})
        log(f"  trial {trial}: {elapsed:.3f}s, {n} tok, ITL={itl:.3f}ms")

    # Profiled run
    log(f"  profiled run...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        torch.cuda.synchronize()
        out = llm.generate([prompt], sp)
        torch.cuda.synchronize()

    # Extract kernel stats
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    log(f"  kernel table:")
    for line in table.split("\n"):
        log(f"    {line}")

    with open(f"{ARTIFACTS}/profile_{arm_name}_kernels.txt", "w") as f:
        f.write(table)

    # Chrome trace
    prof.export_chrome_trace(f"{ARTIFACTS}/profile_{arm_name}_trace.json")

    # Structured kernel data
    kernels = []
    for evt in prof.key_averages():
        if getattr(evt, "self_cuda_time_total", 0) > 0 or getattr(evt, "cuda_time_total", 0) > 0:
            kernels.append({
                "name": evt.key,
                "cuda_total_us": round(getattr(evt, "self_cuda_time_total", 0), 2),
                "cuda_avg_us": round(getattr(evt, "self_cuda_time_total", 0) / max(evt.count, 1), 2),
                "count": evt.count,
                "cpu_total_us": round(evt.cpu_time_total, 2),
            })
    kernels.sort(key=lambda x: x["cuda_total_us"], reverse=True)

    with open(f"{ARTIFACTS}/profile_{arm_name}_kernels.json", "w") as f:
        json.dump(kernels, f, indent=2)

    itls = [r["itl_ms"] for r in timing]
    avg = sum(itls) / len(itls)

    result = {
        "arm": arm_name, "kv_cache_dtype": kv_cache_dtype,
        "avg_itl_ms": round(avg, 3),
        "timing_trials": timing,
        "n_kernel_types": len(kernels),
        "top5_kernels": kernels[:5],
        "total_cuda_us": round(sum(k["cuda_total_us"] for k in kernels), 2),
    }

    with open(f"{ARTIFACTS}/stage1_{arm_name}_inproc_results.json", "w") as f:
        json.dump(result, f, indent=2)

    log(f"  avg ITL: {avg:.3f}ms, {len(kernels)} kernel types, total CUDA: {result['total_cuda_us']:.0f}us")

    del llm, prof
    gc.collect()
    torch.cuda.empty_cache()
    return result

def main():
    arm_name = sys.argv[1] if len(sys.argv) > 1 else "fp16_graph"
    kv_dtype = "auto" if arm_name == "fp16_graph" else "int4_fused"

    log(f"=== Stage 1 in-process profile: {arm_name} ===")
    append_event("PROCESS_STARTED", f"S1_INPROC_{arm_name}", f"in-process profile for {arm_name}")

    os.makedirs(ARTIFACTS, exist_ok=True)
    result = profile_arm(arm_name, kv_dtype)
    append_event("CHECKPOINT", f"S1_{arm_name}_DONE",
                 f"avg_itl={result['avg_itl_ms']}ms, {result['n_kernel_types']} kernel types")
    log(f"=== {arm_name} complete ===")
    os._exit(0)

if __name__ == "__main__":
    main()
