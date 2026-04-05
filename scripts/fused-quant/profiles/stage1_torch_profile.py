#!/usr/bin/env python3
"""Stage 1: torch.profiler kernel comparison — FP16 graph vs fused graph.

Since nsys is blocked by container kernel restrictions, use torch.profiler
which uses CUPTI directly. This gives per-kernel timing for the decode phase.
"""
import time, json, os, sys, gc, csv, io

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
    """Profile one arm using torch.profiler."""
    import torch
    import torch.profiler
    from vllm import LLM, SamplingParams

    log(f"--- ARM: {arm_name} (kv_cache_dtype={kv_cache_dtype}) ---")
    log(f"  creating LLM...")

    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=False,
    )

    prompt_512 = " ".join(["token"] * 512)
    sp = SamplingParams(max_tokens=256, temperature=0)

    # Warmup
    log(f"  warmup...")
    for i in range(3):
        llm.generate([prompt_512], sp)
        log(f"  warmup {i+1}/3")

    # Timing runs (without profiler overhead)
    log(f"  timing runs (5 trials)...")
    timing_results = []
    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = llm.generate([prompt_512], sp)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n = len(outputs[0].outputs[0].token_ids)
        itl = (elapsed * 1000) / max(n, 1)
        timing_results.append({"trial": trial, "elapsed_s": round(elapsed, 4),
                               "tokens": n, "itl_ms": round(itl, 3)})
        log(f"  trial {trial}: {elapsed:.3f}s, {n} tok, ITL={itl:.3f}ms")

    # Profiled run (1 iteration with torch.profiler)
    log(f"  profiled run (1 iteration with torch.profiler)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        with_stack=False,
        record_shapes=True,
    ) as prof:
        torch.cuda.synchronize()
        outputs = llm.generate([prompt_512], sp)
        torch.cuda.synchronize()

    # Extract kernel stats
    log(f"  extracting kernel stats...")

    # Top CUDA kernels by total time
    table = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=30
    )
    log(f"  top kernels:")
    for line in table.split("\n"):
        log(f"    {line}")

    # Save raw table
    with open(f"{ARTIFACTS}/torch_profile_{arm_name}_kernels.txt", "w") as f:
        f.write(table)

    # Save chrome trace for detailed analysis
    prof.export_chrome_trace(f"{ARTIFACTS}/torch_profile_{arm_name}_trace.json")
    log(f"  chrome trace saved")

    # Extract kernel summary as structured data
    kernel_data = []
    for evt in prof.key_averages():
        if evt.device_type is not None and evt.cuda_time_total > 0:
            kernel_data.append({
                "name": evt.key,
                "cuda_time_total_us": round(evt.cuda_time_total, 2),
                "cuda_time_avg_us": round(evt.cuda_time, 2),
                "count": evt.count,
                "cpu_time_total_us": round(evt.cpu_time_total, 2),
            })
    kernel_data.sort(key=lambda x: x["cuda_time_total_us"], reverse=True)

    # Save kernel data
    with open(f"{ARTIFACTS}/torch_profile_{arm_name}_kernels.json", "w") as f:
        json.dump(kernel_data, f, indent=2)

    itls = [r["itl_ms"] for r in timing_results]
    avg_itl = sum(itls) / len(itls)

    result = {
        "arm": arm_name,
        "kv_cache_dtype": kv_cache_dtype,
        "avg_itl_ms": round(avg_itl, 3),
        "timing_trials": timing_results,
        "n_kernel_types": len(kernel_data),
        "top_kernel": kernel_data[0]["name"] if kernel_data else None,
        "total_cuda_us": round(sum(k["cuda_time_total_us"] for k in kernel_data), 2),
    }

    log(f"  avg ITL: {avg_itl:.3f} ms, {len(kernel_data)} kernel types, "
        f"total CUDA: {result['total_cuda_us']:.0f} us")

    # Cleanup
    del llm, prof
    gc.collect()
    torch.cuda.empty_cache()

    return result

def main():
    arm_name = sys.argv[1] if len(sys.argv) > 1 else "fp16_graph"
    kv_dtype = "auto" if arm_name == "fp16_graph" else "int4_fused"

    log(f"=== Stage 1 torch.profiler: {arm_name} ===")
    append_event("PROCESS_STARTED", f"S1_TORCH_PROF_{arm_name}",
                 f"torch.profiler kernel comparison for {arm_name}")

    os.makedirs(ARTIFACTS, exist_ok=True)
    result = profile_arm(arm_name, kv_dtype)

    # Save results
    with open(f"{ARTIFACTS}/stage1_{arm_name}_torch_results.json", "w") as f:
        json.dump(result, f, indent=2)

    append_event("CHECKPOINT", f"S1_{arm_name}_DONE",
                 f"{arm_name} profiled: avg_itl={result['avg_itl_ms']}ms, "
                 f"{result['n_kernel_types']} kernel types")

    log(f"=== {arm_name} complete ===")

    # Force exit to avoid hanging EngineCore cleanup
    os._exit(0)

if __name__ == "__main__":
    main()
