#!/usr/bin/env python3
"""Stage 1: NSYS decode comparison via offline inference with cudaProfilerApi capture.

For each arm (FP16 graph, fused graph):
1. Create vLLM LLM with CUDA graphs enabled
2. Warmup (compile graphs)
3. cudaProfilerStart → steady-state decode → cudaProfilerStop
4. nsys wraps the whole process with --capture-range=cudaProfilerApi

This gives clean per-kernel traces for exactly the decode phase.
"""
import time, json, os, sys, gc

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

def profile_arm(arm_name, kv_cache_dtype, enforce_eager):
    """Profile one arm using offline LLM inference."""
    import torch
    from vllm import LLM, SamplingParams

    log(f"--- ARM: {arm_name} (kv_cache_dtype={kv_cache_dtype}, enforce_eager={enforce_eager}) ---")

    # Create LLM
    log(f"  creating LLM...")
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
    )

    # Warmup: compile CUDA graphs
    log(f"  warmup (compiling CUDA graphs)...")
    sp_warmup = SamplingParams(max_tokens=20, temperature=0)
    for i in range(3):
        llm.generate(["Hello world warmup test"], sp_warmup)
        log(f"  warmup {i+1}/3 done")

    # Extra warmup with the actual prompt shape
    prompt_512 = " ".join(["token"] * 512)
    sp_decode = SamplingParams(max_tokens=256, temperature=0)
    for i in range(2):
        llm.generate([prompt_512], sp_decode)
        log(f"  shape warmup {i+1}/2 done")

    torch.cuda.synchronize()
    time.sleep(1)

    # Profile: bracket with cudaProfiler API
    log(f"  starting profiled decode (5 iterations)...")
    torch.cuda.cudart().cudaProfilerStart()

    results = []
    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = llm.generate([prompt_512], sp_decode)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n_tokens = len(outputs[0].outputs[0].token_ids)
        itl_ms = (elapsed * 1000) / max(n_tokens, 1)
        results.append({
            "trial": trial,
            "elapsed_s": round(elapsed, 4),
            "tokens": n_tokens,
            "itl_ms": round(itl_ms, 3),
        })
        log(f"  trial {trial}: {elapsed:.3f}s, {n_tokens} tok, ITL={itl_ms:.3f}ms")

    torch.cuda.cudart().cudaProfilerStop()
    log(f"  profiled decode complete")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return results

def main():
    arm_name = sys.argv[1] if len(sys.argv) > 1 else "fp16_graph"

    arms = {
        "fp16_graph": ("auto", False),
        "fused_graph": ("int4_fused", False),
    }

    if arm_name not in arms:
        print(f"Unknown arm: {arm_name}. Choose from: {list(arms.keys())}")
        sys.exit(1)

    kv_dtype, enforce_eager = arms[arm_name]
    log(f"=== Stage 1 offline profile: {arm_name} ===")
    append_event("PROCESS_STARTED", f"S1_{arm_name}", f"starting offline NSYS profile for {arm_name}")

    results = profile_arm(arm_name, kv_dtype, enforce_eager)

    # Save results
    results_file = f"{ARTIFACTS}/stage1_{arm_name}_results.json"
    with open(results_file, "w") as f:
        json.dump({"arm": arm_name, "trials": results}, f, indent=2)
    log(f"  results saved to {results_file}")

    # Summary
    itls = [r["itl_ms"] for r in results if r.get("itl_ms")]
    if itls:
        avg_itl = sum(itls) / len(itls)
        log(f"  avg ITL: {avg_itl:.3f} ms over {len(itls)} trials")

    append_event("CHECKPOINT", f"S1_{arm_name}_DONE", f"{arm_name} profile complete, avg ITL={avg_itl:.3f}ms" if itls else f"{arm_name} profile complete")

if __name__ == "__main__":
    main()
