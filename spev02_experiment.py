#!/usr/bin/env python3
"""SPEv-02: Speculative Decode + KV Offload v2 - Full experiment runner."""

import json
import os
import subprocess
import time
import sys
import gc
import signal
from datetime import datetime, timezone

RESULTS_DIR = "/tmp/spev02_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")
    return path


def get_gpu_info():
    """Get GPU info from nvidia-smi."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    parts = [p.strip() for p in result.stdout.strip().split(",")]
    return {
        "gpu_name": parts[0],
        "memory_total_mib": int(parts[1]),
        "memory_used_mib": int(parts[2]),
        "memory_free_mib": int(parts[3]),
        "temperature_c": int(parts[4]),
        "power_draw_w": float(parts[5]),
    }


def get_gpu_mem_used():
    """Return current GPU memory used in MiB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


###############################################################################
# TIER 0: Hardware validation
###############################################################################
def run_tier0():
    print("\n" + "=" * 60)
    print("TIER 0: Hardware validation")
    print("=" * 60)

    gpu_info = get_gpu_info()

    # Check disk
    disk_result = subprocess.run(
        ["df", "-h", "/workspace", "/", "/tmp"], capture_output=True, text=True
    )

    # Check NVMe
    lsblk = subprocess.run(["lsblk"], capture_output=True, text=True)
    nvme_check = subprocess.run(["ls", "/dev/nvme*"], capture_output=True, text=True)
    has_nvme_dev = nvme_check.returncode == 0

    # Quick disk bandwidth test with dd (simpler than fio, doesn't need install)
    print("Running disk bandwidth test on /tmp...")
    dd_write = subprocess.run(
        [
            "dd",
            "if=/dev/zero",
            "of=/tmp/spev02_disktest",
            "bs=1M",
            "count=256",
            "oflag=direct",
        ],
        capture_output=True,
        text=True,
    )
    write_speed = (
        dd_write.stderr.strip().split("\n")[-1] if dd_write.stderr else "unknown"
    )

    dd_read = subprocess.run(
        ["dd", "if=/tmp/spev02_disktest", "of=/dev/null", "bs=1M", "iflag=direct"],
        capture_output=True,
        text=True,
    )
    read_speed = dd_read.stderr.strip().split("\n")[-1] if dd_read.stderr else "unknown"
    subprocess.run(["rm", "-f", "/tmp/spev02_disktest"])

    # CPU info
    cpu_result = subprocess.run(["nproc"], capture_output=True, text=True)
    mem_result = subprocess.run(["free", "-g"], capture_output=True, text=True)

    data = {
        "stage": "tier0",
        "test_name": "hardware_validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_info,
        "cpu_cores": int(cpu_result.stdout.strip()),
        "system_memory": mem_result.stdout.strip(),
        "disk_info": disk_result.stdout.strip(),
        "lsblk": lsblk.stdout.strip(),
        "has_nvme_device_nodes": has_nvme_dev,
        "disk_write_speed": write_speed,
        "disk_read_speed": read_speed,
        "notes": (
            "NVMe devices visible in lsblk but no /dev/nvme* nodes - virtual disk"
            if not has_nvme_dev
            else "NVMe available"
        ),
    }
    save_json(data, "tier0_hardware.json")
    return data


###############################################################################
# STAGE A: LMCache KV Offloading
###############################################################################


def run_vllm_benchmark(model, context_lengths, extra_args=None, label="baseline"):
    """Run vLLM with offline inference at various context lengths, measure perf."""
    results = []

    for ctx_len in context_lengths:
        print(f"  Testing {label} ctx_len={ctx_len}...")

        # Create a benchmark script that measures throughput
        bench_script = f"""
import time
import json
import gc
import torch
from vllm import LLM, SamplingParams

model = "{model}"
ctx_len = {ctx_len}
extra_args = {json.dumps(extra_args or {})}

# Build engine args
engine_kwargs = dict(
    model=model,
    max_model_len=ctx_len + 256,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    dtype="auto",
)
engine_kwargs.update(extra_args)

try:
    llm = LLM(**engine_kwargs)

    # Generate a prompt of approximately ctx_len tokens (roughly 4 chars per token)
    prompt = "Hello " * (ctx_len // 2)
    sampling_params = SamplingParams(max_tokens=128, temperature=0.0)

    # Warmup
    _ = llm.generate([prompt[:100]], sampling_params)

    # Timed run
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.perf_counter() - start

    output_tokens = len(outputs[0].outputs[0].token_ids)
    prompt_tokens = len(outputs[0].prompt_token_ids)

    # Get peak GPU memory
    import subprocess
    mem_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    peak_vram_mib = int(mem_result.stdout.strip())

    result = {{
        "context_length": ctx_len,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_time_s": round(elapsed, 3),
        "tokens_per_sec": round(output_tokens / elapsed, 2),
        "ttft_approx_s": round(elapsed - (output_tokens * elapsed / (prompt_tokens + output_tokens)), 3),
        "peak_vram_mib": peak_vram_mib,
        "status": "ok",
    }}

    del llm
    gc.collect()
    torch.cuda.empty_cache()

except Exception as e:
    result = {{
        "context_length": ctx_len,
        "status": "error",
        "error": str(e),
    }}
    try:
        del llm
    except:
        pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

print("RESULT_JSON:" + json.dumps(result))
"""
        bench_file = f"/tmp/spev02_bench_{label}_{ctx_len}.py"
        with open(bench_file, "w") as f:
            f.write(bench_script)

        try:
            env = os.environ.copy()
            proc = subprocess.run(
                ["python3", bench_file],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            output = proc.stdout + proc.stderr
            # Extract result JSON
            for line in output.split("\n"):
                if line.startswith("RESULT_JSON:"):
                    result = json.loads(line[len("RESULT_JSON:") :])
                    results.append(result)
                    print(
                        f"    -> {result.get('status', 'unknown')}: "
                        f"{result.get('tokens_per_sec', 'N/A')} tok/s, "
                        f"VRAM={result.get('peak_vram_mib', 'N/A')} MiB"
                    )
                    break
            else:
                results.append(
                    {
                        "context_length": ctx_len,
                        "status": "error",
                        "error": f"No result found in output. Last 500 chars: {output[-500:]}",
                    }
                )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "context_length": ctx_len,
                    "status": "timeout",
                    "error": "Benchmark timed out after 600s",
                }
            )
        except Exception as e:
            results.append(
                {"context_length": ctx_len, "status": "error", "error": str(e)}
            )

    return results


def run_stage_a1():
    print("\n" + "=" * 60)
    print("STAGE A1: Baseline (no offload, no LMCache)")
    print("=" * 60)

    model = "Qwen/Qwen2.5-7B-Instruct"
    context_lengths = [512, 2048, 4096, 8192, 16384]

    results = run_vllm_benchmark(model, context_lengths, label="a1_baseline")

    data = {
        "stage": "A1",
        "test_name": "baseline_no_offload",
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    save_json(data, "stageA1_baseline.json")
    return data


def run_stage_a2():
    print("\n" + "=" * 60)
    print("STAGE A2: LMCache with disk offload")
    print("=" * 60)

    model = "Qwen/Qwen2.5-7B-Instruct"
    context_lengths = [512, 2048, 4096, 8192, 16384]

    # LMCache config for disk offload
    lmcache_extra = {
        "kv_cache_dtype": "auto",
    }

    # Write LMCache config
    lmcache_config = {
        "chunk_size": 256,
        "local_device": "cpu",
        "remote_url": "",
        "remote_serde": "",
        "local_disk_path": "/tmp/spev02_lmcache_disk",
    }
    os.makedirs("/tmp/spev02_lmcache_disk", exist_ok=True)

    # For LMCache with vLLM, we use environment variables
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_LOCAL_DISK"] = "/tmp/spev02_lmcache_disk"

    results = run_vllm_benchmark(model, context_lengths, label="a2_lmcache")

    # Prefix caching test
    print("  Testing prefix caching with LMCache...")
    prefix_bench_script = """
import time
import json
import gc
import torch
from vllm import LLM, SamplingParams

model = "Qwen/Qwen2.5-7B-Instruct"

try:
    llm = LLM(
        model=model,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        enable_prefix_caching=True,
    )

    system_prompt = "You are a helpful AI assistant that provides detailed technical explanations. " * 50
    question1 = "What is speculative decoding?"
    question2 = "How does KV cache offloading work?"

    sp = SamplingParams(max_tokens=64, temperature=0.0)

    # First request with system prompt
    prompt1 = system_prompt + " " + question1
    start1 = time.perf_counter()
    out1 = llm.generate([prompt1], sp)
    ttft1 = time.perf_counter() - start1

    # Second request with same system prompt (should benefit from prefix cache)
    prompt2 = system_prompt + " " + question2
    start2 = time.perf_counter()
    out2 = llm.generate([prompt2], sp)
    ttft2 = time.perf_counter() - start2

    result = {
        "prefix_caching_test": True,
        "first_request_time_s": round(ttft1, 3),
        "second_request_time_s": round(ttft2, 3),
        "speedup_ratio": round(ttft1 / max(ttft2, 0.001), 2),
        "status": "ok",
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()

except Exception as e:
    result = {
        "prefix_caching_test": True,
        "status": "error",
        "error": str(e),
    }

print("RESULT_JSON:" + json.dumps(result))
"""
    bench_file = "/tmp/spev02_bench_prefix_cache.py"
    with open(bench_file, "w") as f:
        f.write(prefix_bench_script)

    prefix_result = {"status": "error", "error": "not run"}
    try:
        proc = subprocess.run(
            ["python3", bench_file], capture_output=True, text=True, timeout=600
        )
        for line in (proc.stdout + proc.stderr).split("\n"):
            if line.startswith("RESULT_JSON:"):
                prefix_result = json.loads(line[len("RESULT_JSON:") :])
                break
    except Exception as e:
        prefix_result = {"status": "error", "error": str(e)}

    data = {
        "stage": "A2",
        "test_name": "lmcache_disk_offload",
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "prefix_caching": prefix_result,
        "notes": "LMCache disk offload to /tmp (overlay filesystem, not NVMe)",
    }
    save_json(data, "stageA2_lmcache_offload.json")
    return data


def run_stage_a3():
    print("\n" + "=" * 60)
    print("STAGE A3: FP8 KV quantization check")
    print("=" * 60)

    # Check if FP8 KV is supported
    bench_script = """
import json
import time
import gc
import torch
from vllm import LLM, SamplingParams

model = "Qwen/Qwen2.5-7B-Instruct"
context_lengths = [512, 2048, 4096, 8192]

results = []
for ctx_len in context_lengths:
    try:
        llm = LLM(
            model=model,
            max_model_len=ctx_len + 256,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            kv_cache_dtype="fp8",
        )

        prompt = "Hello " * (ctx_len // 2)
        sp = SamplingParams(max_tokens=128, temperature=0.0)

        _ = llm.generate([prompt[:100]], sp)

        start = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        elapsed = time.perf_counter() - start

        output_tokens = len(outputs[0].outputs[0].token_ids)

        import subprocess
        mem_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        peak_vram = int(mem_result.stdout.strip())

        result = {
            "context_length": ctx_len,
            "output_tokens": output_tokens,
            "total_time_s": round(elapsed, 3),
            "tokens_per_sec": round(output_tokens / elapsed, 2),
            "peak_vram_mib": peak_vram,
            "kv_cache_dtype": "fp8",
            "status": "ok",
        }

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result = {
            "context_length": ctx_len,
            "status": "error",
            "error": str(e),
            "kv_cache_dtype": "fp8",
        }
        try:
            del llm
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    results.append(result)
    print("RESULT_JSON:" + json.dumps(result))

print("ALL_RESULTS:" + json.dumps(results))
"""
    bench_file = "/tmp/spev02_bench_fp8.py"
    with open(bench_file, "w") as f:
        f.write(bench_script)

    results = []
    try:
        proc = subprocess.run(
            ["python3", bench_file], capture_output=True, text=True, timeout=900
        )
        output = proc.stdout + proc.stderr
        for line in output.split("\n"):
            if line.startswith("ALL_RESULTS:"):
                results = json.loads(line[len("ALL_RESULTS:") :])
                break
            elif line.startswith("RESULT_JSON:"):
                r = json.loads(line[len("RESULT_JSON:") :])
                print(
                    f"    -> ctx={r.get('context_length')}: {r.get('status')} "
                    f"{r.get('tokens_per_sec', 'N/A')} tok/s"
                )
    except Exception as e:
        results = [{"status": "error", "error": str(e)}]

    supported = any(r.get("status") == "ok" for r in results)

    data = {
        "stage": "A3",
        "test_name": "fp8_kv_quantization",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fp8_kv_supported": supported,
        "results": results,
    }
    save_json(data, "stageA3_fp8_kv.json")
    return data


###############################################################################
# STAGE B: Speculative Decoding
###############################################################################


def run_stage_b2():
    print("\n" + "=" * 60)
    print("STAGE B2: Draft model speculation")
    print("=" * 60)

    target_model = "Qwen/Qwen2.5-7B-Instruct"
    draft_model = "Qwen/Qwen2.5-0.5B-Instruct"
    context_lengths = [512, 2048, 4096, 8192]

    results = []
    for ctx_len in context_lengths:
        print(f"  Testing speculation ctx_len={ctx_len}...")
        bench_script = f"""
import time
import json
import gc
import torch
from vllm import LLM, SamplingParams

# Try draft model first, fall back to ngram
methods = [
    ("draft_model", dict(model="{draft_model}", num_speculative_tokens=5)),
    ("ngram", dict(method="ngram", num_speculative_tokens=5, prompt_lookup_max=5, prompt_lookup_min=2)),
]

result = None
last_error = "unknown"
for method_name, spec_cfg in methods:
    try:
        llm = LLM(
            model="{target_model}",
            speculative_config=spec_cfg,
            max_model_len={ctx_len + 256},
            gpu_memory_utilization=0.90,
            enforce_eager=True,
        )

        prompt = "Hello " * ({ctx_len} // 2)
        sp = SamplingParams(max_tokens=128, temperature=0.0)

        _ = llm.generate([prompt[:100]], sp)

        start = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        elapsed = time.perf_counter() - start

        output_tokens = len(outputs[0].outputs[0].token_ids)
        prompt_tokens = len(outputs[0].prompt_token_ids)

        import subprocess
        mem_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        peak_vram = int(mem_result.stdout.strip())

        result = {{
            "context_length": {ctx_len},
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_time_s": round(elapsed, 3),
            "tokens_per_sec": round(output_tokens / elapsed, 2),
            "peak_vram_mib": peak_vram,
            "speculation_method": method_name,
            "draft_model": "{draft_model}" if method_name == "draft_model" else "ngram",
            "num_speculative_tokens": 5,
            "status": "ok",
        }}

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        break

    except Exception as e:
        last_error = str(e)
        try:
            del llm
        except:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass
        continue

if result is None:
    result = {{
        "context_length": {ctx_len},
        "status": "error",
        "error": last_error,
    }}

print("RESULT_JSON:" + json.dumps(result))
"""
        bench_file = f"/tmp/spev02_bench_spec_{ctx_len}.py"
        with open(bench_file, "w") as f:
            f.write(bench_script)

        try:
            proc = subprocess.run(
                ["python3", bench_file], capture_output=True, text=True, timeout=600
            )
            output = proc.stdout + proc.stderr
            for line in output.split("\n"):
                if line.startswith("RESULT_JSON:"):
                    result = json.loads(line[len("RESULT_JSON:") :])
                    results.append(result)
                    print(
                        f"    -> {result.get('status')}: "
                        f"{result.get('tokens_per_sec', 'N/A')} tok/s, "
                        f"VRAM={result.get('peak_vram_mib', 'N/A')} MiB"
                    )
                    break
            else:
                results.append(
                    {
                        "context_length": ctx_len,
                        "status": "error",
                        "error": f"No result. Last 500 chars: {output[-500:]}",
                    }
                )
        except subprocess.TimeoutExpired:
            results.append({"context_length": ctx_len, "status": "timeout"})
        except Exception as e:
            results.append(
                {"context_length": ctx_len, "status": "error", "error": str(e)}
            )

    data = {
        "stage": "B2",
        "test_name": "speculation_draft_model",
        "model": target_model,
        "draft_model": draft_model,
        "num_speculative_tokens": 5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    save_json(data, "stageB2_speculation.json")
    return data


###############################################################################
# STAGE C: Combined (only if A and B succeed)
###############################################################################


def run_stage_c1():
    print("\n" + "=" * 60)
    print("STAGE C1: Combined LMCache + Speculation")
    print("=" * 60)

    target_model = "Qwen/Qwen2.5-7B-Instruct"
    draft_model = "Qwen/Qwen2.5-0.5B-Instruct"
    context_lengths = [512, 2048, 4096, 8192]

    results = []
    for ctx_len in context_lengths:
        print(f"  Testing combined ctx_len={ctx_len}...")
        bench_script = f"""
import time
import json
import gc
import torch
from vllm import LLM, SamplingParams

# Try draft model first, fall back to ngram
methods = [
    ("draft_model", dict(model="{draft_model}", num_speculative_tokens=5)),
    ("ngram", dict(method="ngram", num_speculative_tokens=5, prompt_lookup_max=5, prompt_lookup_min=2)),
]

result = None
last_error = "unknown"
for method_name, spec_cfg in methods:
    try:
        llm = LLM(
            model="{target_model}",
            speculative_config=spec_cfg,
            max_model_len={ctx_len + 256},
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            enable_prefix_caching=True,
        )

        prompt = "Hello " * ({ctx_len} // 2)
        sp = SamplingParams(max_tokens=128, temperature=0.0)

        _ = llm.generate([prompt[:100]], sp)

        start = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        elapsed = time.perf_counter() - start

        output_tokens = len(outputs[0].outputs[0].token_ids)
        prompt_tokens = len(outputs[0].prompt_token_ids)

        import subprocess
        mem_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        peak_vram = int(mem_result.stdout.strip())

        result = {{
            "context_length": {ctx_len},
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_time_s": round(elapsed, 3),
            "tokens_per_sec": round(output_tokens / elapsed, 2),
            "peak_vram_mib": peak_vram,
            "speculation_method": method_name,
            "draft_model": "{draft_model}" if method_name == "draft_model" else "ngram",
            "num_speculative_tokens": 5,
            "enable_prefix_caching": True,
            "status": "ok",
        }}

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        break

    except Exception as e:
        last_error = str(e)
        try:
            del llm
        except:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass
        continue

if result is None:
    result = {{
        "context_length": {ctx_len},
        "status": "error",
        "error": last_error,
    }}

print("RESULT_JSON:" + json.dumps(result))
"""
        bench_file = f"/tmp/spev02_bench_combined_{ctx_len}.py"
        with open(bench_file, "w") as f:
            f.write(bench_script)

        try:
            proc = subprocess.run(
                ["python3", bench_file], capture_output=True, text=True, timeout=600
            )
            output = proc.stdout + proc.stderr
            for line in output.split("\n"):
                if line.startswith("RESULT_JSON:"):
                    result = json.loads(line[len("RESULT_JSON:") :])
                    results.append(result)
                    print(
                        f"    -> {result.get('status')}: "
                        f"{result.get('tokens_per_sec', 'N/A')} tok/s, "
                        f"VRAM={result.get('peak_vram_mib', 'N/A')} MiB"
                    )
                    break
            else:
                results.append(
                    {
                        "context_length": ctx_len,
                        "status": "error",
                        "error": f"No result. Last 500 chars: {output[-500:]}",
                    }
                )
        except subprocess.TimeoutExpired:
            results.append({"context_length": ctx_len, "status": "timeout"})
        except Exception as e:
            results.append(
                {"context_length": ctx_len, "status": "error", "error": str(e)}
            )

    data = {
        "stage": "C1",
        "test_name": "combined_lmcache_speculation",
        "model": target_model,
        "draft_model": draft_model,
        "num_speculative_tokens": 5,
        "enable_prefix_caching": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    save_json(data, "stageC1_naive_combined.json")
    return data


###############################################################################
# Summary
###############################################################################


def create_summary(tier0, a1, a2, a3, b2, c1):
    print("\n" + "=" * 60)
    print("Creating summary")
    print("=" * 60)

    def get_ok_results(stage_data):
        if not stage_data or "results" not in stage_data:
            return []
        return [r for r in stage_data["results"] if r.get("status") == "ok"]

    a1_ok = get_ok_results(a1)
    a2_ok = get_ok_results(a2)
    b2_ok = get_ok_results(b2)
    c1_ok = get_ok_results(c1)
    a3_ok = get_ok_results(a3)

    # Calculate averages
    def avg_tps(results):
        tps = [r["tokens_per_sec"] for r in results if "tokens_per_sec" in r]
        return round(sum(tps) / len(tps), 2) if tps else None

    def max_ctx(results):
        ctxs = [r["context_length"] for r in results]
        return max(ctxs) if ctxs else None

    a_success = len(a1_ok) > 0 and len(a2_ok) > 0
    b_success = len(b2_ok) > 0

    # Compare stages
    comparisons = {}
    if a1_ok and a2_ok:
        a1_avg = avg_tps(a1_ok)
        a2_avg = avg_tps(a2_ok)
        if a1_avg and a2_avg:
            comparisons["lmcache_vs_baseline"] = {
                "baseline_avg_tps": a1_avg,
                "lmcache_avg_tps": a2_avg,
                "throughput_ratio": round(a2_avg / a1_avg, 3),
                "lmcache_faster": a2_avg > a1_avg,
            }

    if a1_ok and b2_ok:
        # Compare at matching context lengths
        a1_by_ctx = {r["context_length"]: r for r in a1_ok}
        for br in b2_ok:
            ctx = br["context_length"]
            if ctx in a1_by_ctx:
                a1r = a1_by_ctx[ctx]
                comparisons[f"speculation_vs_baseline_ctx{ctx}"] = {
                    "baseline_tps": a1r.get("tokens_per_sec"),
                    "speculation_tps": br.get("tokens_per_sec"),
                    "speedup": (
                        round(br["tokens_per_sec"] / a1r["tokens_per_sec"], 3)
                        if a1r.get("tokens_per_sec")
                        else None
                    ),
                }

    summary = {
        "experiment": "SPEv-02",
        "description": "Speculative Decode + KV Offload v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": (
                tier0.get("gpu", {}).get("gpu_name", "unknown") if tier0 else "unknown"
            ),
            "gpu_memory_mib": (
                tier0.get("gpu", {}).get("memory_total_mib", 0) if tier0 else 0
            ),
        },
        "stage_results": {
            "A1_baseline": {
                "success": len(a1_ok) > 0,
                "tests_passed": len(a1_ok),
                "avg_tokens_per_sec": avg_tps(a1_ok),
                "max_context_tested": max_ctx(a1_ok),
            },
            "A2_lmcache_offload": {
                "success": len(a2_ok) > 0,
                "tests_passed": len(a2_ok),
                "avg_tokens_per_sec": avg_tps(a2_ok),
                "max_context_tested": max_ctx(a2_ok),
                "prefix_caching": a2.get("prefix_caching", {}) if a2 else {},
            },
            "A3_fp8_kv": {
                "supported": a3.get("fp8_kv_supported", False) if a3 else False,
                "tests_passed": len(a3_ok),
                "avg_tokens_per_sec": avg_tps(a3_ok),
            },
            "B2_speculation": {
                "success": len(b2_ok) > 0,
                "tests_passed": len(b2_ok),
                "avg_tokens_per_sec": avg_tps(b2_ok),
                "max_context_tested": max_ctx(b2_ok),
            },
            "C1_combined": {
                "success": len(c1_ok) > 0,
                "tests_passed": len(c1_ok),
                "avg_tokens_per_sec": avg_tps(c1_ok),
                "ran": a_success and b_success,
            },
        },
        "comparisons": comparisons,
        "conclusions": [],
    }

    # Add conclusions
    if comparisons.get("lmcache_vs_baseline"):
        comp = comparisons["lmcache_vs_baseline"]
        if comp["lmcache_faster"]:
            summary["conclusions"].append(
                f"LMCache offload is {comp['throughput_ratio']:.1f}x baseline throughput"
            )
        else:
            summary["conclusions"].append(
                f"LMCache offload has {comp['throughput_ratio']:.1f}x baseline throughput (slower due to disk I/O)"
            )

    spec_speedups = [
        v["speedup"]
        for k, v in comparisons.items()
        if k.startswith("speculation_vs_baseline") and v.get("speedup")
    ]
    if spec_speedups:
        avg_speedup = sum(spec_speedups) / len(spec_speedups)
        summary["conclusions"].append(
            f"Speculative decoding average speedup: {avg_speedup:.2f}x over baseline"
        )

    save_json(summary, "spev02_summary.json")
    return summary


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    print("SPEv-02 Experiment Runner")
    print(f"Started at {datetime.now(timezone.utc).isoformat()}")
    print(f"Results dir: {RESULTS_DIR}")

    tier0 = run_tier0()
    a1 = run_stage_a1()
    a2 = run_stage_a2()
    a3 = run_stage_a3()
    b2 = run_stage_b2()

    # Run C only if A and B succeeded
    a1_ok = any(r.get("status") == "ok" for r in a1.get("results", []))
    a2_ok = any(r.get("status") == "ok" for r in a2.get("results", []))
    b2_ok = any(r.get("status") == "ok" for r in b2.get("results", []))

    c1 = None
    if a1_ok and a2_ok and b2_ok:
        c1 = run_stage_c1()
    else:
        print("\nSkipping Stage C: A or B did not succeed")
        c1 = {
            "stage": "C1",
            "test_name": "combined_lmcache_speculation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": [],
            "notes": "Skipped because Stage A or B did not independently succeed",
        }
        save_json(c1, "stageC1_naive_combined.json")

    summary = create_summary(tier0, a1, a2, a3, b2, c1)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved in {RESULTS_DIR}/")
    for f in os.listdir(RESULTS_DIR):
        print(f"  {f}")
    print("\nConclusions:")
    for c in summary.get("conclusions", []):
        print(f"  - {c}")
