#!/usr/bin/env python3
"""Stage 1: NSYS steady-state decode comparison — FP16 graph vs fused graph.

Uses nsys launch/start/stop to capture clean decode-phase traces
for both configurations, then summarizes kernel topology.
"""
import subprocess, time, json, os, sys, signal, traceback

RUN_ROOT = "/tmp/acp-runs/20260404T2339Z-residgap-h100-cldopus-a1"
ARTIFACTS = f"{RUN_ROOT}/artifacts"
NSYS = "/opt/nvidia/nsight-compute/2025.2.0/host/target-linux-x64/nsys"
MODEL = "marin-community/marin-8b-base"
PORT = 8199
BASE_URL = f"http://localhost:{PORT}"

ARMS = [
    {
        "name": "fp16_graph",
        "kv_cache_dtype": "auto",
        "enforce_eager": False,
        "label": "FP16 graph (FLASH_ATTN, FULL+PIECEWISE)",
    },
    {
        "name": "fused_graph",
        "kv_cache_dtype": "int4_fused",
        "enforce_eager": False,
        "label": "Fused INT4 graph (fused_int4, FULL+PIECEWISE)",
    },
]

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

def kill_servers():
    os.system("pkill -9 -f 'vllm.entrypoints' 2>/dev/null")
    os.system(f"{NSYS} shutdown 2>/dev/null")
    time.sleep(3)

def wait_server(timeout=180):
    import urllib.request
    for i in range(timeout):
        try:
            req = urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
            if req.getcode() == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def warmup(n=5):
    """Send short requests to compile CUDA graphs."""
    import urllib.request
    for i in range(n):
        data = json.dumps({
            "model": MODEL,
            "prompt": "Hello world test warmup",
            "max_tokens": 20,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/v1/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            log(f"  warmup {i+1}/{n}: HTTP {resp.getcode()}")
        except Exception as e:
            log(f"  warmup {i+1}/{n}: ERROR {e}")
    # Extra sleep to let graphs settle
    time.sleep(2)

def decode_benchmark(n_trials=5, prompt_tokens=512, max_tokens=256):
    """Send decode-heavy requests and collect timing."""
    import urllib.request
    prompt = " ".join(["token"] * prompt_tokens)
    results = []
    for trial in range(n_trials):
        data = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/v1/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            body = json.loads(resp.read())
            elapsed = time.time() - t0
            usage = body.get("usage", {})
            ct = usage.get("completion_tokens", 0)
            pt = usage.get("prompt_tokens", 0)
            itl_ms = (elapsed * 1000) / max(ct, 1) if ct > 0 else None
            results.append({
                "trial": trial, "elapsed_s": round(elapsed, 4),
                "completion_tokens": ct, "prompt_tokens": pt,
                "itl_ms": round(itl_ms, 3) if itl_ms else None,
            })
            log(f"  trial {trial}: {elapsed:.3f}s, {ct} tokens, ITL={itl_ms:.3f}ms")
        except Exception as e:
            log(f"  trial {trial}: ERROR {e}")
            results.append({"trial": trial, "error": str(e)})
    return results

def run_nsys_arm(arm):
    """Run one NSYS profiled arm: launch server under nsys, warmup, profile decode."""
    name = arm["name"]
    kv_dtype = arm["kv_cache_dtype"]
    enforce_eager = arm["enforce_eager"]
    nsys_session = f"s1_{name}"
    nsys_output = f"{ARTIFACTS}/nsys_{name}"

    log(f"--- ARM: {arm['label']} ---")
    kill_servers()

    # Build server command
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--max-model-len", "4096",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
        "--kv-cache-dtype", kv_dtype,
        "--port", str(PORT),
    ]
    if enforce_eager:
        server_cmd.append("--enforce-eager")

    # Launch server under nsys (profiling paused initially)
    nsys_cmd = [
        NSYS, "launch",
        "--session-new", nsys_session,
        "--trace", "cuda,nvtx",
        "--sample", "none",
    ] + server_cmd

    log(f"  launching nsys session '{nsys_session}'...")
    server_log = open(f"{ARTIFACTS}/server-{name}.log", "w")
    proc = subprocess.Popen(
        nsys_cmd, stdout=server_log, stderr=subprocess.STDOUT,
        cwd="/data/vllm", preexec_fn=os.setsid,
    )
    log(f"  server PID={proc.pid}")

    # Wait for server ready
    if not wait_server(180):
        log(f"  FAIL: server did not start in 180s")
        kill_servers()
        return None

    log(f"  server ready, warming up...")
    warmup(5)

    # Start NSYS profiling
    log(f"  starting NSYS capture...")
    start_cmd = [NSYS, "start", "--session", nsys_session]
    subprocess.run(start_cmd, timeout=10, capture_output=True)

    # Run decode benchmark during capture
    log(f"  running decode benchmark (5 trials, decode-heavy-512)...")
    bench_results = decode_benchmark(n_trials=5, prompt_tokens=512, max_tokens=256)

    # Stop NSYS profiling
    log(f"  stopping NSYS capture...")
    stop_cmd = [NSYS, "stop", "--session", nsys_session,
                "--output", nsys_output]
    stop_result = subprocess.run(stop_cmd, timeout=120, capture_output=True, text=True)
    log(f"  nsys stop: {stop_result.returncode}")
    if stop_result.stdout:
        log(f"  nsys stdout: {stop_result.stdout.strip()[:200]}")
    if stop_result.stderr:
        log(f"  nsys stderr: {stop_result.stderr.strip()[:200]}")

    # Shutdown nsys session
    subprocess.run([NSYS, "shutdown", "--session", nsys_session],
                   timeout=30, capture_output=True)

    # Kill server
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except:
        pass
    kill_servers()

    # Check for nsys report file
    nsys_file = f"{nsys_output}.nsys-rep"
    has_nsys = os.path.exists(nsys_file)
    nsys_size = os.path.getsize(nsys_file) if has_nsys else 0
    log(f"  nsys report: exists={has_nsys}, size={nsys_size}")

    # Generate nsys stats if report exists
    stats_summary = None
    if has_nsys and nsys_size > 0:
        try:
            stats_cmd = [NSYS, "stats", "--report", "cuda_gpu_kern_sum",
                         "--format", "csv", nsys_file]
            stats_result = subprocess.run(stats_cmd, timeout=120, capture_output=True, text=True)
            stats_file = f"{ARTIFACTS}/nsys_stats_{name}_kern.csv"
            with open(stats_file, "w") as f:
                f.write(stats_result.stdout)
            stats_summary = stats_result.stdout[:2000]
            log(f"  kernel stats saved to {stats_file}")

            # Also get CUDA API stats
            api_cmd = [NSYS, "stats", "--report", "cuda_api_sum",
                       "--format", "csv", nsys_file]
            api_result = subprocess.run(api_cmd, timeout=120, capture_output=True, text=True)
            api_file = f"{ARTIFACTS}/nsys_stats_{name}_api.csv"
            with open(api_file, "w") as f:
                f.write(api_result.stdout)
            log(f"  API stats saved to {api_file}")
        except Exception as e:
            log(f"  stats generation failed: {e}")

    return {
        "arm": name,
        "label": arm["label"],
        "benchmark": bench_results,
        "nsys_report": nsys_file if has_nsys else None,
        "nsys_size_bytes": nsys_size,
        "stats_summary": stats_summary,
    }

def main():
    log("=== Stage 1: NSYS steady-state decode comparison ===")
    append_event("CHECKPOINT", "S1_NSYS_PROFILE", "starting Stage 1 NSYS decode comparison")

    os.makedirs(ARTIFACTS, exist_ok=True)
    results = {}

    for arm in ARMS:
        try:
            result = run_nsys_arm(arm)
            if result:
                results[arm["name"]] = result
            else:
                results[arm["name"]] = {"arm": arm["name"], "error": "server failed to start"}
        except Exception as e:
            log(f"  ARM {arm['name']} FAILED: {e}")
            traceback.print_exc()
            results[arm["name"]] = {"arm": arm["name"], "error": str(e)}
            kill_servers()

    # Save combined results
    results_file = f"{ARTIFACTS}/stage1_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Stage 1 results saved to {results_file}")

    # Summary comparison
    fp16 = results.get("fp16_graph", {})
    fused = results.get("fused_graph", {})
    fp16_bench = fp16.get("benchmark", [])
    fused_bench = fused.get("benchmark", [])

    fp16_itls = [t["itl_ms"] for t in fp16_bench if t.get("itl_ms")]
    fused_itls = [t["itl_ms"] for t in fused_bench if t.get("itl_ms")]

    if fp16_itls and fused_itls:
        fp16_avg = sum(fp16_itls) / len(fp16_itls)
        fused_avg = sum(fused_itls) / len(fused_itls)
        delta = fused_avg - fp16_avg
        ratio = fused_avg / fp16_avg if fp16_avg > 0 else None
        log(f"=== STAGE 1 SUMMARY ===")
        log(f"  FP16 graph avg ITL: {fp16_avg:.3f} ms")
        log(f"  Fused graph avg ITL: {fused_avg:.3f} ms")
        log(f"  Delta: {delta:+.3f} ms ({(delta/fp16_avg*100):+.1f}%)")
        log(f"  Ratio: {ratio:.4f}x")

    append_event("CHECKPOINT", "S1_COMPLETE", "Stage 1 NSYS profiling complete")
    log("=== Stage 1 complete ===")

if __name__ == "__main__":
    main()
