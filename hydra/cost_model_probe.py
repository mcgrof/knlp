# SPDX-License-Identifier: MIT
"""hydra: state-dependent model-residency cost probe.

Measures the C(m, state) table a residency-aware query router needs:
what it actually costs, on this box, to bring a model from each
residency state to serving-ready.

Sub-commands:
  bandwidth    O_DIRECT sequential-read bandwidth of a file (raw NVMe
               floor, page cache bypassed).
  filecold     Buffered read of a model snapshot, cold (pages dropped
               via posix_fadvise DONTNEED) vs warm re-read.
  engine       One vLLM engine bring-up trial in this process: import,
               LLM() init (weight load + engine build), and first
               token. Run one trial per process.
  sleep-probe  Does vLLM sleep mode work on this GPU stack? Times
               sleep/wake at levels 1 and 2; either answer is data.
  ladder       Orchestrates engine trials (fresh subprocess per trial)
               across a model ladder x {cold,warm} and aggregates
               medians/IQRs into cost_model.json.

Cold state needs no root: model file pages are dropped with
posix_fadvise(POSIX_FADV_DONTNEED) (clean read-only pages drop
immediately); raw bandwidth uses O_DIRECT. Full drop_caches runs are a
separate, explicitly authorized step.
"""

import argparse
import json
import mmap
import os
import statistics
import subprocess
import sys
import time

DEFAULT_LADDER = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")


def hf_hub_dir():
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(home, "hub")


def resolve_snapshot(model_id):
    """Latest local snapshot dir for a model id, or None."""
    name = "models--" + model_id.replace("/", "--")
    snaps = os.path.join(hf_hub_dir(), name, "snapshots")
    if not os.path.isdir(snaps):
        return None
    cands = [os.path.join(snaps, d) for d in os.listdir(snaps)]
    cands = [d for d in cands if os.path.isdir(d)]
    return max(cands, key=os.path.getmtime) if cands else None


def weight_files(snapshot_dir):
    out = []
    for root, _dirs, files in os.walk(snapshot_dir):
        for f in files:
            if f.endswith(WEIGHT_SUFFIXES):
                p = os.path.join(root, f)
                out.append(os.path.realpath(p))
    return sorted(set(out))


def fadvise_dontneed(paths):
    dropped = 0
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            dropped += os.path.getsize(p)
        finally:
            os.close(fd)
    return dropped


def read_buffered(paths, chunk_mib=8):
    buf = bytearray(chunk_mib * 1024 * 1024)
    view = memoryview(buf)
    total = 0
    t0 = time.perf_counter()
    for p in paths:
        with open(p, "rb", buffering=0) as f:
            while True:
                n = f.readinto(view)
                if not n:
                    break
                total += n
    return total, time.perf_counter() - t0


def read_direct(path, block_mib=16, max_gib=None):
    """O_DIRECT sequential read; returns (bytes, seconds)."""
    bs = block_mib * 1024 * 1024
    limit = int(max_gib * 1024**3) if max_gib else None
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, bs)  # anonymous maps are page-aligned
    total = 0
    t0 = time.perf_counter()
    try:
        while True:
            try:
                n = os.readv(fd, [buf])
            except OSError:
                break  # unaligned tail; stop at last full block
            if n <= 0:
                break
            total += n
            if limit and total >= limit:
                break
    finally:
        buf.close()
        os.close(fd)
    return total, time.perf_counter() - t0


def gbps(nbytes, secs):
    return (nbytes / 1e9) / secs if secs > 0 else 0.0


def cmd_bandwidth(args):
    res = []
    for _ in range(args.trials):
        n, dt = read_direct(args.file, args.block_mib, args.max_gib)
        res.append({"bytes": n, "seconds": dt, "gbps": gbps(n, dt)})
    out = {"file": args.file, "method": "o_direct", "trials": res}
    print(json.dumps(out))


def cmd_filecold(args):
    snap = resolve_snapshot(args.model)
    if not snap:
        sys.exit(f"no local snapshot for {args.model}")
    files = weight_files(snap)
    trials = []
    for _ in range(args.trials):
        fadvise_dontneed(files)
        cn, ct = read_buffered(files)
        wn, wt = read_buffered(files)
        trials.append(
            {
                "cold_seconds": ct,
                "cold_gbps": gbps(cn, ct),
                "warm_seconds": wt,
                "warm_gbps": gbps(wn, wt),
                "bytes": cn,
            }
        )
    print(json.dumps({"model": args.model, "snapshot": snap, "trials": trials}))


def _engine_trial(args):
    """One bring-up: returns the timing decomposition dict."""
    snap = resolve_snapshot(args.model)
    files = weight_files(snap) if snap else []
    rec = {
        "model": args.model,
        "state": args.state,
        "enforce_eager": args.enforce_eager,
        "weight_bytes": sum(os.path.getsize(f) for f in files),
    }
    if args.state == "cold":
        rec["dropped_bytes"] = fadvise_dontneed(files)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    t0 = time.perf_counter()
    from vllm import LLM, SamplingParams  # noqa: PLC0415  heavy on purpose

    rec["import_seconds"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    rec["init_seconds"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=1, temperature=0.0),
    )
    rec["first_token_seconds"] = time.perf_counter() - t2
    rec["output_ok"] = bool(out and out[0].outputs)
    rec["total_seconds"] = time.perf_counter() - t0
    return rec


def cmd_engine(args):
    print(json.dumps(_engine_trial(args)))


def cmd_sleep_probe(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_SERVER_DEV_MODE", "1")
    rec = {"model": args.model, "levels": {}}
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enable_sleep_mode=True,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    for level in (1, 2):
        cell = {}
        try:
            t0 = time.perf_counter()
            llm.sleep(level=level)
            cell["sleep_seconds"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            llm.wake_up()
            cell["wake_seconds"] = time.perf_counter() - t1
            out = llm.generate(["2+2="], sp)
            cell["generate_ok"] = bool(out and out[0].outputs)
            cell["supported"] = True
        except Exception as e:  # noqa: BLE001  the failure IS the datum
            cell["supported"] = False
            cell["error"] = f"{type(e).__name__}: {e}"
        rec["levels"][str(level)] = cell
    print(json.dumps(rec))


def _aggregate(vals):
    if not vals:
        return None
    q = statistics.quantiles(vals, n=4) if len(vals) >= 2 else [vals[0]] * 3
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "iqr": q[2] - q[0],
        "min": min(vals),
        "max": max(vals),
    }


def cmd_ladder(args):
    models = args.models.split(",") if args.models else DEFAULT_LADDER
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    jsonl = os.path.join(outdir, "engine_trials.jsonl")
    cells = {}
    with open(jsonl, "a") as log:
        for model in models:
            for state in ("cold", "warm"):
                key = f"{model}|{state}"
                cells[key] = []
                for trial in range(args.trials):
                    cmd = [
                        sys.executable,
                        os.path.abspath(__file__),
                        "engine",
                        "--model",
                        model,
                        "--state",
                        state,
                        "--gpu-mem-util",
                        str(args.gpu_mem_util),
                        "--max-model-len",
                        str(args.max_model_len),
                    ]
                    if args.enforce_eager:
                        cmd.append("--enforce-eager")
                    r = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=1800
                    )
                    line = r.stdout.strip().splitlines()[-1] if r.stdout else ""
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, IndexError):
                        rec = {
                            "model": model,
                            "state": state,
                            "error": r.stderr[-2000:],
                        }
                    rec["trial"] = trial
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    cells[key].append(rec)
                    print(
                        f"[{key}] trial {trial}: "
                        f"init={rec.get('init_seconds', 'ERR')}",
                        file=sys.stderr,
                    )

    summary = {}
    for key, recs in cells.items():
        ok = [r for r in recs if "init_seconds" in r]
        summary[key] = {
            "trials": len(recs),
            "failed": len(recs) - len(ok),
            "import_seconds": _aggregate([r["import_seconds"] for r in ok]),
            "init_seconds": _aggregate([r["init_seconds"] for r in ok]),
            "first_token_seconds": _aggregate([r["first_token_seconds"] for r in ok]),
            "total_seconds": _aggregate([r["total_seconds"] for r in ok]),
            "weight_bytes": ok[0]["weight_bytes"] if ok else None,
        }
    path = os.path.join(outdir, "cost_model.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"wrote": path, "cells": len(summary)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bandwidth")
    b.add_argument("--file", required=True)
    b.add_argument("--block-mib", type=int, default=16)
    b.add_argument("--max-gib", type=float, default=None)
    b.add_argument("--trials", type=int, default=3)
    b.set_defaults(fn=cmd_bandwidth)

    fc = sub.add_parser("filecold")
    fc.add_argument("--model", required=True)
    fc.add_argument("--trials", type=int, default=5)
    fc.set_defaults(fn=cmd_filecold)

    e = sub.add_parser("engine")
    e.add_argument("--model", required=True)
    e.add_argument("--state", choices=("cold", "warm"), default="warm")
    e.add_argument("--enforce-eager", action="store_true")
    e.add_argument("--gpu-mem-util", type=float, default=0.85)
    e.add_argument("--max-model-len", type=int, default=2048)
    e.set_defaults(fn=cmd_engine)

    s = sub.add_parser("sleep-probe")
    s.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    s.add_argument("--enforce-eager", action="store_true")
    s.add_argument("--gpu-mem-util", type=float, default=0.85)
    s.add_argument("--max-model-len", type=int, default=2048)
    s.set_defaults(fn=cmd_sleep_probe)

    l = sub.add_parser("ladder")
    l.add_argument("--models", default=None, help="comma-separated ids")
    l.add_argument("--trials", type=int, default=5)
    l.add_argument("--enforce-eager", action="store_true")
    l.add_argument("--gpu-mem-util", type=float, default=0.85)
    l.add_argument("--max-model-len", type=int, default=2048)
    l.add_argument("--outdir", required=True)
    l.set_defaults(fn=cmd_ladder)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
