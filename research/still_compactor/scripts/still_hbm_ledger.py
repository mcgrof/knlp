#!/usr/bin/env python3
"""STILL - byte-accurate HBM ledger + replacement/streaming microbenchmark.

The decisive systems gate. Allocates REAL Qwen3-4B-shaped KV tensors on the
W7900 and measures peak HBM for two compaction schedules, so we learn whether
the raw source-KV construction cliff is avoidable BEFORE building the kernel or
training anything.

Two schedules, per the the taxonomy:
  (i)  post-prefill-replace : hold the FULL source KV (all T tokens) resident,
       allocate the compact KV, then free the source. Peak includes full source.
  (ii) chunked-streaming    : process T in chunks; hold only one chunk's source
       KV plus the accumulating compact KV. Peak source term is bounded by the
       chunk, not by T.

Both schedules end with the SAME total compact KV (T/c tokens), so any peak
difference is purely the source residency -- exactly the crux.

Each (schedule, T, c, C) cell runs in a fresh subprocess (clean allocator).
Concurrency C models C requests simultaneously at their worst-case moment.
OOM is a valid result and is reported, not worked around.

Usage:
  python3 m1_hbm_ledger.py --out-dir <dir> [--device cuda:0]
  python3 m1_hbm_ledger.py --cell <schedule> <T> <c> <C> --chunk <n>   # child
"""
import argparse, json, os, subprocess, sys

# Qwen3-4B KV geometry (verified 2026-07-21)
L, HKV, HD = 36, 8, 128
BYTES_PER_TOKEN = HKV * HD * 2 * 2 * L        # K+V, bf16, all layers = 147456
GiB = 2 ** 30
BASE_WEIGHTS_GIB = 7.49                        # Qwen3-4B bf16
COMPACTOR_GIB = 50e6 * 2 / GiB                 # ~50M params bf16, once/replica


def analytic_peak_gib(schedule, T, c, C, chunk):
    """Analytic model the measurement is checked against."""
    src_full = BYTES_PER_TOKEN * T / GiB
    compact_total = BYTES_PER_TOKEN * round(T / c) / GiB
    src_chunk = BYTES_PER_TOKEN * min(chunk, T) / GiB
    fixed = BASE_WEIGHTS_GIB + COMPACTOR_GIB
    if schedule == "post_prefill_replace":
        return fixed + C * (src_full + compact_total)
    else:  # chunked_streaming: one chunk source + full accumulated compact
        return fixed + C * (src_chunk + compact_total)


def _alloc_kv(n_tokens, dev):
    """Allocate a Qwen3-4B KV cache for n_tokens: 36 layers x (K,V) bf16."""
    import torch
    t = []
    for _ in range(L):
        k = torch.empty((n_tokens, HKV, HD), dtype=torch.bfloat16, device=dev)
        v = torch.empty((n_tokens, HKV, HD), dtype=torch.bfloat16, device=dev)
        t.append((k, v))
    return t


def run_cell(schedule, T, c, C, chunk, dev):
    import torch
    di = int(dev.split(":")[1]) if ":" in dev else 0  # cuda mem APIs want index
    torch.cuda.set_device(di)              # init HIP ctx (cold mem APIs fail)
    torch.cuda.reset_peak_memory_stats(di)
    result = {"schedule": schedule, "T": T, "c": c, "C": C, "chunk": chunk}
    try:
        # frozen base weights + compactor weights, resident for the whole op
        base = torch.empty(int(BASE_WEIGHTS_GIB * GiB // 2),
                           dtype=torch.bfloat16, device=dev)
        comp = torch.empty(int(COMPACTOR_GIB * GiB // 2),
                           dtype=torch.bfloat16, device=dev)
        t_compact = round(T / c)
        live = [base, comp]
        if schedule == "post_prefill_replace":
            # worst case: all C requests hold source+compact simultaneously
            srcs = [_alloc_kv(T, dev) for _ in range(C)]
            comps = [_alloc_kv(t_compact, dev) for _ in range(C)]
            live += srcs + comps
            torch.cuda.synchronize(di)
            peak = torch.cuda.max_memory_allocated(di)
            del srcs                       # source freed after atomic replace
        else:  # chunked_streaming
            # each request: one chunk source + accumulating compact; worst case
            # is the last chunk (max accumulated compact) held by all C at once.
            n_chunks = max(1, (T + chunk - 1) // chunk)
            comp_per_chunk = max(1, t_compact // n_chunks)
            chunk_srcs = [_alloc_kv(min(chunk, T), dev) for _ in range(C)]
            accum = [_alloc_kv(comp_per_chunk * n_chunks, dev) for _ in range(C)]
            live += chunk_srcs + accum
            torch.cuda.synchronize(di)
            peak = torch.cuda.max_memory_allocated(di)
        free, total = torch.cuda.mem_get_info(di)
        result.update(
            oom=False,
            measured_peak_gib=round(peak / GiB, 3),
            reserved_gib=round(torch.cuda.max_memory_reserved(di) / GiB, 3),
            hbm_used_gib=round((total - free) / GiB, 3),
        )
        del live
        torch.cuda.empty_cache()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # noqa
        msg = str(e).split("\n")[0][:120]
        result.update(oom="out of memory" in str(e).lower(), error=msg)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    result["analytic_peak_gib"] = round(
        analytic_peak_gib(schedule, T, c, C, chunk), 3)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--cell", nargs=4)  # schedule T c C  (child mode)
    args = ap.parse_args()

    if args.cell:                        # child: one cell, print JSON
        import torch  # noqa
        sch, T, c, C = args.cell
        r = run_cell(sch, int(T), int(c), int(C), args.chunk, args.device)
        print("CELL_JSON " + json.dumps(r))
        return

    # parent: orchestrate cells in isolated subprocesses
    os.makedirs(args.out_dir, exist_ok=True)
    Ts = [4096, 8192, 16384, 32768, 65536]
    cs = [32, 50]
    Cs = [1, 2, 4]
    schedules = ["post_prefill_replace", "chunked_streaming"]
    rows = []
    for sch in schedules:
        for T in Ts:
            for c in cs:
                for C in Cs:
                    cmd = [sys.executable, os.path.abspath(__file__),
                           "--device", args.device, "--chunk", str(args.chunk),
                           "--cell", sch, str(T), str(c), str(C)]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    row = None
                    for ln in p.stdout.splitlines():
                        if ln.startswith("CELL_JSON "):
                            row = json.loads(ln[len("CELL_JSON "):])
                    if row is None:
                        row = {"schedule": sch, "T": T, "c": c, "C": C,
                               "oom": None, "error": "no output",
                               "stderr": p.stderr.splitlines()[-1:]}
                    rows.append(row)
                    tag = ("OOM" if row.get("oom") else
                           f"{row.get('measured_peak_gib','?')}GiB")
                    print(f"{sch:22} T={T:>6} c={c} C={C}: {tag}"
                          f"  (analytic {row.get('analytic_peak_gib','?')})")
    with open(os.path.join(args.out_dir, "hbm_ledger.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out_dir}/hbm_ledger.json ({len(rows)} cells)")


if __name__ == "__main__":
    main()
