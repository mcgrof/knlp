#!/usr/bin/env python3
"""Send prompts with long distinct prefixes to vLLM and time each request.

Each of --prompts prefixes is --prefix-tokens random token IDs fixed by the
prompt index, so a request carries exactly prefix/256 full LMCache chunks.
Each request appends a 16-token suffix fixed by (prompt, pass) and asks
for one token, so its latency is the time to first token. Prompts go in a
fixed round-robin order: with an LMCache L1 smaller than the working set,
every prefix leaves L1 before it comes back and loads from L2 again.

Phases:
  populate   one request per prompt, then wait until the L2 directory
             holds every chunk (LMCache stores asynchronously)
  run        for --seconds (or --passes full passes), --concurrency
             requests in flight, starting at --start-at (epoch seconds)

Each record names the request's prompt, pass, latency, prompt tokens,
cached tokens and generated text. The NFS client's per-operation RPC
counts for the mount around the phase come from /proc/self/mountstats.
"""

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pnfs")
)
import mountstats  # noqa: E402

CHUNK = 256


def tokens(seed, n):
    rng = random.Random(seed)
    # ordinary text token IDs; Llama 3 reserves 128000 and above
    return [rng.randrange(1000, 120000) for _ in range(n)]


def request(url, model, prompt_ids):
    body = json.dumps(
        {"model": model, "prompt": prompt_ids, "max_tokens": 1, "temperature": 0}
    ).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        doc = json.load(r)
    lat = time.perf_counter() - t0
    usage = doc.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    return {
        "latency_s": lat,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "text": doc["choices"][0]["text"],
    }


def count_files(path):
    return sum(len(f) for _, _, f in os.walk(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["populate", "run"])
    p.add_argument("--out", required=True)
    p.add_argument("--mount", required=True, help="a path on the mount under test")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--prefix-tokens", type=int, default=8192)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--passes", type=int, default=0)
    p.add_argument("--pass-base", type=int, default=0)
    p.add_argument("--start-at", type=float, default=0)
    p.add_argument("--l2", default="", help="populate: L2 directory to watch")
    a = p.parse_args()
    if a.prefix_tokens % CHUNK:
        sys.exit(f"--prefix-tokens must be a multiple of {CHUNK}")
    prefixes = [tokens(f"prefix:{i}", a.prefix_tokens) for i in range(a.prompts)]
    work = queue.Queue()
    if a.phase == "populate":
        order = [(i, a.pass_base) for i in range(a.prompts)]
    else:
        passes = a.passes or 1000
        order = [(i, a.pass_base + k) for k in range(passes) for i in range(a.prompts)]
    for item in order:
        work.put(item)
    if a.start_at:
        time.sleep(max(0, a.start_at - time.time()))
    s0 = mountstats.snapshot(a.mount)
    t_start = time.time()
    deadline = t_start + a.seconds if a.seconds else float("inf")
    records, lock = [], threading.Lock()

    def worker():
        while time.time() < deadline:
            try:
                i, k = work.get_nowait()
            except queue.Empty:
                return
            ids = prefixes[i] + tokens(f"suffix:{i}:{k}", 16)
            t_send = time.time()
            try:
                r = request(a.url, a.model, ids)
            except Exception as e:  # recorded; the phase continues
                r = {"error": repr(e)}
            r.update({"prompt": i, "pass": k, "t_send": t_send})
            with lock:
                records.append(r)

    th = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    t_end = time.time()
    s1 = mountstats.snapshot(a.mount)
    l2_files = None
    if a.phase == "populate" and a.l2:
        want = a.prompts * (a.prefix_tokens // CHUNK)
        last, stable = -1, 0
        for _ in range(600):
            n = count_files(a.l2)
            stable = stable + 1 if n == last else 0
            last = n
            if n >= want and stable >= 5:
                break
            time.sleep(2)
        l2_files = last
    rpcs = mountstats.delta(s0, s1) or {}
    doc = {
        "phase": a.phase,
        "concurrency": a.concurrency,
        "prompts": a.prompts,
        "prefix_tokens": a.prefix_tokens,
        "t_start": t_start,
        "t_end": t_end,
        "wall_s": t_end - t_start,
        "requests": len(records),
        "errors": sum(1 for r in records if "error" in r),
        "nfs_rpcs": rpcs,
        "l2_files": l2_files,
        "records": records,
    }
    with open(a.out, "w") as f:
        json.dump(doc, f)
    ok = [r for r in records if "error" not in r]
    lat = sorted(r["latency_s"] for r in ok)
    if not lat:
        print(f"{a.phase}: no successful request; errors {records[:2]}")
        sys.exit(1)
    print(
        f"{a.phase} c{a.concurrency}: {len(ok)} ok, {doc['errors']} errors, "
        f"p50 {lat[len(lat) // 2] * 1e3:.1f} ms, {len(ok) / doc['wall_s']:.2f} req/s, "
        f"cached {sorted(set(r['cached_tokens'] for r in ok))}, "
        f"rpcs { {k: v['calls'] for k, v in rpcs.items()} }, l2_files {l2_files}"
    )


if __name__ == "__main__":
    main()
