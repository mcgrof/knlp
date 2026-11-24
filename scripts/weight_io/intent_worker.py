#!/usr/bin/env python3
"""Load weights the way a given loader does, for intent tracing.

Modes:
  safetensors  vLLM's default reader: safe_open() each shard and
               get_tensor() every tensor, then copy it into a GPU
               parameter. The shard is memory-mapped; the reader makes
               no read calls on it.
  experts-mmap one MoE layer's selected experts through safe_open() and
               get_tensor(), as an expert-offload fetch would.
  experts-pread the same experts read with one buffered pread() per
               tensor into host memory, then copied to the GPU.

Prints RESULT {json} with t0/t1 monotonic ns around the timed part and
a digest check of every loaded tensor against --ref.
"""

import argparse, hashlib, json, os, struct, sys, time
from pathlib import Path

DT = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "I32": "int32",
    "U8": "uint8",
    "I8": "int8",
    "I64": "int64",
}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", default="")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    import torch
    from safetensors import safe_open

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    want = None
    if a.mode.startswith("experts"):
        ex = [int(x) for x in a.experts.split(",")]
        pre = [f"model.layers.{a.layer}.mlp.experts.{e}." for e in ex]
        want = lambda k: any(k.startswith(p) for p in pre)
    plan = []  # (file, name, meta, data_start)
    for f in a.files:
        base, h = header(f)
        for k, m in h.items():
            if want is None or want(k):
                plan.append((f, k, m, base))
    params = {
        k: torch.empty(m["shape"], dtype=getattr(torch, DT[m["dtype"]]), device=dev)
        for _, k, m, _ in plan
    }
    torch.cuda.synchronize()
    out = {
        "mode": a.mode,
        "tensors": len(plan),
        "bytes": sum(
            m["data_offsets"][1] - m["data_offsets"][0] for _, _, m, _ in plan
        ),
    }
    out["t0_mono_ns"] = time.monotonic_ns()
    t0 = time.perf_counter()
    if a.mode in ("safetensors", "experts-mmap"):
        by_file = {}
        for f, k, _, _ in plan:
            by_file.setdefault(f, []).append(k)
        for f, names in by_file.items():
            with safe_open(f, framework="pt") as fh:
                for k in names:
                    params[k].copy_(fh.get_tensor(k))
    elif a.mode == "experts-pread":
        fds = {}
        for f, k, m, base in plan:
            fd = fds.get(f) or fds.setdefault(f, os.open(f, os.O_RDONLY))
            s, e = m["data_offsets"]
            buf = os.pread(fd, e - s, base + s)
            t = torch.frombuffer(bytearray(buf), dtype=params[k].dtype)
            params[k].copy_(t.view(params[k].shape))
        for fd in fds.values():
            os.close(fd)
    else:
        sys.exit(f"unknown mode {a.mode}")
    torch.cuda.synchronize()
    out["load_s"] = time.perf_counter() - t0
    out["t1_mono_ns"] = time.monotonic_ns()
    ref = json.loads(Path(a.ref).read_text())
    bad = [
        k
        for k, t in params.items()
        if hashlib.sha256(
            t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        ).hexdigest()
        != ref[k]
    ]
    out["verified"] = len(params)
    out["mismatched"] = bad
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
