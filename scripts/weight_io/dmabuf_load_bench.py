#!/usr/bin/env python3
"""Compare fastsafetensors' DMA-BUF copier with its other readers.

The DMA-BUF copier reads safetensors shards from NVMe straight into GPU
memory through io_uring DMA-BUF fixed buffers. The nogds reader, which
vLLM's --load-format fastsafetensors uses when cuFile is absent, reads
into pinned host memory and copies each chunk to the GPU. This script
loads one checkpoint with each reader and reports load time, bytes read
from the drive and whether every tensor matches the file byte for byte.

Each trial is a fresh process, so every trial pays the start-up costs a
serving engine pays. Before each trial the page cache is dropped (this
needs root or password-less sudo; otherwise the trial is marked warm).
The worker calls ParallelLoader the way vLLM's fastsafetensors weight
iterator does on one GPU and copies each tensor into a preallocated
parameter, as vLLM's model loader does. The model must fit in one GPU
with room for two shards.

Readers:
  nogds   ParallelLoader(nogds=True): host bounce buffers.
  dmabuf  ParallelLoader(nogds=False) with FASTSAFETENSORS_USE_DMABUF=1,
          the switch an engine that only passes nogds can use. Strict
          mode is on, so a refused export or registration is reported
          as an error instead of silently falling back to nogds.
  gds     ParallelLoader(nogds=False) without the switch: cuFile when it
          is installed (NVIDIA only), otherwise nogds.

The JSON output records the storage geometry and GPU architecture next
to the results, and no drive serial numbers.

Example:
  sudo python3 dmabuf_load_bench.py --model /models/Qwen2.5-7B-Instruct \\
      --trials 3 --out result.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return 8 + n, meta


def file_digests(path):
    base, meta = header(path)
    out = {}
    with open(path, "rb") as f:
        for name, m in meta.items():
            lo, hi = m["data_offsets"]
            f.seek(base + lo)
            h, left = hashlib.sha256(), hi - lo
            while left:
                b = f.read(min(left, 64 << 20))
                h.update(b)
                left -= len(b)
            out[name] = h.hexdigest()
    return out


def shard_files(model):
    p = Path(model)
    files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no .safetensors files under {model}")
    return [str(f) for f in files]


# Storage description: geometry only, never identity.


def sysfs_block(path):
    st = os.stat(os.path.realpath(path))
    node = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
    return node.resolve() if node.exists() else None


def read_sys(p):
    try:
        return p.read_text().strip()
    except OSError:
        return None


def mount_of(path):
    real, best = os.path.realpath(path), ("", "?", "?")
    with open("/proc/self/mountinfo") as f:
        for line in f:
            left, right = line.split(" - ", 1)
            mnt = left.split()[4]
            fstype, source = right.split()[:2]
            if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(
                mnt
            ) > len(best[0]):
                best = (mnt, fstype, source)
    return {"mount": best[0], "fstype": best[1]}


def storage_info(path):
    node = sysfs_block(path)
    info = mount_of(path)
    if node is None:
        return info
    disk = node.parent if (node / "partition").exists() else node
    q = disk / "queue"
    info["block_device"] = disk.name
    for k in (
        "logical_block_size",
        "physical_block_size",
        "max_sectors_kb",
        "max_hw_sectors_kb",
        "max_segments",
        "max_hw_dmabuf_sectors_kb",
        "nr_requests",
    ):
        v = read_sys(q / k)
        if v is not None:
            info[k] = v
    size = read_sys(disk / "size")
    if size:
        info["capacity_gb"] = round(int(size) * 512 / 1e9)
    pci = disk / "device" / "device"
    for k in ("current_link_speed", "current_link_width"):
        v = read_sys(pci / k)
        if v is not None:
            info["nvme_" + k] = v
    return info


def sectors_read(path):
    node = sysfs_block(path)
    stat = read_sys(node / "stat") if node else None
    return int(stat.split()[2]) * 512 if stat else None


def one_file_per_device(files):
    """One shard per block device, for counting bytes read per device."""
    seen = {}
    for f in files:
        seen.setdefault(str(sysfs_block(f)), f)
    return list(seen.values())


def drop_caches():
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True
    except OSError:
        pass
    r = subprocess.run(
        ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        capture_output=True,
    )
    return r.returncode == 0


# Worker: one load in a fresh process.


def gpu_info(torch, device):
    if device.type != "cuda":
        return {"type": device.type}
    p = torch.cuda.get_device_properties(device)
    info = {
        "runtime": "rocm" if torch.version.hip else "cuda",
        "runtime_version": torch.version.hip or torch.version.cuda,
        "name": p.name,
        "arch": getattr(p, "gcnArchName", None) or f"sm_{p.major}{p.minor}",
        "memory_gib": round(p.total_memory / 2**30, 1),
    }
    if torch.version.cuda:
        info["cuda_dma_buf_supported"] = cuda_dma_buf_supported(device.index or 0)
    return info


def cuda_dma_buf_supported(index):
    """CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, which gates CUDA's export."""
    import ctypes

    try:
        cu = ctypes.CDLL("libcuda.so.1")
        dev, val = ctypes.c_int(), ctypes.c_int()
        if cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), index):
            return None
        if cu.cuDeviceGetAttribute(ctypes.byref(val), 124, dev):
            return None
        return bool(val.value)
    except OSError:
        return None


def worker(args):
    import torch

    from fastsafetensors import SingleGroup
    from fastsafetensors.parallel_loader import ParallelLoader

    try:
        from fastsafetensors.copier.registry import copier_class_of
    except ImportError:
        copier_class_of = None

    files = args.files
    device = torch.device(args.device)
    out = {"reader": args.reader, "gpu": gpu_info(torch, device)}
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)
    params = {}
    for f in files:
        for k, m in header(f)[1].items():
            if m["dtype"] not in DTYPES:
                raise SystemExit(f"unsupported dtype {m['dtype']} in {f}")
            dtype = getattr(torch, DTYPES[m["dtype"]])
            params[k] = torch.empty(m["shape"], dtype=dtype, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()

    devs = one_file_per_device(files)
    before = [sectors_read(f) for f in devs]
    t0 = time.perf_counter()
    pl = ParallelLoader(
        pg=SingleGroup(),
        hf_weights_files=files,
        queue_size=0,
        use_tqdm_on_load=False,
        device=str(device),
        nogds=args.reader == "nogds",
    )
    if copier_class_of is not None:
        out["copier"] = copier_class_of(pl.loader.copier_constructor).__name__
    try:
        for name, tensor in pl.iterate_weights():
            params[name].copy_(tensor)
    finally:
        pl.close()
    sync()
    out["load_s"] = time.perf_counter() - t0
    after = [sectors_read(f) for f in devs]
    if None not in before and None not in after:
        out["drive_bytes_read"] = sum(a - b for a, b in zip(after, before))
    out["bytes"] = sum(t.nbytes for t in params.values())
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.ref:
        ref = json.loads(Path(args.ref).read_text())
        bad = []
        for k, t in params.items():
            b = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            if hashlib.sha256(b).hexdigest() != ref[k]:
                bad.append(k)
        out["verified"] = len(params)
        out["mismatched"] = bad
    print("RESULT " + json.dumps(out), flush=True)


# Driver.


def fst_info():
    try:
        import fastsafetensors
    except ImportError:
        sys.exit("fastsafetensors is not importable")
    src = Path(fastsafetensors.__file__).resolve().parent
    info = {"version": getattr(fastsafetensors, "__version__", None)}
    r = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if r.returncode == 0:
        info["commit"] = r.stdout.strip()
    return info


def run_trial(args, reader, ref):
    env = dict(os.environ)
    for k in ("FASTSAFETENSORS_USE_DMABUF", "FASTSAFETENSORS_DMABUF_STRICT"):
        env.pop(k, None)
    if reader == "dmabuf":
        env["FASTSAFETENSORS_USE_DMABUF"] = "1"
        env["FASTSAFETENSORS_DMABUF_STRICT"] = "1"
    cold = drop_caches() if args.drop_caches else False
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", reader]
    cmd += ["--device", args.device] + (["--ref", ref] if ref else [])
    r = subprocess.run(cmd + args.files, env=env, capture_output=True, text=True)
    line = next(
        (x for x in r.stdout.splitlines()[::-1] if x.startswith("RESULT ")), None
    )
    if r.returncode or line is None:
        tail = (r.stderr or r.stdout).strip().splitlines()[-8:]
        return {"reader": reader, "cold": cold, "error": "\n".join(tail)}
    res = json.loads(line[len("RESULT ") :])
    res["cold"] = cold
    res["gb_per_s"] = res["bytes"] / res["load_s"] / 1e9
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="checkpoint directory or one shard")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--readers", default="nogds,dmabuf")
    ap.add_argument(
        "--verify",
        choices=("first", "all", "none"),
        default="first",
        help="check tensors against the file on the first trial of each "
        "reader (default), on every trial, or never",
    )
    ap.add_argument("--no-drop-caches", dest="drop_caches", action="store_false")
    ap.add_argument("--out", default="dmabuf-load-bench.json")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--ref", help=argparse.SUPPRESS)
    ap.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        args.reader = args.worker
        return worker(args)
    if not args.model:
        ap.error("--model is required")

    args.files = shard_files(args.model)
    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    report = {
        "schema": "knlp-dmabuf-load-bench-v1",
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "fastsafetensors": fst_info(),
        "checkpoint": {
            "name": Path(args.model).resolve().name,
            "shards": len(args.files),
            "bytes": sum(os.path.getsize(f) for f in args.files),
        },
        "storage": {Path(f).name: storage_info(f) for f in args.files},
        "trials": [],
    }
    print(f"{len(args.files)} shards, {report['checkpoint']['bytes'] / 1e9:.2f} GB")

    ref = None
    if args.verify != "none":
        print("hashing every tensor from the files for the byte check ...")
        digests = {}
        for f in args.files:
            digests.update(file_digests(f))
        fd, ref = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fp:
            json.dump(digests, fp)

    try:
        for t in range(args.trials):
            order = readers[t % len(readers) :] + readers[: t % len(readers)]
            for reader in order:
                check = args.verify == "all" or (args.verify == "first" and t == 0)
                res = run_trial(args, reader, ref if check else None)
                res["trial"] = t
                report["trials"].append(res)
                if "gpu" in res:
                    report.setdefault("gpu", res.pop("gpu"))
                if "error" in res:
                    print(f"trial {t} {reader}: ERROR\n{res['error']}")
                    continue
                print(
                    f"trial {t} {reader:7s} {res['load_s']:7.3f} s "
                    f"{res['gb_per_s']:6.2f} GB/s "
                    f"copier={res.get('copier', '?')} cold={res['cold']}"
                    + (
                        f" mismatched={len(res['mismatched'])}/{res['verified']}"
                        if "verified" in res
                        else ""
                    )
                )
    finally:
        if ref:
            os.unlink(ref)

    summary = {}
    for reader in readers:
        ok = [r for r in report["trials"] if r["reader"] == reader and "load_s" in r]
        summary[reader] = {
            "trials": len(ok),
            "errors": sum(
                1 for r in report["trials"] if r["reader"] == reader and "error" in r
            ),
            "load_s_median": statistics.median(r["load_s"] for r in ok) if ok else None,
            "load_s_range": (
                [min(r["load_s"] for r in ok), max(r["load_s"] for r in ok)]
                if ok
                else None
            ),
            "mismatched_tensors": sum(len(r.get("mismatched", [])) for r in ok),
            "all_cold": all(r["cold"] for r in ok) if ok else None,
        }
    report["summary"] = summary
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
