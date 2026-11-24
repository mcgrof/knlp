#!/usr/bin/env python3
"""Record what each weight loader asks for and what the drive does.

For each model and loader this script runs cold loads under four tracers
and keeps the traces for analyze_intent.py:

  nvme_tp_monitor   every NVMe command on the drive, with its size
  syscall_monitor   the loader's own system calls, with requested sizes
  mmap_readamp      demand faults and page-cache fills per file
  load_io.bt        io_uring setup, buffer registration and request sizes

The loader process starts stopped, so the per-process tracers attach
before its first system call. The page cache is dropped before each
load, and every load checks the SHA-256 of each tensor it loaded.

Loaders:

  safetensors   vLLM's default reader: safe_open() and get_tensor()
                (intent_worker.py)
  nogds         fastsafetensors through a host bounce buffer
  dmabuf        the fastsafetensors DMA-BUF copier: io_uring READ_FIXED
                straight into GPU memory
  experts-mmap-N, experts-pread-N
                N routed experts of one MoE layer, through safe_open()
                or one pread() per tensor (with --moe)

The tracers come from https://github.com/SamsungDS/ebpf-syscall and run
under sudo; the loaders run as the calling user. The DMA-BUF loader needs
a kernel with io_uring DMA-BUF file reads and the fastsafetensors branch
2026-10-28-iouring-dma-buf on PYTHONPATH.

Example:

  python3 trace_intent.py --tools ~/ebpf-syscall --disk nvme0n1 \\
      --model qwen2.5-7b=/models/Qwen2.5-7B-Instruct \\
      --model emo=/models/Emo_1b14b --moe emo --out /data/intent
"""

import argparse, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import dmabuf_load_bench as B
import traced_replay as T

FULL = ("safetensors", "nogds", "dmabuf")


def env_for(loader):
    env = dict(os.environ)
    for k in ("FASTSAFETENSORS_USE_DMABUF", "FASTSAFETENSORS_DMABUF_STRICT"):
        env.pop(k, None)
    if loader == "dmabuf":
        env["FASTSAFETENSORS_USE_DMABUF"] = "1"
        env["FASTSAFETENSORS_DMABUF_STRICT"] = "1"
    return env


def cmd_for(a, loader, files, ref):
    py = a.python
    if loader in ("nogds", "dmabuf"):
        return [
            py,
            str(D / "dmabuf_load_bench.py"),
            "--worker",
            loader,
            "--device",
            "cuda:0",
            "--ref",
            ref,
            *files,
        ]
    if loader == "safetensors":
        return [py, str(D / "intent_worker.py"), "safetensors", "--ref", ref, *files]
    mode, n = loader.rsplit("-", 1)  # experts-mmap-8 -> experts-mmap, 8
    return [
        py,
        str(D / "intent_worker.py"),
        mode,
        "--layer",
        str(a.layer),
        "--experts",
        a.experts[n],
        "--ref",
        ref,
        *files,
    ]


def parse(out):
    line = next((x for x in out.splitlines()[::-1] if x.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else None


def load(a, loader, files, ref, rd):
    tools = Path(a.tools)
    rec = {"loader": loader, "cold": B.drop_caches()}
    rd.mkdir(parents=True, exist_ok=True)
    nv, _ = T.start_tracer(
        [
            str(tools / "nvme_tp_monitor"),
            "--disk",
            a.disk,
            "--dur",
            str(a.max_seconds),
            "--jsonl",
            str(rd / "device.jsonl"),
        ],
        "attached",
    )
    procs = [("nvme", nv)]
    log = open(rd / "worker.log", "w")
    child = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            'kill -STOP $$; exec "$@"',
            "sh",
            *cmd_for(a, loader, files, ref),
        ],
        env=env_for(loader),
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
    )
    try:
        for name, cmd, ready in (
            (
                "bpftrace",
                ["bpftrace", "-q", str(D / "load_io.bt"), str(child.pid)],
                "attached",
            ),
            (
                "syscall_monitor",
                [
                    str(tools / "syscall_monitor"),
                    "--pid",
                    str(child.pid),
                    "--jsonl-out",
                    str(rd / "syscalls.jsonl"),
                    "--duration",
                    str(a.max_seconds),
                    "--label",
                    f"{rd.parent.name}-{loader}",
                    "--quiet",
                ],
                "attached",
            ),
            (
                "mmap_readamp",
                [
                    str(tools / "mmap_readamp"),
                    "--pid",
                    str(child.pid),
                    "--dur",
                    str(a.max_seconds),
                    "--jsonl",
                    str(rd / "faults.jsonl"),
                ],
                "tracing",
            ),
        ):
            p, _ = T.start_tracer(cmd, ready)
            procs.append((name, p))
        os.kill(child.pid, signal.SIGCONT)
        out, _ = child.communicate()
    except BaseException:
        child.kill()
        raise
    finally:
        time.sleep(1.0)
        for name, p in procs[::-1]:
            (rd / f"{name}.out").write_text(T.stop_tracer(p, grace=120))
        for f in ("device.jsonl", "syscalls.jsonl", "faults.jsonl"):
            if (rd / f).exists():
                T.readable_trace(rd / f)
        log.close()
    res = parse(out)
    rec.update(res or {"error": (rd / "worker.log").read_text()[-3000:]})
    rec["rc"] = child.returncode
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="a safetensors checkpoint directory; repeat for more",
    )
    ap.add_argument(
        "--moe",
        action="append",
        default=[],
        metavar="NAME",
        help="also run expert fetches on this model",
    )
    ap.add_argument("--layer", type=int, default=3, help="MoE layer for expert fetches")
    ap.add_argument(
        "--experts-1", default="17", help="expert ids for the 1-expert fetch"
    )
    ap.add_argument(
        "--experts-8",
        default="3,17,29,42,64,88,101,120",
        help="expert ids for the 8-expert fetch",
    )
    ap.add_argument("--tools", required=True, help="built ebpf-syscall tree")
    ap.add_argument("--disk", required=True, help="block device holding the models")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-seconds", type=int, default=900)
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter with torch, safetensors and fastsafetensors",
    )
    a = ap.parse_args()
    a.experts = {"1": a.experts_1, "8": a.experts_8}
    top = Path(a.out)
    top.mkdir(parents=True, exist_ok=True)
    for spec in a.model:
        name, mdir = spec.split("=", 1)
        files = B.shard_files(mdir)
        md = top / name
        md.mkdir(exist_ok=True)
        meta = {
            "model": name,
            "dir": str(Path(mdir).resolve()),
            "shards": len(files),
            "bytes": sum(os.path.getsize(f) for f in files),
            "storage": {Path(f).name: B.storage_info(f) for f in files},
        }
        (md / "meta.json").write_text(json.dumps(meta, indent=1))
        print(
            f"== {name}: {meta['shards']} shards {meta['bytes']/1e9:.2f} GB", flush=True
        )
        digests = {}
        for f in files:
            digests.update(B.file_digests(f))
        fd, ref = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fp:
            json.dump(digests, fp)
        loaders = list(FULL)
        if name in a.moe:
            loaders += [
                f"experts-{m}-{n}" for m in ("mmap", "pread") for n in a.experts
            ]
        recs = []
        try:
            for rep in range(1, a.reps + 1):
                order = loaders if rep % 2 else loaders[::-1]
                for loader in order:
                    rec = load(a, loader, files, ref, md / f"{loader}-rep{rep}")
                    rec["rep"] = rep
                    recs.append(rec)
                    (md / "loads.json").write_text(json.dumps(recs, indent=1))
                    bad = rec.get("mismatched")
                    print(
                        f"{name} {loader:16s} rep{rep} load_s={rec.get('load_s')} "
                        f"cold={rec['cold']} verified={rec.get('verified')} "
                        f"mismatched={None if bad is None else len(bad)} "
                        f"err={'error' in rec}",
                        flush=True,
                    )
        finally:
            os.unlink(ref)


if __name__ == "__main__":
    main()
