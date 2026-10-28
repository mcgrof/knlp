#!/usr/bin/env python3
"""Native model load followed by resident decode, with trace windows.

This is the negative control for weight-storage reads: after a native
`from_pretrained` load of a small causal language model onto the GPU,
ordinary prefill and decode must cause zero weight reads from storage.
The script records the load window and then N timed generation trials,
each as a window with the same fields the range replay uses, so the
traced wrapper can attribute NVMe commands per window. The checkpoint
file's extents are recorded so any read inside the model file during a
decode window is attributed to it rather than hidden.

The load uses the library's own loader (safetensors, buffered or mmap
as the library chooses); this is the native cold or warm startup arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from replay_ranges import fiemap, device_lbas, resident_pages  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument(
        "--cache",
        choices=["cold", "warm"],
        default="cold",
        help="page-cache state of the shard files before the native load",
    )
    ap.add_argument("--part-start-sectors", type=int, required=True)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    mdir = Path(args.model_dir)
    shards = sorted(mdir.glob("*.safetensors"))
    # page-cache control on every shard
    fds = [os.open(str(s), os.O_RDONLY) for s in shards]
    sizes = [os.fstat(fd).st_size for fd in fds]
    whole = [[(0, sz)] for sz in sizes]
    resid_before = []
    for fd, sz, w in zip(fds, sizes, whole):
        if args.cache == "cold":
            for _ in range(4):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                r = resident_pages(fd, w, sz)
                if r[0] == 0:
                    break
                time.sleep(0.2)
        else:
            with open(fd, "rb", closefd=False) as fh:
                while fh.read(64 << 20):
                    pass
            r = resident_pages(fd, w, sz)
        resid_before.append(r)
    extents = [fiemap(fd, sz) for fd, sz in zip(fds, sizes)]
    lbas = []
    for ex, w in zip(extents, whole):
        lbas.extend(device_lbas(ex, w, args.part_start_sectors))

    dev = torch.device("cuda")
    time.sleep(args.settle)
    t_load0 = time.monotonic_ns()
    model = AutoModelForCausalLM.from_pretrained(
        mdir, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    torch.cuda.synchronize()
    t_load1 = time.monotonic_ns()
    time.sleep(args.settle)
    t_load2 = time.monotonic_ns()
    resid_after_load = [
        resident_pages(fd, w, sz)[0] for fd, sz, w in zip(fds, sizes, whole)
    ]

    torch.manual_seed(0)
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (1, args.prompt_tokens), device=dev)
    trials = []
    for t in range(args.trials):
        time.sleep(args.settle)
        t0 = time.monotonic_ns()
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.new_tokens,
                min_new_tokens=args.new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        t1 = time.monotonic_ns()
        time.sleep(args.settle)
        t2 = time.monotonic_ns()
        trials.append(
            {
                "trial": t,
                "t_start_ns": t0,
                "t_end_ns": t1,
                "t_drain_end_ns": t2,
                "elapsed_ns": t1 - t0,
                "F_bytes": 0,
                "F_kind": "none_expected",
                "resident_pages_before": None,
                "target_pages": None,
                "generated_tokens": int(out.shape[1] - ids.shape[1]),
            }
        )
        print(
            f"trial {t}: {args.new_tokens} tokens in {(t1 - t0) / 1e6:.1f} ms",
            flush=True,
        )
    # the load window is reported as trial -2 so the wrapper attributes it too
    trials.insert(
        0,
        {
            "trial": -2,
            "t_start_ns": t_load0,
            "t_end_ns": t_load1,
            "t_drain_end_ns": t_load2,
            "elapsed_ns": t_load1 - t_load0,
            "F_bytes": sum(sizes),
            "F_kind": "native_loader_file_bytes",
            "resident_pages_before": sum(r[0] for r in resid_before),
            "target_pages": sum(r[1] for r in resid_before),
            "resident_pages_after_load": sum(resid_after_load),
        },
    )
    doc = {
        "file": str(shards[0]),
        "files": [str(s) for s in shards],
        "file_size": sum(sizes),
        "inode": os.fstat(fds[0]).st_ino,
        "label": f"native-load-{args.cache}-then-resident-decode",
        "repo": str(mdir),
        "unit": "whole_model",
        "arm": "native_loader",
        "cache": args.cache,
        "qd": 0,
        "chunk": 0,
        "align": 4096,
        "mode": "resident_infer",
        "n_ranges": len(whole),
        "W_required_bytes": sum(sizes),
        "ranges": [(0, sz) for sz in sizes],
        "extents": [e for ex in extents for e in ex],
        "device_lbas": lbas,
        "part_start_sectors": args.part_start_sectors,
        "gpu_alloc_after_load": torch.cuda.memory_allocated(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "trials": trials,
    }
    Path(args.out).write_text(json.dumps(doc))


if __name__ == "__main__":
    main()
