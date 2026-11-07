#!/usr/bin/env python3
"""Write one layer's selected routed experts into a contiguous file (CPU).

The pool from calibrate_pool.py keeps 32 of 127 routed experts per layer.
In the original checkpoint those experts are scattered across the layer's
1.5 GiB run, so a reader that allocates one buffer per contiguous span
would also allocate the unselected holes. This script copies exactly the
selected experts' tensors of one layer, unchanged, into a new safetensors
file, then checks each written tensor against the BF16 manifest hash.
The packed file is the I/O representation every reader then uses, so a
comparison between readers is not affected by the packing.

With --stacked the file instead holds the two tensors the fused MoE kernel
(fused_moe.py) consumes: ``ROUTED_W13`` [experts, 2*inter, hidden] with
each expert's gate rows then up rows, and ``ROUTED_W2`` [experts, hidden,
inter], in pool order. The bytes are the same experts' weights; every
slice is checked against the manifest hash.

With --align N the header is padded with spaces, which the safetensors
format allows, so that the tensor data starts at a multiple of N bytes.
A reader that loads the whole file into one page-aligned GPU buffer and
uses the tensors in place (the DMA-BUF copier) then gets tensors at
N-aligned addresses; without it they start wherever the header ends,
which can slow kernels that read them. The tensor bytes are unchanged.

Usage:
    pack_pool_layer.py MODEL_DIR POOL.json BF16_MANIFEST.json OUT.safetensors \
        [--layer 15] [--stacked] [--align 4096]
"""

import argparse
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    EXPERT_MATS,
    expert_tensor_name,
    file_sha256,
    stacked_name,
    tensor_sha256,
    write_json,
)  # noqa


def data_offset(path):
    with open(path, "rb") as f:
        return 8 + struct.unpack("<Q", f.read(8))[0]


def align_data(path, align):
    """Pad the header with spaces so the tensor data starts N-aligned."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
        body = f.read()
    pad = (-(8 + n)) % align
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", n + pad))
        f.write(header + b" " * pad)
        f.write(body)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("pool")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--stacked", action="store_true")
    ap.add_argument("--align", type=int, default=0)
    args = ap.parse_args()

    with open(args.pool) as f:
        pool = json.load(f)
    with open(args.manifest) as f:
        man = json.load(f)["tensors"]
    routed = pool["layers"][str(args.layer)]["routed"]
    names = [expert_tensor_name(args.layer, e, m) for e in routed for m in EXPERT_MATS]
    t0 = time.time()
    out = {}
    by_shard = {}
    for n in names:
        by_shard.setdefault(man[n]["shard"], []).append(n)
    for shard, ns in by_shard.items():
        with safe_open(os.path.join(args.model_dir, shard), framework="pt") as f:
            for n in ns:
                out[n] = f.get_tensor(n).contiguous()
    bad = [n for n in names if tensor_sha256(out[n]) != man[n]["bf16_sha256"]]
    if bad:
        sys.exit(f"source tensors differ from manifest: {bad[:4]}")
    meta = {"format": "pt", "layer": str(args.layer)}
    if args.stacked:
        g = lambda e, m: out[expert_tensor_name(args.layer, e, m)]  # noqa
        stacked = {
            stacked_name(args.layer, "w13"): torch.stack(
                [torch.cat([g(e, "gate_proj"), g(e, "up_proj")], 0) for e in routed]
            ).contiguous(),
            stacked_name(args.layer, "w2"): torch.stack(
                [g(e, "down_proj") for e in routed]
            ).contiguous(),
        }
        meta["layout"] = "stacked"
        save_file(stacked, args.out, metadata=meta)
    else:
        save_file(out, args.out, metadata=meta)
    if args.align:
        align_data(args.out, args.align)
    with safe_open(args.out, framework="pt") as f:
        if args.stacked:
            got = {}
            w13 = f.get_tensor(stacked_name(args.layer, "w13"))
            w2 = f.get_tensor(stacked_name(args.layer, "w2"))
            inter = w13.shape[1] // 2
            for i, e in enumerate(routed):
                got[expert_tensor_name(args.layer, e, "gate_proj")] = w13[i, :inter]
                got[expert_tensor_name(args.layer, e, "up_proj")] = w13[i, inter:]
                got[expert_tensor_name(args.layer, e, "down_proj")] = w2[i]
        else:
            got = {n: f.get_tensor(n) for n in names}
        bad = [n for n in names if tensor_sha256(got[n]) != man[n]["bf16_sha256"]]
    if bad:
        sys.exit(f"written tensors differ from manifest: {bad[:4]}")
    payload = sum(man[n]["nbytes"] for n in names)
    info = {
        "layer": args.layer,
        "routed_expert_ids": routed,
        "tensors": len(names),
        "payload_bytes": payload,
        "file_bytes": os.path.getsize(args.out),
        "layout": "stacked" if args.stacked else "per-expert",
        "data_offset": data_offset(args.out),
        "file_sha256": file_sha256(args.out),
        "pool_file": os.path.abspath(args.pool),
        "seconds": round(time.time() - t0, 2),
    }
    write_json(args.out + ".json", info)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
