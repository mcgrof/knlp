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

Usage:
    pack_pool_layer.py MODEL_DIR POOL.json BF16_MANIFEST.json OUT.safetensors \
        [--layer 15]
"""

import argparse
import json
import os
import sys
import time

from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    EXPERT_MATS,
    expert_tensor_name,
    file_sha256,
    tensor_sha256,
    write_json,
)  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("pool")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=15)
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
    save_file(out, args.out, metadata={"format": "pt", "layer": str(args.layer)})
    with safe_open(args.out, framework="pt") as f:
        bad = [
            n for n in names if tensor_sha256(f.get_tensor(n)) != man[n]["bf16_sha256"]
        ]
    if bad:
        sys.exit(f"written tensors differ from manifest: {bad[:4]}")
    payload = sum(man[n]["nbytes"] for n in names)
    info = {
        "layer": args.layer,
        "routed_expert_ids": routed,
        "tensors": len(names),
        "payload_bytes": payload,
        "file_bytes": os.path.getsize(args.out),
        "file_sha256": file_sha256(args.out),
        "pool_file": os.path.abspath(args.pool),
        "seconds": round(time.time() - t0, 2),
    }
    write_json(args.out + ".json", info)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
