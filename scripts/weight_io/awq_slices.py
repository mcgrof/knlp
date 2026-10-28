#!/usr/bin/env python3
"""Emit the byte ranges that a set of selected neurons occupies in AWQ tensors.

AutoAWQ GEMM packs INT4 weights across output features: qweight has shape
[K, N/8] int32, so one output neuron j is nibble (j % 8) of column j // 8
in every one of the K rows. Its bytes are therefore K separate 4-byte
words with a stride of N/2 bytes. The matching scales ([K/G, N] fp16) and
qzeros ([K/G, N/8] int32) add K/G words each.

For the down projection the roles swap: an input neuron j of down_proj is
row j of its qweight, which is one contiguous N/2-byte run.

This script reads the real shard header, computes those ranges for a
chosen neuron set and writes a JSON list of [abs_start, length] with the
useful-byte accounting (W_code, W_meta) beside it. The union of 4-byte
words that fall in one int32 (eight neighbouring neurons) is taken, so a
clustered selection shares words.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path


def shard_headers(model_dir: Path):
    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            hl = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hl))
        base = 8 + hl
        for name, t in hdr.items():
            if name == "__metadata__":
                continue
            s, e = t["data_offsets"]
            out[name] = (str(f), base + s, e - s, t["dtype"], t["shape"])
    return out


def merge(ranges):
    ranges = sorted(ranges)
    out = []
    for a, n in ranges:
        if out and a <= out[-1][0] + out[-1][1]:
            b = max(out[-1][0] + out[-1][1], a + n)
            out[-1] = (out[-1][0], b - out[-1][0])
        else:
            out.append((a, n))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument(
        "--proj", choices=["gate_proj", "up_proj", "down_proj"], required=True
    )
    ap.add_argument(
        "--neurons", type=int, required=True, help="number of neurons to select"
    )
    ap.add_argument("--pattern", choices=["clustered", "scattered"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    h = shard_headers(Path(args.model_dir))
    base = f"model.layers.{args.layer}.mlp.{args.proj}"
    qw = h[base + ".qweight"]
    qz = h[base + ".qzeros"]
    sc = h[base + ".scales"]
    file = qw[0]
    assert qz[0] == file and sc[0] == file, "tensors span shards"
    K, N8 = qw[4]
    N = N8 * 8
    KG = qz[4][0]
    G = K // KG
    if args.proj == "down_proj":
        # input neurons: rows of qweight (contiguous), rows of scales/qzeros groups
        total = K
    else:
        total = N
    rng = random.Random(args.seed)
    if args.pattern == "clustered":
        start = rng.randrange(0, total - args.neurons)
        sel = list(range(start, start + args.neurons))
    else:
        sel = sorted(rng.sample(range(total), args.neurons))

    ranges = []
    w_code = w_meta = 0
    if args.proj == "down_proj":
        row_bytes = N8 * 4
        for j in sel:
            ranges.append((qw[1] + j * row_bytes, row_bytes))
            w_code += N // 2  # N nibbles
        groups = sorted({j // G for j in sel})
        for g in groups:
            ranges.append((sc[1] + g * N * 2, N * 2))
            ranges.append((qz[1] + g * N8 * 4, N8 * 4))
            w_meta += N * 2 + N8 * 4
    else:
        cols = sorted({j // 8 for j in sel})
        for k in range(K):
            rowoff = qw[1] + k * N8 * 4
            for c in cols:
                ranges.append((rowoff + c * 4, 4))
        w_code = len(sel) * K // 2  # nibbles
        for g in range(KG):
            for j in sel:
                ranges.append((sc[1] + (g * N + j) * 2, 2))
            for c in cols:
                ranges.append((qz[1] + (g * N8 + c) * 4, 4))
        w_meta = len(sel) * KG * 2 + len(cols) * KG * 4
    merged = merge(ranges)
    doc = {
        "file": file,
        "layer": args.layer,
        "proj": args.proj,
        "pattern": args.pattern,
        "neurons": (
            sel
            if len(sel) <= 32
            else {"n": len(sel), "first": sel[:4], "last": sel[-4:]}
        ),
        "n_neurons": len(sel),
        "K": K,
        "N": N,
        "group_size": G,
        "W_code": w_code,
        "W_meta": w_meta,
        "W_required": sum(n for _, n in merged),
        "n_ranges": len(merged),
        "ranges": merged,
    }
    Path(args.out).write_text(json.dumps(doc))
    print(
        f"{args.proj} {args.pattern} n={len(sel)}: W_code={w_code} W_meta={w_meta} W_required={doc['W_required']} ranges={len(merged)}"
    )


if __name__ == "__main__":
    main()
