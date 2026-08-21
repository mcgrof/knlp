# SPDX-License-Identifier: GPL-2.0
"""Measure token-axis structure of captured KV blocks.

The first KV-House question: do real KV caches carry temporal
structure that a token-shuffled control does not? Emits one CSV row
per (sample, layer, kv head, tensor, block size, block index) with
the real-vs-shuffled metric pairs from kv_house.temporal_stats. The
claim under test is supported only where the real ordering is
materially stronger than the shuffled ordering; the "random" text
class is the second control (structureless input).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from kv_house.temporal_stats import block_stats  # noqa: E402


def iter_blocks(x, block_size, max_blocks):
    """Evenly spaced full blocks across the context."""
    t = x.shape[0]
    n = t // block_size
    if n == 0:
        return
    take = min(n, max_blocks)
    for j in range(take):
        idx = j * n // take
        yield idx, x[idx * block_size : (idx + 1) * block_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--block-sizes", default="8,16,32")
    ap.add_argument("--max-blocks", type=int, default=8)
    ap.add_argument("--max-heads", type=int, default=0, help="0 = all kv heads")
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap_dir = Path(args.capture_dir)
    manifest = json.loads((cap_dir / "manifest.json").read_text())
    block_sizes = [int(b) for b in args.block_sizes.split(",")]

    rows = []
    for entry in manifest:
        rec = torch.load(cap_dir / entry["file"], map_location="cpu")
        for layer in sorted(rec)[:: args.layer_stride]:
            tensors = rec[layer]
            n_kv = tensors["k_post"].shape[0]
            heads = range(n_kv if not args.max_heads else min(n_kv, args.max_heads))
            for h in heads:
                for tname in ("k_post", "k_pre", "v"):
                    if tname not in tensors:
                        continue
                    x = tensors[tname][h].float()
                    for b in block_sizes:
                        for j, blk in iter_blocks(x, b, args.max_blocks):
                            row = {
                                "file": entry["file"],
                                "class": entry["class"],
                                "ctx": entry["ctx"],
                                "layer": layer,
                                "head": h,
                                "tensor": tname,
                                "block_size": b,
                                "block_index": j,
                            }
                            row.update(block_stats(blk, x, seed=j))
                            rows.append(row)
        print(f"measured {entry['file']}: {len(rows)} rows total")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
