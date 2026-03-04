#!/usr/bin/env python3
"""Phase 1: Storage microbenchmark for QK Router."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.qk_router.blocking import compute_block_bytes
from lib.qk_router.storage import run_storage_microbench
from lib.qk_router.utils import save_json


def main():
    run_root = os.environ.get("RUN_ROOT", "/mnt/tmpfs/knlp/results/qk_router_01")
    tmpfs_store = os.environ.get("TMPFS_STORE", "/mnt/tmpfs/qk_router_01")
    sfs_store = os.environ.get("SFS_STORE", "/mnt/SFS-hugging/hub/qk_router_01")

    # Qwen2.5-0.5B: 2 KV heads, head_dim=64, block_size=128, bf16
    num_kv_heads = 2
    head_dim = 64
    block_size = 128
    dtype_bytes = 2  # bf16
    num_layers = 24

    block_bytes_per_layer = compute_block_bytes(
        num_kv_heads, head_dim, block_size, dtype_bytes
    )
    total_block_bytes = block_bytes_per_layer * num_layers

    print("=" * 60)
    print("QK Router Phase 1: Storage Microbench")
    print("=" * 60)
    print(f"Block size: {block_size} tokens")
    print(f"Bytes per layer: {block_bytes_per_layer:,}")
    print(f"Total block bytes (all layers): {total_block_bytes:,}")
    print(f"Total block KB: {total_block_bytes / 1024:.1f}")
    print()

    results = run_storage_microbench(tmpfs_store, sfs_store, total_block_bytes)
    results["block_config"] = {
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "dtype_bytes": dtype_bytes,
        "num_layers": num_layers,
        "bytes_per_layer": block_bytes_per_layer,
        "total_block_bytes": total_block_bytes,
    }

    out_path = os.path.join(run_root, "storage_microbench.json")
    save_json(results, out_path)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n--- Summary ---")
    for tier, label in [("tmpfs", "Tier 1 (warm)"), ("sfs", "Tier 2 (cold)")]:
        if tier in results and "sequential" in results[tier]:
            r = results[tier]["sequential"]
            if "error" not in r:
                print(
                    f"{label}: p50={r['p50_us']:.0f}us "
                    f"p95={r['p95_us']:.0f}us "
                    f"throughput={r['throughput_mb_s']:.1f}MB/s"
                )
            else:
                print(f"{label}: ERROR - {r['error']}")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
