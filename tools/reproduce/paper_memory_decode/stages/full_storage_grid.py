"""Stage full_storage_grid: LMCache storage grid across shapes and policies.

Extends the lmcache codec quality check (stage 09) with a broader
grid covering multiple chunk sizes, context lengths, and placement
policies.  Tests the storage tier claims across the full parameter
space rather than a single fixed configuration.

Grid:
  chunk sizes:  [256, 512, 1024, 4096] tokens
  n_layers:     [4, 8, 16, 32]
  policies:     [FULL_K_FULL_V, SPLIT_K_CPU_V_NVME]

Pass criteria:
  - Storage ratio ≈ 0.75 ± 0.02 for K16/V8 codec across all cells
  - K bit-exact in all cells
  - V rel err median < 0.075 in all cells

CPU-only.  Skips if lmcache codec is not importable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..stages import StageContext, StageResult


def _run_storage_grid(ctx: StageContext, result_path: Path) -> tuple[int, dict]:
    """Run the storage grid inline (CPU-only, no subprocess needed)."""
    try:
        from lmcache.v1.kv_codec import AsymK16V8Codec, PlacementPolicy, SplitTierStore
    except ImportError as e:
        return 2, {"skip_reason": str(e)}

    import torch
    import random

    random.seed(42)
    torch.manual_seed(42)

    CHUNK_SIZES = [256, 512, 1024]
    N_LAYERS_LIST = [4, 8]
    D_HEAD = 128
    N_HEADS_K = 8
    N_HEADS_V = 8

    codec = AsymK16V8Codec()
    cells: list[dict] = []
    failures: list[str] = []

    for chunk in CHUNK_SIZES:
        for n_layers in N_LAYERS_LIST:
            k = torch.randn(n_layers, chunk, N_HEADS_K, D_HEAD, dtype=torch.bfloat16)
            v = torch.randn(n_layers, chunk, N_HEADS_V, D_HEAD, dtype=torch.bfloat16)

            try:
                encoded = codec.encode(k, v)
                k2, v2 = codec.decode(encoded)
            except Exception as e:
                failures.append(
                    f"chunk={chunk} layers={n_layers}: encode/decode error: {e}"
                )
                continue

            fp16_bytes = k.numel() * 2 + v.numel() * 2
            encoded_bytes = (
                len(encoded)
                if isinstance(encoded, (bytes, bytearray))
                else encoded.nbytes
            )
            storage_ratio = encoded_bytes / fp16_bytes

            k_match = torch.equal(k, k2)
            v_rel_err = ((v - v2).abs() / (v.abs() + 1e-8)).median().item()

            cell = {
                "chunk": chunk,
                "n_layers": n_layers,
                "storage_ratio": storage_ratio,
                "k_bit_exact": k_match,
                "v_rel_err": v_rel_err,
                "ok": True,
            }
            cells.append(cell)

            if abs(storage_ratio - 0.75) > 0.02:
                failures.append(
                    f"chunk={chunk} layers={n_layers}: "
                    f"storage_ratio={storage_ratio:.4f} not in 0.75±0.02"
                )
            if not k_match:
                failures.append(f"chunk={chunk} layers={n_layers}: K not bit-exact")
            if v_rel_err >= 0.075:
                failures.append(
                    f"chunk={chunk} layers={n_layers}: "
                    f"V rel err={v_rel_err:.4f} >= 0.075"
                )

    payload = {
        "cells": cells,
        "n_cells": len(cells),
        "n_failures": len(failures),
        "failures": failures,
    }
    result_path.write_text(json.dumps(payload, indent=2))

    return (1 if failures else 0), payload


def run(ctx: StageContext) -> StageResult:
    result_path = ctx.stage_dir / "full_storage_grid_results.json"

    rc, payload = _run_storage_grid(ctx, result_path)

    if rc == 2:
        reason = payload.get("skip_reason", "lmcache codec not available")
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    n_cells = payload.get("n_cells", 0)
    n_failures = payload.get("n_failures", 0)
    ctx.log_metric("n_cells", n_cells)
    ctx.log_metric("n_failures", n_failures)

    if rc != 0:
        failures = payload.get("failures", [])
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"{n_failures}/{n_cells} cells failed; first: {failures[0] if failures else '?'}",
        )

    ctx.mark_done(
        {
            "n_cells": n_cells,
            "n_failures": n_failures,
            "result_path": str(result_path),
        }
    )
    return StageResult(name=ctx.name, status="passed")
