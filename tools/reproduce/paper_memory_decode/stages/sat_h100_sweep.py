"""Stage sat_h100_sweep: batch × context throughput grid for the saturation model.

Runs gate_sat_sweep.py to measure tok/s for Qwen2.5-7B across a grid of
batch sizes [1, 2, 4, 8, 16, 32] and context lengths [2K, 4K, 8K, 16K, 32K].
Results are written to the stage dir as sat_sweep_results.json.

Pass criteria:
  - At least 80% of the (batch, ctx) cells must yield a valid measurement.

Requires an H100 or equivalent GPU with ≥80 GB HBM.
Expected runtime: 60-120 min on a single H100 SXM.
Skips with a structured skip entry when no GPU is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_sat_sweep.py"


def run(ctx: StageContext) -> StageResult:
    import torch

    if not torch.cuda.is_available():
        ctx.mark_skipped("no CUDA GPU available")
        return StageResult(
            name=ctx.name, status="skipped", reason="no CUDA GPU available"
        )

    result_path = ctx.stage_dir / "sat_sweep_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_RESULT_PATH": str(result_path),
            "KNLP_MODEL_ID": ctx.cfg.qwen25_7b,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        timeout=7200,  # 2 h ceiling
    )

    # Parse headline metrics from stdout.
    n_cells = n_ok = n_failed = None
    try:
        text = ctx.stdout_path.read_text()
        import re

        m = re.search(r"SAT_SWEEP_CELLS=(\d+)", text)
        if m:
            n_cells = int(m.group(1))
        m = re.search(r"SAT_SWEEP_OK=(\d+)", text)
        if m:
            n_ok = int(m.group(1))
        m = re.search(r"SAT_SWEEP_FAILED=(\d+)", text)
        if m:
            n_failed = int(m.group(1))
    except Exception:
        pass

    if n_cells is not None:
        ctx.log_metric("n_cells", n_cells)
    if n_ok is not None:
        ctx.log_metric("n_ok", n_ok)
    if n_failed is not None:
        ctx.log_metric("n_failed", n_failed)

    if rc == 2:
        reason = "gate skipped (no GPU)"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_sat_sweep.py returned rc={rc}; "
            f"cells={n_cells} ok={n_ok} failed={n_failed}",
        )

    ctx.mark_done(
        {
            "n_cells": n_cells,
            "n_ok": n_ok,
            "n_failed": n_failed,
            "result_path": str(result_path),
        }
    )
    return StageResult(name=ctx.name, status="passed")
