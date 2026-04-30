"""Stage sat_hill_fit: Hill saturation model fit.

Reads the sweep JSON produced by sat_h100_sweep and fits a Hill curve
to tok/s vs batch size for each context length:

    S(B) = S_max * B / (K_m + B)

Also fits a unified B*T model over the full (batch, ctx) grid.
Results written as sat_hillfit_results.json in the stage dir.

Pass criteria: min R² across context lengths > 0.85.

Skips if the sweep results from sat_h100_sweep are not present.
This stage is CPU-only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_sat_hillfit.py"


def run(ctx: StageContext) -> StageResult:
    # Locate sweep results from the previous stage.
    run_dir = ctx.run_dir
    sweep_results = list(run_dir.glob("stages/sat_h100_sweep/sat_sweep_results.json"))
    if not sweep_results:
        reason = "sat_h100_sweep results not found; run sat_h100_sweep first"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    sweep_path = sweep_results[0]
    result_path = ctx.stage_dir / "sat_hillfit_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_SWEEP_PATH": str(sweep_path),
            "KNLP_RESULT_PATH": str(result_path),
        },
        timeout=300,
    )

    r2_min = r2_mean = bt_r2 = None
    try:
        text = ctx.stdout_path.read_text()
        m = re.search(r"HILL_R2_MIN=([0-9.]+)", text)
        if m:
            r2_min = float(m.group(1))
        m = re.search(r"HILL_R2_MEAN=([0-9.]+)", text)
        if m:
            r2_mean = float(m.group(1))
        m = re.search(r"BT_R2=([0-9.]+)", text)
        if m:
            bt_r2 = float(m.group(1))
    except Exception:
        pass

    if r2_min is not None:
        ctx.log_metric("hill_r2_min", r2_min)
    if r2_mean is not None:
        ctx.log_metric("hill_r2_mean", r2_mean)
    if bt_r2 is not None:
        ctx.log_metric("bt_r2", bt_r2)

    if rc == 2:
        reason = "sweep data missing; skipping fit"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_sat_hillfit.py returned rc={rc}; r2_min={r2_min}",
        )

    ctx.mark_done(
        {
            "r2_min": r2_min,
            "r2_mean": r2_mean,
            "bt_r2": bt_r2,
            "result_path": str(result_path),
        }
    )
    return StageResult(name=ctx.name, status="passed")
