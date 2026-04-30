"""Stage sat_figures: generate saturation model figures.

Reads sweep + Hill fit JSON and generates two publication-quality figures
via gate_sat_figures.py:

  fig_sat_curves.pdf   — tok/s vs batch size per context length with
                         Hill model overlay
  fig_sat_surface.pdf  — 2D heatmap over the (batch, ctx) grid

Figures are written to the stage dir.  This stage skips gracefully if
matplotlib is not installed (figures are informational, not gating).
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_sat_figures.py"


def run(ctx: StageContext) -> StageResult:
    run_dir = ctx.run_dir
    sweep_results = list(run_dir.glob("stages/sat_h100_sweep/sat_sweep_results.json"))
    fit_results = list(run_dir.glob("stages/sat_hill_fit/sat_hillfit_results.json"))

    if not sweep_results or not fit_results:
        missing = []
        if not sweep_results:
            missing.append("sat_h100_sweep")
        if not fit_results:
            missing.append("sat_hill_fit")
        reason = f"prerequisite stage results missing: {missing}"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    fig_dir = ctx.stage_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_SWEEP_PATH": str(sweep_results[0]),
            "KNLP_FIT_PATH": str(fit_results[0]),
            "KNLP_FIG_DIR": str(fig_dir),
        },
        timeout=120,
    )

    figs = list(fig_dir.glob("*.pdf")) + list(fig_dir.glob("*.png"))

    if rc == 2:
        # matplotlib not available — not a hard failure.
        reason = "matplotlib not available; figures skipped"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_sat_figures.py returned rc={rc}",
        )

    ctx.log_metric("figures_generated", len(figs))
    ctx.mark_done(
        {
            "figures": [str(f) for f in figs],
            "fig_dir": str(fig_dir),
        }
    )
    return StageResult(name=ctx.name, status="passed")
