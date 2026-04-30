"""Stage sat_b_vs_bt_model_comparison: compare B vs B*T saturation models.

Reads sweep + Hill fit JSON and compares two model families:

  B-model (per-context):   S(B | T) = S_max(T) * B / (K_m(T) + B)
  B*T-model (unified):     S(B, T) = S_max_bt * (B*T) / (K_m_bt + B*T)

The paper claim is that B*T is a better predictor of throughput because
batch size and context length both scale HBM reads linearly and are
interchangeable in the memory-bandwidth-bound regime.

Pass criteria: B*T model wins on at least 50% of cells.

Skips if sweep or fit results are not present.  CPU-only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_sat_b_vs_bt.py"


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

    result_path = ctx.stage_dir / "sat_b_vs_bt_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_SWEEP_PATH": str(sweep_results[0]),
            "KNLP_FIT_PATH": str(fit_results[0]),
            "KNLP_RESULT_PATH": str(result_path),
        },
        timeout=120,
    )

    bt_win_frac = rss_ratio = km_spearman = None
    try:
        text = ctx.stdout_path.read_text()
        for tag, var in [
            ("BT_WIN_FRACTION", "bt_win_frac"),
            ("RSS_RATIO", "rss_ratio"),
            ("KM_T_SPEARMAN", "km_spearman"),
        ]:
            m = re.search(rf"{tag}=([0-9.\-]+)", text)
            if m:
                locals()[var] = float(m.group(1))
        bt_win_frac = locals().get("bt_win_frac")
        rss_ratio = locals().get("rss_ratio")
        km_spearman = locals().get("km_spearman")
    except Exception:
        pass

    for name, val in [
        ("bt_win_fraction", bt_win_frac),
        ("rss_ratio", rss_ratio),
        ("km_t_spearman", km_spearman),
    ]:
        if val is not None:
            ctx.log_metric(name, val)

    if rc == 2:
        reason = "prerequisite data missing"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_sat_b_vs_bt.py returned rc={rc}; "
            f"km_t_spearman={km_spearman}",
        )

    ctx.mark_done(
        {
            "bt_win_fraction": bt_win_frac,
            "rss_ratio": rss_ratio,
            "km_t_spearman": km_spearman,
            "result_path": str(result_path),
        }
    )
    return StageResult(name=ctx.name, status="passed")
