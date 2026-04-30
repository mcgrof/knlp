"""Stage static_calibration_battery: static vs dynamic FP8 calibration.

Runs gate_static_calibration.py to verify the paper's finding that
static FP8 KV calibration does not improve over dynamic (per-tensor)
scaling for Qwen2.5-7B.  The Qwen K-fragility collapse persists under
static calibration because the error is in the K-component quantization,
not in scale selection.

Pass criteria:
  - FP8-static GSM8K accuracy ≤ FP8-sym + 5pp (static not better)
  - FP16 and FP8-sym both run successfully

Requires an sm89+ GPU.  Expected runtime: 30-60 min.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_static_calibration.py"


def run(ctx: StageContext) -> StageResult:
    import torch

    if not torch.cuda.is_available():
        ctx.mark_skipped("no CUDA GPU available")
        return StageResult(
            name=ctx.name, status="skipped", reason="no CUDA GPU available"
        )

    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        reason = f"GPU sm{cap[0]}{cap[1]} < sm89; FP8 e4m3 requires H100/H200/B200"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    result_path = ctx.stage_dir / "static_calibration_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_RESULT_PATH": str(result_path),
            "KNLP_MODEL_ID": ctx.cfg.qwen25_7b,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        timeout=7200,
    )

    metrics: dict = {}
    try:
        text = ctx.stdout_path.read_text()
        for key in ["FP16_ACC", "SYM_ACC", "CALIB_ACC"]:
            m = re.search(rf"STATIC_{key}=([0-9.]+)", text)
            if m:
                metrics[key.lower()] = float(m.group(1))
    except Exception:
        pass

    for k, v in metrics.items():
        ctx.log_metric(k, v)

    if rc == 2:
        reason = "gate skipped"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_static_calibration.py returned rc={rc}; metrics={metrics}",
        )

    ctx.mark_done({**metrics, "result_path": str(result_path)})
    return StageResult(name=ctx.name, status="passed")
