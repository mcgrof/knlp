"""Stage cross_gpu_lanes: cross-GPU consistency check.

Runs gate_cross_gpu.py on whatever GPU is present to verify the core
asym K16/V8 quality result (Qwen K-fragility, asym matches FP16)
reproduces and records the GPU identity for the run manifest.

Unlike stage 07 (which is H100-specific), this stage adapts to the
available hardware:
  - sm89+ (H100/H200/B200): tests all three configs including asym
  - sm80 (A100):            tests FP16 and FP8-sym only; asym skipped
  - sm80-:                  skips entirely

Pass criteria match the core Qwen quality gate.
Expected runtime: 15-30 min.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_cross_gpu.py"


def run(ctx: StageContext) -> StageResult:
    import torch

    if not torch.cuda.is_available():
        ctx.mark_skipped("no CUDA GPU available")
        return StageResult(
            name=ctx.name, status="skipped", reason="no CUDA GPU available"
        )

    cap = torch.cuda.get_device_capability()
    if cap < (8, 0):
        reason = f"GPU sm{cap[0]}{cap[1]} < sm80; FP8 requires sm80+"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    result_path = ctx.stage_dir / "cross_gpu_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_RESULT_PATH": str(result_path),
            "KNLP_MODEL_ID": ctx.cfg.qwen25_7b,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        timeout=5400,
    )

    metrics: dict = {}
    try:
        text = ctx.stdout_path.read_text()
        for key in ["NAME", "COMPUTE", "FP16_ACC", "SYM_ACC", "ASYM_ACC"]:
            m = re.search(rf"CROSS_GPU_{key}=(.+)", text)
            if m:
                val = m.group(1).strip()
                try:
                    metrics[key.lower()] = float(val)
                except ValueError:
                    metrics[key.lower()] = val
    except Exception:
        pass

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            ctx.log_metric(k, v)

    if rc == 2:
        reason = "gate skipped (GPU not supported)"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_cross_gpu.py returned rc={rc}; metrics={metrics}",
        )

    ctx.mark_done({**metrics, "result_path": str(result_path)})
    return StageResult(name=ctx.name, status="passed")
