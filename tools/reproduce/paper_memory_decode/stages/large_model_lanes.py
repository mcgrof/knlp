"""Stage large_model_lanes: 70B-class model sanity on 4×H100 (TP=4).

Runs gate_large_model.py to verify the asym K16/V8 stack works for
a 70B-class model under tensor parallelism.  Tests Qwen2.5-72B-Instruct
with TP=4 across both FP16 and asym KV configurations.

Pass criteria:
  - FP16 GSM8K spot-check ≥ 80% (5 problems)
  - Asym K16/V8 within 10pp of FP16

Hardware requirement: exactly 4 GPUs, each ≥70 GB.
Skips gracefully on any other hardware configuration.
Expected runtime: 30-60 min (model load dominates).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_large_model.py"

REQUIRED_GPUS = 4


def run(ctx: StageContext) -> StageResult:
    import torch

    if not torch.cuda.is_available():
        ctx.mark_skipped("no CUDA GPU available")
        return StageResult(
            name=ctx.name, status="skipped", reason="no CUDA GPU available"
        )

    n_gpu = torch.cuda.device_count()
    if n_gpu < REQUIRED_GPUS:
        reason = f"only {n_gpu} GPUs; large model lanes require {REQUIRED_GPUS}"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        reason = f"GPU sm{cap[0]}{cap[1]} < sm89; asym KV requires H100/H200/B200"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    result_path = ctx.stage_dir / "large_model_results.json"
    large_model = ctx.cfg.raw.get(
        "CONFIG_KNLP_LARGE_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"
    )

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_RESULT_PATH": str(result_path),
            "KNLP_LARGE_MODEL_ID": large_model,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        timeout=7200,
    )

    metrics: dict = {}
    try:
        text = ctx.stdout_path.read_text()
        for key in ["FP16_ACC", "ASYM_ACC", "ASYM_DELTA"]:
            m = re.search(rf"LARGE_{key}=([0-9.\-]+)", text)
            if m:
                metrics[key.lower()] = float(m.group(1))
    except Exception:
        pass

    for k, v in metrics.items():
        ctx.log_metric(k, v)

    if rc == 2:
        reason = "gate skipped (hardware requirements not met)"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_large_model.py returned rc={rc}; metrics={metrics}",
        )

    ctx.mark_done({**metrics, "model": large_model, "result_path": str(result_path)})
    return StageResult(name=ctx.name, status="passed")
