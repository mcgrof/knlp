"""Stage niah_16k_32k: Needle In A Haystack retrieval at 16K and 32K.

Runs gate_niah.py to test long-context retrieval accuracy across
FP16, FP8-sym, and asym K16/V8 KV configurations.

Pass criteria:
  - FP16 retrieval accuracy ≥ 90% at both 16K and 32K
  - Asym K16/V8 within 5pp of FP16 at both lengths
  - FP8-sym degradation is expected and not gating

Requires sm89+ GPU (H100/H200/B200).  Skips on sm89- hardware.
Expected runtime: 30-60 min on a single H100 SXM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_niah.py"


def run(ctx: StageContext) -> StageResult:
    import torch

    if not torch.cuda.is_available():
        ctx.mark_skipped("no CUDA GPU available")
        return StageResult(
            name=ctx.name, status="skipped", reason="no CUDA GPU available"
        )

    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        reason = f"GPU sm{cap[0]}{cap[1]} < sm89; asym KV requires H100/H200/B200"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    result_path = ctx.stage_dir / "niah_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_RESULT_PATH": str(result_path),
            "KNLP_MODEL_ID": ctx.cfg.qwen25_7b,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        timeout=5400,  # 90 min ceiling
    )

    metrics: dict = {}
    try:
        text = ctx.stdout_path.read_text()
        for key in [
            "FP16_16K",
            "FP16_32K",
            "ASYM_16K",
            "ASYM_32K",
            "SYM_16K",
            "SYM_32K",
            "ASYM_DELTA_16K",
            "ASYM_DELTA_32K",
        ]:
            m = re.search(rf"NIAH_{key}=([0-9.\-]+)", text)
            if m:
                metrics[key.lower()] = float(m.group(1))
    except Exception:
        pass

    for k, v in metrics.items():
        ctx.log_metric(k, v)

    if rc == 2:
        reason = "gate skipped (GPU capability)"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_niah.py returned rc={rc}; metrics={metrics}",
        )

    ctx.mark_done({**metrics, "result_path": str(result_path)})
    return StageResult(name=ctx.name, status="passed")
