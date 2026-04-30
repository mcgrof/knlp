"""Stage spec_decode_interaction: speculative decoding × KV quant composition.

Runs gate_spec_decode.py to measure how n-gram speculation (k=5)
composes with FP8-sym KV quantization.  Reproduces the composition
ratio analysis from the paper's speculative decoding section:

    rho = S_combined / (S_spec_speedup × S_quant_speedup)

Pass criteria:
  - At least one super-multiplicative configuration (rho > 1.1) found
    at long context (T ≥ 8K) with small batch (B=1)

Requires an sm89+ GPU.  Expected runtime: 3-6 h on a single H100 SXM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_spec_decode.py"


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

    result_path = ctx.stage_dir / "spec_decode_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "KNLP_RESULT_PATH": str(result_path),
            "KNLP_MODEL_QWEN25_7B": ctx.cfg.qwen25_7b,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        timeout=21600,  # 6 h ceiling
    )

    metrics: dict = {}
    try:
        text = ctx.stdout_path.read_text()
        for key in ["N_CONFIGS", "SUB_MULT", "SUPER_MULT", "MEAN_RHO"]:
            m = re.search(rf"SPEC_{key}=([0-9.]+)", text)
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
            reason=f"gate_spec_decode.py returned rc={rc}; metrics={metrics}",
        )

    ctx.mark_done({**metrics, "result_path": str(result_path)})
    return StageResult(name=ctx.name, status="passed")
