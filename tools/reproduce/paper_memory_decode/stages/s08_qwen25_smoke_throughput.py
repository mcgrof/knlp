"""Stage 08: Qwen2.5-7B smoke throughput measurement.

Runs gate_qwen_smoke.py to measure tok/s for FP16 / FP8-sym / asym
K16/V8.  Throughput is informational; this stage does not gate the run
on specific tok/s values (hardware varies too much).

Reference results (2026-04-27 H100 SXM, enforce_eager=True):
  FP16      102.6 tok/s
  FP8-sym    97.9 tok/s
  Asym K16/V8 105.8 tok/s
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_qwen_smoke.py"


def run(ctx: StageContext) -> StageResult:
    import torch

    if not torch.cuda.is_available():
        ctx.mark_skipped("no GPU; throughput test requires GPU")
        return StageResult(
            name=ctx.name,
            status="skipped",
            reason="no GPU; throughput test requires GPU",
        )

    result_path = ctx.stage_dir / "qwen_smoke_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "FLASHINFER_EXPERIMENTAL_ASYM_PREFILL": "1",
            "KNLP_MODEL_QWEN": ctx.cfg.model_qwen25_7b,
            "KNLP_RESULT_PATH": str(result_path),
        },
        timeout=3600,
    )

    fp16_tps = sym_tps = asym_tps = None
    try:
        text = ctx.stdout_path.read_text()
        for tag in ("FP16_TOKS", "SYM_TOKS", "ASYM_TOKS"):
            m = re.search(rf"{tag}=([0-9.]+)", text)
            if m:
                val = float(m.group(1))
                if tag == "FP16_TOKS":
                    fp16_tps = val
                elif tag == "SYM_TOKS":
                    sym_tps = val
                elif tag == "ASYM_TOKS":
                    asym_tps = val
    except Exception:
        pass

    for name, val in [
        ("fp16_tok_per_s", fp16_tps),
        ("sym_tok_per_s", sym_tps),
        ("asym_tok_per_s", asym_tps),
    ]:
        if val is not None:
            ctx.log_metric(name, val)

    if result_path.exists():
        ctx.telemetry.log_artifact(result_path, "throughput_results")

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_qwen_smoke.py returned rc={rc}",
        )

    ctx.mark_done(
        {
            "fp16_tok_per_s": fp16_tps,
            "sym_tok_per_s": sym_tps,
            "asym_tok_per_s": asym_tps,
        }
    )
    return StageResult(name=ctx.name, status="passed")
