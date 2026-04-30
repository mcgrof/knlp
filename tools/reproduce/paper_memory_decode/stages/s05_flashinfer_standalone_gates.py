"""Stage 05: FlashInfer asymmetric K/V standalone gate.

Runs gate_flashinfer_asym.py to verify the asym-prefill-refactor-stage
FlashInfer fork supports BF16-K / FP8-V paged KV attention.

Pass criteria:
  - decode median rel err < 0.10  (measured: 0.0254 on H100)
  - prefill median rel err < 0.10 (measured: 0.0255 on H100 after FI-4d/FI-5)

Requires sm89+ GPU (H100, H200, B200, A100 with FP8 support).
Skips with structured skip entry when GPU is sm89- or absent.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_flashinfer_asym.py"

DECODE_ERR_BOUND = 0.10
PREFILL_ERR_BOUND = 0.10


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

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "FLASHINFER_EXPERIMENTAL_ASYM_PREFILL": "1",
        },
        timeout=300,
    )

    # Parse metrics from stdout log.
    decode_err = prefill_err = None
    try:
        text = ctx.stdout_path.read_text()
        m = re.search(r"DECODE_REL_ERR=([0-9.]+)", text)
        if m:
            decode_err = float(m.group(1))
        m = re.search(r"PREFILL_REL_ERR=([0-9.]+)", text)
        if m:
            prefill_err = float(m.group(1))
    except Exception:
        pass

    if decode_err is not None:
        ctx.log_metric("decode_rel_err", decode_err)
    if prefill_err is not None:
        ctx.log_metric("prefill_rel_err", prefill_err)

    if rc == 2:
        # Gate script exited with 2 = skip (GPU not capable).
        reason = "gate skipped (GPU capability)"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_flashinfer_asym.py returned rc={rc}; "
            f"decode_err={decode_err} prefill_err={prefill_err}",
        )

    # Check bounds.
    failures = []
    if decode_err is not None and decode_err >= DECODE_ERR_BOUND:
        failures.append(f"decode rel err {decode_err:.4f} >= bound {DECODE_ERR_BOUND}")
    if prefill_err is not None and prefill_err >= PREFILL_ERR_BOUND:
        failures.append(
            f"prefill rel err {prefill_err:.4f} >= bound {PREFILL_ERR_BOUND}"
        )
    if failures:
        return StageResult(name=ctx.name, status="failed", reason="; ".join(failures))

    ctx.mark_done(
        {
            "decode_rel_err": decode_err,
            "prefill_rel_err": prefill_err,
        }
    )
    return StageResult(name=ctx.name, status="passed")
