import sys
"""Stage 06: vLLM asymmetric KV cache writer correctness gate.

Runs gate_writer_asym.py to verify the K-lossless / V-FP8 write
contract is upheld.  Runs on CPU; no GPU, no FlashInfer, no vLLM
needed.

Pass criteria (all four sub-tests must pass):
  - K bit-exact at every written slot
  - V dequantizes within FP8 e4m3 noise (median rel err < 0.075)
  - K and V do not alias across their respective buffer halves
  - slot_mapping=-1 tokens are correctly skipped
"""

from __future__ import annotations

from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_writer_asym.py"


def run(ctx: StageContext) -> StageResult:
    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        timeout=120,
    )

    if rc == 2:
        reason = "gate skipped (torch.float8_e4m3fn unavailable)"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_writer_asym.py returned rc={rc}; "
            "see stages/06_vllm_writer_gate/stdout.log",
        )

    ctx.log_metric("writer_gate_passed", 1)
    ctx.mark_done({"writer_gate_passed": True})
    return StageResult(name=ctx.name, status="passed")
