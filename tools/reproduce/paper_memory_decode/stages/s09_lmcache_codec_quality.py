import sys
"""Stage 09: LMCache AsymK16V8Codec quality gate.

Runs gate_lmcache_codec.py to verify the storage-tier claims:
  - AsymK16V8Codec gives storage ratio ≈ 0.75 vs FP16 baseline
  - K is preserved bit-exactly
  - V dequantizes within FP8 e4m3 noise (median rel err < 0.075)

This tests the LMCache storage tier only (CPU, no GPU, no vLLM).
The vLLM runtime E2E path (Milestones 2-6) is tracked separately and
is not yet complete in the public branches.

Reference results (2026-04-25 H100, 24-cell grid):
  storage_ratio median = 0.7500
  K bit-exact = True across all cells
  V rel err median = 0.0217
"""

from __future__ import annotations

import re
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_lmcache_codec.py"


def run(ctx: StageContext) -> StageResult:
    result_path = ctx.stage_dir / "lmcache_codec_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={"KNLP_RESULT_PATH": str(result_path)},
        timeout=300,
    )

    ratio = k_exact = v_err = nvme_ratio = None
    try:
        text = ctx.stdout_path.read_text()
        for tag in (
            "STORAGE_RATIO_MEDIAN",
            "K_BIT_EXACT",
            "V_REL_ERR_MEDIAN",
            "SPLIT_TIER_NVME_RATIO",
        ):
            m = re.search(rf"{tag}=([0-9.NA/]+)", text)
            if m and m.group(1) not in ("N/A", ""):
                try:
                    val = float(m.group(1))
                    if tag == "STORAGE_RATIO_MEDIAN":
                        ratio = val
                    elif tag == "K_BIT_EXACT":
                        k_exact = int(val)
                    elif tag == "V_REL_ERR_MEDIAN":
                        v_err = val
                    elif tag == "SPLIT_TIER_NVME_RATIO":
                        nvme_ratio = val
                except ValueError:
                    pass
    except Exception:
        pass

    for name, val in [
        ("storage_ratio_median", ratio),
        ("k_bit_exact", k_exact),
        ("v_rel_err_median", v_err),
        ("split_tier_nvme_ratio", nvme_ratio),
    ]:
        if val is not None:
            ctx.log_metric(name, val)

    if result_path.exists():
        ctx.telemetry.log_artifact(result_path, "codec_results")

    if rc == 2:
        reason = "lmcache not importable; install LMCache (stage 04)"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_lmcache_codec.py returned rc={rc}",
        )

    ctx.mark_done(
        {
            "storage_ratio_median": ratio,
            "k_bit_exact": k_exact,
            "v_rel_err_median": v_err,
        }
    )
    return StageResult(name=ctx.name, status="passed")
