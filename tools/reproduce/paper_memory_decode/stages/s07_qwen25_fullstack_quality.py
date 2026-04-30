"""Stage 07: Qwen2.5-7B full-stack quality battery.

Runs gate_qwen_quality.py to measure GSM8K accuracy (n=200) and
HF PPL@2K across FP16 / FP8-sym / asym K16/V8 via the modified
vLLM + FlashInfer serving stack.

Pass criteria:
  - asym GSM8K accuracy within 1.5pp of FP16 baseline
  - asym must not show the Qwen K-fragility collapse (acc > 0.5)
  - FP8-sym is expected to collapse (acc ≈ 2%) — this is the paper claim

Reference results (2026-04-27 H100):
  FP16      acc=90.5%
  FP8-sym   acc=2.0%   (Qwen K-fragility confirmed)
  Asym K16/V8 acc=90.0% (matches FP16)
"""

from __future__ import annotations

import sys
import json
import re
from pathlib import Path

from ..stages import StageContext, StageResult

_GATE = Path(__file__).resolve().parents[1] / "gate_qwen_quality.py"

ASYM_DELTA_THRESHOLD = 0.015  # asym GSM8K must be within 1.5pp of FP16
ASYM_MIN_ACC = 0.50  # asym must not collapse (fp8_sym collapses to ~2%)


def run(ctx: StageContext) -> StageResult:
    import torch

    if not torch.cuda.is_available():
        ctx.mark_skipped("no GPU; Qwen2.5-7B requires GPU inference")
        return StageResult(
            name=ctx.name,
            status="skipped",
            reason="no GPU; Qwen2.5-7B requires GPU inference",
        )

    result_path = ctx.stage_dir / "qwen_quality_results.json"

    rc = ctx.run_subprocess(
        [sys.executable, str(_GATE)],
        extra_env={
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "FLASHINFER_EXPERIMENTAL_ASYM_PREFILL": "1",
            "KNLP_MODEL_QWEN": ctx.cfg.model_qwen25_7b,
            "KNLP_GSM8K_N": "200",
            "KNLP_RESULT_PATH": str(result_path),
        },
        timeout=7200,  # quality battery can take 1-2 h cold model download
    )

    # Parse structured tags from stdout.
    fp16_acc = asym_acc = sym_acc = delta = ppl = None
    try:
        text = ctx.stdout_path.read_text()
        for tag, var in [
            ("FP16_GSM8K_ACC", None),
            ("ASYM_GSM8K_ACC", None),
            ("SYM_GSM8K_ACC", None),
            ("ASYM_FP16_DELTA", None),
            ("HF_PPL_2K", None),
        ]:
            m = re.search(rf"{tag}=([0-9.]+)", text)
            if m:
                val = float(m.group(1))
                if tag == "FP16_GSM8K_ACC":
                    fp16_acc = val
                elif tag == "ASYM_GSM8K_ACC":
                    asym_acc = val
                elif tag == "SYM_GSM8K_ACC":
                    sym_acc = val
                elif tag == "ASYM_FP16_DELTA":
                    delta = val
                elif tag == "HF_PPL_2K":
                    ppl = val
    except Exception:
        pass

    for name, val in [
        ("fp16_gsm8k_acc", fp16_acc),
        ("asym_gsm8k_acc", asym_acc),
        ("sym_gsm8k_acc", sym_acc),
        ("asym_fp16_delta", delta),
        ("hf_ppl_2k", ppl),
    ]:
        if val is not None:
            ctx.log_metric(name, val)

    if result_path.exists():
        ctx.telemetry.log_artifact(result_path, "quality_results")

    if rc not in (0, 1):
        # rc=1 means threshold breach caught by gate script itself.
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"gate_qwen_quality.py crashed (rc={rc})",
        )

    # Our own threshold checks (independent of gate script).
    failures = []
    if asym_acc is not None:
        if asym_acc < ASYM_MIN_ACC:
            failures.append(
                f"asym GSM8K acc={asym_acc:.3f} < collapse threshold "
                f"{ASYM_MIN_ACC} (FP8-sym collapse level is ~0.02)"
            )
        if fp16_acc and fp16_acc > 0:
            actual_delta = abs(asym_acc - fp16_acc) / fp16_acc
            if actual_delta > ASYM_DELTA_THRESHOLD:
                failures.append(
                    f"asym GSM8K delta {actual_delta:.4f} > "
                    f"threshold {ASYM_DELTA_THRESHOLD} "
                    f"(fp16={fp16_acc:.3f} asym={asym_acc:.3f})"
                )

    if rc == 1 and not failures:
        failures.append("gate_qwen_quality.py reported failure (rc=1)")

    if failures:
        return StageResult(name=ctx.name, status="failed", reason="; ".join(failures))

    ctx.mark_done(
        {
            "fp16_gsm8k_acc": fp16_acc,
            "asym_gsm8k_acc": asym_acc,
            "sym_gsm8k_acc": sym_acc,
            "asym_fp16_delta": delta,
            "hf_ppl_2k": ppl,
        }
    )
    return StageResult(name=ctx.name, status="passed")
