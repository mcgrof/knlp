"""Stage turboquant_stress: quantization pipeline stress test.

Exercises the FP8 quantization pipeline under adversarial tensor
statistics to verify numerical robustness:

  - Near-zero tensors (underflow risk)
  - Large-magnitude tensors (overflow / saturation)
  - Mixed-magnitude tensors (scale selection stress)
  - High-entropy random tensors
  - Structured tensors (attention patterns, spike at position 0)

For each condition, verifies:
  - Encode/decode round-trip V rel err < 0.15 (relaxed for adversarial)
  - K round-trip is bit-exact
  - No NaN/Inf produced

CPU-only.  Skips if lmcache codec is not importable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ..stages import StageContext, StageResult

CHUNK = 512
N_LAYERS = 4
N_HEADS = 8
D_HEAD = 128
V_REL_ERR_BOUND = 0.15


def _make_tensors(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (N_LAYERS, CHUNK, N_HEADS, D_HEAD)
    if mode == "near_zero":
        k = torch.randn(*shape, dtype=torch.bfloat16) * 1e-4
        v = torch.randn(*shape, dtype=torch.bfloat16) * 1e-4
    elif mode == "large_magnitude":
        k = torch.randn(*shape, dtype=torch.bfloat16) * 100.0
        v = torch.randn(*shape, dtype=torch.bfloat16) * 100.0
    elif mode == "mixed":
        k = torch.randn(*shape, dtype=torch.bfloat16)
        k[0, : CHUNK // 2] *= 1e-3
        k[0, CHUNK // 2 :] *= 50.0
        v = k.clone()
    elif mode == "high_entropy":
        k = torch.empty(*shape, dtype=torch.bfloat16).uniform_(-1, 1)
        v = torch.empty(*shape, dtype=torch.bfloat16).uniform_(-1, 1)
    elif mode == "attention_spike":
        # Attention pattern: spike at position 0, small elsewhere.
        k = torch.randn(*shape, dtype=torch.bfloat16) * 0.1
        v = torch.randn(*shape, dtype=torch.bfloat16) * 0.1
        k[:, 0] = 5.0
        v[:, 0] = 5.0
    else:
        k = torch.randn(*shape, dtype=torch.bfloat16)
        v = torch.randn(*shape, dtype=torch.bfloat16)
    return k, v


def run(ctx: StageContext) -> StageResult:
    try:
        from lmcache.v1.kv_codec import AsymK16V8Codec
    except ImportError as e:
        reason = f"lmcache codec not importable: {e}"
        ctx.mark_skipped(reason)
        return StageResult(name=ctx.name, status="skipped", reason=reason)

    codec = AsymK16V8Codec()
    modes = ["near_zero", "large_magnitude", "mixed", "high_entropy", "attention_spike"]
    results: list[dict] = []
    failures: list[str] = []

    for mode in modes:
        k, v = _make_tensors(mode)
        cell: dict = {"mode": mode}
        try:
            encoded = codec.encode(k, v)
            k2, v2 = codec.decode(encoded)
            k_ok = torch.equal(k, k2)
            has_nan = torch.isnan(v2).any().item() or torch.isinf(v2).any().item()
            v_err = ((v - v2).abs() / (v.abs() + 1e-8)).median().item()
            cell.update(
                {
                    "k_bit_exact": k_ok,
                    "v_rel_err": v_err,
                    "has_nan_inf": has_nan,
                    "ok": k_ok and not has_nan and v_err < V_REL_ERR_BOUND,
                }
            )
            if not k_ok:
                failures.append(f"{mode}: K not bit-exact")
            if has_nan:
                failures.append(f"{mode}: NaN/Inf in V output")
            if v_err >= V_REL_ERR_BOUND:
                failures.append(f"{mode}: V rel err={v_err:.4f} >= {V_REL_ERR_BOUND}")
        except Exception as e:
            cell["error"] = str(e)
            failures.append(f"{mode}: exception: {e}")

        results.append(cell)
        status = "OK" if cell.get("ok") else "FAIL"
        ctx.log_metric(f"{mode}_ok", 1 if cell.get("ok") else 0)
        print(f"  {mode}: {status}  v_err={cell.get('v_rel_err', 'NA')}", flush=True)

    result_path = ctx.stage_dir / "turboquant_stress_results.json"
    result_path.write_text(
        json.dumps(
            {"modes": results, "n_failures": len(failures), "failures": failures},
            indent=2,
        )
    )

    ctx.log_metric("n_modes", len(modes))
    ctx.log_metric("n_failures", len(failures))

    if failures:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"{len(failures)}/{len(modes)} modes failed; first: {failures[0]}",
        )

    ctx.mark_done(
        {
            "n_modes": len(modes),
            "n_failures": 0,
            "result_path": str(result_path),
        }
    )
    return StageResult(name=ctx.name, status="passed")
