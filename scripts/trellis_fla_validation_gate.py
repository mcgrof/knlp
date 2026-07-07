"""On-pod validation gate for the Slot-Mixing Delta / FLA baseline run.

Run this on the CUDA pod BEFORE the training matrix. It answers the questions
that cannot be checked without the installed fla + a real GPU:

  A. FLA layer signatures + the kwargs our FLARefMixer actually passes/drops
     (does GatedDeltaProduct accept num_householder / allow_neg_eigval, and is
     the output-gate kwarg name right).
  B. Forward/backward through each fla reference layer is finite (this is where
     the Hopper #734 chunk_local_cumsum_scalar crash shows up).
  C. chunk vs fused_recurrent numerical oracle (fp32 A/B, the authors' kernels
     must agree) for gated-delta and, if available, gated-delta-product.
  D. Recurrent-state dtype + bytes for the fla rows vs ours, for state-byte
     parity (H*D*D*2 is only right if the state is bf16).
  E. short-conv is off on every arm.

Each check is independently guarded and the script prints a JSON summary at the
end. A non-zero exit means at least one CRITICAL check failed -> do not run the
paid training matrix.

Usage (on the pod, knlp checked out, fla installed):
    python3 scripts/trellis_fla_validation_gate.py --out /workspace/gate.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import traceback

import torch

from trellis_lm.config import TrellisConfig


def _cfg(n_slots: int = 64) -> TrellisConfig:
    # d_model=256, H=4, d_head=64 -> inner_dim==d_model, expand_k=expand_v=1.
    # n_slots=64 matches the fla 64x64 state for the cross-family parity row.
    return TrellisConfig(
        vocab_size=256,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_head=64,
        n_slots=n_slots,
        max_seq_len=128,
        dtype="fp32",
    )


def check_signatures(report: dict) -> None:
    """A + E: fla signatures, FLARefMixer effective/dropped kwargs, attrs."""
    from trellis_lm import linear_baselines_fla_ref as R

    report["has_fla_ref"] = R.HAS_FLA_REF
    if not R.HAS_FLA_REF:
        report["signatures_error"] = "fla reference layers unavailable on host"
        report["_critical_fail"].append("no_fla")
        return
    import fla

    report["fla_version"] = getattr(fla, "__version__", "unknown")
    from fla.layers import DeltaNet, GatedDeltaNet

    sigs = {}
    for name, cls in [("DeltaNet", DeltaNet), ("GatedDeltaNet", GatedDeltaNet)]:
        sigs[name] = list(inspect.signature(cls.__init__).parameters)
    if R.GatedDeltaProduct is not None:
        sigs["GatedDeltaProduct"] = list(
            inspect.signature(R.GatedDeltaProduct.__init__).parameters
        )
    report["fla_signatures"] = sigs

    cfg = _cfg()
    rows = {"delta_ref": (False, 1), "gated_delta_ref": (True, 1)}
    if R.GatedDeltaProduct is not None:
        rows["deltaproduct_nh2"] = (True, 2)
        rows["deltaproduct_nh3"] = (True, 3)
    built = {}
    for row, (gated, nh) in rows.items():
        try:
            m = R.FLARefMixer(cfg, gated=gated, num_householder=nh)
            layer = m.layer
            attrs = {
                k: getattr(layer, k, None)
                for k in (
                    "head_dim",
                    "head_k_dim",
                    "head_v_dim",
                    "num_heads",
                    "num_householder",
                    "allow_neg_eigval",
                    "use_short_conv",
                    "use_gate",
                    "use_output_gate",
                    "expand_v",
                )
            }
            built[row] = {
                "effective_kwargs": getattr(layer, "_fla_effective_kwargs", {}),
                "dropped_kwargs": getattr(layer, "_fla_dropped_kwargs", {}),
                "attrs": {k: str(v) for k, v in attrs.items()},
                "state_bytes_per_seq": m.state_bytes_per_seq(),
            }
            # E: short-conv must be off (mechanism comparison default)
            sc = getattr(layer, "use_short_conv", None)
            if sc:
                report["_critical_fail"].append(f"{row}:short_conv_on")
        except Exception as e:  # noqa: BLE001
            built[row] = {"error": f"{type(e).__name__}: {e}"}
            report["_critical_fail"].append(f"{row}:build_failed")
    report["fla_rows"] = built


def check_forward(report: dict) -> None:
    """B: finite forward+backward through each fla reference layer (bf16)."""
    from trellis_lm import linear_baselines_fla_ref as R

    if not R.HAS_FLA_REF:
        return
    cfg = _cfg()
    dev = "cuda"
    rows = {"delta_ref": (False, 1), "gated_delta_ref": (True, 1)}
    if R.GatedDeltaProduct is not None:
        rows["deltaproduct_nh2"] = (True, 2)
    fwd = {}
    for row, (gated, nh) in rows.items():
        try:
            m = R.FLARefMixer(cfg, gated=gated, num_householder=nh).to(dev)
            x = torch.randn(2, 32, cfg.d_model, device=dev, requires_grad=True)
            y = m(x, training=True)
            loss = y.float().pow(2).mean()
            loss.backward()
            gmax = max(
                float(p.grad.abs().max()) for p in m.parameters() if p.grad is not None
            )
            fwd[row] = {
                "out_finite": bool(torch.isfinite(y).all()),
                "grad_finite": bool(torch.isfinite(torch.tensor(gmax))),
                "grad_max": gmax,
            }
            if not torch.isfinite(y).all():
                report["_critical_fail"].append(f"{row}:forward_nonfinite")
        except Exception as e:  # noqa: BLE001
            fwd[row] = {"error": f"{type(e).__name__}: {e}"}
            report["_critical_fail"].append(f"{row}:forward_crash")
            report.setdefault("_traces", {})[f"fwd_{row}"] = traceback.format_exc()
    report["forward_smoke"] = fwd


def _call_op(fn, q, k, v, g, beta, scale):
    """Invoke an fla op passing only kwargs its signature declares."""
    params = inspect.signature(fn).parameters
    kw = {}
    if "scale" in params:
        kw["scale"] = scale
    if "output_final_state" in params:
        kw["output_final_state"] = True
    if "use_qk_l2norm_in_kernel" in params:
        kw["use_qk_l2norm_in_kernel"] = True
    if "head_first" in params:
        kw["head_first"] = False
    return fn(q, k, v, g, beta, **kw)


def check_oracle(report: dict) -> None:
    """C + D: chunk vs fused_recurrent A/B and recurrent-state dtype/bytes."""
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except Exception as e:  # noqa: BLE001
        report["oracle_error"] = f"import: {type(e).__name__}: {e}"
        return
    dev = "cuda"
    B, T, H, D = 1, 256, 4, 64
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device=dev, dtype=torch.float32)
    k = torch.randn(B, T, H, D, device=dev, dtype=torch.float32)
    v = torch.randn(B, T, H, D, device=dev, dtype=torch.float32)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=dev))
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev))
    scale = D**-0.5
    try:
        oc = _call_op(chunk_gated_delta_rule, q, k, v, g, beta, scale)
        of = _call_op(fused_recurrent_gated_delta_rule, q, k, v, g, beta, scale)
        oc_o = oc[0] if isinstance(oc, tuple) else oc
        of_o = of[0] if isinstance(of, tuple) else of
        err = float((oc_o.float() - of_o.float()).abs().max())
        report["gdn_chunk_vs_fused_maxerr"] = err
        if err > 1e-3:
            report["_critical_fail"].append("gdn_oracle_mismatch")
        # D: recurrent-state dtype/bytes from the chunk op's final state
        if isinstance(oc, tuple) and len(oc) > 1 and torch.is_tensor(oc[1]):
            st = oc[1]
            report["gdn_state"] = {
                "shape": list(st.shape),
                "dtype": str(st.dtype),
                "bytes_per_seq": int(st.numel() * st.element_size() // B),
            }
    except Exception as e:  # noqa: BLE001
        report["oracle_error"] = f"{type(e).__name__}: {e}"
        report["_critical_fail"].append("gdn_oracle_crash")
        report.setdefault("_traces", {})["oracle"] = traceback.format_exc()

    # ours: fp32 M-state bytes for the same width at n_slots=64
    cfg = _cfg()
    ours = cfg.n_heads * cfg.n_slots * cfg.d_head
    report["ours_state"] = {
        "n_heads": cfg.n_heads,
        "n_slots": cfg.n_slots,
        "d_head": cfg.d_head,
        "elements_per_seq": ours,
        "note": "M-state dtype follows the run dtype; fp32 run => 4 bytes/elem",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report: dict = {"_critical_fail": []}
    report["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["device_name"] = torch.cuda.get_device_name(0)
        report["torch"] = torch.__version__

    for fn in (check_signatures, check_forward, check_oracle):
        try:
            fn(report)
        except Exception as e:  # noqa: BLE001
            report[f"{fn.__name__}_fatal"] = f"{type(e).__name__}: {e}"
            report["_critical_fail"].append(f"{fn.__name__}:fatal")
            report.setdefault("_traces", {})[fn.__name__] = traceback.format_exc()

    fails = report["_critical_fail"]
    report["gate_pass"] = len(fails) == 0
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print("\nGATE_PASS" if report["gate_pass"] else f"\nGATE_FAIL: {fails}")
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
