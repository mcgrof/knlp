"""Compare stored execution-device samples with independent CPU references."""

import os
import argparse
import json
from pathlib import Path
import time

import torch

from research.tinystories_depth.data import sha
from research.tinystories_depth.diagnostics import DoublyStochastic, limit
from research.tinystories_depth.qualify import reference_sink
from .coordinator import atomic


def errors(a, b):
    a, b = a.double().reshape(-1, 16), b.double().reshape(-1, 16)
    norms = b.norm(dim=-1)
    absolute = (a - b).norm(dim=-1)
    relative = absolute / norms.clamp_min(1e-300)
    passed = torch.where(norms >= 1e-8, relative <= 1e-3, absolute <= 1e-6)
    return [
        {
            "reference_norm": n.item(),
            "absolute_l2": d.item(),
            "relative_l2": r.item() if n >= 1e-8 else None,
            "pass": bool(p),
        }
        for n, d, r, p in zip(norms, absolute, relative, passed)
    ]


def actual(root, jobid):
    root = Path(root)
    result = json.loads((root / "records/runs" / jobid / "RESULT.json").read_text())
    assert result["status"] == "complete"
    out = root / "records/numerics"
    out.mkdir(exist_ok=True)
    dest = out / (jobid + ".json")
    assert not dest.exists(), "Do not overwrite sampled misses"
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    record = {
        "job_id": jobid,
        "status": "running",
        "snapshots": [],
        "finite_misses": [],
        "reference": "independent FP64 finite20, eps1e-6; relative L2<=1e-3 for norm>=1e-8, otherwise absolute L2<=1e-6",
        "converged_reference": "repaired inherited stationary-limit oracle; diagnostic only; bounded400 log sweeps +200 damped Newton steps",
        "device_comparison": "stored HIP finite output/VJP versus CPU reference, not a CPU recreation of the HIP result",
    }
    before = time.perf_counter()
    for receipt in result["diagnostics"]:
        assert sha(receipt["path"]) == receipt["sha256"]
        payload = torch.load(receipt["path"], map_location="cpu", weights_only=False)
        samples = payload["samples"]
        assert len(samples) == 2 * result["metadata"]["job"]["depth"]
        z = torch.stack([s["logits"] for s in samples]).double().requires_grad_()
        cot = torch.stack([s["finite_cotangent"] for s in samples]).double()
        same = reference_sink(z)
        grad = torch.autograd.grad((same * cot).sum(), z)[0]
        native = torch.stack([s["carry"] for s in samples])
        native_grad = torch.stack([s["execution_device_finite_vjp"] for s in samples])
        output_errors, vjp_errors = errors(native, same), errors(native_grad, grad)
        rows = []
        n = samples[0]["logits"].shape[0]
        for i, s in enumerate(samples):
            rows.append(
                {
                    "sublayer": s["sublayer"],
                    "indices": s["indices"].tolist(),
                    "finite_output": output_errors[i * n : (i + 1) * n],
                    "finite_vjp": vjp_errors[i * n : (i + 1) * n],
                    "row_residual_max": (s["carry"].sum(-1) - 1).abs().max().item(),
                    "column_residual_max": (s["carry"].sum(-2) - 1).abs().max().item(),
                    "minimum_carry_entry": s["carry"].min().item(),
                    "mean_energy": s["mean_energy"],
                    "difference_energy": s["difference_energy"],
                    "write_rms": s["write_rms"],
                    "write_l2_change": s["write_l2_change"],
                    "immediate_write_sum_change_max": s[
                        "immediate_write_sum_change_max"
                    ],
                    "effective_coefficient_perturbation_rms": s[
                        "effective_coefficient_perturbation_rms"
                    ],
                    "minimum_original_margin": s["margins"].min().item(),
                    "base_write": s["base_write"].tolist(),
                    "coded_write": s["coded_write"].tolist(),
                    "depth_code": s.get("depth_code", torch.zeros(4)).tolist(),
                    "strength": s.get("strength", 0),
                }
            )
            for name, values in (
                ("output", rows[-1]["finite_output"]),
                ("vjp", rows[-1]["finite_vjp"]),
            ):
                for k, value in enumerate(values):
                    if not value["pass"]:
                        record["finite_misses"].append(
                            {
                                "step": payload["step"],
                                "sublayer": s["sublayer"],
                                "matrix_index": k,
                                "token_index": s["indices"][k].item(),
                                "comparison": name,
                                **value,
                            }
                        )
        diagnostic = {"status": "unavailable"}
        try:
            zz = z.detach().requires_grad_()
            converged = limit(zz)
            cg = torch.autograd.grad((converged * cot).sum(), zz)[0]
            diagnostic = {
                "status": "complete",
                "solver": DoublyStochastic.method,
                "iterations": DoublyStochastic.count,
                "output": errors(native, converged),
                "vjp": errors(native_grad, cg),
            }
        except (RuntimeError, AssertionError) as exc:
            diagnostic["error"] = repr(exc)
        record["snapshots"].append(
            {
                "step": payload["step"],
                "payload": receipt,
                "sublayers": rows,
                "converged_diagnostic": diagnostic,
            }
        )
    record["finite_reference_pass"] = not record["finite_misses"]
    record["status"] = "complete"
    record["cpu_seconds"] = time.perf_counter() - before
    atomic(dest, record)
    print(
        json.dumps(
            {
                "job": jobid,
                "finite_misses": len(record["finite_misses"]),
                "cpu_seconds": record["cpu_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id", required=True)
    args = p.parse_args()
    actual(args.root, args.job_id)
