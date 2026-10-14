"""Replay one retained miss without training or changing numerical policy."""

import os
import argparse
import time
from pathlib import Path

import torch

from research.tinystories_depth.model import mhc
from research.tinystories_depth.qualify import reference_sink
from .diagnostics import errors
from .stage_c_common import read, sha, write


def audit(payload_path, expected_sha, destination, device):
    assert sha(payload_path) == expected_sha
    assert not Path(destination).exists()
    started = time.perf_counter()
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = torch.load(payload_path, map_location="cpu", weights_only=False)
    s = p["samples"][17]
    assert s["sublayer"] == 17 and s["indices"][14].item() == 1910
    z = s["logits"].double().requires_grad_()
    cot = s["finite_cotangent"].double()
    ref = reference_sink(z)
    vref = torch.autograd.grad((ref * cot).sum(), z)[0]
    retained = errors(s["execution_device_finite_vjp"], vref)[14]
    assert not retained["pass"]
    assert abs(retained["relative_l2"] - 0.0015140281925691116) < 1e-10
    comparisons = {}
    tensors = {"reference_output": ref.detach(), "reference_vjp": vref}
    for name, dev, dtype, independent in (
        ("cpu_production_fp32", "cpu", torch.float32, False),
        ("cpu_production_fp64", "cpu", torch.float64, False),
        ("cpu_independent_fp32", "cpu", torch.float32, True),
        ("execution_production_fp32", device, torch.float32, False),
    ):
        x = s["logits"].to(dev, dtype).detach().requires_grad_()
        g = s["finite_cotangent"].to(dev, dtype)
        y = (
            reference_sink(x)
            if independent
            else mhc.sinkhorn_normalize_ref(x, repeat=20, eps=1e-6)
        )
        v = torch.autograd.grad((y * g).sum(), x)[0]
        comparisons[name] = {
            "output_error": errors(y.cpu(), ref)[14],
            "vjp_error": errors(v.cpu(), vref)[14],
            "output_bit_identical_to_hip": torch.equal(y.cpu(), s["carry"]),
            "vjp_bit_identical_to_hip": torch.equal(
                v.cpu(), s["execution_device_finite_vjp"]
            ),
        }
        tensors[name + "_output"] = y.detach().cpu()
        tensors[name + "_vjp"] = v.cpu()
    # A single saved 4x4 matrix is a disposable finite-difference fixture.
    x = z[14].detach()
    g = cot[14]
    direction = torch.randn(
        (4, 4), generator=torch.Generator().manual_seed(76390), dtype=torch.float64
    )
    direction /= direction.norm()
    exact = (vref[14] * direction).sum().item()
    estimates = []
    for eps in (1e-2, 1e-3, 1e-4):
        plus = (reference_sink(x + eps * direction) * g).sum().item()
        minus = (reference_sink(x - eps * direction) * g).sum().item()
        numeric = (plus - minus) / (2 * eps)
        estimates.append(
            {
                "eps": eps,
                "numeric": numeric,
                "absolute_error": abs(numeric - exact),
                "pass": abs(numeric - exact) <= 1e-10 + 1e-3 * abs(exact),
            }
        )
    assert sum(x["pass"] for x in estimates) >= 2
    fp64 = comparisons["cpu_production_fp64"]
    assert fp64["vjp_error"]["relative_l2"] < 1e-7
    if device == "cuda" and torch.version.hip:
        assert comparisons["execution_production_fp32"]["output_bit_identical_to_hip"]
        assert comparisons["execution_production_fp32"]["vjp_bit_identical_to_hip"]
    tpath = Path(destination).with_suffix(".pt")
    torch.save({"saved_sample": s, "replay": tensors}, tpath)
    write(
        destination,
        {
            "status": "complete",
            "job_id": "B-extension-V01-76313",
            "step": 2000,
            "payload_sha256": expected_sha,
            "replay_tensors": {"path": str(tpath), "sha256": sha(tpath)},
            "device": device,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "retained_miss": retained,
            "comparisons": comparisons,
            "finite_difference": {
                "direction_seed": 76390,
                "autograd": exact,
                "estimates": estimates,
            },
            "disposition": "The same finite function agrees in FP64. FP32 small-gradient error remains a real tolerance failure; no incorrect expression or reference was found in this bounded fixture.",
            "implementation_invalidated": False,
            "qualification_resolved": False,
            "production_changed": False,
            "oracle_changed": False,
            "thresholds_changed": False,
            "infinite_projection_audit": "not performed",
            "seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--payload", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    audit(a.payload, a.sha256, a.destination, a.device)
