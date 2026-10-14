"""Check only the new endpoint against the unchanged finite20 reference."""

import os
import argparse
from pathlib import Path
import time

import torch

from research.tinystories_depth.qualify import reference_sink
from .coordinator import atomic, read, sha
from .diagnostics import errors


def actual(root, jobid):
    r = Path(root) / "records"
    result = read(r / "runs" / jobid / "RESULT.json")
    assert result["status"] == "complete" and result["completed_step"] == 2000
    dest = r / "numerics" / (jobid + ".json")
    assert not dest.exists()
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    before = time.perf_counter()
    record = {
        "job_id": jobid,
        "snapshots": [],
        "finite_misses": [],
        "reference": "unchanged independent FP64 finite20 eps1e-6; relative<=1e-3 for norm>=1e-8, else absolute<=1e-6",
        "prior_diagnostics_reused": result["prior_diagnostics_reused"],
        "infinite_projection_audit": "not repeated",
    }
    assert len(result["diagnostics"]) == 1
    for receipt in result["diagnostics"]:
        assert receipt["step"] == 2000 and sha(receipt["path"]) == receipt["sha256"]
        payload = torch.load(receipt["path"], map_location="cpu", weights_only=False)
        samples = payload["samples"]
        assert len(samples) == 64
        z = torch.stack([s["logits"] for s in samples]).double().requires_grad_()
        cot = torch.stack([s["finite_cotangent"] for s in samples]).double()
        output = reference_sink(z)
        vjp = torch.autograd.grad((output * cot).sum(), z)[0]
        n = samples[0]["logits"].shape[0]
        output_errors = errors(torch.stack([s["carry"] for s in samples]), output)
        vjp_errors = errors(
            torch.stack([s["execution_device_finite_vjp"] for s in samples]), vjp
        )
        sublayers = []
        for i, s in enumerate(samples):
            row = {
                "sublayer": s["sublayer"],
                "indices": s["indices"].tolist(),
                "finite_output": output_errors[i * n : (i + 1) * n],
                "finite_vjp": vjp_errors[i * n : (i + 1) * n],
                "row_residual_max": (s["carry"].sum(-1) - 1).abs().max().item(),
                "column_residual_max": (s["carry"].sum(-2) - 1).abs().max().item(),
                "minimum_carry_entry": s["carry"].min().item(),
                "mean_energy": s["mean_energy"],
                "difference_energy": s["difference_energy"],
                "base_write": s["base_write"].tolist(),
                "coded_write": s["coded_write"].tolist(),
                "depth_code": s.get("depth_code", torch.zeros(4)).tolist(),
                "strength": s.get("strength", 0),
            }
            for name, values in (
                ("output", row["finite_output"]),
                ("vjp", row["finite_vjp"]),
            ):
                for k, value in enumerate(values):
                    if not value["pass"]:
                        record["finite_misses"].append(
                            {
                                "step": 2000,
                                "sublayer": s["sublayer"],
                                "matrix_index": k,
                                "token_index": s["indices"][k].item(),
                                "comparison": name,
                                **value,
                            }
                        )
            sublayers.append(row)
        record["snapshots"].append(
            {"step": 2000, "payload": receipt, "sublayers": sublayers}
        )
    record.update(
        status="complete",
        finite_reference_pass=not record["finite_misses"],
        cpu_seconds=time.perf_counter() - before,
    )
    atomic(dest, record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id", required=True)
    args = p.parse_args()
    actual(args.root, args.job_id)
