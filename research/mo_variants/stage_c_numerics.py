"""All prescribed actual CUDA routing samples against the finite20 oracle."""

import os
import argparse
import time
from pathlib import Path

import torch

from research.tinystories_depth.qualify import reference_sink
from .diagnostics import errors
from .stage_c_common import read, sha, write


def actual(root, jobid):
    root = Path(root)
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    assert not torch.cuda.is_initialized()
    r = root / "records"
    result = read(r / "runs" / jobid / "RESULT.json")
    assert result["status"] == "complete"
    dest = r / "numerics" / (jobid + ".json")
    assert not dest.exists()
    before = time.perf_counter()
    record = {
        "job_id": jobid,
        "snapshots": [],
        "finite_misses": [],
        "reference": "unchanged FP64 finite20 eps1e-6; relative<=1e-3 when norm>=1e-8, otherwise absolute<=1e-6",
        "infinite_projection_audit": "not performed",
    }
    for receipt in result["diagnostics"]:
        # Export remaps pod paths to the verified local checkpoint root.
        path = root / "checkpoints" / jobid / Path(receipt["path"]).name
        assert sha(path) == receipt["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload["samples"]
        assert len(samples) == 64
        z = torch.stack([s["logits"] for s in samples]).double().requires_grad_()
        cot = torch.stack([s["finite_cotangent"] for s in samples]).double()
        ref = reference_sink(z)
        vjp = torch.autograd.grad((ref * cot).sum(), z)[0]
        n = samples[0]["logits"].shape[0]
        eo = errors(torch.stack([s["carry"] for s in samples]), ref)
        ev = errors(
            torch.stack([s["execution_device_finite_vjp"] for s in samples]), vjp
        )
        rows = []
        for i, s in enumerate(samples):
            row = {
                "sublayer": s["sublayer"],
                "indices": s["indices"].tolist(),
                "finite_output": eo[i * n : (i + 1) * n],
                "finite_vjp": ev[i * n : (i + 1) * n],
                "row_residual_max": (s["carry"].sum(-1) - 1).abs().max().item(),
                "column_residual_max": (s["carry"].sum(-2) - 1).abs().max().item(),
                "minimum_carry_entry": s["carry"].min().item(),
                "mean_energy": s["mean_energy"],
                "difference_energy": s["difference_energy"],
                "write_rms": s["write_rms"],
                "write_l2_change": s["write_l2_change"],
                "minimum_original_margin": s["margins"].min().item(),
                "effective_coefficient_perturbation_rms": s[
                    "effective_coefficient_perturbation_rms"
                ],
                "base_write": s["base_write"].tolist(),
                "coded_write": s["coded_write"].tolist(),
                "depth_code": s.get("depth_code", torch.zeros(4)).tolist(),
            }
            for name in ("finite_output", "finite_vjp"):
                for k, val in enumerate(row[name]):
                    if not val["pass"]:
                        record["finite_misses"].append(
                            {
                                "step": payload["step"],
                                "sublayer": i,
                                "matrix_index": k,
                                "token_index": s["indices"][k].item(),
                                "comparison": name,
                                **val,
                            }
                        )
            rows.append(row)
        record["snapshots"].append(
            {"step": payload["step"], "payload": receipt, "sublayers": rows}
        )
    assert [x["step"] for x in record["snapshots"]] == [0, 50, 500, 1000, 2000, 4000]
    record.update(
        status="complete",
        finite_reference_pass=not record["finite_misses"],
        cpu_seconds=time.perf_counter() - before,
    )
    write(dest, record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id", required=True)
    a = p.parse_args()
    actual(a.root, a.job_id)
