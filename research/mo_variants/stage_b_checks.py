"""Bounded Stage B resume proofs and replay of retained high-LR misses."""

import argparse
import gc
import math
import os
from importlib.metadata import distributions
from pathlib import Path
import platform
import time

import torch

from research.tinystories_depth.model import digest, loss, mhc
from research.tinystories_depth.qualify import reference_sink
from research.tinystories_depth.train import state_hash
from .preflight import clients
from .coordinator import atomic, read, sha
from .diagnostics import errors
from .job import checkpoint
from .model import Model
from .stage_b_prepare import ROWS, verify_source


def checks(root):
    root = Path(root)
    r = root / "records"
    authorization = read(r / "STAGE-B-RESOURCE-DECISION.json")
    assert authorization["scope"] == "B" and authorization["status"] == "approved"
    verify_source(root)

    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clients()
    started = time.perf_counter()
    atomic(
        r / "STAGE-B-PREFLIGHT.json",
        {
            "device": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "kernel": platform.release(),
            "module_paths": {"torch": torch.__file__, "mhc": mhc.__file__},
            "package_pins": sorted(
                f"{d.metadata['Name']}=={d.version}" for d in distributions()
            ),
            "confirmation_arrays_evaluated": False,
        },
    )
    assert torch.__version__ == "2.9.1+rocm6.4"
    resume = []
    cp = root / "checkpoints"
    cp.mkdir(exist_ok=True)
    for variant in ROWS:
        generator = torch.Generator().manual_seed(76390)
        data = [
            (
                torch.randint(0, 128, (2, 8), generator=generator).cuda(),
                torch.randint(0, 128, (2, 8), generator=generator).cuda(),
            )
            for _ in range(5)
        ]
        reference = Model(variant, 4, 76390, vocab=128).cuda()
        partial = Model(variant, 4, 76390, vocab=128).cuda()

        def optimizer(m):
            return torch.optim.AdamW(
                m.parameters(), lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0
            )

        a, b = optimizer(reference), optimizer(partial)
        torch.manual_seed(76390)
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

        def advance(m, o, first, last):
            for step in range(first, last + 1):
                m.step = step
                for group in o.param_groups:
                    group["lr"] = 0.003 * min(step / 100, 1)
                o.zero_grad(set_to_none=True)
                x, y = data[step - 98]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    value = loss(m(x), y)
                value.backward()
                o.step()
                # Probe both RNG streams so save/restore is tested independently
                # of whether the production architecture currently uses dropout.
                torch.rand(2)
                torch.rand(2, device="cuda")

        advance(reference, a, 98, 102)
        end_cpu, end_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        advance(partial, b, 98, 100)
        receipt = checkpoint(
            partial,
            b,
            100,
            {"fixture": True, "variant": variant},
            cp / f"resume-{variant}.pt",
        )
        saved = torch.load(receipt["path"], map_location="cpu", weights_only=False)
        resumed = Model(variant, 4, 76390, vocab=128).cuda()
        c = optimizer(resumed)
        resumed.load_state_dict(saved["model"], strict=True)
        c.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_gpu"])
        advance(resumed, c, 101, 102)
        assert state_hash(reference.state_dict()) == state_hash(resumed.state_dict())
        assert state_hash(a.state_dict()) == state_hash(c.state_dict())
        assert torch.equal(torch.get_rng_state(), end_cpu)
        assert torch.equal(torch.cuda.get_rng_state(), end_gpu)
        assert (
            reference.code_map == resumed.code_map
            and reference.mapping_lookup == resumed.mapping_lookup
        )
        resume.append(
            {
                "variant": variant,
                "parameter_state_bit_identical": True,
                "optimizer_state_bit_identical": True,
                "cpu_gpu_rng_bit_identical": True,
                "global_steps": [98, 102],
                "interruption_step": 100,
                "scheduler_warmup_boundary_preserved": True,
                "data_cursor_preserved": True,
                "code_map_preserved": True,
                "fixture_checkpoint": receipt,
                "prior_applicable_fixture": "RESUME-FIXTURE-cuda.json",
            }
        )
        del reference, partial, resumed, a, b, c, saved, data
        gc.collect()
        torch.cuda.empty_cache()
    atomic(
        r / "STAGE-B-RESUME-FIXTURE.json",
        {
            "status": "complete",
            "scope": "eight tiny-vocabulary operator/control fixtures; official states untouched",
            "rows": resume,
            "seconds": time.perf_counter() - started,
        },
    )
    replay = []
    for variant in ("V19", "V20"):
        jobid = f"A-pilot-{variant}-76301-lr0.01"
        historical = read(r / "numerics" / (jobid + ".json"))
        receipt = next(
            x["payload"] for x in historical["snapshots"] if x["step"] == 500
        )
        assert sha(receipt["path"]) == receipt["sha256"]
        payload = torch.load(receipt["path"], map_location="cpu", weights_only=False)
        z64 = (
            torch.stack([s["logits"] for s in payload["samples"]])
            .double()
            .requires_grad_()
        )
        cot64 = torch.stack(
            [s["finite_cotangent"] for s in payload["samples"]]
        ).double()
        ref = reference_sink(z64)
        ref_vjp = torch.autograd.grad((ref * cot64).sum(), z64)[0]
        for miss in historical["finite_misses"]:
            assert miss["step"] == 500 and miss["comparison"] == "vjp"
            i, k = miss["sublayer"], miss["matrix_index"]
            s = payload["samples"][i]
            z = s["logits"].cuda().requires_grad_()
            cot = s["finite_cotangent"].cuda()
            native = mhc.sinkhorn_normalize_ref(z, repeat=20, eps=1e-6)
            vjp = torch.autograd.grad((native * cot).sum(), z)[0]
            zz = s["logits"].requires_grad_()
            cpu_native = mhc.sinkhorn_normalize_ref(zz, repeat=20, eps=1e-6)
            cpu_vjp = torch.autograd.grad(
                (cpu_native * s["finite_cotangent"]).sum(), zz
            )[0]
            recorded_error = errors(s["execution_device_finite_vjp"], ref_vjp[i])[k]
            assert not recorded_error["pass"]
            assert math.isclose(
                recorded_error["absolute_l2"],
                miss["absolute_l2"],
                rel_tol=1e-6,
                abs_tol=1e-15,
            )
            record = {
                "job_id": jobid,
                "historical_miss": miss,
                "payload": receipt,
                "stored_hip_vjp_error": recorded_error,
                "replayed_hip_output_error": errors(native.cpu(), ref[i])[k],
                "replayed_hip_vjp_error": errors(vjp.cpu(), ref_vjp[i])[k],
                "native_cpu_fp32_vjp_error": errors(cpu_vjp, ref_vjp[i])[k],
                "hip_output_bit_identical_to_stored": torch.equal(
                    native.cpu(), s["carry"]
                ),
                "hip_vjp_bit_identical_to_stored": torch.equal(
                    vjp.cpu(), s["execution_device_finite_vjp"]
                ),
                "logits": s["logits"][k].tolist(),
                "cotangent": s["finite_cotangent"][k].tolist(),
                "stored_hip_output": s["carry"][k].tolist(),
                "replayed_hip_output": native[k].detach().cpu().tolist(),
                "stored_hip_vjp": s["execution_device_finite_vjp"][k].tolist(),
                "replayed_hip_vjp": vjp[k].detach().cpu().tolist(),
                "fp64_finite20_reference_output": ref[i, k].detach().tolist(),
                "fp64_finite20_reference_vjp": ref_vjp[i, k].tolist(),
            }
            assert (
                record["hip_output_bit_identical_to_stored"]
                and record["hip_vjp_bit_identical_to_stored"]
            ), "replay implementation/runtime mismatch; hold affected row"
            replay.append(record)
    assert len(replay) == 6
    atomic(
        r / "NUMERICAL-REPLAY.json",
        {
            "status": "complete",
            "diagnostic_source_sha256": sha(
                root / "code/research/mo_variants/stage_b_checks.py"
            ),
            "production_changed": False,
            "oracle_changed": False,
            "thresholds_changed": False,
            "historical_misses_resolved": False,
            "selected_recipe_blocker_detected": False,
            "scope": "saved unused .01-LR payloads replayed without retraining; native CPU comparison is diagnostic only",
            "samples": replay,
        },
    )
    clients()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    checks(p.parse_args().root)
