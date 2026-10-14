"""Disposable controls, exact resume and H100 cost qualification."""

import argparse
import gc
import os
import platform
import statistics
import subprocess
import time
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import torch

from research.tinystories_depth.model import digest, loss
from research.tinystories_depth.probe import batch, evaluate, finite
from research.tinystories_depth.qualify import run as inherited_fidelity
from research.tinystories_depth.train import state_hash
from .job import checkpoint, snapshot
from .model import Model
from .stage_c_common import MODEL_SHA, ROWS, read, sha, write
from .stage_c_quantize import groupwise, gpu_fds


def fixtures(root, device, cost=False):
    root = Path(root)
    target = root / "tests" / ("qualification-" + device)
    target.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(root / "code/research/mo_variants/model.py") == MODEL_SHA
    if device == "cpu":
        assert not torch.cuda.is_initialized()
        assert not gpu_fds()
    else:
        assert torch.__version__ == "2.9.1+cu128" and torch.version.hip is None
        assert "H100" in torch.cuda.get_device_name(0)
    inherited_fidelity(target, device)
    record = {
        "status": "running",
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": sorted(
            f"{d.metadata['Name']}=={d.version}" for d in distributions()
        ),
        "scope": "disposable fixtures and discovery-array timing, no confirmation outcome",
        "rows": [],
        "quantizer_checks": [],
        "cost_cells": [],
    }
    for row, variant, mapping in ROWS:
        model = Model(variant, 4, 76390, vocab=128, mapping=mapping).to(device)
        partial = Model(variant, 4, 76390, vocab=128, mapping=mapping).to(device)

        def opt(m):
            return torch.optim.AdamW(
                m.parameters(), lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0
            )

        a, b = opt(model), opt(partial)
        g = torch.Generator().manual_seed(76390)
        batches = [
            (
                torch.randint(0, 128, (2, 8), generator=g).to(device),
                torch.randint(0, 128, (2, 8), generator=g).to(device),
            )
            for _ in range(5)
        ]
        torch.manual_seed(76390)
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state() if device == "cuda" else None

        def advance(m, o, first, last):
            for step in range(first, last + 1):
                m.step = step
                for group in o.param_groups:
                    group["lr"] = 0.003 * min(step / 100, 1)
                o.zero_grad(set_to_none=True)
                x, y = batches[step - 98]
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    value = loss(m(x), y)
                value.backward()
                finite(m)
                o.step()
                finite(m, o)
                torch.rand(2)
                if device == "cuda":
                    torch.rand(2, device="cuda")

        advance(model, a, 98, 102)
        end_cpu = torch.get_rng_state()
        end_gpu = torch.cuda.get_rng_state() if device == "cuda" else None
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng)
        advance(partial, b, 98, 100)
        saved = {
            "model": partial.state_dict(),
            "optimizer": b.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
        }
        path = target / (row + "-resume.pt")
        torch.save(saved, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        resumed = Model(variant, 4, 76390, vocab=128, mapping=mapping).to(device)
        c = opt(resumed)
        resumed.load_state_dict(saved["model"], strict=True)
        c.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state(saved["gpu_rng"])
        advance(resumed, c, 101, 102)
        assert state_hash(model.state_dict()) == state_hash(resumed.state_dict())
        assert state_hash(a.state_dict()) == state_hash(c.state_dict())
        assert torch.equal(torch.get_rng_state(), end_cpu)
        if device == "cuda":
            assert torch.equal(torch.cuda.get_rng_state(), end_gpu)
        assert model.code_map == resumed.code_map
        record["rows"].append(
            {
                "row": row,
                "mapping": mapping,
                "parameter_and_optimizer_bit_identical": True,
                "cpu_gpu_rng_bit_identical": True,
                "warmup_boundary_steps": [98, 102],
                "code_maps_preserved": True,
                "checkpoint_sha256": sha(path),
            }
        )
        del model, partial, resumed, a, b, c, saved, batches
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    w = torch.arange(2 * 131, dtype=torch.float32).reshape(2, 131) - 128
    for bits in (4, 3):
        q, s = groupwise(w, bits)
        assert s.shape == (2, 2, 1) and q.shape == w.shape
        assert torch.equal(
            s[:, -1, 0], w[:, 128:].abs().amax(-1) / (2 ** (bits - 1) - 1)
        )
        z, sz = groupwise(torch.zeros(2, 131), bits)
        assert not z.any() and not sz.any()
        record["quantizer_checks"].append(
            {
                "bits": bits,
                "partial_group_scale_ignores_padding": True,
                "zero_rows_preserved": True,
            }
        )
    if cost:
        assert device == "cuda"
        record["gpu_inventory"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
        )
        record["cgroup_cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        train = np.load(root / "data/train.npy", mmap_mode="r")
        order = np.load(root / "data/order-76390.npy")
        curve = np.load(root / "data/curve.npy", mmap_mode="r")
        x, y = batch(train, order, 0, "cuda")
        for row in ("V00", "V01", "V30"):
            torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            model = Model(row, 32, 76390).cuda()
            o = opt(model)
            torch.cuda.synchronize()
            setup = time.perf_counter() - before
            times = []
            for step in range(1, 26):
                torch.cuda.synchronize()
                before = time.perf_counter()
                model.step = step
                o.zero_grad(set_to_none=True)
                for group in o.param_groups:
                    group["lr"] = 0.003 * min(step / 100, 1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    value = loss(model(x), y)
                value.backward()
                finite(model)
                o.step()
                finite(model, o)
                torch.cuda.synchronize()
                if step > 5:
                    times.append(time.perf_counter() - before)
            before = time.perf_counter()
            evaluate(model, curve, "cuda")
            evaluation = time.perf_counter() - before
            before = time.perf_counter()
            snapshot(model, x, 25, target / (row + "-routing.pt"), {"disposable": True})
            routing = time.perf_counter() - before
            before = time.perf_counter()
            receipt = checkpoint(
                model, o, 25, {"disposable": True}, target / (row + "-cost.pt")
            )
            saving = time.perf_counter() - before
            record["cost_cells"].append(
                {
                    "row": row,
                    "setup_seconds": setup,
                    "step_mean_seconds": statistics.mean(times),
                    "step_samples_seconds": times,
                    "curve_eval_seconds": evaluation,
                    "routing_snapshot_seconds": routing,
                    "checkpoint_seconds": saving,
                    "checkpoint_bytes": receipt["bytes"],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            del model, o
            gc.collect()
            torch.cuda.empty_cache()
    record["status"] = "complete"
    write(root / "records" / ("STAGE-C-QUALIFICATION-" + device + ".json"), record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--cost", action="store_true")
    a = p.parse_args()
    fixtures(a.root, a.device, a.cost)
