# SPDX-License-Identifier: MIT
"""Synthetic smoke and disposable training/validation timing, never final targets."""

import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from research.layerrope_beta.model import Model
from research.layerrope_beta.time_budget.state import runtime, synchronize
from research.layerrope_beta.time_budget.train import train
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.kv_cache import evaluate, parity
from .diagnostics import measure
from .protocol import freeze


def smoke(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "research/layerrope_beta/tests/test_qknorm.py",
        ],
        env={**os.environ, "KNLP_QKNORM_DEVICE": "cpu"},
        capture_output=True,
        text=True,
    )
    (root / "tests.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("QKNorm smoke qualification failed")
    p = freeze()
    data = root / "data"
    data.mkdir()
    x = np.random.default_rng(p["probe_seed"]).integers(0, 97, 513, dtype=np.uint16)
    for n in ("train", "validation"):
        np.save(data / (n + ".npy"), x)
    write(
        data / "TRAINING.json",
        dict(files={n: sha(data / n) for n in ("train.npy", "validation.npy")}),
    )
    # Exercise the real shared trainer, endpoint hook and independent resume test.
    p["equal_token_updates"] = 2
    for arm in p["arms"]:
        job = dict(
            **arm,
            seed=p["probe_seed"],
            stage="qualification",
            depth=2,
            vocab=97,
            window=16,
            batch_size=2,
        )
        r = train(
            data,
            job,
            root / arm["name"],
            device="cpu",
            smoke=True,
            contract=p,
            diagnostic=measure,
            stop_after_updates=3,
        )
        if r["history"]["update_landmarks"][0]["step"] != 2:
            raise RuntimeError("Update endpoint not saved")
        model = Model(
            job["arm"], 2, job["seed"], beta=0.0, vocab=97, qknorm=job["qknorm"]
        ).eval()
        state = torch.load(
            r["last_checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(state["model"])
        values = {
            c["name"]: evaluate(model, x, c, "cpu", window=16) for c in p["cache_cases"]
        }
        write(root / (arm["name"] + "-CACHE.json"), values)
    write(
        root / "REPORT.json",
        dict(
            status="complete",
            scientific_results=False,
            arms=8,
            policies=10,
            tests=result.stdout,
        ),
    )


def qualify(card_path, resource_path, root):
    from .run import resource_gate

    card, resource = resource_gate(card_path, resource_path, "qualify")
    if resource["gpu_count"] != 1:
        raise ValueError(
            "Qualify each single-GPU runtime separately before multi-GPU use"
        )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "research/layerrope_beta/tests/test_qknorm.py",
        ],
        env={**os.environ, "KNLP_QKNORM_DEVICE": "cuda"},
        capture_output=True,
        text=True,
    )
    (root / "operator-tests.log").write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise RuntimeError("GPU operator qualification failed")
    measurements = []
    p = card["protocol"]
    tokens = np.load(Path(card["data"]) / "training/validation.npy", mmap_mode="r")
    for arm in p["arms"]:
        if time.monotonic() - began >= resource["max_seconds"]:
            raise TimeoutError("Qualification time cap")
        torch.cuda.reset_peak_memory_stats()
        job = dict(
            **arm,
            seed=p["probe_seed"],
            stage="qualification",
            depth=32,
            vocab=50257,
            window=256,
            batch_size=8,
            max_process_seconds=600,
        )
        r = train(
            Path(card["data"]) / "training",
            job,
            root / arm["name"],
            contract=p,
            diagnostic=measure,
            stop_after_updates=20,
        )
        model = (
            Model(arm["arm"], 32, p["probe_seed"], beta=0.0, qknorm=arm["qknorm"])
            .to("cuda")
            .eval()
        )
        state = torch.load(
            r["last_checkpoint"]["path"], map_location="cuda", weights_only=False
        )
        model.load_state_dict(state["model"])
        ids = torch.as_tensor(
            np.array(tokens[: 8 * 256], dtype=np.int64).reshape(8, 256), device="cuda"
        )
        parity_result = parity(model, ids, "cuda")
        with torch.no_grad(), torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(ids)
            synchronize("cuda")
        backend = sorted(
            {x.key for x in profiler.key_averages() if "scaled_dot_product" in x.key}
        )
        cells = []
        for c in p["cache_cases"]:
            v = evaluate(model, tokens, c, "cuda")
            cells.append(
                dict(
                    policy=c["name"],
                    seconds=v["seconds"],
                    predicted_tokens=v["predicted_tokens"],
                )
            )
        measurements.append(
            dict(
                arm=arm,
                initial_parameter_hashes=r["initial_parameter_hashes"],
                common_initial_weights=r["common_initial_weights"],
                training_seconds=r["training_seconds"],
                steps=r["completed_steps"],
                median_update_seconds=float(
                    np.median(
                        [u["update_seconds"] for u in r["history"]["updates"][-10:]]
                    )
                ),
                checkpoint_bytes=Path(r["last_checkpoint"]["path"]).stat().st_size,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                backend_operations=backend,
                parity=parity_result,
                evaluation=cells,
            )
        )
        write(root / "PROGRESS.json", measurements)
        del model, state
        torch.cuda.empty_cache()
    for arm in ("R", "L", "M", "ML"):
        off, on = [
            next(
                m
                for m in measurements
                if m["arm"]["arm"] == arm and m["arm"]["qknorm"] == q
            )
            for q in (False, True)
        ]
        if off["initial_parameter_hashes"] != {
            n: v
            for n, v in on["initial_parameter_hashes"].items()
            if not n.endswith("qknorm_gain")
        }:
            raise RuntimeError("Full-depth initial weights are not paired")
    if not all(
        m["common_initial_weights"] == measurements[0]["common_initial_weights"]
        for m in measurements
    ):
        raise RuntimeError("Full-depth architecture weights are not paired")
    projected_evaluation_seconds = sum(
        c["seconds"] * (p["final_predicted_tokens"] / c["predicted_tokens"]) * 6
        for m in measurements
        for c in m["evaluation"]
    )
    training_seconds = 24 * p["primary_seconds"]
    checkpoint_bytes = sum(m["checkpoint_bytes"] * 12 for m in measurements)
    # Uniform validation at nine time landmarks plus the update endpoint.
    offline_validation = sum(
        next(c["seconds"] for c in m["evaluation"] if c["policy"] == "clean") * 30
        for m in measurements
    )
    projection = dict(
        training_seconds=training_seconds,
        evaluation_seconds=projected_evaluation_seconds,
        offline_validation_seconds=offline_validation,
        scientific_checkpoint_bytes_upper=checkpoint_bytes,
        export_reserve_seconds=checkpoint_bytes / 7_000_000 + 900,
        contingency_fraction=0.25,
    )
    write(
        root / "QUALIFICATION.json",
        dict(
            status="qualified",
            card_sha256=sha(card_path),
            runtime=runtime("cuda"),
            measurements=measurements,
            projection=projection,
            seconds=time.monotonic() - began,
            backend_policy=dict(
                flash=torch.backends.cuda.flash_sdp_enabled(),
                math=torch.backends.cuda.math_sdp_enabled(),
                memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            ),
            precision="FP32 master/norm/gains/routing/CE, BF16 matrix operations; no TF32",
            final_targets_opened=False,
        ),
    )
    return read(root / "QUALIFICATION.json")
