"""Freeze the declared shortlist and verify every exact continuation state."""

import os
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import statistics as st
import time

import numpy as np
import torch

from research.tinystories_depth.model import digest
from research.tinystories_depth.train import state_hash
from .coordinator import atomic, read, sha
from .model import Model

ROWS = ["V00", "V01", "V02", "V03", "V09", "V19", "V20", "V30"]
SEEDS = [76311, 76312, 76313]
GUIDANCE = "depth-code-continuation"


def verify_source(root):
    r = Path(root) / "records"
    original = read(r / "SOURCE.json")
    extension = read(r / "STAGE-B-SOURCE.json")
    assert sha(r / "SOURCE.json") == extension["stage_a_source_manifest_sha256"]
    for name, expected in {
        **original["implemented_files"],
        **extension["files"],
    }.items():
        assert sha(Path(root) / "code" / name) == expected, name


def verify_checkpoint(root, job, previous, saved):
    r = Path(root) / "records"
    receipt = previous["checkpoint"]
    assert previous["status"] == "complete" and saved["step"] == 1000
    assert receipt["step"] == 1000 and sha(receipt["path"]) == receipt["sha256"]
    prior = saved["metadata"]["job"]
    assert all(prior[k] == job[k] for k in ("variant", "depth", "seed"))
    assert prior["stage"] == "A-discovery"
    lr = read(r / "SELECTED-LRS.json")[job["variant"]]["lr"]
    assert lr == 0.003 and saved["metadata"]["selected_lr"] == lr
    assert saved["metadata"]["selected_lrs_sha256"] == sha(r / "SELECTED-LRS.json")
    assert saved["metadata"]["data_order_sha256"] == sha(
        Path(root) / f"data/order-{job['seed']}.npy"
    )
    assert saved["metadata"]["curve_sha256"] == sha(Path(root) / "data/curve.npy")
    assert saved["metadata"]["source"] == read(r / "SOURCE.json")
    for key, expected in (
        ("model", "parameter_sha256"),
        ("optimizer", "optimizer_sha256"),
    ):
        assert state_hash(saved[key]) == receipt[expected]
    for key in ("rng_cpu", "rng_gpu"):
        assert digest(saved[key]) == receipt[key + "_sha256"]
    assert all(
        torch.isfinite(v).all()
        for v in saved["model"].values()
        if v.is_floating_point()
    )
    for value in saved["optimizer"]["state"].values():
        assert all(
            torch.isfinite(v).all() for v in value.values() if torch.is_tensor(v)
        )
        assert float(value["step"]) == 1000
    assert all(
        g["lr"] == lr
        and tuple(g["betas"]) == (0.9, 0.95)
        and g["eps"] == 1e-8
        and g["weight_decay"] == 0
        for g in saved["optimizer"]["param_groups"]
    )


def prepare(root):
    root = Path(root)
    r = root / "records"
    began = time.perf_counter()
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    verify_source(root)
    assert not (r / "STAGE-B-SELECTION.json").exists()
    source = read(r / "STAGE-A-RESULTS.json")
    ranks = source["rankings"]
    eps = {(e["variant"], e["seed"]): e for e in source["endpoints"]}
    scored = []
    for row in ranks:
        regrets = [
            eps[row["variant"], s]["gpu_clean_endpoint_ce"]
            - eps["V00", s]["gpu_clean_endpoint_ce"]
            for s in SEEDS
        ]
        scored.append(
            {
                "variant": row["variant"],
                "primary_mean": row["primary_ce500_1000"]["mean"],
                "primary_worst_seed": row["primary_ce500_1000"]["max"],
                "int4_mean": row["int4_ce"]["mean"],
                "int4_worst_seed": row["int4_ce"]["max"],
                "mean_clean_regret_vs_M": st.mean(regrets),
                "worst_clean_regret_vs_M": max(regrets),
                "quantization_eligible": st.mean(regrets) <= 0.02
                and max(regrets) <= 0.05,
                "historical_finite_samples_pass": row["qualified_finite_samples"],
            }
        )
    clean = sorted(
        scored, key=lambda x: (x["primary_mean"], x["primary_worst_seed"], x["variant"])
    )[:4]
    quant = sorted(
        [x for x in scored if x["quantization_eligible"]],
        key=lambda x: (x["int4_mean"], x["int4_worst_seed"], x["variant"]),
    )[:4]
    selected = sorted(
        {x["variant"] for x in clean + quant} | {"V00", "V01", "V02", "V03"}
    )
    assert selected == ROWS, selected
    registry = read(r / "JOBS.json")["jobs"]
    jobs = sorted(
        [j for j in registry if j["stage"] == "B" and j["variant"] in selected],
        key=lambda j: (j["seed"], j["variant"]),
    )
    assert len(jobs) == 24
    data = read(r / "DATA.json")
    for name in ("train.npy", *(f"order-{s}.npy" for s in SEEDS)):
        assert sha(root / "data" / name) == data["training_and_order_files"][name]
    assert sha(root / "data/curve.npy") == data["verified_inherited_files"]["curve.npy"]
    orders = []
    for seed in SEEDS:
        order = np.load(root / f"data/order-{seed}.npy")
        assert len(order) >= 16000
        orders.append(
            {
                "seed": seed,
                "sha256": sha(root / f"data/order-{seed}.npy"),
                "windows": len(order),
                "consumed_prefix_windows": 8000,
                "prefix_sha256": digest(torch.from_numpy(order[:8000])),
                "regenerated": False,
            }
        )
    receipts = []
    for job in jobs:
        previous = read(r / "runs" / job["resume_from"] / "RESULT.json")
        receipt = previous["checkpoint"]
        saved = torch.load(receipt["path"], map_location="cpu", weights_only=False)
        verify_checkpoint(root, job, previous, saved)
        model = Model(job["variant"], 32, job["seed"])
        assert model.code_map == previous["identities"]["code_map"]
        assert model.mapping_lookup == previous["identities"]["mapping_lookup"]
        initial_buffers = dict(model.named_buffers())
        assert all(
            torch.equal(value, saved["model"][name])
            for name, value in initial_buffers.items()
            if name in saved["model"]
        )
        model.load_state_dict(saved["model"], strict=True)
        assert state_hash(model.state_dict()) == receipt["parameter_sha256"]
        receipts.append(
            {
                "job_id": job["id"],
                "parent_job": job["resume_from"],
                "checkpoint": receipt,
                "state_finite": True,
                "optimizer_steps": 1000,
                "global_data_cursor_windows": 8000,
                "model_optimizer_rng_hashes_verified": True,
                "row_buffers_code_map_verified": True,
                "nonpersistent_buffers_rebuilt_from_unchanged_model": [
                    name for name in initial_buffers if name not in saved["model"]
                ],
                "selected_lr": 0.003,
                "selected_lrs_sha256": sha(r / "SELECTED-LRS.json"),
            }
        )
        del model, saved
    scope = []
    for variant in selected:
        stages = []
        for job in registry:
            if job["variant"] != variant or not job["stage"].startswith("A-"):
                continue
            n = read(r / "numerics" / (job["id"] + ".json"))
            stages.append(
                {
                    "job_id": job["id"],
                    "stage": job["stage"],
                    "seed": job["seed"],
                    "lr": job["lr"],
                    "selected_recipe": job["stage"] == "A-discovery"
                    or job.get("lr") == 0.003,
                    "finite_reference_pass": n["finite_reference_pass"],
                    "misses": n["finite_misses"],
                    "record_sha256": sha(r / "numerics" / (job["id"] + ".json")),
                }
            )
        fixtures = {
            d: {
                "source_sha256": sha(
                    r / f"variant-fixtures-final/VARIANT-FIDELITY-{d}.json"
                ),
                "full_network": read(
                    r / f"variant-fixtures-final/VARIANT-FIDELITY-{d}.json"
                )["full_network"][variant],
            }
            for d in ("cpu", "cuda")
        }
        assert all(x["finite_reference_pass"] for x in stages if x["selected_recipe"])
        scope.append(
            {
                "variant": variant,
                "fixtures": fixtures,
                "stages": stages,
                "historical_row_flag": all(x["finite_reference_pass"] for x in stages),
                "selected_recipe_sampled_pass": True,
                "qualification": (
                    "exploratory continuation; unresolved historical misses still block qualified nomination"
                    if variant in ("V19", "V20")
                    else "recorded samples pass"
                ),
            }
        )
    atomic(
        r / "STAGE-B-SELECTION.json",
        {
            "guidance_commit": GUIDANCE,
            "reviewed_results_commit": "discovery",
            "rule": read(r / "PLAN.json")["conditional_stages"]["B"],
            "ranked_source_values": scored,
            "clean_nominees": [x["variant"] for x in clean],
            "quantization_nominees": [x["variant"] for x in quant],
            "variants": selected,
            "jobs": jobs,
            "source_sha256": sha(r / "SOURCE.json"),
            "data_sha256": sha(r / "DATA.json"),
            "results_sha256": sha(r / "STAGE-A-RESULTS.json"),
            "numerical_caveats": "V19/V20 unused .01 pilots have six historical VJP misses; selected .003 samples pass; no threshold waiver.",
        },
    )
    atomic(
        r / "NUMERICAL-SCOPE.json",
        {
            "rows": scope,
            "thresholds_unchanged": True,
            "infinite_projection_audit": "not repeated",
        },
    )
    atomic(
        r / "STAGE-B-RESUME-INPUTS.json",
        {
            "status": "verified",
            "checkpoints": receipts,
            "orders": orders,
            "training_sha256": sha(root / "data/train.npy"),
            "discovery_array_sha256": sha(root / "data/curve.npy"),
            "confirmation_arrays_evaluated": False,
        },
    )
    durations = [
        read(r / "runs" / j["resume_from"] / "RESULT.json")["end_to_end_seconds"]
        for j in jobs
    ]
    atomic(
        r / "STAGE-B-PLAN.json",
        {
            "guidance_commit": GUIDANCE,
            "stage": "B",
            "job_count": 24,
            "ordering": "seed-major, numeric variant",
            "global_steps": [1000, 2000],
            "additional_optimizer_steps": 24000,
            "primary": "mean CE at 1000,1100,...,2000 inclusive; separate endpoint",
            "quantization": read(r / "PLAN.json")["quantization"],
            "targets": read(r / "PLAN.json")["decisions"]["convergence"],
            "matched_elapsed_budgets_seconds": [180, 300, 450, 600],
            "diagnostics": "new endpoint finite20 only; historical snapshots reused",
            "forecast": {
                "basis_stage_a_selected_elapsed_seconds": sum(durations),
                "gpu_hours": [2.5, 3],
                "cpu_hours": [0.3, 0.6],
                "additional_endpoint_bytes_estimate": sum(
                    x["checkpoint"]["bytes"] for x in receipts
                ),
            },
            "disk_free_bytes": shutil.disk_usage(root).free,
            "disk_stop_free_bytes": 50 * 2**30,
            "stages_c_d_enabled": False,
            "automatic_retries": False,
            "cpu_preparation_seconds": time.perf_counter() - began,
            "frozen_before_stage_b_outcomes": True,
        },
    )
    assert shutil.disk_usage(root).free >= 50 * 2**30
    print(
        json.dumps(
            {
                "status": "prepared",
                "rows": selected,
                "checkpoints": len(receipts),
                "order_windows": orders[0]["windows"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    prepare(p.parse_args().root)
