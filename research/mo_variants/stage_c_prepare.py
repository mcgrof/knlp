"""Freeze all confirmation rows, maps, claims and numerical qualifications."""

import os
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from .model import Model
from .stage_c_common import GUIDANCE, MODEL_SHA, ROWS, SEEDS, read, sha, write


def prepare(root):
    """Freeze the seven-row recipe from locally prepared public data."""
    root = Path(root)
    r = root / "records"
    if (r / "STAGE-C-PLAN.json").exists():
        raise FileExistsError("Use a new output directory for a new confirmation")
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation must use an independent CPU process")
    # Historical flags are retained; a new qualification run cannot waive them.
    scoped = [
        {
            "variant": f"V{n:02}",
            "historical_and_B_pass": n not in (1, 19, 20),
            "qualification_is_sampled_not_uniform": True,
            "misses": (
                [{"scope": "historical finite-function tolerance miss"}]
                if n in (1, 19, 20)
                else []
            ),
        }
        for n in range(32)
    ]
    write(
        r / "STAGE-C-NUMERICAL-SCOPE.json",
        {
            "status": "complete",
            "rows": scoped,
            "thresholds_unchanged": True,
            "production_math_changed": False,
            "infinite_projection_audit": "not performed",
        },
    )
    selection = {
        "frozen": True,
        "training_nominee": "V30",
        "quantization_nominee": "V02",
        "raw_training_and_quantization_nominee": "V20",
        "numerical_gate_unchanged": True,
        "rows": [a for a, _, _ in ROWS],
        "V01": "faithful implementation; numerically flagged comparison baseline",
    }
    write(r / "SELECTION.json", selection)
    data = read(r / "DATA.json")
    for n in ("train.npy", "curve.npy", "DOCUMENTS.json"):
        assert sha(root / "data" / n) == data["verified_inherited_files"][n]
    for n, h in data["confirmation_files"].items():
        assert sha(root / "data/confirmation" / n) == h
    docs = read(root / "data/confirmation/DOCUMENTS.json")
    assert not any(docs["overlap_counts"].values())
    jobs, maps = [], []
    for seed in SEEDS:
        order_sha = data["training_and_order_files"][f"order-{seed}.npy"]
        assert sha(root / "data" / f"order-{seed}.npy") == order_sha
        initial = None
        for row, variant, mapping in ROWS:
            # Vocabulary has no effect on lookup RNG or routing initialization.
            m = Model(variant, 32, seed, vocab=128, mapping=mapping)
            if initial is None:
                initial = m.common_hashes
            assert initial == m.common_hashes
            map_record = {
                "row": row,
                "seed": seed,
                "code_map": m.code_map,
                "mapping_lookup": m.mapping_lookup,
            }
            if row.startswith("V30"):
                ordered = Model("V30", 32, seed, vocab=128)
                delta = []
                for site in range(2):
                    differences = [
                        (
                            torch.outer(m.code(l, site), m.code(l, site))
                            - torch.outer(ordered.code(l, site), ordered.code(l, site))
                        )
                        .abs()
                        .max()
                        .item()
                        / 4
                        for l in range(32)
                    ]
                    if mapping == "shuffle":
                        assert m.mapping_lookup[site] != list(range(32))
                        assert max(differences) > 1e-4
                    if mapping == "constant":
                        assert m.code_map[site] == [15] * 32
                    delta.append(differences)
                map_record[
                    "carry_logit_perturbation_absolute_difference_from_ordered"
                ] = delta
                map_record["sign_symmetry_checked_on_actual_outer_products"] = True
                del ordered
            maps.append(map_record)
            jobs.append(
                {
                    "id": f"C-{row}-{seed}",
                    "stage": "C",
                    "row": row,
                    "variant": variant,
                    "mapping": mapping,
                    "seed": seed,
                    "depth": 32,
                    "vocab": 50257,
                    "steps": 4000,
                    "lr": 0.003,
                    "order_sha256": order_sha,
                    "code_map": m.code_map,
                    "mapping_lookup": m.mapping_lookup,
                    "worker": (seed - SEEDS[0]) % 2,
                }
            )
            del m
            gc.collect()
    write(
        r / "STAGE-C-CONTROL-MAPS.json",
        {
            "frozen_before_outcomes": True,
            "maps": maps,
            "rng_namespaces_unchanged": True,
            "constant_capacity": "two constant phase lookups; slope and intercept are not separately identifiable at one depth index",
        },
    )
    claims = [
        {
            "name": f"{candidate}-{kind}-vs-{baseline}",
            "candidate": candidate,
            "baseline": baseline,
            "metric": metric,
        }
        for candidate, kind, metric in (
            ("V30", "training", "primary_ce"),
            ("V02", "quantization", "primary_int4_ce"),
        )
        for baseline in ("V00", "V01")
    ]
    claims += [
        {
            "name": f"V30-ordered-carry-vs-{c}",
            "candidate": "V30",
            "baseline": c,
            "metric": "primary_ce",
        }
        for c in ("V30-S", "V30-C")
    ]
    plan = {
        "stage": "C",
        "guidance": GUIDANCE,
        "frozen_before_confirmation_outcomes": True,
        "D_enabled": False,
        "jobs": jobs,
        "seeds": SEEDS,
        "nominees": selection,
        "selection_sha256": sha(r / "SELECTION.json"),
        "maps_sha256": sha(r / "STAGE-C-CONTROL-MAPS.json"),
        "numerical_scope_sha256": sha(r / "STAGE-C-NUMERICAL-SCOPE.json"),
        "source_model_sha256": MODEL_SHA,
        "data": data,
        "confirmation_document_receipt_sha256": sha(
            root / "data/confirmation/DOCUMENTS.json"
        ),
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0,
            "warmup_steps": 100,
        },
        "precision": "inherited BF16 blocks/activations, FP32 masters/routing/norms/Sinkhorn/CE; TF32 disabled; compile disabled",
        "predicted_tokens_per_update": 2048,
        "primary_steps": list(range(2000, 4001, 100)),
        "early_window_steps": list(range(1000, 2001, 100)),
        "final_array_predicted_tokens": 131072,
        "numerical_steps": [0, 50, 500, 1000, 2000, 4000],
        "checkpoint_steps": [1000, 2000, 3000, 4000],
        "training_rule": {
            "all_primary_pairs_negative": True,
            "mean_gain_min": 0.005,
            "lower_final_mean": True,
            "max_final_regression": 0.01,
        },
        "quantization_rule": {
            "all_int4_pairs_negative": True,
            "mean_damage_gain": {"V00": 0.01, "V01": 0.005},
            "minimum_damage_wins": 7,
            "max_clean_regression": 0.01,
        },
        "mechanism_rule": {
            "controls": ["V30-S", "V30-C"],
            "all_primary_pairs_negative": True,
            "mean_gain_min": 0.005,
        },
        "claim_family": claims,
        "inference": "one-sided exact paired sign tests, Holm correction over all six declared claims; paired seed t intervals are descriptive",
        "target_rules": {
            "primary": 3.0,
            "diagnostic": 4.0,
            "consecutive_evaluations": 2,
            "interpolation": False,
            "extra_targets": [],
        },
        "matched_time_seconds": [100, 200, 400, 600, 900, 1200],
        "matched_time_method": "last supported scheduled observation, no interpolation or extrapolation",
        "endpoint_suite": [
            "clean",
            "rowwise8",
            "rowwise4",
            "rowwise3",
            "rowwise2",
            "group128-4",
            "group128-3",
            "blocks4",
            "blocks3",
            "rowwise4-scalars-bf16-storage",
            "rowwise4-routing-code-bf16-storage",
        ],
        "quantizer_pass_rule": "all eight seeds add <=0.1 nats over own CPU clean CE on identical final array",
        "infrastructure_recovery": {
            "nonfinite_failure": "retain and stop; no outcome retries",
            "transport_loss": "sync prior durable state, resume exact hashes once on same GPU identity; no automatic replacement pod",
            "training_restart": "at most one exact infrastructure resume per job; never change seed, LR, code map or data",
            "checkpoint_integrity_failure": "stop study",
        },
        "resource": {
            "provider": "configured by the caller",
            "gpu_type": "NVIDIA H100 80GB HBM3",
            "gpu_count": 2,
            "paired_seed_block_placement": "complete blocks alternate between the two fixed GPU indices",
            "CPU_endpoints": "independent CPU process, 8 threads",
            "free_space_watermark_bytes": 50 * 2**30,
        },
    }
    write(r / "STAGE-C-PLAN.json", plan)
    print(
        json.dumps(
            {
                "status": "frozen",
                "jobs": len(jobs),
                "rows": selection["rows"],
                "confirmation_evaluated": False,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    prepare(a.root)
