# SPDX-License-Identifier: MIT
"""Freeze a fixed-recipe intervention without changing the older study."""

from research.layerrope_beta.time_budget.protocol import freeze as previous
from research.tinystories_depth.kv_cache import bytes_per_token

LRS = {"R": 0.001, "L": 0.01, "M": 0.003, "ML": 0.01}


def freeze(seeds=(77811, 77812, 77813), probe_seed=77890):
    if len(seeds) != 3 or len(set(seeds)) != 3 or probe_seed in seeds:
        raise ValueError("Three distinct final seeds and a separate probe are required")
    p = previous()
    for key in (
        "pilot_seed",
        "lr_grid",
        "lr_tie_epsilon",
        "lr_extensions",
        "inference",
        "primary_contrasts",
        "weight_policies",
        "endpoint_evaluations_per_final",
        "supporting_cache_names",
        "declared_cache_menu",
    ):
        p.pop(key, None)
    menu = [
        (16, 16),
        (3, 16),
        (2, 16),
        (16, 3),
        (16, 2),
        (3, 3),
        (3, 2),
        (2, 3),
        (2, 2),
    ]
    cache = [
        dict(
            name="clean" if (k, v) == (16, 16) else f"symmetric-K{k}V{v}",
            k_bits=k,
            v_bits=v,
            method="symmetric",
        )
        for k, v in menu
    ]
    cache.append(dict(name="affine-K2V2", k_bits=2, v_bits=2, method="affine"))
    p.update(
        schema="layerrope-qknorm-fixed-recipe.v1",
        final_seeds=list(seeds),
        probe_seed=probe_seed,
        arms=[
            dict(name=r + ("-on" if q else "-off"), arm=r, qknorm=q, lr=lr)
            for r, lr in LRS.items()
            for q in (False, True)
        ],
        equal_token_updates=1000,
        cache_cases=cache,
        qknorm=dict(
            epsilon=1e-12,
            norm_accumulation="FP32",
            initialization="log2(256*256-256)",
            scalar="one unconstrained FP32 parameter per attention module, shared heads",
            application="gain cast to Q dtype, Q multiplied by gain, SDPA scale=1.0",
            order="project, head split, normalize Q/K, positional RoPE, round new K/V, attention",
            weight_decay=0.0,
        ),
        optimizer=dict(
            type="AdamW", betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0, clipping=None
        ),
        diagnostic=dict(
            allowed_keys_min=16,
            maximum_probability=0.99,
            normalized_entropy=0.1,
            loss_spike_nats=1.0,
            preceding_updates=100,
            sample="first two validation windows; offline dropout-free",
        ),
        tolerance=dict(
            full_window_noquant="bit exact",
            attention_bf16_max_abs=0.004,
            cached_model_bf16_max_abs=0.125,
            fp64_atol=1e-10,
            fp64_rtol=1e-9,
        ),
        screening=dict(
            paired_mean_improvement_nats=0.01,
            every_pair_improves=True,
            own_clean_damage_limit=0.1,
            common_quality_R_off_margin=0.1,
            interpretation="Three-seed exploratory screen; no population winner or token-bootstrap intervals",
        ),
        max_aggregate_gpu_hours=12,
        max_total_usd=60,
        fixed_recipe_limitation="Learning rates inherited from no-QKNorm tuning; no equal tuning claim",
    )
    return p


def menu():
    return [dict(**c, storage=bytes_per_token(c)) for c in freeze()["cache_cases"]]
