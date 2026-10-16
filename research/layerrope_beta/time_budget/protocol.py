# SPDX-License-Identifier: MIT
"""Immutable time, tuning, endpoint and paired-median comparison contracts."""

import math
from research.layerrope_beta.measure import policies
from research.tinystories_depth.kv_cache import cases, bytes_per_token

ROWS = ("R", "L", "M", "ML")
PILOT_SEED = 77601
FINAL_SEEDS = tuple(range(77611, 77619))
PROBE_SEED = 77690
LR_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03)
LANDMARKS = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 615.0, 700.0, 787.0)
CONTRASTS = (("R", "M"), ("L", "M"), ("ML", "M"), ("R", "ML"), ("L", "ML"))
CACHE_NAMES = (
    "clean",
    "symmetric-K3V3",
    "symmetric-K3V2",
    "symmetric-K2V3",
    "symmetric-K2V2",
    "affine-K2V2",
    "fp4_e2m1-K4V4",
    "fp8_e4m3fn-K8V8",
)
WEIGHT_NAMES = (
    "all-FP4-weights",
    "all-INT4-weights",
    "all-FP8-weights",
    "all-INT8-weights",
    "block-W4A4",
    "block-W8A8",
    "residual-streams-FP4",
    "residual-streams-FP8",
)


def cache_menu():
    full = cases() + [
        dict(name="fp4_e2m1-K4V4", k_bits=4, v_bits=4, method="fp4_e2m1"),
        dict(name="fp8_e4m3fn-K8V8", k_bits=8, v_bits=8, method="fp8_e4m3fn"),
    ]
    selected = [next(x for x in full if x["name"] == name) for name in CACHE_NAMES]
    floor = {x["name"] for x in full if bytes_per_token(x)["ideal_total_bytes"] <= 3072}
    if not floor <= set(CACHE_NAMES):
        raise ValueError("Reduced menu omits a declared cell at or below the floor")
    return selected, [dict(**x, storage=bytes_per_token(x)) for x in full]


def freeze():
    cache, full = cache_menu()
    return dict(
        schema="layerrope-equal-training-time.v1",
        rows=list(ROWS),
        beta_initialization=0.0,
        pilot_seed=PILOT_SEED,
        final_seeds=list(FINAL_SEEDS),
        probe_seed=PROBE_SEED,
        lr_grid=list(LR_GRID),
        lr_tie_epsilon=1e-9,
        lr_extensions=dict(low=0.0001, high=0.1, common_round_limit=1),
        landmarks=list(LANDMARKS),
        primary_seconds=787.0,
        supporting_seconds=615.0,
        depth=32,
        vocab=50257,
        window=256,
        train_batch=8,
        eval_batch=32,
        validation_predicted_tokens=32768,
        final_predicted_tokens=131072,
        warmup_updates=100,
        routing_cadence=25,
        finite_cadence=25,
        routing_error_limit=1e-4,
        routing_domain_limit=1e-7,
        routing_rule="Observed constraint miss limits qualified mechanism claims; retain empirical CE and every miss",
        clock="Synchronized complete-update cumulative time including batch preparation, transfer, checks and warm-up; offline work separate",
        overshoot_fraction=0.01,
        cache_cases=cache,
        declared_cache_menu=full,
        supporting_cache_names=[
            "clean",
            "symmetric-K3V3",
            "symmetric-K3V2",
            "affine-K2V2",
        ],
        weight_policies=[
            next(p for p in policies() if p["name"] == n) for n in WEIGHT_NAMES
        ],
        endpoint_evaluations_per_final=20,
        primary_contrasts=[
            dict(first=a, second=b, endpoint=e)
            for a, b in CONTRASTS
            for e in ("clean", "symmetric-K3V2")
        ],
        inference=dict(
            target="population median of paired seed difference",
            upper_bound="max of all eight paired differences",
            family_lower_coverage=1 - 10 / 256,
            margins=dict(
                practical_improvement=-0.01,
                directional_improvement=0.0,
                noninferiority=0.01,
            ),
        ),
        limits=[
            "Conditional on frozen recipe, hardware and exact evaluation data",
            "Fake quantization is not packed serving memory or kernel throughput",
            "No automatic seed expansion, beta search or historical continuation",
        ],
    )


def choose_lrs(results, grid=LR_GRID, epsilon=1e-9):
    """Require every trial to report; invalid numerical trials cannot win."""
    selected = {}
    for row in ROWS:
        trials = [x for x in results if x["row"] == row]
        if sorted(x["lr"] for x in trials) != sorted(grid):
            raise ValueError("Incomplete or duplicate pilot grid: " + row)
        valid = [
            x
            for x in trials
            if x["status"] == "complete" and math.isfinite(x["validation_ce"])
        ]
        if not valid:
            raise ValueError("No valid pilot: " + row)
        best = min(x["validation_ce"] for x in valid)
        selected[row] = min(
            x["lr"] for x in valid if x["validation_ce"] <= best + epsilon
        )
    return selected


def edge_extensions(selected):
    # Apply the same bounded extension to every architecture.
    extra = []
    if min(selected.values()) == min(LR_GRID):
        extra.append(0.0001)
    if max(selected.values()) == max(LR_GRID):
        extra.append(0.1)
    return extra


def median_claim(differences):
    if len(differences) != 8 or not all(math.isfinite(x) for x in differences):
        raise ValueError("All eight valid paired seeds are required")
    upper = max(differences)
    statement = "unresolved"
    for name, margin in [
        ("practical_median_improvement", -0.01),
        ("directional_median_improvement", 0.0),
        ("median_noninferiority", 0.01),
    ]:
        if upper < margin:
            statement = name
            break
    return dict(
        upper=upper,
        statement=statement,
        simultaneous_lower_coverage=1 - 10 / 256,
        target="median of paired seed effect; not population mean or equivalence",
    )
