# SPDX-License-Identifier: MIT
"""Analyze the frozen paired family without dropping seeds or inventing floors."""

import math
import statistics
from research.layerrope_beta.time_budget.protocol import freeze, median_claim


def analyze(records):
    p = freeze()
    index = {(r["row"], r["seed"]): r for r in records}
    expected = {(r, s) for r in p["rows"] for s in p["final_seeds"]}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError("All 32 distinct final models are required")
    for r in records:
        for t, names in [
            ("615.0", p["supporting_cache_names"]),
            ("787.0", [c["name"] for c in p["cache_cases"]]),
        ]:
            result = r["endpoints"][t]
            if len(result["cache"]) != len(names) or {
                c["name"] for c in result["cache"]
            } != set(names):
                raise ValueError("Incomplete cache endpoint menu")
            values = [c["ce"] for c in result["cache"]]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite endpoint CE")
        weights = r["endpoints"]["787.0"]["weight_activation_residual"]
        if len(weights) != len(p["weight_policies"]) or {
            w["name"] for w in weights
        } != {w["name"] for w in p["weight_policies"]}:
            raise ValueError("Incomplete weight/residual menu")
        if not all(math.isfinite(w["ce"]) for w in weights):
            raise ValueError("Nonfinite weight/residual CE")

    def cache(row, seed, case):
        return next(
            c
            for c in index[row, seed]["endpoints"]["787.0"]["cache"]
            if c["name"] == case
        )

    contrasts = []
    for c in p["primary_contrasts"]:
        differences = [
            cache(c["first"], s, c["endpoint"])["ce"]
            - cache(c["second"], s, c["endpoint"])["ce"]
            for s in p["final_seeds"]
        ]
        contrasts.append(
            dict(
                **c,
                seeds=p["final_seeds"],
                differences=differences,
                observed_mean=statistics.mean(differences),
                observed_median=statistics.median(differences),
                observed_range=[min(differences), max(differences)],
                **median_claim(differences)
            )
        )
    floors = []
    descriptive = []
    for row in p["rows"]:
        passes = []
        for case in p["declared_cache_menu"]:
            if case["storage"]["ideal_total_bytes"] > 3072:
                continue
            damages = [
                cache(row, s, case["name"])["ce"] - cache(row, s, "clean")["ce"]
                for s in p["final_seeds"]
            ]
            passes.append(
                dict(
                    case=case["name"],
                    ideal_bytes_per_token=case["storage"]["ideal_total_bytes"],
                    own_clean_damage=damages,
                    all_seed_pass=all(d <= 0.1 for d in damages),
                )
            )
        passed = [c["ideal_bytes_per_token"] for c in passes if c["all_seed_pass"]]
        floors.append(
            dict(
                row=row,
                cases=passes,
                observed_floor=min(passed) if passed else None,
                scope="Minimum in declared menu if established at/below 3072; no packed-memory claim",
            )
        )
        for s in p["final_seeds"]:
            descriptive.append(
                dict(
                    row=row,
                    seed=s,
                    v_bit_penalty=cache(row, s, "symmetric-K3V2")["ce"]
                    - cache(row, s, "symmetric-K3V3")["ce"],
                    equal_byte_allocation_difference=cache(row, s, "symmetric-K3V2")[
                        "ce"
                    ]
                    - cache(row, s, "symmetric-K2V3")["ce"],
                )
            )
    return dict(
        contrasts=contrasts,
        floors=floors,
        descriptive=descriptive,
        scope="Conditional population median of paired seed effect; empirical CE retained despite routing misses",
        routing_constraint_pass=all(r["routing_constraint_pass"] for r in records),
        records=records,
    )
