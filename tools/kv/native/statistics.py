# SPDX-License-Identifier: MIT
"""Paired question statistics; sparse transitions retain exact uncertainty."""

import random
from scipy.stats import beta


def bounds(count, total, alpha=0.05):
    lower = 0.0 if count == 0 else float(beta.ppf(alpha, count, total - count + 1))
    upper = (
        1.0 if count == total else float(beta.ppf(1 - alpha, count + 1, total - count))
    )
    return [lower, upper]


def paired(arm, baseline, margin=0.02, seed=20261021):
    if set(arm) != set(baseline) or not arm:
        raise ValueError("paired question IDs must match exactly")
    keys = sorted(arm)
    total = len(keys)
    losses = sum(bool(baseline[k]) and not arm[k] for k in keys)
    gains = sum(bool(arm[k]) and not baseline[k] for k in keys)
    differences = [int(arm[k]) - int(baseline[k]) for k in keys]
    rng = random.Random(seed)
    samples = sorted(
        sum(differences[rng.randrange(total)] for _ in keys) / total
        for _ in range(2000)
    )
    import numpy as np

    bootstrap = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    loss_bounds = bounds(losses, total)
    gain_bounds = bounds(gains, total)
    loss_bonferroni = bounds(losses, total, alpha=0.025)
    gain_bonferroni = bounds(gains, total, alpha=0.025)
    return {
        "cases": total,
        "baseline_accuracy": sum(bool(baseline[k]) for k in keys) / total,
        "arm_accuracy": sum(bool(arm[k]) for k in keys) / total,
        "correct_to_wrong": losses,
        "wrong_to_correct": gains,
        "net_accuracy_change": (gains - losses) / total,
        "paired_question_bootstrap_95_two_sided": bootstrap,
        "loss_probability_95_one_sided_bounds": loss_bounds,
        "gain_probability_95_one_sided_bounds": gain_bounds,
        "net_accuracy_change_95_exact_bonferroni": [
            gain_bonferroni[0] - loss_bonferroni[1],
            gain_bonferroni[1] - loss_bonferroni[0],
        ],
        "loss_margin": margin,
        "supported_margin": loss_bounds[1] <= margin,
        "bootstrap_seed": seed,
        "bootstrap_repetitions": 2000,
        "question_ids_are_resampling_units": True,
    }
