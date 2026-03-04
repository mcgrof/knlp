"""Latency-aware scheduler: combines semantic score with systems utility."""

import numpy as np


def compute_utility(
    semantic_scores: np.ndarray,
    reuse_scores: np.ndarray,
    fetch_latencies_us: np.ndarray,
    block_bytes: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
) -> np.ndarray:
    """Compute utility score for each block.

    utility_i = alpha * semantic_i + beta * reuse_i
                - gamma * fetch_latency_i - delta * bandwidth_cost_i

    All features are normalized to [0, 1] before combining.
    """
    n = len(semantic_scores)
    if n == 0:
        return np.array([])

    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    sem_norm = normalize(semantic_scores)
    reuse_norm = normalize(reuse_scores)
    lat_norm = normalize(fetch_latencies_us)
    bw_norm = normalize(block_bytes.astype(float))

    utility = alpha * sem_norm + beta * reuse_norm - gamma * lat_norm - delta * bw_norm
    return utility


def schedule_prefetches(
    semantic_scores: np.ndarray,
    reuse_scores: np.ndarray,
    fetch_latencies_us: np.ndarray,
    block_bytes: np.ndarray,
    resident_blocks: set,
    in_flight: set,
    prefetch_budget: int = 2,
    exploration_slots: int = 1,
    exploration_range: tuple = (9, 32),
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
    seed: int = 42,
    step: int = 0,
) -> list[int]:
    """Choose blocks to prefetch.

    Returns list of block IDs to prefetch (up to prefetch_budget + exploration_slots).
    """
    n = len(semantic_scores)
    utility = compute_utility(
        semantic_scores,
        reuse_scores,
        fetch_latencies_us,
        block_bytes,
        alpha,
        beta,
        gamma,
        delta,
    )

    # Exclude already-resident and in-flight blocks
    candidates = []
    for i in range(n):
        if i not in resident_blocks and i not in in_flight:
            candidates.append(i)

    if not candidates:
        return []

    # Sort candidates by utility descending
    cand_utility = [(i, utility[i]) for i in candidates]
    cand_utility.sort(key=lambda x: x[1], reverse=True)

    # Top prefetch_budget by utility
    prefetch = [c[0] for c in cand_utility[:prefetch_budget]]

    # Exploration slots: pick from exploration_range band
    if exploration_slots > 0:
        rng = np.random.RandomState(seed + step)
        exp_lo, exp_hi = exploration_range
        exp_lo = min(exp_lo, len(cand_utility))
        exp_hi = min(exp_hi, len(cand_utility))
        if exp_lo < exp_hi:
            exp_candidates = [c[0] for c in cand_utility[exp_lo:exp_hi]]
            exp_candidates = [c for c in exp_candidates if c not in prefetch]
            if exp_candidates:
                n_exp = min(exploration_slots, len(exp_candidates))
                exp_picks = rng.choice(exp_candidates, size=n_exp, replace=False)
                prefetch.extend(exp_picks.tolist())

    return prefetch
