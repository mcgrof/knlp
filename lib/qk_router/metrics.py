"""Metrics computation for QK Router experiments."""

import numpy as np


def compute_recall_at_m(predicted: list[int], actual: list[int], m: int) -> float:
    """Recall@M: fraction of actual items in top-M predictions."""
    if not actual:
        return 1.0
    top_m = set(predicted[:m])
    return len(top_m & set(actual)) / len(actual)


def compute_precision_at_m(predicted: list[int], actual: list[int], m: int) -> float:
    """Precision@M: fraction of top-M predictions that are actual."""
    if m == 0:
        return 1.0
    top_m = set(predicted[:m])
    return len(top_m & set(actual)) / m


def aggregate_replay_results(results: list[dict]) -> dict:
    """Aggregate replay results across multiple requests."""
    if not results:
        return {}

    policies = {}
    for r in results:
        agg = r["aggregate"]
        pol = agg["policy"]
        if pol not in policies:
            policies[pol] = []
        policies[pol].append(agg)

    summary = {}
    for pol, runs in policies.items():
        summary[pol] = {
            "num_requests": len(runs),
            "avg_missed_rate": float(np.mean([r["missed_rate"] for r in runs])),
            "avg_wasted_rate": float(np.mean([r["wasted_rate"] for r in runs])),
            "avg_stall_us": float(np.mean([r["avg_stall_us"] for r in runs])),
            "avg_p50_decode_us": float(
                np.mean([r["p50_decode_latency_us"] for r in runs])
            ),
            "avg_p95_decode_us": float(
                np.mean([r["p95_decode_latency_us"] for r in runs])
            ),
            "avg_overlap_frac": float(np.mean([r["avg_overlap_frac"] for r in runs])),
            "total_misses": sum(r["total_misses"] for r in runs),
            "total_needed": sum(r["total_needed_blocks"] for r in runs),
        }

    return summary


def compute_reuse_distance_histogram(traces: list[dict], num_blocks: int) -> dict:
    """Compute block reuse-distance histogram from traces."""
    last_seen = {}
    distances = []

    for trace in traces:
        step = trace["decode_step"]
        for bid in trace["needed_blocks_mass"]:
            if bid in last_seen:
                dist = step - last_seen[bid]
                distances.append(dist)
            last_seen[bid] = step

    if not distances:
        return {"distances": [], "histogram": {}}

    arr = np.array(distances)
    bins = [1, 2, 4, 8, 16, 32, 64]
    hist = {}
    for i in range(len(bins)):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else float("inf")
        count = int(np.sum((arr >= lo) & (arr < hi)))
        hist[f"{lo}-{int(hi) if hi != float('inf') else 'inf'}"] = count

    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "histogram": hist,
    }
