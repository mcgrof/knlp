"""Replay simulator: step-by-step replay of decode traces with prefetch policies."""

import numpy as np
from dataclasses import dataclass, field

from .router import score_blocks_cosine, router_predict
from .scheduler import schedule_prefetches


@dataclass
class ReplayState:
    """State tracked during replay simulation."""

    resident_blocks: set = field(default_factory=set)
    in_flight: dict = field(default_factory=dict)  # block_id -> completion_step
    fetch_completion_times: dict = field(default_factory=dict)
    bandwidth_used_bytes: float = 0.0
    total_stall_us: float = 0.0
    per_step_metrics: list = field(default_factory=list)


@dataclass
class StepMetrics:
    """Metrics for a single replay step."""

    step: int
    needed_blocks: list
    resident_needed: list
    missed_blocks: list
    prefetched_blocks: list
    stall_us: float
    fetch_overlap_achieved: float
    wasted_prefetches: int
    decode_latency_us: float


def replay_traces(
    traces: list[dict],
    q_vectors: list[np.ndarray],
    k_summaries: np.ndarray,
    block_tiers: list[int],
    tier_latencies_us: dict,
    block_bytes: int,
    policy: str = "no_prefetch",
    initial_resident: set = None,
    prefetch_budget: int = 2,
    exploration_slots: int = 1,
    exploration_range: tuple = (9, 32),
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
    score_fn: str = "cosine",
    aggregation: str = "mean_across_layers",
    seed: int = 42,
    top_m: int = 8,
) -> dict:
    """Replay decode traces with a given prefetch policy.

    Args:
        traces: list of trace dicts from trace_collect
        q_vectors: list of [num_layers, head_dim] Q vectors
        k_summaries: [num_blocks, num_layers, head_dim]
        block_tiers: per-block tier assignment [0, 1, 2]
        tier_latencies_us: {0: gpu_lat, 1: tmpfs_lat, 2: sfs_lat}
        block_bytes: bytes per block payload
        policy: prefetch policy name
        initial_resident: blocks initially resident (tier 0)

    Returns:
        dict with aggregate and per-step metrics
    """
    num_blocks = (
        k_summaries.shape[0]
        if k_summaries is not None
        else traces[0]["prefix_block_count"]
    )
    num_steps = len(traces)

    if initial_resident is None:
        initial_resident = {i for i, t in enumerate(block_tiers) if t == 0}

    state = ReplayState(resident_blocks=set(initial_resident))

    # Build fetch latency array based on tier
    fetch_latencies = np.array(
        [tier_latencies_us.get(block_tiers[i], 0) for i in range(num_blocks)]
    )
    block_bytes_arr = np.full(num_blocks, block_bytes, dtype=np.float64)

    # Track last-used step for recency scoring
    last_used = np.full(num_blocks, -1, dtype=np.int64)

    total_misses = 0
    total_needed = 0
    total_wasted = 0
    total_stall = 0.0
    total_prefetched = 0
    step_metrics = []

    for step_idx in range(num_steps):
        trace = traces[step_idx]
        needed = set(trace["needed_blocks_mass"])
        total_needed += len(needed)

        # Complete in-flight transfers
        completed = set()
        for bid, comp_step in list(state.in_flight.items()):
            if comp_step <= step_idx:
                state.resident_blocks.add(bid)
                completed.add(bid)
        for bid in completed:
            del state.in_flight[bid]

        # Check which needed blocks are resident
        resident_needed = needed & state.resident_blocks
        missed = needed - state.resident_blocks - set(state.in_flight.keys())

        # Stall: pay latency for missed blocks that aren't prefetched
        stall = 0.0
        for bid in missed:
            stall += fetch_latencies[bid]
            # On-demand fetch makes it resident
            state.resident_blocks.add(bid)
        total_stall += stall
        total_misses += len(missed)

        # Prefetch overlap: blocks that arrived just in time
        inflight_needed = needed & set(state.in_flight.keys())
        overlap = len(resident_needed) + len(inflight_needed)
        overlap_frac = overlap / len(needed) if len(needed) > 0 else 1.0

        # Issue prefetches for next step
        prefetch_list = []
        if policy == "no_prefetch":
            prefetch_list = []
        elif policy == "recency_only_top_m":
            # Rank by recency (most recently used first), prefetch top-M
            recency = np.zeros(num_blocks)
            for bid in range(num_blocks):
                if last_used[bid] >= 0:
                    recency[bid] = 1.0 / (step_idx - last_used[bid] + 1)
            sorted_idx = np.argsort(recency)[::-1]
            for bid in sorted_idx:
                bid = int(bid)
                if bid not in state.resident_blocks and bid not in state.in_flight:
                    prefetch_list.append(bid)
                    if len(prefetch_list) >= top_m:
                        break
        elif policy == "semantic_top_m":
            if (
                q_vectors is not None
                and k_summaries is not None
                and step_idx < len(q_vectors)
            ):
                top_idx, _ = router_predict(
                    q_vectors[step_idx], k_summaries, top_m, score_fn, aggregation
                )
                prefetch_list = [
                    b
                    for b in top_idx
                    if b not in state.resident_blocks and b not in state.in_flight
                ]
        elif policy == "utility_aware":
            if (
                q_vectors is not None
                and k_summaries is not None
                and step_idx < len(q_vectors)
            ):
                scores = score_blocks_cosine(
                    q_vectors[step_idx], k_summaries, aggregation
                )
                reuse = np.zeros(num_blocks)
                for bid in range(num_blocks):
                    if last_used[bid] >= 0:
                        reuse[bid] = 1.0 / (step_idx - last_used[bid] + 1)
                prefetch_list = schedule_prefetches(
                    scores,
                    reuse,
                    fetch_latencies,
                    block_bytes_arr,
                    state.resident_blocks,
                    set(state.in_flight.keys()),
                    prefetch_budget,
                    0,
                    exploration_range,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    seed,
                    step_idx,
                )
        elif policy == "utility_aware_plus_exploration":
            if (
                q_vectors is not None
                and k_summaries is not None
                and step_idx < len(q_vectors)
            ):
                scores = score_blocks_cosine(
                    q_vectors[step_idx], k_summaries, aggregation
                )
                reuse = np.zeros(num_blocks)
                for bid in range(num_blocks):
                    if last_used[bid] >= 0:
                        reuse[bid] = 1.0 / (step_idx - last_used[bid] + 1)
                prefetch_list = schedule_prefetches(
                    scores,
                    reuse,
                    fetch_latencies,
                    block_bytes_arr,
                    state.resident_blocks,
                    set(state.in_flight.keys()),
                    prefetch_budget,
                    exploration_slots,
                    exploration_range,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    seed,
                    step_idx,
                )

        # Register prefetches as in-flight
        wasted = 0
        for bid in prefetch_list:
            # Estimate completion: current step + ceil(fetch_time / decode_time)
            fetch_steps = max(1, int(np.ceil(fetch_latencies[bid] / 1000)))
            state.in_flight[bid] = step_idx + fetch_steps
            total_prefetched += 1

        # Update last-used for needed blocks
        for bid in needed:
            last_used[bid] = step_idx

        # Count wasted prefetches from previous steps
        # A prefetch was wasted if the block arrived but was never needed
        for bid in completed:
            if bid not in needed:
                wasted += 1
                total_wasted += 1

        baseline_lat = trace.get("baseline_latency_us", 0)
        decode_lat = baseline_lat + stall

        step_metrics.append(
            {
                "step": step_idx,
                "needed": list(needed),
                "resident_needed": list(resident_needed),
                "missed": list(missed),
                "prefetched": prefetch_list,
                "stall_us": stall,
                "overlap_frac": overlap_frac,
                "wasted": wasted,
                "decode_latency_us": decode_lat,
            }
        )

    # Aggregate metrics
    decode_lats = [s["decode_latency_us"] for s in step_metrics]
    stalls = [s["stall_us"] for s in step_metrics]

    agg = {
        "policy": policy,
        "num_steps": num_steps,
        "total_needed_blocks": total_needed,
        "total_misses": total_misses,
        "missed_rate": total_misses / max(total_needed, 1),
        "total_prefetched": total_prefetched,
        "total_wasted_prefetches": total_wasted,
        "wasted_rate": total_wasted / max(total_prefetched, 1),
        "total_stall_us": total_stall,
        "avg_stall_us": total_stall / max(num_steps, 1),
        "p50_decode_latency_us": float(np.median(decode_lats)) if decode_lats else 0,
        "p95_decode_latency_us": (
            float(np.percentile(decode_lats, 95)) if decode_lats else 0
        ),
        "avg_overlap_frac": float(np.mean([s["overlap_frac"] for s in step_metrics])),
    }

    return {"aggregate": agg, "per_step": step_metrics}
