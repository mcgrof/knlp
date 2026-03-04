#!/usr/bin/env python3
"""Phase 11: Future-Q ablation (conditional, only if one-step results justify)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.qk_router.future_q import LinearQPredictor
from lib.qk_router.replay import replay_traces
from lib.qk_router.router import score_blocks_cosine
from lib.qk_router.scheduler import schedule_prefetches
from lib.qk_router.metrics import aggregate_replay_results
from lib.qk_router.blocking import build_block_map, assign_tiers, compute_block_bytes
from lib.qk_router.utils import load_json, save_json


def replay_with_future_q(
    traces,
    q_vectors,
    k_summaries,
    block_tiers,
    tier_latencies,
    block_bytes,
    horizon=1,
    window_size=4,
    seed=42,
):
    """Replay with future-Q predicted queries for the slow tier."""
    num_blocks = k_summaries.shape[0]
    predictor = LinearQPredictor(window_size=window_size, horizon=horizon)

    # Use the standard replay but with predicted Q for prefetch decisions
    from lib.qk_router.replay import ReplayState

    state = ReplayState(
        resident_blocks={i for i, t in enumerate(block_tiers) if t == 0}
    )

    fetch_latencies = np.array(
        [tier_latencies.get(block_tiers[i], 0) for i in range(num_blocks)]
    )
    block_bytes_arr = np.full(num_blocks, block_bytes, dtype=np.float64)
    last_used = np.full(num_blocks, -1, dtype=np.int64)

    step_metrics = []
    total_misses = 0
    total_needed = 0
    total_stall = 0.0

    for step_idx, trace in enumerate(traces):
        needed = set(trace["needed_blocks_mass"])
        total_needed += len(needed)

        # Complete in-flight
        completed = set()
        for bid, comp_step in list(state.in_flight.items()):
            if comp_step <= step_idx:
                state.resident_blocks.add(bid)
                completed.add(bid)
        for bid in completed:
            del state.in_flight[bid]

        resident_needed = needed & state.resident_blocks
        missed = needed - state.resident_blocks - set(state.in_flight.keys())

        stall = sum(fetch_latencies[bid] for bid in missed)
        for bid in missed:
            state.resident_blocks.add(bid)
        total_stall += stall
        total_misses += len(missed)

        # Update predictor with current Q
        if step_idx < len(q_vectors):
            predictor.update(q_vectors[step_idx])

        # Use predicted future Q for prefetch
        predicted_q = predictor.predict()
        if predicted_q is not None:
            scores = score_blocks_cosine(predicted_q, k_summaries)
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
                prefetch_budget=2,
                exploration_slots=1,
                seed=seed,
                step=step_idx,
            )
            for bid in prefetch_list:
                fetch_steps = max(1, int(np.ceil(fetch_latencies[bid] / 1000)))
                state.in_flight[bid] = step_idx + fetch_steps

        for bid in needed:
            last_used[bid] = step_idx

        baseline_lat = trace.get("baseline_latency_us", 0)
        step_metrics.append(
            {
                "step": step_idx,
                "stall_us": stall,
                "decode_latency_us": baseline_lat + stall,
                "missed": list(missed),
                "needed": list(needed),
            }
        )

    decode_lats = [s["decode_latency_us"] for s in step_metrics]
    num_steps = len(traces)

    return {
        "aggregate": {
            "policy": f"future_q_h{horizon}",
            "num_steps": num_steps,
            "total_needed_blocks": total_needed,
            "total_misses": total_misses,
            "missed_rate": total_misses / max(total_needed, 1),
            "total_prefetched": 0,
            "total_wasted_prefetches": 0,
            "wasted_rate": 0,
            "total_stall_us": total_stall,
            "avg_stall_us": total_stall / max(num_steps, 1),
            "p50_decode_latency_us": (
                float(np.median(decode_lats)) if decode_lats else 0
            ),
            "p95_decode_latency_us": (
                float(np.percentile(decode_lats, 95)) if decode_lats else 0
            ),
            "avg_overlap_frac": 0,
        },
        "per_step": step_metrics,
    }


def main():
    run_root = os.environ.get("RUN_ROOT", "/mnt/tmpfs/knlp/results/qk_router_01")
    prefix_length = int(os.environ.get("PREFIX_LENGTH", "4096"))
    block_size = int(os.environ.get("BLOCK_SIZE", "128"))
    num_requests = int(os.environ.get("NUM_REQUESTS", "64"))

    num_kv_heads = 2
    head_dim = 64
    num_layers = 24
    dtype_bytes = 2

    block_bytes_per_layer = compute_block_bytes(
        num_kv_heads, head_dim, block_size, dtype_bytes
    )
    total_block_bytes = block_bytes_per_layer * num_layers

    # Check if one-step results justify future-Q
    replay_path = os.path.join(run_root, "replay_results.json")
    if not os.path.exists(replay_path):
        print("No replay results found. Run replay ablation first.")
        return

    replay = load_json(replay_path)
    medium = replay.get("storage_medium", {})
    rec = medium.get("recency_only_top_m", {})
    sem = medium.get("semantic_top_m", {})

    if not rec or not sem:
        print("Insufficient replay data. Skipping future-Q.")
        return

    rec_miss = rec.get("avg_missed_rate", 1)
    sem_miss = sem.get("avg_missed_rate", 1)
    delta = rec_miss - sem_miss

    if delta < 0.01:
        print(
            f"Semantic delta vs recency = {delta:.3f}. "
            "One-step signal too weak to justify future-Q. Skipping."
        )
        save_json(
            {"skipped": True, "reason": f"semantic delta {delta:.3f} < 0.01"},
            os.path.join(run_root, "future_q_results.json"),
        )
        return

    print("=" * 60)
    print("QK Router Phase 11: Future-Q Ablation")
    print("=" * 60)

    # Load storage bench
    from scripts.qk_router_replay_ablation import load_storage_bench

    tier_latencies = load_storage_bench(run_root)

    # storage_medium tier assignments
    blocks = build_block_map(
        prefix_length, block_size, num_layers, num_kv_heads, head_dim
    )
    assign_tiers(blocks, 16, 16)
    block_tiers = [b.tier for b in blocks]

    k_sum = np.load(os.path.join(run_root, "k_summaries", "direct_centroid.npy"))

    future_q_results = {}
    for horizon in [1, 2, 4]:
        print(f"\n--- Horizon {horizon} ---")
        results_list = []

        for req_idx in range(num_requests):
            trace_path = os.path.join(
                run_root, "traces", f"trace_req{req_idx:03d}.json"
            )
            q_path = os.path.join(run_root, "q_vectors", f"q_req{req_idx:03d}.npy")
            if not os.path.exists(trace_path) or not os.path.exists(q_path):
                continue

            traces = load_json(trace_path)
            q_vecs = np.load(q_path)

            result = replay_with_future_q(
                traces,
                q_vecs,
                k_sum,
                block_tiers,
                tier_latencies,
                total_block_bytes,
                horizon=horizon,
            )
            results_list.append(result)

        if results_list:
            agg = aggregate_replay_results(results_list)
            key = f"future_q_h{horizon}"
            if key in agg:
                future_q_results[key] = agg[key]
                r = agg[key]
                print(
                    f"  missed={r['avg_missed_rate']:.3f} "
                    f"stall={r['avg_stall_us']:.0f}us "
                    f"p95={r['avg_p95_decode_us']:.0f}us"
                )

    save_json(future_q_results, os.path.join(run_root, "future_q_results.json"))
    print("\nFuture-Q ablation complete.")


if __name__ == "__main__":
    main()
