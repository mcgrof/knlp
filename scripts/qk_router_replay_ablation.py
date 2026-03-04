#!/usr/bin/env python3
"""Phase 5-10: Replay ablation across policies, summaries, and storage regimes."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.qk_router.replay import replay_traces
from lib.qk_router.router import compute_router_metrics, score_blocks_cosine
from lib.qk_router.metrics import (
    aggregate_replay_results,
    compute_reuse_distance_histogram,
)
from lib.qk_router.blocking import assign_tiers, build_block_map, compute_block_bytes
from lib.qk_router.utils import save_json, load_json, Timer


def load_storage_bench(run_root: str) -> dict:
    """Load storage microbench results and return tier latencies."""
    path = os.path.join(run_root, "storage_microbench.json")
    if not os.path.exists(path):
        print("WARNING: storage_microbench.json not found, using defaults")
        return {0: 0.0, 1: 50.0, 2: 5000.0}

    data = load_json(path)
    tier1_lat = 50.0  # default
    tier2_lat = 5000.0

    if "tmpfs" in data and "sequential" in data["tmpfs"]:
        r = data["tmpfs"]["sequential"]
        if "p50_us" in r:
            tier1_lat = r["p50_us"]

    if "sfs" in data and "sequential" in data["sfs"]:
        r = data["sfs"]["sequential"]
        if "p50_us" in r:
            tier2_lat = r["p50_us"]

    return {0: 0.0, 1: tier1_lat, 2: tier2_lat}


def run_router_only_metrics(
    traces: list[dict],
    q_vectors: np.ndarray,
    k_summaries: np.ndarray,
    summary_mode: str,
) -> dict:
    """Compute router-only recall/precision metrics."""
    num_blocks = k_summaries.shape[0]
    m_values = [2, 4, 8, 16]

    all_metrics = {f"recall@{m}": [] for m in m_values}
    all_metrics.update({f"precision@{m}": [] for m in m_values})
    all_metrics["score_separation"] = []

    for step_idx, trace in enumerate(traces):
        if step_idx >= len(q_vectors):
            break
        q = q_vectors[step_idx]
        scores = score_blocks_cosine(q, k_summaries)
        needed = trace["needed_blocks_mass"]
        sorted_idx = np.argsort(scores)[::-1]

        metrics = compute_router_metrics(
            sorted_idx.tolist(), needed, scores, num_blocks, m_values
        )
        for k, v in metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)

    # Average across steps
    result = {}
    for k, vals in all_metrics.items():
        if vals:
            result[k] = float(np.mean(vals))
    result["summary_mode"] = summary_mode
    return result


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
    num_prefix_blocks = prefix_length // block_size

    print("=" * 60)
    print("QK Router Phase 5-10: Replay Ablation")
    print("=" * 60)

    # Load storage bench
    tier_latencies = load_storage_bench(run_root)
    print(f"Tier latencies (us): {tier_latencies}")

    # Storage regimes
    storage_regimes = {
        "storage_mild": {"tier1_budget": 28, "tier2_budget": 4},
        "storage_medium": {"tier1_budget": 16, "tier2_budget": 16},
        "storage_harsh": {"tier1_budget": 4, "tier2_budget": 28},
    }

    # Summary modes
    summary_modes = [
        "direct_centroid",
        "random_summary",
        "first_k_real",
        "sampled_real_geometry",
    ]

    # Policies
    policies = [
        "no_prefetch",
        "recency_only_top_m",
        "semantic_top_m",
        "utility_aware",
        "utility_aware_plus_exploration",
    ]

    # Load K summaries
    k_summaries_all = {}
    for mode in summary_modes:
        path = os.path.join(run_root, "k_summaries", f"{mode}.npy")
        if os.path.exists(path):
            k_summaries_all[mode] = np.load(path)
            print(f"Loaded K-summary: {mode} {k_summaries_all[mode].shape}")
        else:
            print(f"WARNING: K-summary not found: {path}")

    # === ROUTER-ONLY METRICS ===
    print("\n" + "=" * 40)
    print("Router-Only Metrics by Summary Mode")
    print("=" * 40)

    router_results = {}
    for mode in summary_modes:
        if mode not in k_summaries_all:
            continue
        k_sum = k_summaries_all[mode]
        mode_metrics_all = []

        for req_idx in range(min(num_requests, 16)):  # sample for speed
            trace_path = os.path.join(
                run_root, "traces", f"trace_req{req_idx:03d}.json"
            )
            q_path = os.path.join(run_root, "q_vectors", f"q_req{req_idx:03d}.npy")
            if not os.path.exists(trace_path) or not os.path.exists(q_path):
                continue
            traces = load_json(trace_path)
            q_vecs = np.load(q_path)
            metrics = run_router_only_metrics(traces, q_vecs, k_sum, mode)
            mode_metrics_all.append(metrics)

        if mode_metrics_all:
            avg = {}
            for k in mode_metrics_all[0]:
                if k == "summary_mode":
                    avg[k] = mode
                    continue
                vals = [m[k] for m in mode_metrics_all if k in m]
                if vals:
                    avg[k] = float(np.mean(vals))
            router_results[mode] = avg
            print(
                f"  {mode}: recall@8={avg.get('recall@8', 0):.3f} "
                f"precision@8={avg.get('precision@8', 0):.3f} "
                f"separation={avg.get('score_separation', 0):.4f}"
            )

    save_json(router_results, os.path.join(run_root, "router_only_metrics.json"))

    # === REPLAY ABLATIONS ===
    print("\n" + "=" * 40)
    print("Replay Ablations")
    print("=" * 40)

    all_replay_results = {}

    for regime_name, regime_cfg in storage_regimes.items():
        print(f"\n--- {regime_name} ---")

        # Build tier assignments
        blocks = build_block_map(
            prefix_length, block_size, num_layers, num_kv_heads, head_dim
        )
        assign_tiers(blocks, regime_cfg["tier1_budget"], regime_cfg["tier2_budget"])
        block_tiers = [b.tier for b in blocks]
        initial_resident = set()  # nothing GPU-resident initially

        regime_results = {}

        for policy in policies:
            # Use direct_centroid for semantic policies
            summary_mode = "direct_centroid"
            k_sum = k_summaries_all.get(summary_mode)

            policy_replay_results = []
            for req_idx in range(num_requests):
                trace_path = os.path.join(
                    run_root, "traces", f"trace_req{req_idx:03d}.json"
                )
                q_path = os.path.join(run_root, "q_vectors", f"q_req{req_idx:03d}.npy")
                if not os.path.exists(trace_path) or not os.path.exists(q_path):
                    continue
                traces = load_json(trace_path)
                q_vecs = np.load(q_path)

                result = replay_traces(
                    traces,
                    q_vecs,
                    k_sum,
                    block_tiers,
                    tier_latencies,
                    total_block_bytes,
                    policy=policy,
                    initial_resident=initial_resident,
                )
                policy_replay_results.append(result)

            if policy_replay_results:
                agg = aggregate_replay_results(policy_replay_results)
                if policy in agg:
                    regime_results[policy] = agg[policy]
                    r = agg[policy]
                    print(
                        f"  {policy}: missed={r['avg_missed_rate']:.3f} "
                        f"stall={r['avg_stall_us']:.0f}us "
                        f"p95={r['avg_p95_decode_us']:.0f}us "
                        f"overlap={r['avg_overlap_frac']:.3f}"
                    )

        all_replay_results[regime_name] = regime_results

    save_json(all_replay_results, os.path.join(run_root, "replay_results.json"))

    # === SUMMARY MODE ABLATION ===
    print("\n" + "=" * 40)
    print("Summary Mode Ablation (storage_medium, semantic_top_m)")
    print("=" * 40)

    summary_ablation_results = {}
    regime_cfg = storage_regimes["storage_medium"]
    blocks = build_block_map(
        prefix_length, block_size, num_layers, num_kv_heads, head_dim
    )
    assign_tiers(blocks, regime_cfg["tier1_budget"], regime_cfg["tier2_budget"])
    block_tiers = [b.tier for b in blocks]

    for mode in summary_modes:
        k_sum = k_summaries_all.get(mode)
        if k_sum is None:
            continue

        policy_replay_results = []
        for req_idx in range(num_requests):
            trace_path = os.path.join(
                run_root, "traces", f"trace_req{req_idx:03d}.json"
            )
            q_path = os.path.join(run_root, "q_vectors", f"q_req{req_idx:03d}.npy")
            if not os.path.exists(trace_path) or not os.path.exists(q_path):
                continue
            traces = load_json(trace_path)
            q_vecs = np.load(q_path)

            result = replay_traces(
                traces,
                q_vecs,
                k_sum,
                block_tiers,
                tier_latencies,
                total_block_bytes,
                policy="semantic_top_m",
                initial_resident=set(),
            )
            policy_replay_results.append(result)

        if policy_replay_results:
            agg = aggregate_replay_results(policy_replay_results)
            if "semantic_top_m" in agg:
                summary_ablation_results[mode] = agg["semantic_top_m"]
                r = agg["semantic_top_m"]
                print(
                    f"  {mode}: missed={r['avg_missed_rate']:.3f} "
                    f"stall={r['avg_stall_us']:.0f}us "
                    f"p95={r['avg_p95_decode_us']:.0f}us"
                )

    save_json(
        summary_ablation_results,
        os.path.join(run_root, "summary_ablation_results.json"),
    )

    # === REUSE DISTANCE ===
    print("\n" + "=" * 40)
    print("Reuse Distance Analysis")
    print("=" * 40)

    all_traces_flat = []
    for req_idx in range(num_requests):
        trace_path = os.path.join(run_root, "traces", f"trace_req{req_idx:03d}.json")
        if os.path.exists(trace_path):
            traces = load_json(trace_path)
            all_traces_flat.extend(traces)

    reuse_data = compute_reuse_distance_histogram(all_traces_flat, num_prefix_blocks)
    save_json(reuse_data, os.path.join(run_root, "reuse_distance.json"))
    if "mean" in reuse_data:
        print(
            f"  Mean reuse distance: {reuse_data['mean']:.1f} steps, "
            f"median: {reuse_data['median']:.1f}, "
            f"p95: {reuse_data['p95']:.1f}"
        )

    print("\nPhase 5-10 complete.")
    print(f"Results saved to {run_root}")


if __name__ == "__main__":
    main()
