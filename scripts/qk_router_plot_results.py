#!/usr/bin/env python3
"""Generate all required plots for QK Router experiment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.qk_router.plots import (
    plot_recall_by_summary,
    plot_precision_by_summary,
    plot_p95_latency_by_policy,
    plot_wasted_bandwidth,
    plot_fetch_overlap,
    plot_recency_vs_semantic,
    plot_reuse_distance_histogram,
)
from lib.qk_router.utils import load_json


def main():
    run_root = os.environ.get("RUN_ROOT", "/mnt/tmpfs/knlp/results/qk_router_01")
    plot_root = os.environ.get("PLOT_ROOT", "/mnt/tmpfs/knlp/plots/qk_router_01")
    os.makedirs(plot_root, exist_ok=True)

    print("=" * 60)
    print("QK Router: Generating Plots")
    print("=" * 60)

    # 1. Router recall by summary mode
    router_path = os.path.join(run_root, "router_only_metrics.json")
    if os.path.exists(router_path):
        router = load_json(router_path)
        p = plot_recall_by_summary(router, plot_root)
        print(f"  Recall by summary: {p}")
        p = plot_precision_by_summary(router, plot_root)
        print(f"  Precision by summary: {p}")

    # 2. p95 latency by policy and regime
    replay_path = os.path.join(run_root, "replay_results.json")
    if os.path.exists(replay_path):
        replay = load_json(replay_path)
        p = plot_p95_latency_by_policy(replay, plot_root)
        print(f"  p95 latency: {p}")

        # Wasted bandwidth (use storage_medium)
        if "storage_medium" in replay:
            p = plot_wasted_bandwidth(replay["storage_medium"], plot_root)
            print(f"  Wasted bandwidth: {p}")
            p = plot_fetch_overlap(replay["storage_medium"], plot_root)
            print(f"  Fetch overlap: {p}")

            # Recency vs semantic comparison
            rec = replay["storage_medium"].get("recency_only_top_m", {})
            sem = replay["storage_medium"].get("semantic_top_m", {})
            if rec and sem:
                # Need router-level recall, use router_only_metrics
                if os.path.exists(router_path):
                    router = load_json(router_path)
                    rec_router = router.get(
                        "first_k_real", router.get("random_summary", {})
                    )
                    sem_router = router.get("direct_centroid", {})
                    p = plot_recency_vs_semantic(rec_router, sem_router, plot_root)
                    print(f"  Recency vs semantic: {p}")

    # 3. Reuse distance histogram
    reuse_path = os.path.join(run_root, "reuse_distance.json")
    if os.path.exists(reuse_path):
        reuse = load_json(reuse_path)
        p = plot_reuse_distance_histogram(reuse, plot_root)
        if p:
            print(f"  Reuse distance: {p}")

    print("\nPlots complete.")


if __name__ == "__main__":
    main()
