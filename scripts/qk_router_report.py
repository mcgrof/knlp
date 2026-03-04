#!/usr/bin/env python3
"""Phase 13: Generate final report for QK Router experiment."""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.qk_router.utils import load_json


def generate_report(run_root: str, doc_path: str):
    """Generate the final markdown report."""

    # Load all results
    env = (
        load_json(os.path.join(run_root, "environment.json"))
        if os.path.exists(os.path.join(run_root, "environment.json"))
        else {}
    )
    manifest = (
        load_json(os.path.join(run_root, "manifest.json"))
        if os.path.exists(os.path.join(run_root, "manifest.json"))
        else {}
    )
    storage = (
        load_json(os.path.join(run_root, "storage_microbench.json"))
        if os.path.exists(os.path.join(run_root, "storage_microbench.json"))
        else {}
    )
    trace_summary = (
        load_json(os.path.join(run_root, "trace_summary.json"))
        if os.path.exists(os.path.join(run_root, "trace_summary.json"))
        else {}
    )
    router_metrics = (
        load_json(os.path.join(run_root, "router_only_metrics.json"))
        if os.path.exists(os.path.join(run_root, "router_only_metrics.json"))
        else {}
    )
    replay_results = (
        load_json(os.path.join(run_root, "replay_results.json"))
        if os.path.exists(os.path.join(run_root, "replay_results.json"))
        else {}
    )
    summary_ablation = (
        load_json(os.path.join(run_root, "summary_ablation_results.json"))
        if os.path.exists(os.path.join(run_root, "summary_ablation_results.json"))
        else {}
    )
    reuse_data = (
        load_json(os.path.join(run_root, "reuse_distance.json"))
        if os.path.exists(os.path.join(run_root, "reuse_distance.json"))
        else {}
    )

    lines = []
    lines.append("# QK Router 01: Signal Check Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 1. Objective
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "Determine whether current decode-step queries Q_t, scored against "
        "a resident geometry-aware K-summary index, can predict which "
        "offloaded KV blocks should be prefetched, and whether this "
        "semantic routing beats trivial baselines."
    )
    lines.append("")

    # 2. Environment
    lines.append("## 2. Environment")
    lines.append("")
    lines.append(
        f"- GPU: {env.get('gpu_name', 'N/A')} ({env.get('gpu_memory_gb', 0)}GB)"
    )
    lines.append(f"- PyTorch: {env.get('torch_version', 'N/A')}")
    lines.append(f"- CUDA: {env.get('cuda_version', 'N/A')}")
    lines.append(f"- Transformers: {env.get('transformers_version', 'N/A')}")
    lines.append(f"- Git commit: {env.get('git_commit', 'N/A')}")
    lines.append("")

    # 3. Model and workload
    lines.append("## 3. Model and Workload")
    lines.append("")
    if manifest:
        m = manifest.get("model", {})
        w = manifest.get("workload", {})
        lines.append(f"- Model: {m.get('name', 'N/A')}")
        lines.append(f"- Layers: {m.get('num_layers', 'N/A')}")
        lines.append(f"- KV heads: {m.get('num_kv_heads', 'N/A')}")
        lines.append(f"- Head dim: {m.get('head_dim', 'N/A')}")
        lines.append(f"- Prefix length: {w.get('prefix_length', 'N/A')}")
        lines.append(f"- Block size: {w.get('block_size', 'N/A')} tokens")
        lines.append(f"- Num prefix blocks: {w.get('num_prefix_blocks', 'N/A')}")
        lines.append(f"- Max new tokens: {w.get('max_new_tokens', 'N/A')}")
        lines.append(f"- Num requests: {w.get('num_requests', 'N/A')}")
        lines.append(f"- Dataset: {w.get('dataset', 'N/A')}")
    lines.append("")

    # 4. Storage microbench
    lines.append("## 4. Storage Microbench Results")
    lines.append("")
    if storage:
        lines.append("| Tier | Path | p50 (us) | p95 (us) | Throughput (MB/s) |")
        lines.append("|------|------|----------|----------|-------------------|")
        for tier, label in [("tmpfs", "/mnt/tmpfs"), ("sfs", "/mnt/SFS-hugging")]:
            if tier in storage and "sequential" in storage[tier]:
                r = storage[tier]["sequential"]
                if "error" not in r:
                    lines.append(
                        f"| {tier} | {label} | "
                        f"{r.get('p50_us', 0):.0f} | "
                        f"{r.get('p95_us', 0):.0f} | "
                        f"{r.get('throughput_mb_s', 0):.1f} |"
                    )
                else:
                    lines.append(f"| {tier} | {label} | ERROR | ERROR | ERROR |")
    lines.append("")

    # 5. Trace construction
    lines.append("## 5. Trace Construction")
    lines.append("")
    if trace_summary:
        lines.append(f"- Requests: {trace_summary.get('num_requests', 'N/A')}")
        lines.append(
            f"- Steps per request: {trace_summary.get('max_new_tokens', 'N/A')}"
        )
        per_req = trace_summary.get("per_request", [])
        if per_req:
            avg_needed = sum(r["avg_needed_blocks"] for r in per_req) / len(per_req)
            lines.append(f"- Avg needed blocks per step: {avg_needed:.1f}")
    lines.append("")

    # 6. K-summary construction
    lines.append("## 6. K-Summary Construction Methods")
    lines.append("")
    lines.append("| Mode | Description |")
    lines.append("|------|-------------|")
    lines.append(
        "| direct_centroid | Per-layer centroid of real block keys, normalized |"
    )
    lines.append("| random_summary | Random normalized vectors (dumb baseline) |")
    lines.append("| first_k_real | Summaries from first 4 blocks only |")
    lines.append("| sampled_real_geometry | Keys sampled across full prefix span |")
    lines.append("")

    # 7. Router-only results
    lines.append("## 7. Router-Only Results")
    lines.append("")
    if router_metrics:
        lines.append(
            "| Mode | Recall@2 | Recall@4 | Recall@8 | Recall@16 | Score Sep |"
        )
        lines.append(
            "|------|----------|----------|----------|-----------|-----------|"
        )
        for mode in [
            "direct_centroid",
            "random_summary",
            "first_k_real",
            "sampled_real_geometry",
        ]:
            if mode in router_metrics:
                r = router_metrics[mode]
                lines.append(
                    f"| {mode} | "
                    f"{r.get('recall@2', 0):.3f} | "
                    f"{r.get('recall@4', 0):.3f} | "
                    f"{r.get('recall@8', 0):.3f} | "
                    f"{r.get('recall@16', 0):.3f} | "
                    f"{r.get('score_separation', 0):.4f} |"
                )
    lines.append("")

    # 8. Replay/scheduler results
    lines.append("## 8. Replay / Scheduler Results")
    lines.append("")
    for regime in ["storage_mild", "storage_medium", "storage_harsh"]:
        if regime in replay_results:
            lines.append(f"### {regime}")
            lines.append("")
            lines.append(
                "| Policy | Missed Rate | Avg Stall (us) | p95 Decode (us) | Overlap |"
            )
            lines.append(
                "|--------|-------------|----------------|-----------------|---------|"
            )
            for policy in [
                "no_prefetch",
                "recency_only_top_m",
                "semantic_top_m",
                "utility_aware",
                "utility_aware_plus_exploration",
            ]:
                if policy in replay_results[regime]:
                    r = replay_results[regime][policy]
                    lines.append(
                        f"| {policy} | "
                        f"{r.get('avg_missed_rate', 0):.3f} | "
                        f"{r.get('avg_stall_us', 0):.0f} | "
                        f"{r.get('avg_p95_decode_us', 0):.0f} | "
                        f"{r.get('avg_overlap_frac', 0):.3f} |"
                    )
            lines.append("")

    # 9. Comparison against recency-only
    lines.append("## 9. Recency-Only vs Semantic Comparison")
    lines.append("")
    if "storage_medium" in replay_results:
        r_rec = replay_results["storage_medium"].get("recency_only_top_m", {})
        r_sem = replay_results["storage_medium"].get("semantic_top_m", {})
        r_util = replay_results["storage_medium"].get(
            "utility_aware_plus_exploration", {}
        )
        if r_rec and r_sem:
            rec_miss = r_rec.get("avg_missed_rate", 0)
            sem_miss = r_sem.get("avg_missed_rate", 0)
            delta = rec_miss - sem_miss
            lines.append(f"- Recency missed rate: {rec_miss:.3f}")
            lines.append(f"- Semantic missed rate: {sem_miss:.3f}")
            lines.append(f"- Delta (recency - semantic): {delta:+.3f}")
            if r_util:
                util_miss = r_util.get("avg_missed_rate", 0)
                lines.append(f"- Utility+exploration missed rate: {util_miss:.3f}")
            lines.append("")
            if delta > 0.05:
                lines.append("Semantic routing significantly outperforms recency-only.")
            elif delta > 0.01:
                lines.append("Semantic routing modestly outperforms recency-only.")
            elif delta > -0.01:
                lines.append("Semantic routing performs comparably to recency-only.")
            else:
                lines.append("Recency-only outperforms semantic routing.")
    lines.append("")

    # 10. Geometry-aware summary findings
    lines.append("## 10. Geometry-Aware Summary Findings")
    lines.append("")
    if summary_ablation:
        lines.append("| Mode | Missed Rate | Avg Stall (us) | p95 Decode (us) |")
        lines.append("|------|-------------|----------------|-----------------|")
        for mode in [
            "direct_centroid",
            "random_summary",
            "first_k_real",
            "sampled_real_geometry",
        ]:
            if mode in summary_ablation:
                r = summary_ablation[mode]
                lines.append(
                    f"| {mode} | "
                    f"{r.get('avg_missed_rate', 0):.3f} | "
                    f"{r.get('avg_stall_us', 0):.0f} | "
                    f"{r.get('avg_p95_decode_us', 0):.0f} |"
                )
        lines.append("")
        dc = summary_ablation.get("direct_centroid", {})
        rs = summary_ablation.get("random_summary", {})
        if dc and rs:
            dc_miss = dc.get("avg_missed_rate", 0)
            rs_miss = rs.get("avg_missed_rate", 0)
            delta = rs_miss - dc_miss
            if delta > 0.05:
                lines.append(
                    "Geometry-aware summaries (direct_centroid) strongly outperform random."
                )
            elif delta > 0.01:
                lines.append("Geometry-aware summaries modestly outperform random.")
            else:
                lines.append(
                    "Geometry-aware summaries show no meaningful advantage over random."
                )
    lines.append("")

    # 12. Limitations
    lines.append("## 12. Limitations")
    lines.append("")
    lines.append("- Single model (Qwen2.5-0.5B, 494M params)")
    lines.append("- Single context length (4K prefix)")
    lines.append("- Simulator-only (no actual block fault-in)")
    lines.append("- Greedy decoding only")
    lines.append("- No learned router (plain QK retrieval only)")
    lines.append("- 128-token blocks only (no 256-token comparison)")
    lines.append("")

    # 13. Recommendation
    lines.append("## 13. Recommendation")
    lines.append("")
    # Determine recommendation from data
    if "storage_medium" in replay_results:
        r_rec = replay_results["storage_medium"].get("recency_only_top_m", {})
        r_sem = replay_results["storage_medium"].get("semantic_top_m", {})
        r_util = replay_results["storage_medium"].get(
            "utility_aware_plus_exploration", {}
        )

        rec_miss = r_rec.get("avg_missed_rate", 1) if r_rec else 1
        sem_miss = r_sem.get("avg_missed_rate", 1) if r_sem else 1
        util_miss = r_util.get("avg_missed_rate", 1) if r_util else 1

        best_sem = min(sem_miss, util_miss)
        delta = rec_miss - best_sem

        dc_recall = router_metrics.get("direct_centroid", {}).get("recall@8", 0)
        rs_recall = router_metrics.get("random_summary", {}).get("recall@8", 0)
        geo_delta = dc_recall - rs_recall

        if delta > 0.1 and geo_delta > 0.1:
            recommendation = "GO"
            reason = (
                f"Semantic routing beats recency-only by {delta:.1%} on missed rate. "
                f"Geometry-aware summaries beat random by {geo_delta:.1%} on recall@8. "
                "Clear signal worth pursuing."
            )
        elif delta > 0.03 or geo_delta > 0.05:
            recommendation = "MAYBE"
            reason = (
                f"Semantic routing beats recency-only by {delta:.1%}. "
                f"Geometry delta: {geo_delta:.1%}. "
                "Signal exists but may be narrow."
            )
        else:
            recommendation = "NO-GO"
            reason = (
                f"Semantic routing delta vs recency: {delta:.1%}. "
                f"Geometry delta: {geo_delta:.1%}. "
                "Insufficient signal to justify further work."
            )
    else:
        recommendation = "INCONCLUSIVE"
        reason = "Insufficient data to make recommendation."

    lines.append(f"**{recommendation}**: {reason}")
    lines.append("")

    # Write report
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {doc_path}")

    # Summary JSON
    summary = {
        "recommendation": recommendation,
        "reason": reason,
        "router_recall_at_8": {
            mode: router_metrics.get(mode, {}).get("recall@8", None)
            for mode in [
                "direct_centroid",
                "random_summary",
                "first_k_real",
                "sampled_real_geometry",
            ]
        },
        "replay_medium": {
            policy: replay_results.get("storage_medium", {})
            .get(policy, {})
            .get("avg_missed_rate", None)
            for policy in [
                "no_prefetch",
                "recency_only_top_m",
                "semantic_top_m",
                "utility_aware",
                "utility_aware_plus_exploration",
            ]
        },
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(run_root, "summary.json")
    import json

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    return recommendation


def main():
    run_root = os.environ.get("RUN_ROOT", "/mnt/tmpfs/knlp/results/qk_router_01")
    doc_path = os.environ.get("DOC_PATH", "/mnt/tmpfs/knlp/docs/qk_router_01.md")
    recommendation = generate_report(run_root, doc_path)
    print(f"\nFinal recommendation: {recommendation}")


if __name__ == "__main__":
    main()
