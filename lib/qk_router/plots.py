"""Plotting utilities for QK Router experiments."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_recall_by_summary(results: dict, plot_root: str, m_values=None):
    """Plot router recall@M by summary mode."""
    if m_values is None:
        m_values = [2, 4, 8, 16]

    fig, axes = plt.subplots(1, len(m_values), figsize=(4 * len(m_values), 4))
    if len(m_values) == 1:
        axes = [axes]

    for ax, m in zip(axes, m_values):
        modes = sorted(results.keys())
        vals = [results[mode].get(f"recall@{m}", 0) for mode in modes]
        bars = ax.bar(
            range(len(modes)), vals, color=["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
        )
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"Recall@{m}")
        ax.set_title(f"Recall@{m}")
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    path = os.path.join(plot_root, "recall_by_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_precision_by_summary(results: dict, plot_root: str, m_values=None):
    """Plot precision@M by summary mode."""
    if m_values is None:
        m_values = [2, 4, 8, 16]

    fig, axes = plt.subplots(1, len(m_values), figsize=(4 * len(m_values), 4))
    if len(m_values) == 1:
        axes = [axes]

    for ax, m in zip(axes, m_values):
        modes = sorted(results.keys())
        vals = [results[mode].get(f"precision@{m}", 0) for mode in modes]
        bars = ax.bar(
            range(len(modes)), vals, color=["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
        )
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"Precision@{m}")
        ax.set_title(f"Precision@{m}")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(plot_root, "precision_by_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_p95_latency_by_policy(results: dict, plot_root: str):
    """Plot p95 simulated decode latency by policy and storage regime."""
    fig, ax = plt.subplots(figsize=(10, 5))

    regimes = sorted(results.keys())
    policies = sorted(set(p for r in results.values() for p in r.keys()))
    x = np.arange(len(policies))
    width = 0.8 / len(regimes)
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    for i, regime in enumerate(regimes):
        vals = [
            results[regime].get(p, {}).get("avg_p95_decode_us", 0) for p in policies
        ]
        ax.bar(x + i * width, vals, width, label=regime, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(regimes) - 1) / 2)
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("p95 Decode Latency (us)")
    ax.set_title("p95 Simulated Decode Latency by Policy and Storage Regime")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plot_root, "p95_latency_by_policy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_wasted_bandwidth(results: dict, plot_root: str):
    """Plot wasted bandwidth by policy."""
    fig, ax = plt.subplots(figsize=(8, 4))
    policies = sorted(results.keys())
    vals = [results[p].get("avg_wasted_rate", 0) for p in policies]
    bars = ax.bar(range(len(policies)), vals, color="#FF5722")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Wasted Prefetch Rate")
    ax.set_title("Wasted Bandwidth by Policy")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    path = os.path.join(plot_root, "wasted_bandwidth.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_fetch_overlap(results: dict, plot_root: str):
    """Plot fetch overlap achieved by policy."""
    fig, ax = plt.subplots(figsize=(8, 4))
    policies = sorted(results.keys())
    vals = [results[p].get("avg_overlap_frac", 0) for p in policies]
    bars = ax.bar(range(len(policies)), vals, color="#2196F3")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg Fetch Overlap")
    ax.set_title("Fetch Overlap Achieved by Policy")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(plot_root, "fetch_overlap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_recency_vs_semantic(
    recency_results: dict, semantic_results: dict, plot_root: str
):
    """Plot recency-only vs semantic recall comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))
    m_values = [2, 4, 8, 16]
    rec_vals = [recency_results.get(f"recall@{m}", 0) for m in m_values]
    sem_vals = [semantic_results.get(f"recall@{m}", 0) for m in m_values]

    x = np.arange(len(m_values))
    width = 0.35
    ax.bar(x - width / 2, rec_vals, width, label="recency_only", color="#FF9800")
    ax.bar(x + width / 2, sem_vals, width, label="semantic", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{m}" for m in m_values])
    ax.set_ylabel("Recall")
    ax.set_title("Recency-Only vs Semantic Recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plot_root, "recency_vs_semantic.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_reuse_distance_histogram(reuse_data: dict, plot_root: str):
    """Plot block reuse-distance histogram."""
    fig, ax = plt.subplots(figsize=(8, 4))
    hist = reuse_data.get("histogram", {})
    if not hist:
        return None

    labels = list(hist.keys())
    values = list(hist.values())
    ax.bar(range(len(labels)), values, color="#9C27B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Block Reuse-Distance Histogram")
    plt.tight_layout()
    path = os.path.join(plot_root, "reuse_distance_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
