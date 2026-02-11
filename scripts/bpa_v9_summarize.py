#!/usr/bin/env python
"""
BPA v9 summary: one-command report from decode benchmark results.

Usage:
    python scripts/bpa_v9_summarize.py bpa_v9_results
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np


def load_results(results_dir):
    path = os.path.join(results_dir, "decode_results.json")
    with open(path) as f:
        return json.load(f)


def aggregate(results):
    """Aggregate across seeds for each (method, seq_len, stress_mode)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["method"], r["seq_len"], r.get("stress_mode", "control"))
        groups[key].append(r)

    agg = []
    for (method, seq_len, stress), runs in sorted(groups.items()):
        agg.append(
            {
                "method": method,
                "seq_len": seq_len,
                "stress_mode": stress,
                "n_seeds": len(runs),
                "prefill_ms": float(np.mean([r["prefill_ms"] for r in runs])),
                "decode_per_token_ms": float(
                    np.mean([r["decode_per_token_ms"] for r in runs])
                ),
                "decode_p95_ms": float(np.mean([r["decode_p95_ms"] for r in runs])),
                "gate_pct_of_total": float(
                    np.mean([r["gate_pct_of_total"] for r in runs])
                ),
                "ppl": float(np.mean([r["ppl"] for r in runs])),
                "ppl_std": float(np.std([r["ppl"] for r in runs])),
                "kv_bytes_read_per_token": float(
                    np.mean([r["kv_bytes_read_per_token"] for r in runs])
                ),
                "kv_kept_mean": float(np.mean([r["kv_kept_mean"] for r in runs])),
                "throughput_toks_per_sec": float(
                    np.mean([r["throughput_toks_per_sec"] for r in runs])
                ),
                "peak_mem_mb": float(np.mean([r["peak_mem_mb"] for r in runs])),
                "W_mean": float(np.mean([r.get("W_mean", 0) for r in runs])),
                "B_far_mean": float(np.mean([r.get("B_far_mean", 0) for r in runs])),
                "region_ppl_early": float(
                    np.mean([r.get("region_ppl_early", 0) for r in runs])
                ),
                "region_ppl_late": float(
                    np.mean([r.get("region_ppl_late", 0) for r in runs])
                ),
            }
        )
    return agg


def print_main_table(agg):
    """Print the main comparison table."""
    control = [a for a in agg if a["stress_mode"] == "control"]
    if not control:
        return

    print("\n" + "=" * 120)
    print("MAIN RESULTS: Dense vs BPA v9 vs Static Top-K (control)")
    print("=" * 120)
    print(
        f"{'Method':15s} {'L':>5s} {'Prefill':>8s} {'Decode':>8s} "
        f"{'p95':>8s} {'Gate%':>6s} {'PPL':>10s} "
        f"{'KV_kept':>8s} {'KV_MB/tok':>9s} {'KV_ratio':>8s} "
        f"{'tok/s':>7s}"
    )
    print("-" * 120)

    for L in sorted(set(a["seq_len"] for a in control)):
        dense = [a for a in control if a["method"] == "dense" and a["seq_len"] == L]
        if not dense:
            continue
        dense_kv = dense[0]["kv_bytes_read_per_token"]
        dense_ppl = dense[0]["ppl"]

        for a in sorted(
            [x for x in control if x["seq_len"] == L],
            key=lambda x: x["method"],
        ):
            kv_mb = a["kv_bytes_read_per_token"] / 1e6
            kv_ratio = a["kv_bytes_read_per_token"] / dense_kv if dense_kv > 0 else 0
            ppl_delta = (a["ppl"] - dense_ppl) / dense_ppl * 100 if dense_ppl > 0 else 0
            ppl_str = f"{a['ppl']:.0f} ({ppl_delta:+.1f}%)"
            print(
                f"{a['method']:15s} {L:5d} "
                f"{a['prefill_ms']:7.0f}ms "
                f"{a['decode_per_token_ms']:7.2f}ms "
                f"{a['decode_p95_ms']:7.2f}ms "
                f"{a['gate_pct_of_total']:5.1f}% "
                f"{ppl_str:>10s} "
                f"{a['kv_kept_mean']:7.0f} "
                f"{kv_mb:8.2f} "
                f"{kv_ratio:7.2f}x "
                f"{a['throughput_toks_per_sec']:6.0f}"
            )
        print()


def print_stress_table(agg):
    """Print stress test comparison."""
    stress = [a for a in agg if a["stress_mode"] != "control"]
    if not stress:
        return

    print("\n" + "=" * 100)
    print("STRESS TESTS: BPA v9 Adaptive Behavior")
    print("=" * 100)
    print(
        f"{'Method':15s} {'L':>5s} {'Stress':>15s} {'PPL':>10s} "
        f"{'Early PPL':>10s} {'Late PPL':>10s} {'KV_kept':>8s} "
        f"{'W_mean':>7s} {'B_far':>6s}"
    )
    print("-" * 100)

    for L in sorted(set(a["seq_len"] for a in stress)):
        for sm in sorted(set(a["stress_mode"] for a in stress)):
            for a in sorted(
                [x for x in stress if x["seq_len"] == L and x["stress_mode"] == sm],
                key=lambda x: x["method"],
            ):
                print(
                    f"{a['method']:15s} {L:5d} {sm:>15s} "
                    f"{a['ppl']:9.0f} "
                    f"{a['region_ppl_early']:9.0f} "
                    f"{a['region_ppl_late']:9.0f} "
                    f"{a['kv_kept_mean']:7.0f} "
                    f"{a['W_mean']:6.0f} "
                    f"{a['B_far_mean']:5.1f}"
                )
        print()


def print_adaptive_summary(agg):
    """Print adaptive controller behavior."""
    bpa = [a for a in agg if a["method"] == "bpa_v9" and a["stress_mode"] == "control"]
    if not bpa:
        return

    print("\n" + "=" * 80)
    print("ADAPTIVE CONTROLLER BEHAVIOR (bpa_v9, control)")
    print("=" * 80)
    print(
        f"{'L':>5s} {'W_mean':>8s} {'B_far':>8s} "
        f"{'KV_kept':>8s} {'KV_savings':>10s}"
    )
    print("-" * 80)

    for a in sorted(bpa, key=lambda x: x["seq_len"]):
        dense = [
            x
            for x in agg
            if x["method"] == "dense"
            and x["seq_len"] == a["seq_len"]
            and x["stress_mode"] == "control"
        ]
        if dense:
            savings = (
                1 - a["kv_bytes_read_per_token"] / dense[0]["kv_bytes_read_per_token"]
            ) * 100
        else:
            savings = 0
        print(
            f"{a['seq_len']:5d} "
            f"{a['W_mean']:7.0f} "
            f"{a['B_far_mean']:7.1f} "
            f"{a['kv_kept_mean']:7.0f} "
            f"{savings:9.1f}%"
        )


def generate_report(agg, results_dir):
    """Generate bpa_v9_final_report.md."""
    report = []
    report.append("# BPA v9: Decode Benchmark + Adaptive Controller\n")
    report.append(
        "> Goal: Prove end-to-end decode wins under realistic workloads "
        "with adaptive W(t) and B_far(t) control.\n"
    )

    report.append("## Model")
    report.append("- GPT2_RGSA (124M params), FineWebEdu, 615 iters")
    report.append("- Decode benchmark: prefill + autoregressive decode")
    report.append("- Position embedding interpolation for L > 1024\n")

    # Main results table
    control = [a for a in agg if a["stress_mode"] == "control"]
    if control:
        report.append("## Main Results: Decode Benchmark\n")
        report.append(
            "| Method | L | Prefill | Decode/tok | p95 | Gate% "
            "| PPL | PPL vs Dense | KV kept | KV MB/tok | KV ratio |"
        )
        report.append(
            "|--------|---|---------|-----------|-----|------"
            "|-----|-------------|---------|-----------|----------|"
        )

        for L in sorted(set(a["seq_len"] for a in control)):
            dense = [a for a in control if a["method"] == "dense" and a["seq_len"] == L]
            if not dense:
                continue
            dense_ppl = dense[0]["ppl"]
            dense_kv = dense[0]["kv_bytes_read_per_token"]

            for a in sorted(
                [x for x in control if x["seq_len"] == L],
                key=lambda x: x["method"],
            ):
                kv_mb = a["kv_bytes_read_per_token"] / 1e6
                kv_ratio = (
                    a["kv_bytes_read_per_token"] / dense_kv if dense_kv > 0 else 0
                )
                ppl_delta = (
                    (a["ppl"] - dense_ppl) / dense_ppl * 100 if dense_ppl > 0 else 0
                )
                report.append(
                    f"| {a['method']} | {L} "
                    f"| {a['prefill_ms']:.0f}ms "
                    f"| {a['decode_per_token_ms']:.2f}ms "
                    f"| {a['decode_p95_ms']:.2f}ms "
                    f"| {a['gate_pct_of_total']:.1f}% "
                    f"| {a['ppl']:.0f} "
                    f"| {ppl_delta:+.1f}% "
                    f"| {a['kv_kept_mean']:.0f} "
                    f"| {kv_mb:.2f} "
                    f"| {kv_ratio:.2f}x |"
                )
        report.append("")

    # Adaptive controller
    bpa = [a for a in agg if a["method"] == "bpa_v9" and a["stress_mode"] == "control"]
    if bpa:
        report.append("## Adaptive Controller Behavior\n")
        report.append("| L | W_mean | B_far_mean | KV kept | KV savings |")
        report.append("|---|--------|-----------|---------|-----------|")
        for a in sorted(bpa, key=lambda x: x["seq_len"]):
            dense_a = [
                x
                for x in control
                if x["method"] == "dense" and x["seq_len"] == a["seq_len"]
            ]
            savings = 0
            if dense_a:
                savings = (
                    1
                    - a["kv_bytes_read_per_token"]
                    / dense_a[0]["kv_bytes_read_per_token"]
                ) * 100
            report.append(
                f"| {a['seq_len']} | {a['W_mean']:.0f} "
                f"| {a['B_far_mean']:.1f} "
                f"| {a['kv_kept_mean']:.0f} "
                f"| {savings:.1f}% |"
            )
        report.append("")

    # Stress tests
    stress = [a for a in agg if a["stress_mode"] != "control"]
    if stress:
        report.append("## Stress Tests\n")
        for sm in sorted(set(a["stress_mode"] for a in stress)):
            report.append(f"### {sm}\n")
            report.append(
                "| Method | L | PPL | Early PPL | Late PPL "
                "| KV kept | W_mean | B_far |"
            )
            report.append(
                "|--------|---|-----|-----------|----------"
                "|---------|--------|-------|"
            )
            for a in sorted(
                [x for x in stress if x["stress_mode"] == sm],
                key=lambda x: (x["seq_len"], x["method"]),
            ):
                report.append(
                    f"| {a['method']} | {a['seq_len']} "
                    f"| {a['ppl']:.0f} "
                    f"| {a['region_ppl_early']:.0f} "
                    f"| {a['region_ppl_late']:.0f} "
                    f"| {a['kv_kept_mean']:.0f} "
                    f"| {a['W_mean']:.0f} "
                    f"| {a['B_far_mean']:.1f} |"
                )
            report.append("")

    # Conclusions placeholder
    report.append("## Conclusions\n")
    report.append("### (1) Does BPA v9 reduce KV traffic during decode?\n")
    report.append("### (2) Is adaptive W(t)/B_far(t) working?\n")
    report.append("### (3) Gate overhead acceptable?\n")
    report.append("### (4) Quality within tolerance?\n")

    report_text = "\n".join(report) + "\n"
    path = os.path.join(results_dir, "bpa_v9_final_report.md")
    with open(path, "w") as f:
        f.write(report_text)
    print(f"\nReport: {path}")


def main():
    results_dir = "bpa_v9_results"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    results = load_results(results_dir)
    print(f"Loaded {len(results)} results from {results_dir}")

    agg = aggregate(results)

    print_main_table(agg)
    print_stress_table(agg)
    print_adaptive_summary(agg)
    generate_report(agg, results_dir)


if __name__ == "__main__":
    main()
