#!/usr/bin/env python
"""
BPA v10 summary: generate report from decode benchmark results.

Usage:
    python scripts/bpa_v10_summarize.py bpa_v10_results
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np


TRAINED_MAX_CTX = 1024


def context_regime(L):
    return "in_range" if L <= TRAINED_MAX_CTX else "extrapolated"


def load_results(results_dir):
    path = os.path.join(results_dir, "decode_results_v10.json")
    with open(path) as f:
        return json.load(f)


def load_tuning_summary(results_dir):
    path = os.path.join(results_dir, "tuning_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def aggregate(results):
    """Aggregate across seeds for each (method, seq_len, stress_mode)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["method"], r["seq_len"], r.get("stress_mode", "control"))
        groups[key].append(r)

    agg = []
    for (method, seq_len, stress), runs in sorted(groups.items()):
        a = {
            "method": method,
            "seq_len": seq_len,
            "context_regime": context_regime(seq_len),
            "stress_mode": stress,
            "n_seeds": len(runs),
        }
        # Average all numeric fields
        float_keys = [
            "prefill_ms",
            "decode_per_token_ms",
            "decode_p95_ms",
            "gate_pct_of_total",
            "ppl",
            "kv_bytes_read_per_token",
            "kv_kept_mean",
            "throughput_toks_per_sec",
            "peak_cpu_rss_mb",
            "peak_gpu_alloc_mb",
            "peak_gpu_reserved_mb",
            "W_mean",
            "k_far_mean",
            "k_far_std",
            "pressure_mean",
            "B_far_raw_mean",
            "region_ppl_early",
            "region_ppl_late",
        ]
        for k in float_keys:
            vals = [r.get(k, 0) for r in runs]
            a[k] = float(np.mean(vals))

        a["ppl_std"] = float(np.std([r.get("ppl", 0) for r in runs]))
        # Integer fields: take max of max, min of min
        a["k_far_max"] = int(max(r.get("k_far_max", 0) for r in runs))
        a["k_far_min"] = int(min(r.get("k_far_min", 0) for r in runs))
        a["W_min_obs"] = int(min(r.get("W_min_obs", 0) for r in runs))
        a["W_max_obs"] = int(max(r.get("W_max_obs", 0) for r in runs))
        agg.append(a)
    return agg


def generate_report(agg, results_dir, tuning=None):
    """Generate bpa_v10_final_report.md."""
    report = []
    report.append("# BPA v10: Decode Benchmark with Matched-Quality Tuning\n")

    # Run meta from first result if available
    results_path = os.path.join(results_dir, "decode_results_v10.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            raw = json.load(f)
        if raw and "run_meta" in raw[0]:
            meta = raw[0]["run_meta"]
            report.append("## Environment\n")
            report.append(f"- Hostname: {meta.get('hostname', '?')}")
            report.append(f"- CPU: {meta.get('cpu', '?')}")
            report.append(f"- GPU: {meta.get('gpu', 'N/A')}")
            report.append(f"- Torch: {meta.get('torch_version', '?')}")
            report.append(f"- Device: {meta.get('device', '?')}")
            report.append(f"- Trained max ctx: {meta.get('trained_max_ctx', '?')}")
            report.append(f"- Git SHA: {meta.get('git_sha', '?')}\n")

    # Separate in_range and extrapolated
    in_range = [a for a in agg if a["context_regime"] == "in_range"]
    extrapolated = [a for a in agg if a["context_regime"] == "extrapolated"]

    # In-range headline tables
    control_ir = [a for a in in_range if a["stress_mode"] == "control"]
    if control_ir:
        report.append("## Headline Results (in_range, L <= 1024)\n")
        report.append(
            "| Method | L | PPL | PPL vs Dense | p50 (ms) | p95 (ms) "
            "| Gate% | KV kept | KV MB/tok | KV ratio "
            "| tok/s | CPU MB | GPU MB |"
        )
        report.append(
            "|--------|---|-----|-------------|----------|----------"
            "|-------|---------|-----------|----------"
            "|-------|--------|--------|"
        )
        for L in sorted(set(a["seq_len"] for a in control_ir)):
            dense = [
                a for a in control_ir if a["method"] == "dense" and a["seq_len"] == L
            ]
            if not dense:
                continue
            dp = dense[0]["ppl"]
            dk = dense[0]["kv_bytes_read_per_token"]

            for a in sorted(
                [x for x in control_ir if x["seq_len"] == L],
                key=lambda x: x["method"],
            ):
                kv_mb = a["kv_bytes_read_per_token"] / 1e6
                kv_ratio = a["kv_bytes_read_per_token"] / dk if dk > 0 else 0
                ppl_d = (a["ppl"] - dp) / dp * 100 if dp > 0 else 0
                report.append(
                    f"| {a['method']} | {L} "
                    f"| {a['ppl']:.0f} "
                    f"| {ppl_d:+.1f}% "
                    f"| {a['decode_per_token_ms']:.2f} "
                    f"| {a['decode_p95_ms']:.2f} "
                    f"| {a['gate_pct_of_total']:.1f}% "
                    f"| {a['kv_kept_mean']:.0f} "
                    f"| {kv_mb:.2f} "
                    f"| {kv_ratio:.2f}x "
                    f"| {a['throughput_toks_per_sec']:.0f} "
                    f"| {a['peak_cpu_rss_mb']:.0f} "
                    f"| {a['peak_gpu_alloc_mb']:.0f} |"
                )
        report.append("")

    # Tuning results
    if tuning:
        report.append("## Matched-Quality Tuning Results\n")
        report.append("| Method | L | Tol% | Status | PPL | PPL delta | p50 (ms) |")
        report.append("|--------|---|------|--------|-----|-----------|----------|")
        for s in tuning:
            method = s.get("method", "bpa_v10")
            L = s.get("L", "?")
            tol = s.get("tol_pct", "?")
            status = s.get("status", "?")
            if status == "PASS":
                m = s["metrics"]
                report.append(
                    f"| {method} | {L} | {tol}% | PASS "
                    f"| {m['ppl']:.0f} "
                    f"| {m['ppl_delta_pct']:+.1f}% "
                    f"| {m['decode_p50_ms']:.2f} |"
                )
            else:
                report.append(f"| {method} | {L} | {tol}% | **FAIL** | - | - | - |")
        report.append("")

    # Adaptivity proof
    stress = [a for a in agg if a["stress_mode"] != "control"]
    if stress:
        report.append("## Adaptivity Proof: Stress Tests\n")
        for sm in sorted(set(a["stress_mode"] for a in stress)):
            report.append(f"### {sm}\n")
            report.append(
                "| L | PPL | Easy PPL | Hard PPL "
                "| k_far_mean | k_far_max | W_mean | KV kept |"
            )
            report.append(
                "|---|-----|----------|----------"
                "|-----------|-----------|--------|---------|"
            )
            for a in sorted(
                [x for x in stress if x["stress_mode"] == sm],
                key=lambda x: x["seq_len"],
            ):
                report.append(
                    f"| {a['seq_len']} "
                    f"| {a['ppl']:.0f} "
                    f"| {a['region_ppl_early']:.0f} "
                    f"| {a['region_ppl_late']:.0f} "
                    f"| {a['k_far_mean']:.1f} "
                    f"| {a['k_far_max']} "
                    f"| {a['W_mean']:.0f} "
                    f"| {a['kv_kept_mean']:.0f} |"
                )
            report.append("")

    # Extrapolated appendix
    control_ex = [a for a in extrapolated if a["stress_mode"] == "control"]
    if control_ex:
        report.append("## Appendix: Extrapolated Results (L > 1024)\n")
        report.append(
            "> These results use position embedding interpolation "
            "and should not be used as headline numbers.\n"
        )
        report.append(
            "| Method | L | PPL | PPL vs Dense | p50 (ms) | KV kept " "| KV ratio |"
        )
        report.append(
            "|--------|---|-----|-------------|----------|---------|" "----------|"
        )
        for L in sorted(set(a["seq_len"] for a in control_ex)):
            dense = [
                a for a in control_ex if a["method"] == "dense" and a["seq_len"] == L
            ]
            if not dense:
                continue
            dp = dense[0]["ppl"]
            dk = dense[0]["kv_bytes_read_per_token"]
            for a in sorted(
                [x for x in control_ex if x["seq_len"] == L],
                key=lambda x: x["method"],
            ):
                kv_ratio = a["kv_bytes_read_per_token"] / dk if dk > 0 else 0
                ppl_d = (a["ppl"] - dp) / dp * 100 if dp > 0 else 0
                report.append(
                    f"| {a['method']} | {L} "
                    f"| {a['ppl']:.0f} "
                    f"| {ppl_d:+.1f}% "
                    f"| {a['decode_per_token_ms']:.2f} "
                    f"| {a['kv_kept_mean']:.0f} "
                    f"| {kv_ratio:.2f}x |"
                )
        report.append("")

    report_text = "\n".join(report) + "\n"
    path = os.path.join(results_dir, "bpa_v10_final_report.md")
    with open(path, "w") as f:
        f.write(report_text)
    print(f"Report: {path}")


def main():
    results_dir = "bpa_v10_results"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    results = load_results(results_dir)
    print(f"Loaded {len(results)} results from {results_dir}")

    agg = aggregate(results)
    tuning = load_tuning_summary(results_dir)
    generate_report(agg, results_dir, tuning)


if __name__ == "__main__":
    main()
