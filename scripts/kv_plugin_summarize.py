#!/usr/bin/env python3
"""
KV Plugin v3 Benchmark Summarizer

Reads benchmark JSON results and generates markdown tables comparing
orthogonal vs PCA/SVD presets across models.

Usage:
    python scripts/kv_plugin_summarize.py key_results/kv_plugin_ortho/
    python scripts/kv_plugin_summarize.py --output summary.md key_results/*.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_results(paths: List[Path]) -> List[Dict]:
    """Load all JSON result files."""
    all_results = []
    for path in paths:
        if path.is_dir():
            json_files = list(path.glob("*.json"))
        else:
            json_files = [path]

        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                    # Handle both single result and list of results
                    if isinstance(data, list):
                        all_results.extend(data)
                    else:
                        all_results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {jf}: {e}")

    return all_results


def get_baseline_ppl(results: List[Dict], model: str) -> Optional[float]:
    """Get baseline (none preset) PPL for a model."""
    for r in results:
        if r.get("model") == model and r.get("preset") == "none":
            ppl = r.get("perplexity")
            if isinstance(ppl, (int, float)):
                return ppl
    return None


def format_ppl_delta(ppl: float, baseline: float) -> str:
    """Format PPL delta as percentage."""
    if baseline is None or baseline == 0:
        return "N/A"
    delta = ((ppl - baseline) / baseline) * 100
    if delta > 0:
        return f"+{delta:.1f}%"
    return f"{delta:.1f}%"


def get_compression_ratio(result: Dict) -> str:
    """Get compression ratio from result."""
    compression = result.get("compression", {})
    if isinstance(compression, dict):
        ratio = compression.get("ratio")
        if ratio:
            return f"{ratio:.0f}x"
    return "N/A"


def generate_main_comparison_table(results: List[Dict]) -> str:
    """Generate the main Δ val PPL comparison table."""
    # Group by model
    models = sorted(set(r.get("model", "unknown") for r in results))

    # Target presets for comparison
    target_presets = [
        "none",
        "balanced",
        "aggressive",
        "orthogonal",
        "orthogonal_aggressive",
    ]

    lines = ["## Validation PPL Comparison: Orthogonal vs PCA/SVD", ""]
    lines.append("| Model | Preset | Compression | Calibration | Δ val PPL |")
    lines.append("|-------|--------|-------------|-------------|-----------|")

    for model in models:
        baseline_ppl = get_baseline_ppl(results, model)
        model_results = [r for r in results if r.get("model") == model]

        # Sort by preset order
        def preset_order(r):
            preset = r.get("preset", "")
            if preset in target_presets:
                return target_presets.index(preset)
            return 100

        model_results = sorted(model_results, key=preset_order)

        for r in model_results:
            preset = r.get("preset", "unknown")
            if preset not in target_presets:
                continue

            ppl = r.get("perplexity")
            if not isinstance(ppl, (int, float)):
                ppl_str = "error"
                delta_str = "N/A"
            elif preset == "none":
                ppl_str = f"{ppl:.2f}"
                delta_str = "baseline"
            else:
                ppl_str = f"{ppl:.2f}"
                delta_str = format_ppl_delta(ppl, baseline_ppl)

            compression = get_compression_ratio(r)
            calib = "no" if r.get("zero_calibration") else "yes"
            if preset == "none":
                calib = "-"

            # Shorten model name
            model_short = model.split("/")[-1]

            lines.append(
                f"| {model_short} | {preset} | {compression} | {calib} | {delta_str} |"
            )

        # Add separator between models
        if model != models[-1]:
            lines.append("|  |  |  |  |  |")

    return "\n".join(lines)


def generate_detailed_metrics_table(results: List[Dict]) -> str:
    """Generate detailed metrics table with memory/speed."""
    lines = ["## Detailed Metrics", ""]
    lines.append(
        "| Model | Preset | PPL | Memory (MB) | TTFT (ms) | Throughput (tok/s) |"
    )
    lines.append(
        "|-------|--------|-----|-------------|-----------|-------------------|"
    )

    for r in results:
        model = r.get("model", "unknown").split("/")[-1]
        preset = r.get("preset", "unknown")

        ppl = r.get("perplexity")
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else "error"

        mem = r.get("memory", {})
        mem_str = (
            f"{mem.get('peak_memory_mb', 'N/A'):.1f}"
            if isinstance(mem, dict)
            else "N/A"
        )

        ttft = r.get("ttft", {})
        ttft_str = (
            f"{ttft.get('avg_ttft_ms', 'N/A'):.2f}" if isinstance(ttft, dict) else "N/A"
        )

        tp = r.get("throughput", {})
        tp_str = (
            f"{tp.get('tokens_per_second', 'N/A'):.1f}"
            if isinstance(tp, dict)
            else "N/A"
        )

        lines.append(
            f"| {model} | {preset} | {ppl_str} | {mem_str} | {ttft_str} | {tp_str} |"
        )

    return "\n".join(lines)


def generate_orthogonal_highlight(results: List[Dict]) -> str:
    """Generate a highlight section for orthogonal compressor."""
    lines = ["## Orthogonal Compressor Highlights", ""]

    # Find orthogonal and balanced results for comparison
    for model in sorted(set(r.get("model", "unknown") for r in results)):
        model_results = [r for r in results if r.get("model") == model]
        baseline_ppl = get_baseline_ppl(results, model)

        ortho = next(
            (r for r in model_results if r.get("preset") == "orthogonal"), None
        )
        balanced = next(
            (r for r in model_results if r.get("preset") == "balanced"), None
        )

        if ortho and balanced and baseline_ppl:
            ortho_ppl = ortho.get("perplexity")
            balanced_ppl = balanced.get("perplexity")

            if isinstance(ortho_ppl, (int, float)) and isinstance(
                balanced_ppl, (int, float)
            ):
                model_short = model.split("/")[-1]
                ortho_delta = ((ortho_ppl - baseline_ppl) / baseline_ppl) * 100
                balanced_delta = ((balanced_ppl - baseline_ppl) / baseline_ppl) * 100
                gap = ortho_delta - balanced_delta

                lines.append(f"**{model_short}**:")
                lines.append(f"- Orthogonal (zero-calib): Δ PPL +{ortho_delta:.1f}%")
                lines.append(f"- Balanced (calibrated):   Δ PPL +{balanced_delta:.1f}%")
                lines.append(f"- Gap: {gap:.1f}% additional PPL for zero-calibration")
                lines.append("")

    if len(lines) == 2:
        lines.append("No orthogonal vs balanced comparisons available.")

    return "\n".join(lines)


def generate_summary_markdown(results: List[Dict]) -> str:
    """Generate full summary markdown document."""
    sections = [
        "# KV Plugin v3 Benchmark Summary",
        "",
        generate_main_comparison_table(results),
        "",
        generate_orthogonal_highlight(results),
        "",
        generate_detailed_metrics_table(results),
        "",
        "---",
        "Generated by `scripts/kv_plugin_summarize.py`",
    ]
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize KV Plugin v3 benchmark results"
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="JSON result files or directories containing them",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output markdown file (default: print to stdout)",
    )
    args = parser.parse_args()

    # Load results
    results = load_results(args.paths)
    if not results:
        print("No results found!")
        return 1

    print(f"Loaded {len(results)} benchmark results")

    # Generate summary
    summary = generate_summary_markdown(results)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
        print(f"Summary written to {args.output}")
    else:
        print("\n" + summary)

    return 0


if __name__ == "__main__":
    exit(main())
