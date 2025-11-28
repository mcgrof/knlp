#!/usr/bin/env python3
"""Compare KVSplice ablation results across W7900, A100, and H100 GPUs."""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# GPU run configurations
RUNS = {
    "W7900": {
        "project": "mcgrof-citizen/gpt2-kvsplice-ablation-50pct-w7900",
        "variants": ["gpt2_adamwspam_mla", "gpt2_adamwspam_mla_kv"],
        "compression_ratio": 0.5,
        "gpu_name": "AMD Radeon Pro W7900",
        "gpu_memory": "48GB",
    },
    "A100-40G": {
        "project": "mcgrof-citizen/gpt2-kvsplice-ablation-70pct-a100-40g",
        "variants": ["gpt2_adamwspam_mla", "gpt2_adamwspam_mla_kv"],
        "compression_ratio": 0.5,  # Intended 0.7 but defaulted to 0.5
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_memory": "40GB",
    },
    "H100": {
        "project": "mcgrof-citizen/gpt2-kvsplice-ablation-h100",
        "variants": ["gpt2_adamwspam_mla", "gpt2_adamwspam_mla_kv"],
        "compression_ratio": 0.5,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_memory": "80GB",
    },
}


def fetch_run_data(project, run_name):
    """Fetch metrics from a W&B run."""
    api = wandb.Api()

    try:
        runs = api.runs(project, filters={"display_name": run_name})
        if not runs:
            print(f"  ⚠️  No run found: {run_name}")
            return None

        run = runs[0]

        # Fetch training history
        history = run.history(samples=10000)

        # Get final metrics
        summary = run.summary._json_dict

        return {
            "history": history,
            "summary": summary,
            "config": run.config,
            "name": run.name,
            "url": run.url,
        }
    except Exception as e:
        print(f"  ❌ Error fetching {run_name}: {e}")
        return None


def main():
    print("=" * 70)
    print("KVSplice Ablation Study: GPU Comparison")
    print("=" * 70)

    all_data = {}

    # Fetch data from all runs
    for gpu_name, gpu_config in RUNS.items():
        print(f"\n{gpu_name} ({gpu_config['gpu_name']}, {gpu_config['gpu_memory']}):")
        print(f"  Project: {gpu_config['project']}")

        gpu_data = {}
        for variant in gpu_config['variants']:
            print(f"  Fetching {variant}...")
            data = fetch_run_data(gpu_config['project'], variant)
            if data:
                gpu_data[variant] = data
                print(f"    ✓ Found run with {len(data['history'])} steps")

        all_data[gpu_name] = {
            "runs": gpu_data,
            "config": gpu_config,
        }

    # Check if all runs have KVSplice metrics
    print("\n" + "=" * 70)
    print("KVSplice Metrics Availability")
    print("=" * 70)

    for gpu_name, gpu_data in all_data.items():
        print(f"\n{gpu_name}:")
        for variant, run_data in gpu_data["runs"].items():
            history = run_data["history"]
            kvsplice_cols = [c for c in history.columns if "kvsplice" in c.lower()]
            if kvsplice_cols:
                print(f"  {variant}: ✓ Found {len(kvsplice_cols)} KVSplice metrics")
                print(f"    Sample metrics: {kvsplice_cols[:5]}")
            else:
                print(f"  {variant}: ⚠️  No KVSplice metrics found")

    # Compare final validation loss and perplexity
    print("\n" + "=" * 70)
    print("Final Training Metrics Comparison")
    print("=" * 70)

    results = []
    for gpu_name, gpu_data in all_data.items():
        for variant, run_data in gpu_data["runs"].items():
            summary = run_data["summary"]
            config = run_data["config"]

            variant_name = "MLA" if "mla_kv" not in variant else "MLA+KVSplice"
            compression = "6x" if variant_name == "MLA" else "12x"

            result = {
                "GPU": gpu_name,
                "Variant": variant_name,
                "Compression": compression,
                "Val Loss": summary.get("val_loss", "N/A"),
                "Val PPL": summary.get("val_perplexity", "N/A"),
                "Iterations": summary.get("iteration", "N/A"),
                "Training Time": f"{summary.get('_runtime', 0) / 3600:.1f}h",
            }
            results.append(result)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Plot comparison graphs
    print("\n" + "=" * 70)
    print("Generating Comparison Graphs")
    print("=" * 70)

    # Create output directory
    output_dir = Path("docs/kvsplice")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Validation loss over time (all GPUs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for gpu_name, gpu_data in all_data.items():
        for variant, run_data in gpu_data["runs"].items():
            history = run_data["history"]
            variant_name = "MLA" if "mla_kv" not in variant else "MLA+KVSplice"
            label = f"{gpu_name} - {variant_name}"

            # Plot validation loss
            if "val_loss" in history.columns and "iteration" in history.columns:
                df_clean = history[["iteration", "val_loss"]].dropna()
                axes[0].plot(df_clean["iteration"], df_clean["val_loss"], label=label, alpha=0.8)

            # Plot validation perplexity
            if "val_perplexity" in history.columns and "iteration" in history.columns:
                df_clean = history[["iteration", "val_perplexity"]].dropna()
                axes[1].plot(df_clean["iteration"], df_clean["val_perplexity"], label=label, alpha=0.8)

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss Across GPUs")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Validation Perplexity")
    axes[1].set_title("Validation Perplexity Across GPUs")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    output_file = output_dir / "gpu_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    # Plot 2: Final metrics bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Group by GPU and variant
    gpus = df["GPU"].unique()
    variants = ["MLA", "MLA+KVSplice"]
    x = np.arange(len(gpus))
    width = 0.35

    for i, variant in enumerate(variants):
        variant_df = df[df["Variant"] == variant]
        val_losses = []
        val_ppls = []

        for gpu in gpus:
            gpu_df = variant_df[variant_df["GPU"] == gpu]
            if len(gpu_df) > 0:
                val_losses.append(gpu_df.iloc[0]["Val Loss"])
                val_ppls.append(gpu_df.iloc[0]["Val PPL"])
            else:
                val_losses.append(None)
                val_ppls.append(None)

        # Remove None values
        val_losses = [v if v != "N/A" else None for v in val_losses]
        val_ppls = [v if v != "N/A" else None for v in val_ppls]

        offset = width * (i - 0.5)
        axes[0].bar(x + offset, val_losses, width, label=variant, alpha=0.8)
        axes[1].bar(x + offset, val_ppls, width, label=variant, alpha=0.8)

    axes[0].set_xlabel("GPU")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Final Validation Loss by GPU")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(gpus)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].set_xlabel("GPU")
    axes[1].set_ylabel("Validation Perplexity")
    axes[1].set_title("Final Validation Perplexity by GPU")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gpus)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale("log")

    plt.tight_layout()
    output_file = output_dir / "gpu_final_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_file}")
    plt.close()

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print(f"Graphs saved to: {output_dir}/")
    print("\nSummary:")
    print(f"  - Tested {len(all_data)} GPUs: {', '.join(all_data.keys())}")
    print(f"  - Each GPU ran 2 variants: MLA (6x) and MLA+KVSplice (12x)")
    print(f"  - All used compression_ratio=0.5 (A100 intended 0.7 but defaulted to 0.5)")


if __name__ == "__main__":
    main()
