#!/usr/bin/env python3
"""
Regenerate summary report using real GPU memory measurements with detailed epoch analysis.
Supports incremental updates when make continue adds new test results.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from glob import glob
import re

# Handle numpy import gracefully
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print(
        "Warning: numpy not found. Some statistics will be unavailable.",
        file=sys.stderr,
    )
    print("Install with: pip install numpy", file=sys.stderr)


def load_json(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return None


def get_gpu_memory_stats(test_dir):
    """Extract GPU memory statistics from monitoring files."""
    gpu_stats = {}

    # Find GPU monitoring files ONLY in the specified test directory
    gpu_files = []

    if os.path.exists(test_dir):
        gpu_files.extend(
            glob(os.path.join(test_dir, "**/gpu_stats_*.json"), recursive=True)
        )
        gpu_files.extend(
            glob(os.path.join(test_dir, "**/gpu_training_*.json"), recursive=True)
        )
        gpu_files.extend(
            glob(os.path.join(test_dir, "**/gpu_inference_*.json"), recursive=True)
        )

    for gpu_file in gpu_files:
        # Extract test name from path
        parent_dir = Path(gpu_file).parent.name
        filename = Path(gpu_file).stem

        # Determine test name from filename or parent dir
        test_name = parent_dir
        if "gpu_training_" in filename:
            # Parse training file: gpu_training_adamwprune_state_70
            parts = filename.replace("gpu_training_", "").split("_")
            if len(parts) >= 3:
                test_name = f"resnet18_{parts[0]}_{parts[1]}_{parts[2]}"
        elif "gpu_stats_" in filename:
            # Already has full name in parent dir
            test_name = parent_dir

        # Load GPU data
        data = load_json(gpu_file)
        if not data:
            continue

        # Extract memory values
        samples = data if isinstance(data, list) else data.get("samples", [])
        if not samples:
            # Check if data has summary field directly
            if isinstance(data, dict) and "summary" in data:
                summary = data["summary"]
                if "mean_memory_mb" in summary:
                    gpu_stats[test_name] = {
                        "mean": summary.get("mean_memory_mb", 0),
                        "max": summary.get("max_memory_mb", 0),
                        "min": summary.get("min_memory_mb", 0),
                        "std": 0,
                    }
            continue

        memory_values = []
        for s in samples:
            if "memory_used" in s:
                val = s["memory_used"]
                # Filter out idle values (< 100 MB usually indicates idle)
                if val > 100:
                    memory_values.append(val)
            elif "memory_mb" in s:
                memory_values.append(s["memory_mb"])
            elif "memory_used_mb" in s:
                memory_values.append(s["memory_used_mb"])

        if memory_values:
            # Calculate statistics with or without numpy
            mean_val = (
                np.mean(memory_values)
                if HAS_NUMPY
                else sum(memory_values) / len(memory_values)
            )
            if HAS_NUMPY:
                std_val = np.std(memory_values)
            else:
                # Manual standard deviation calculation
                variance = sum((x - mean_val) ** 2 for x in memory_values) / len(
                    memory_values
                )
                std_val = variance**0.5

            # Store under multiple possible keys for matching
            gpu_stats[test_name] = {
                "mean": mean_val,
                "max": max(memory_values),
                "min": min(memory_values),
                "std": std_val,
            }

            # Also store under alternative keys for AdamWPrune
            if "adamwprune" in test_name.lower():
                # Store under both state and movement variants
                alt_name = test_name.replace("_state_", "_movement_")
                if alt_name != test_name:
                    gpu_stats[alt_name] = gpu_stats[test_name]
                alt_name = test_name.replace("_movement_", "_state_")
                if alt_name != test_name:
                    gpu_stats[alt_name] = gpu_stats[test_name]

    return gpu_stats


def analyze_accuracy_details(metrics_file):
    """Analyze detailed accuracy information from training metrics."""
    metrics = load_json(metrics_file)
    if not metrics:
        return None

    test_accs = metrics.get("test_accuracy", [])
    if not test_accs:
        # Try to extract from epochs data if available
        epochs = metrics.get("epochs", [])
        if epochs and isinstance(epochs[0], dict):
            # Try to get test_accuracy from each epoch
            test_accs = []
            for epoch in epochs:
                # Look for various possible field names
                acc = (
                    epoch.get("test_accuracy")
                    or epoch.get("accuracy")
                    or epoch.get("val_accuracy")
                )
                if acc is not None:
                    test_accs.append(acc)

        if not test_accs:
            # Fallback to top-level accuracy fields
            final_acc = metrics.get("final_accuracy", metrics.get("best_accuracy", 0))
            return {
                "best_accuracy": final_acc,
                "best_epoch": None,
                "final_accuracy": final_acc,
                "final_epoch": None,
                "final_sparsity": metrics.get("final_sparsity", 0),
                "has_epoch_data": False,
            }

    # Find best accuracy and when it occurred
    best_acc = max(test_accs)
    best_epoch = test_accs.index(best_acc) + 1
    final_acc = test_accs[-1]
    final_epoch = len(test_accs)

    # Get sparsity information
    sparsities = metrics.get("sparsity", [])
    final_sparsity = metrics.get("final_sparsity", sparsities[-1] if sparsities else 0)

    # For state pruning, find best accuracy at target sparsity
    best_at_target = None
    best_at_target_epoch = None
    if sparsities and len(sparsities) == len(test_accs):
        target_sparsity = final_sparsity
        # Find accuracies at or near target sparsity (within 1%)
        at_target = [
            (i, test_accs[i])
            for i in range(len(test_accs))
            if sparsities[i] >= target_sparsity - 0.01
        ]
        if at_target:
            best_at_target_idx, best_at_target = max(at_target, key=lambda x: x[1])
            best_at_target_epoch = best_at_target_idx + 1
            best_sparsity_at_peak = sparsities[best_at_target_idx]
        else:
            best_sparsity_at_peak = (
                sparsities[best_epoch - 1]
                if best_epoch <= len(sparsities)
                else final_sparsity
            )
    else:
        best_sparsity_at_peak = final_sparsity

    # Check stability (standard deviation of last 10 epochs)
    if len(test_accs) >= 10:
        last_10 = test_accs[-10:]
        if HAS_NUMPY:
            stability = np.std(last_10)
        else:
            # Manual standard deviation calculation
            mean_last_10 = sum(last_10) / len(last_10)
            variance = sum((x - mean_last_10) ** 2 for x in last_10) / len(last_10)
            stability = variance**0.5
    else:
        stability = 0

    return {
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "best_sparsity": (
            sparsities[best_epoch - 1]
            if sparsities and best_epoch <= len(sparsities)
            else final_sparsity
        ),
        "best_at_target": best_at_target,
        "best_at_target_epoch": best_at_target_epoch,
        "final_accuracy": final_acc,
        "final_epoch": final_epoch,
        "final_sparsity": final_sparsity,
        "degradation": best_acc - final_acc,
        "stability": stability,
        "has_epoch_data": True,
    }


def update_all_results(results_dir):
    """Update all_results.json from individual test metrics files."""
    all_results = []

    # Find all test directories
    for test_dir in sorted(os.listdir(results_dir)):
        # Skip non-test directories (look for model names)
        if not (test_dir.startswith("resnet") or test_dir.startswith("lenet")):
            continue

        metrics_file = os.path.join(results_dir, test_dir, "training_metrics.json")
        if not os.path.exists(metrics_file):
            continue

        # Parse test name
        # Patterns:
        #   model_optimizer_pruning_sparsity (e.g., lenet5_adamw_magnitude_70)
        #   model_optimizer_none (e.g., lenet5_adamw_none)
        #   model_optimizer_variant_pruning_sparsity (e.g., lenet5_adamwprune_bitter0_state_70)
        #   model_optimizer_variant_none (e.g., lenet5_adamwprune_bitter0_none)
        parts = test_dir.split("_")
        if len(parts) < 3:
            continue

        model = parts[0]
        optimizer = parts[1]

        # Handle sparsity and pruning method
        last_part = parts[-1]
        if last_part == "none":
            sparsity = 0.0
            pruning = "none"
        elif last_part.isdigit():
            sparsity = int(last_part) / 100.0
            pruning = parts[-2]  # Second to last is pruning method
        else:
            # Unknown pattern, skip
            continue

        # Load and analyze metrics
        metrics = load_json(metrics_file)
        if not metrics:
            continue

        details = analyze_accuracy_details(metrics_file)

        result = {
            "test_id": test_dir,  # Add test_id for directory mapping
            "model": model,
            "optimizer": optimizer,
            "pruning_method": pruning,
            "target_sparsity": sparsity,
            "final_accuracy": details["final_accuracy"],
            "best_accuracy": details["best_accuracy"],
            "best_epoch": details["best_epoch"],
            "final_sparsity": details["final_sparsity"],
            "test_accuracy": (
                metrics.get("test_accuracy", [])[-1]
                if metrics.get("test_accuracy")
                else details["final_accuracy"]
            ),
        }
        all_results.append(result)

    # Save updated results
    all_results_file = os.path.join(results_dir, "all_results.json")
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def regenerate_summary(results_dir, output_file="summary_report.txt"):
    """Regenerate summary report with real GPU memory data and detailed analysis."""

    # Update all_results.json first
    all_results = update_all_results(results_dir)

    if not all_results:
        print(f"Error: No test results found in {results_dir}")
        return False

    # Get GPU memory stats
    gpu_stats = get_gpu_memory_stats(results_dir)

    # Process results with detailed analysis
    test_results = []
    accuracy_details = {}

    for result in all_results:
        # Use test_id from result (preserves full directory name including variant)
        test_name = result.get("test_id")
        if not test_name:
            # Fallback to constructing test ID if not present
            model = result.get("model", "unknown")
            optimizer = result.get("optimizer", "unknown")
            pruning = result.get("pruning_method", "none")
            sparsity = int(result.get("target_sparsity", 0) * 100)
            test_name = f"{model}_{optimizer}_{pruning}_{sparsity}"

        model = result.get("model", "unknown")
        optimizer = result.get("optimizer", "unknown")
        pruning = result.get("pruning_method", "none")

        # Get detailed accuracy analysis
        metrics_file = os.path.join(results_dir, test_name, "training_metrics.json")
        if os.path.exists(metrics_file):
            details = analyze_accuracy_details(metrics_file)
            accuracy_details[test_name] = details
        else:
            details = {
                "best_accuracy": result.get(
                    "best_accuracy", result.get("final_accuracy", 0)
                ),
                "final_accuracy": result.get("final_accuracy", 0),
                "best_epoch": result.get("best_epoch"),
                "final_sparsity": result.get(
                    "final_sparsity", result.get("target_sparsity", 0)
                ),
            }
            accuracy_details[test_name] = details

        # Determine which accuracy to use for ranking
        # For state pruning with gradual sparsity, use best at target
        # For movement/magnitude pruning, use best overall
        if pruning == "state" and details.get("best_at_target"):
            display_accuracy = details["best_at_target"]
            accuracy_note = f"@{int(details['final_sparsity']*100)}%"
        else:
            display_accuracy = details["best_accuracy"]
            accuracy_note = ""

        # Get GPU stats
        gpu_mean = gpu_stats.get(test_name, {}).get("mean", 0)
        gpu_max = gpu_stats.get(test_name, {}).get("max", 0)

        test_results.append(
            {
                "test_id": test_name,
                "model": model,
                "optimizer": optimizer,
                "pruning": pruning,
                "accuracy": display_accuracy,
                "accuracy_note": accuracy_note,
                "best_accuracy": details["best_accuracy"],
                "best_epoch": details.get("best_epoch"),
                "final_accuracy": details["final_accuracy"],
                "sparsity": details["final_sparsity"],
                "gpu_mean": gpu_mean,
                "gpu_max": gpu_max,
                "status": "Success",
            }
        )

    # Sort by accuracy for ranking
    test_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # Generate report
    timestamp = datetime.now().isoformat()

    with open(os.path.join(results_dir, output_file), "w") as f:
        f.write("Test Matrix Summary Report (With Detailed Analysis)\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"From: {results_dir}\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        total_tests = len(test_results)
        successful = len([t for t in test_results if t["status"] == "Success"])
        failed = total_tests - successful

        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")

        # Results table with detailed accuracy info
        f.write("Detailed Results Table:\n")
        f.write("-" * 120 + "\n")
        f.write(
            f"{'Test ID':<40} {'Best Acc':<10} {'Final Acc':<10} {'Epoch':<8} {'Sparsity':<10} {'GPU (MB)':<12} {'Status':<10}\n"
        )
        f.write("-" * 120 + "\n")

        for result in test_results:
            test_id = result["test_id"][:40]
            best_acc = (
                f"{result['best_accuracy']:.2f}%" if result["best_accuracy"] else "N/A"
            )
            final_acc = (
                f"{result['final_accuracy']:.2f}%"
                if result["final_accuracy"]
                else "N/A"
            )
            epoch = f"@{result['best_epoch']}" if result["best_epoch"] else "N/A"
            sparsity = f"{result['sparsity']*100:.1f}%"
            gpu = f"{result['gpu_mean']:.1f}" if result["gpu_mean"] > 0 else "N/A"
            status = "✓" if result["status"] == "Success" else "✗"

            f.write(
                f"{test_id:<40} {best_acc:<10} {final_acc:<10} {epoch:<8} {sparsity:<10} {gpu:<12} {status:<10}\n"
            )

        f.write("-" * 120 + "\n\n")

        # Accuracy Analysis Section
        f.write("Accuracy Analysis:\n")
        f.write("-" * 80 + "\n")

        # Group by optimizer
        by_optimizer = {}
        for result in test_results:
            opt = result["optimizer"]
            if opt not in by_optimizer:
                by_optimizer[opt] = []
            by_optimizer[opt].append(result)

        f.write("Performance by Optimizer:\n\n")
        for opt in sorted(by_optimizer.keys()):
            results = by_optimizer[opt]
            best_result = max(results, key=lambda x: x["best_accuracy"])

            f.write(f"{opt.upper()}:\n")
            f.write(f"  Best accuracy: {best_result['best_accuracy']:.2f}%")
            if best_result["best_epoch"]:
                f.write(f" (epoch {best_result['best_epoch']})")
            f.write(f"\n")
            f.write(f"  Final accuracy: {best_result['final_accuracy']:.2f}%\n")

            # Check if state pruning
            if best_result["pruning"] == "state":
                details = accuracy_details.get(best_result["test_id"], {})
                if details.get("best_at_target"):
                    f.write(
                        f"  Best at target sparsity: {details['best_at_target']:.2f}%"
                    )
                    if details.get("best_at_target_epoch"):
                        f.write(f" (epoch {details['best_at_target_epoch']})")
                    f.write(f"\n")

            # Stability analysis
            details = accuracy_details.get(best_result["test_id"], {})
            if details.get("stability") is not None:
                f.write(
                    f"  Stability (std last 10 epochs): {details['stability']:.2f}%\n"
                )
            if details.get("degradation") is not None and details["degradation"] != 0:
                f.write(f"  Degradation from peak: {details['degradation']:.2f}%\n")
            f.write("\n")

        # Best performers with fair comparison
        f.write("\nFair Comparison (at target sparsity):\n")
        f.write("-" * 80 + "\n")

        rank = 1
        for result in test_results[:10]:
            details = accuracy_details.get(result["test_id"], {})

            # Determine which accuracy to display
            if result["pruning"] == "state" and details.get("best_at_target"):
                acc = details["best_at_target"]
                note = (
                    f" (best at {int(round(details['final_sparsity']*100))}% sparsity)"
                )
            else:
                acc = result["best_accuracy"]
                note = ""

            f.write(f"{rank}. {result['optimizer']}: {acc:.2f}%{note}")
            # Use appropriate epoch based on which accuracy we're showing
            if result["pruning"] == "state" and details.get("best_at_target_epoch"):
                f.write(f" @ epoch {details['best_at_target_epoch']}")
            elif details.get("best_epoch"):
                f.write(f" @ epoch {details['best_epoch']}")
            f.write(f"\n")
            rank += 1

        # GPU Memory Analysis
        if any(r["gpu_mean"] > 0 for r in test_results):
            f.write("\n" + "GPU Memory Analysis:\n")
            f.write("-" * 80 + "\n")

            # Sort by GPU memory
            gpu_results = [r for r in test_results if r["gpu_mean"] > 0]
            gpu_results.sort(key=lambda x: x["gpu_mean"])

            f.write("Memory Usage Ranking:\n")
            for i, result in enumerate(gpu_results[:5], 1):
                f.write(f"{i}. {result['test_id']}: {result['gpu_mean']:.1f} MB")
                f.write(f" (accuracy: {result['best_accuracy']:.2f}%)\n")

            # Memory efficiency
            f.write("\nMemory Efficiency (Accuracy per GB):\n")
            for result in gpu_results[:5]:
                efficiency = result["best_accuracy"] / (result["gpu_mean"] / 1024)
                f.write(f"  {result['optimizer']}: {efficiency:.2f}%/GB\n")

    print(f"Summary report regenerated: {os.path.join(results_dir, output_file)}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python regenerate_summary_with_gpu.py <test_results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)

    regenerate_summary(results_dir)


if __name__ == "__main__":
    main()
