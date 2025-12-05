#!/usr/bin/env python3
"""
Full KV Plugin Evaluation Pipeline

One command to run all important experiments for the KV compression paper.

Runs:
- eval_ppl.py on GPT-2-124M, Qwen2.5-0.5B, Qwen2.5-7B
- eval_tasks.py on GSM8K, Winogrande, PIQA, RTE, CommonsenseQA
- eval_performance.py for context lengths 1k, 4k, 8k, 16k, 32k

For each model, sweeps configs:
- baseline (no compression)
- orthogonal (6x compression)
- orthogonal_int8 (12x - V only)
- orthogonal_int8_kv (12x - K+V)
- orthogonal_int4 (24x - V only)
- orthogonal_int4_kv (24x - K+V)

Stores results in JSONL/CSV under results/.

Usage:
    # Run all experiments (full)
    python scripts/run_all_kv_experiments.py --output-dir results/

    # Quick validation run (small samples)
    python scripts/run_all_kv_experiments.py --quick --output-dir results/

    # Single model only
    python scripts/run_all_kv_experiments.py --models gpt2 --output-dir results/

    # Single config only
    python scripts/run_all_kv_experiments.py --configs baseline orthogonal --output-dir results/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

# Models to evaluate
MODELS = {
    "gpt2": "openai-community/gpt2",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

# Configs to sweep (preset name -> description)
CONFIGS = {
    "none": {
        "preset": "none",
        "description": "FP16 baseline (no compression)",
        "compression": 1.0,
    },
    "orthogonal": {
        "preset": "orthogonal",
        "description": "Orthogonal 6x (no quant)",
        "compression": 6.0,
    },
    "orthogonal_int8": {
        "preset": "orthogonal_int8",
        "description": "Orthogonal + int8 V (12x)",
        "compression": 12.0,
    },
    "orthogonal_int8_kv": {
        "preset": "orthogonal_int8_kv",
        "description": "Orthogonal + int8 K+V (12x)",
        "compression": 12.0,
    },
    "orthogonal_int4": {
        "preset": "orthogonal_int4",
        "description": "Orthogonal + int4 V (24x)",
        "compression": 24.0,
    },
    "orthogonal_int4_kv": {
        "preset": "orthogonal_int4_kv",
        "description": "Orthogonal + int4 K+V (24x)",
        "compression": 24.0,
    },
}

# Tasks to evaluate
TASKS = ["gsm8k", "winogrande", "piqa", "rte", "commonsense_qa"]

# Context lengths for performance evaluation
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]

# Quick mode settings (for validation)
QUICK_SETTINGS = {
    "ppl_samples": 10,
    "task_samples": 20,
    "context_lengths": [512, 1024, 2048],
}

# Full mode settings
FULL_SETTINGS = {
    "ppl_samples": 100,
    "task_samples": 100,
    "context_lengths": CONTEXT_LENGTHS,
}


@dataclass
class ExperimentResult:
    """Single experiment result."""

    model: str
    config: str
    compression: float
    timestamp: str

    # Perplexity results
    ppl_wikitext: Optional[float] = None
    ppl_c4: Optional[float] = None

    # Task accuracy results
    gsm8k_acc: Optional[float] = None
    winogrande_acc: Optional[float] = None
    piqa_acc: Optional[float] = None
    rte_acc: Optional[float] = None
    commonsense_qa_acc: Optional[float] = None

    # Performance results (at reference context length)
    ttft_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    kv_memory_mb: Optional[float] = None
    peak_memory_gb: Optional[float] = None

    # Status
    status: str = "pending"
    error: Optional[str] = None


def run_command(cmd: List[str], timeout: int = 3600) -> Dict:
    """Run a command and return JSON output."""
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"    Error: {result.stderr[:500]}")
            return {"error": result.stderr[:500]}

        # Try to parse JSON from output or output file
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def run_ppl_evaluation(
    model_name: str,
    preset: str,
    output_file: str,
    max_samples: int = 100,
) -> Dict:
    """Run perplexity evaluation."""
    cmd = [
        sys.executable,
        "scripts/eval_ppl.py",
        "--model",
        model_name,
        "--preset",
        preset,
        "--datasets",
        "wikitext2",
        "--max-samples",
        str(max_samples),
        "--seq-len",
        "1024",
        "--output",
        output_file,
    ]
    result = run_command(cmd)

    # Load results from output file
    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)
    return result


def run_task_evaluation(
    model_name: str,
    preset: str,
    tasks: List[str],
    output_file: str,
    max_samples: int = 100,
) -> Dict:
    """Run task accuracy evaluation."""
    cmd = [
        sys.executable,
        "scripts/eval_tasks.py",
        "--model",
        model_name,
        "--preset",
        preset,
        "--tasks",
        *tasks,
        "--max-samples",
        str(max_samples),
        "--output",
        output_file,
    ]
    result = run_command(cmd)

    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)
    return result


def run_performance_evaluation(
    model_name: str,
    preset: str,
    context_lengths: List[int],
    output_file: str,
) -> Dict:
    """Run performance evaluation."""
    cmd = [
        sys.executable,
        "scripts/eval_performance.py",
        "--model",
        model_name,
        "--preset",
        preset,
        "--context-lengths",
        *[str(c) for c in context_lengths],
        "--output",
        output_file,
    ]
    result = run_command(cmd)

    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)
    return result


def run_experiment(
    model_key: str,
    config_key: str,
    output_dir: str,
    settings: Dict,
) -> ExperimentResult:
    """Run full experiment for one model/config combination."""
    model_name = MODELS[model_key]
    config = CONFIGS[config_key]
    preset = config["preset"]

    result = ExperimentResult(
        model=model_key,
        config=config_key,
        compression=config["compression"],
        timestamp=datetime.now().isoformat(),
    )

    # Create output directory
    exp_dir = os.path.join(output_dir, model_key, config_key)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {model_key} ({model_name})")
    print(f"Config: {config_key} - {config['description']}")
    print(f"{'='*60}")

    # 1. Perplexity evaluation
    print("\n[1/3] Running perplexity evaluation...")
    ppl_file = os.path.join(exp_dir, "ppl_results.json")
    try:
        ppl_results = run_ppl_evaluation(
            model_name,
            preset,
            ppl_file,
            settings["ppl_samples"],
        )
        if "datasets" in ppl_results:
            if "wikitext2" in ppl_results["datasets"]:
                result.ppl_wikitext = ppl_results["datasets"]["wikitext2"].get(
                    "perplexity"
                )
            if "c4" in ppl_results["datasets"]:
                result.ppl_c4 = ppl_results["datasets"]["c4"].get("perplexity")
    except Exception as e:
        print(f"  PPL evaluation failed: {e}")
        result.error = str(e)

    # 2. Task evaluation (skip for quick mode if model is large)
    if model_key != "qwen-7b" or settings["task_samples"] >= 50:
        print("\n[2/3] Running task evaluation...")
        task_file = os.path.join(exp_dir, "task_results.json")
        try:
            # Run subset of tasks for faster evaluation
            quick_tasks = ["winogrande", "piqa"]
            task_results = run_task_evaluation(
                model_name,
                preset,
                quick_tasks,
                task_file,
                settings["task_samples"],
            )
            if "tasks" in task_results:
                tasks = task_results["tasks"]
                result.winogrande_acc = tasks.get("winogrande", {}).get("accuracy")
                result.piqa_acc = tasks.get("piqa", {}).get("accuracy")
        except Exception as e:
            print(f"  Task evaluation failed: {e}")
    else:
        print("\n[2/3] Skipping task evaluation (quick mode + large model)")

    # 3. Performance evaluation
    print("\n[3/3] Running performance evaluation...")
    perf_file = os.path.join(exp_dir, "perf_results.json")
    try:
        perf_results = run_performance_evaluation(
            model_name,
            preset,
            settings["context_lengths"],
            perf_file,
        )
        if "ttft" in perf_results and perf_results["ttft"]:
            # Use middle context length as reference
            mid_idx = len(perf_results["ttft"]) // 2
            result.ttft_ms = perf_results["ttft"][mid_idx].get("ttft_mean_ms")
        if "throughput" in perf_results and perf_results["throughput"]:
            mid_idx = len(perf_results["throughput"]) // 2
            result.tokens_per_sec = perf_results["throughput"][mid_idx].get(
                "tokens_per_second"
            )
            result.peak_memory_gb = perf_results["throughput"][mid_idx].get(
                "peak_memory_gb"
            )
        if "kv_cache" in perf_results and perf_results["kv_cache"]:
            mid_idx = len(perf_results["kv_cache"]) // 2
            result.kv_memory_mb = perf_results["kv_cache"][mid_idx].get("kv_cache_mb")
    except Exception as e:
        print(f"  Performance evaluation failed: {e}")

    result.status = "completed"
    return result


def save_results_jsonl(results: List[ExperimentResult], output_file: str):
    """Save results as JSONL (one JSON object per line)."""
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"Saved JSONL: {output_file}")


def save_results_csv(results: List[ExperimentResult], output_file: str):
    """Save results as CSV."""
    if not results:
        return

    fieldnames = list(asdict(results[0]).keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Saved CSV: {output_file}")


def print_summary_table(results: List[ExperimentResult]):
    """Print summary table to console."""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

    # Group by model
    by_model = {}
    for r in results:
        if r.model not in by_model:
            by_model[r.model] = []
        by_model[r.model].append(r)

    for model, model_results in by_model.items():
        print(f"\n--- {model} ---")
        print(
            f"{'Config':<20} {'Comp':>6} {'PPL':>8} {'WG':>6} {'PIQA':>6} "
            f"{'TTFT':>8} {'Tok/s':>8} {'KV MB':>8}"
        )
        print("-" * 90)

        for r in model_results:
            ppl = f"{r.ppl_wikitext:.2f}" if r.ppl_wikitext else "N/A"
            wg = f"{r.winogrande_acc*100:.1f}" if r.winogrande_acc else "N/A"
            piqa = f"{r.piqa_acc*100:.1f}" if r.piqa_acc else "N/A"
            ttft = f"{r.ttft_ms:.1f}" if r.ttft_ms else "N/A"
            tps = f"{r.tokens_per_sec:.0f}" if r.tokens_per_sec else "N/A"
            kv = f"{r.kv_memory_mb:.1f}" if r.kv_memory_mb else "N/A"

            print(
                f"{r.config:<20} {r.compression:>5.0f}x {ppl:>8} {wg:>6} {piqa:>6} "
                f"{ttft:>8} {tps:>8} {kv:>8}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run full KV plugin evaluation pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=list(CONFIGS.keys()),
        choices=list(CONFIGS.keys()),
        help="Configs to evaluate",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer samples, shorter contexts)",
    )
    parser.add_argument(
        "--skip-tasks",
        action="store_true",
        help="Skip task evaluation",
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="Skip performance evaluation",
    )
    args = parser.parse_args()

    # Select settings
    settings = QUICK_SETTINGS if args.quick else FULL_SETTINGS

    print("=" * 60)
    print("KV PLUGIN FULL EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {args.models}")
    print(f"Configs: {args.configs}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"PPL samples: {settings['ppl_samples']}")
    print(f"Task samples: {settings['task_samples']}")
    print(f"Context lengths: {settings['context_lengths']}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all experiments
    all_results = []
    total_experiments = len(args.models) * len(args.configs)
    current = 0

    start_time = time.time()

    for model_key in args.models:
        for config_key in args.configs:
            current += 1
            print(f"\n[Experiment {current}/{total_experiments}]")

            result = run_experiment(
                model_key,
                config_key,
                args.output_dir,
                settings,
            )
            all_results.append(result)

            # Save intermediate results
            save_results_jsonl(
                all_results,
                os.path.join(args.output_dir, "all_results.jsonl"),
            )

    elapsed = time.time() - start_time

    # Save final results
    save_results_jsonl(
        all_results,
        os.path.join(args.output_dir, "all_results.jsonl"),
    )
    save_results_csv(
        all_results,
        os.path.join(args.output_dir, "all_results.csv"),
    )

    # Print summary
    print_summary_table(all_results)

    print(f"\n{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
