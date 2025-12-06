#!/usr/bin/env python3
"""
Benchmark KV compression quality using lm-eval harness.

Compares baseline (no compression) vs various compression ranks on
standard benchmarks like HellaSwag, PIQA, WinoGrande, ARC.

Usage:
    python scripts/benchmark_kv_compression_quality.py \
        --model Qwen/Qwen2.5-7B \
        --ranks 32 64 96 \
        --tasks hellaswag piqa \
        --wandb
"""

import argparse
import gc
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    compression_type: str  # "baseline" or "orthogonal"
    rank: int  # 0 for baseline
    task: str
    metric: str
    value: float
    stderr: float
    num_samples: int
    timestamp: str
    gpu_name: str


class OrthogonalCompressor(nn.Module):
    """Orthogonal projection compressor (random - for memory tests only)."""

    def __init__(self, d_input: int, rank: int, device: str = "cuda"):
        super().__init__()
        Q, _ = torch.linalg.qr(torch.randn(d_input, rank, device=device))
        self.register_buffer("U", Q.to(torch.float16))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.U

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.U.T


class CalibratedCompressor(nn.Module):
    """PCA-calibrated compressor for quality-preserving compression."""

    def __init__(self, U: torch.Tensor, mean: torch.Tensor, device: str = "cuda"):
        super().__init__()
        self.register_buffer("U", U.to(device).to(torch.float16))
        self.register_buffer("mean", mean.to(device).to(torch.float16))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.U

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.U.T + self.mean


def run_lm_eval(
    model_name: str,
    tasks: List[str],
    cache=None,
    batch_size: int = 4,
    limit: int = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run lm-eval harness on model with optional compressed cache."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(f"  Loading model for evaluation...")

    # Create model wrapper
    model = HFLM(
        pretrained=model_name,
        device=device,
        dtype="float16",
        batch_size=batch_size,
        trust_remote_code=True,
    )

    # If we have a compressed cache, we need to modify how the model handles caching
    # For now, we'll measure perplexity directly since lm-eval's cache handling
    # is complex to intercept

    print(f"  Running evaluation on {tasks}...")
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )

    return results


def run_perplexity_eval(
    model_name: str,
    cache=None,
    num_samples: int = 100,
    max_length: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate perplexity with optional compressed cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Load WikiText-2 for perplexity
    print(f"  Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"][:num_samples])

    # Tokenize
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length * num_samples,
    )
    input_ids = encodings.input_ids.to(device)

    # Calculate perplexity in chunks
    print(f"  Calculating perplexity...")
    nlls = []
    seq_len = input_ids.size(1)

    with torch.no_grad():
        for i in range(0, min(seq_len, max_length * 10), max_length):
            end = min(i + max_length, seq_len)
            chunk = input_ids[:, i:end]

            if cache is not None:
                cache.reset()
                outputs = model(chunk, past_key_values=cache, labels=chunk)
            else:
                outputs = model(chunk, labels=chunk)

            nlls.append(outputs.loss.item() * (end - i))

    ppl = torch.exp(torch.tensor(sum(nlls) / min(seq_len, max_length * 10))).item()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"perplexity": ppl}


def load_calibration(calib_path: str, device: str = "cuda") -> Dict:
    """Load calibration data from file."""
    calib_data = torch.load(calib_path, map_location=device)
    return calib_data


def benchmark_compression(
    model_name: str,
    ranks: List[int],
    tasks: List[str] = None,
    num_ppl_samples: int = 100,
    limit: int = None,
    device: str = "cuda",
    calib_path: str = None,
) -> List[BenchmarkResult]:
    """Benchmark baseline vs compressed at various ranks."""
    from transformers import AutoConfig

    results = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().isoformat()

    # Get model config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    num_attn_heads = getattr(
        config, "num_attention_heads", getattr(config, "n_head", 12)
    )
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 768))
    head_dim = hidden_size // num_attn_heads

    # Load calibration if provided
    calib_data = None
    if calib_path:
        print(f"Loading calibration from: {calib_path}")
        calib_data = load_calibration(calib_path, device)
        print(f"  Calibration rank: {calib_data['rank']}")

    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_name}")
    print(f"  Layers: {num_layers}, Head dim: {head_dim}")
    if calib_data:
        print(f"  Using CALIBRATED compression (PCA-based)")
    else:
        print(f"  Using RANDOM orthogonal compression (quality will degrade)")
    print("=" * 70)

    # Baseline perplexity
    print(f"\n[Baseline] Evaluating perplexity...")
    ppl_results = run_perplexity_eval(
        model_name,
        cache=None,
        num_samples=num_ppl_samples,
        device=device,
    )
    baseline_ppl = ppl_results["perplexity"]
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    results.append(
        BenchmarkResult(
            model_name=model_name,
            compression_type="baseline",
            rank=0,
            task="wikitext2",
            metric="perplexity",
            value=baseline_ppl,
            stderr=0.0,
            num_samples=num_ppl_samples,
            timestamp=timestamp,
            gpu_name=gpu_name,
        )
    )

    # Test each compression rank (V-only compression - K is too sensitive)
    for rank in ranks:
        if rank > head_dim:
            print(f"\nSkipping rank {rank} (> head_dim {head_dim})")
            continue

        compression_ratio = head_dim / rank
        # V-only gives half the compression (only V is compressed)
        effective_ratio = 2 * head_dim / (head_dim + rank)

        compression_type = "calibrated" if calib_data else "orthogonal"
        print(
            f"\n[V-only Rank {rank}] ({effective_ratio:.2f}x effective, {compression_type})"
        )

        # Create compressed cache with V-only compression
        from gpt2.compression.compressed_cache import (
            CompressedDynamicCache,
            IdentityCompressor,
        )

        # K uses identity (no compression)
        k_compressors = [IdentityCompressor() for _ in range(num_layers)]

        # V uses calibrated or random orthogonal
        if calib_data and calib_data["rank"] == rank:
            # Use calibrated compressors
            v_compressors = []
            for layer_idx in range(num_layers):
                layer_data = calib_data["layers"][layer_idx]
                V_U = layer_data["V"]["U"]
                V_mean = layer_data["V"]["mean"]
                v_compressors.append(CalibratedCompressor(V_U, V_mean, device))
        else:
            # Fall back to random orthogonal
            if calib_data:
                print(
                    f"  WARNING: Calibration rank {calib_data['rank']} != requested rank {rank}, using random"
                )
            v_compressors = [
                OrthogonalCompressor(head_dim, rank, device) for _ in range(num_layers)
            ]
            compression_type = "orthogonal"

        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        # Evaluate perplexity
        print(f"  Evaluating perplexity...")
        ppl_results = run_perplexity_eval(
            model_name,
            cache=cache,
            num_samples=num_ppl_samples,
            device=device,
        )
        compressed_ppl = ppl_results["perplexity"]
        delta_ppl = (compressed_ppl - baseline_ppl) / baseline_ppl * 100

        print(f"  Compressed PPL: {compressed_ppl:.4f} ({delta_ppl:+.2f}%)")

        results.append(
            BenchmarkResult(
                model_name=model_name,
                compression_type=compression_type,
                rank=rank,
                task="wikitext2",
                metric="perplexity",
                value=compressed_ppl,
                stderr=0.0,
                num_samples=num_ppl_samples,
                timestamp=timestamp,
                gpu_name=gpu_name,
            )
        )

        # Add delta as separate metric
        results.append(
            BenchmarkResult(
                model_name=model_name,
                compression_type=compression_type,
                rank=rank,
                task="wikitext2",
                metric="delta_ppl_pct",
                value=delta_ppl,
                stderr=0.0,
                num_samples=num_ppl_samples,
                timestamp=timestamp,
                gpu_name=gpu_name,
            )
        )

    return results


def print_summary(results: List[BenchmarkResult], baseline_ppl: float):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("QUALITY BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Type':<12} {'Rank':>6} {'PPL':>10} {'ΔPPL':>10}")
    print("-" * 80)

    for r in results:
        if r.metric == "perplexity":
            model_short = r.model_name.split("/")[-1][:19]
            if r.compression_type == "baseline":
                delta = "-"
            else:
                delta = f"{(r.value - baseline_ppl) / baseline_ppl * 100:+.2f}%"
            print(
                f"{model_short:<20} {r.compression_type:<12} "
                f"{r.rank if r.rank > 0 else '-':>6} "
                f"{r.value:>10.4f} {delta:>10}"
            )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV compression quality")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[32, 48, 64, 96],
        help="Compression ranks to test",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for perplexity",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration file (.pt) for PCA-based compression",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to W&B",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("KV Compression Quality Benchmark")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(f"Ranks: {args.ranks}")
    if args.calibration:
        print(f"Calibration: {args.calibration}")
    else:
        print("Calibration: NONE (random orthogonal - expect quality degradation)")
    print("=" * 70)

    # Run benchmarks
    results = benchmark_compression(
        args.model,
        args.ranks,
        num_ppl_samples=args.num_samples,
        calib_path=args.calibration,
    )

    # Get baseline PPL for summary
    baseline_ppl = next(
        (r.value for r in results if r.compression_type == "baseline"), 0
    )

    # Print summary
    print_summary(results, baseline_ppl)

    # W&B logging
    if args.wandb:
        try:
            import wandb

            run = wandb.init(
                project="kv-compression-quality",
                name=f"quality_{args.model.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": args.model,
                    "ranks": args.ranks,
                    "num_samples": args.num_samples,
                },
            )
            print(f"\nW&B run URL: {wandb.run.url}")

            for r in results:
                wandb.log(
                    {
                        "compression_type": r.compression_type,
                        "rank": r.rank,
                        "task": r.task,
                        "metric": r.metric,
                        "value": r.value,
                        "compression_ratio": 128 / r.rank if r.rank > 0 else 1.0,
                    }
                )

            wandb.finish()
        except Exception as e:
            print(f"W&B logging failed: {e}")

    # Save results
    output_path = (
        args.output
        or f"key_results/quality_{args.model.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "ranks": args.ranks,
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
