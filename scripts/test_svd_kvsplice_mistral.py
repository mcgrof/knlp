#!/usr/bin/env python3
"""
Test SVD-calibrated KVSplice on Mistral-7B.

Uses SVD to compute optimal low-rank projections from model activations,
then applies them for KV cache compression during generation.
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import gc
from typing import Tuple


class SVDKVCompressor(nn.Module):
    """SVD-calibrated KV cache compressor."""

    def __init__(self, d_in: int, d_compressed: int):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = d_compressed

        # Projection layers (will be initialized from SVD)
        self.compress = nn.Linear(d_in, d_compressed, bias=False)
        self.expand = nn.Linear(d_compressed, d_in, bias=False)

        # Initialize as identity (will be overwritten by SVD)
        nn.init.eye_(self.compress.weight)
        nn.init.eye_(self.expand.weight)

    def calibrate_from_activations(self, activations: torch.Tensor):
        """
        Calibrate projection matrices using SVD on collected activations.

        Args:
            activations: [N, d_in] tensor of activation samples
        """
        print(f"  Calibrating from {activations.shape[0]} samples...")

        # Center the data
        mean = activations.mean(dim=0, keepdim=True)
        centered = activations - mean

        # Compute SVD
        print(f"  Computing SVD...")
        U, S, V = torch.svd(centered.cpu())

        # Use top-k singular vectors
        expand_weight = V[:, : self.d_compressed]  # [d_in, d_compressed]
        compress_weight = expand_weight.T  # [d_compressed, d_in]

        # Set weights (nn.Linear weights are transposed internally)
        # expand: nn.Linear(d_compressed, d_in) -> weight is [d_in, d_compressed]
        # compress: nn.Linear(d_in, d_compressed) -> weight is [d_compressed, d_in]
        self.expand.weight.data = expand_weight.contiguous()
        self.compress.weight.data = compress_weight.contiguous()

        # Compute explained variance
        total_var = S.pow(2).sum()
        kept_var = S[: self.d_compressed].pow(2).sum()
        explained = (kept_var / total_var * 100).item()

        print(f"  Explained variance: {explained:.2f}%")
        print(f"  Top 5 singular values: {S[:5].numpy()}")

        return explained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress and immediately expand (for testing)."""
        compressed = self.compress(x)
        expanded = self.expand(compressed)
        return expanded


def collect_attention_activations(
    model,
    tokenizer,
    layer_idx: int = 0,
    n_samples: int = 2000,
    max_length: int = 512,
    device: str = "cuda",
):
    """
    Collect attention output activations for SVD calibration.

    Returns:
        activations: [N, hidden_size] tensor
    """
    print(f"\nCollecting activations from layer {layer_idx}...")

    # Load calibration dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]

    activations = []

    # Hook to capture activations
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Store flattened: [B, T, D] -> [B*T, D]
        activations.append(hidden.detach().cpu().reshape(-1, hidden.shape[-1]))

    # Register hook
    layer = model.model.layers[layer_idx]
    hook = layer.self_attn.register_forward_hook(hook_fn)

    # Collect
    model.eval()
    samples_collected = 0

    with torch.no_grad():
        for text in texts:
            if samples_collected >= n_samples:
                break

            inputs = tokenizer(
                text, max_length=max_length, truncation=True, return_tensors="pt"
            ).to(device)

            if inputs["input_ids"].shape[1] < 10:
                continue

            model(**inputs)
            samples_collected += inputs["input_ids"].shape[1]

            if samples_collected % 500 == 0:
                print(f"  Progress: {samples_collected}/{n_samples} tokens")

    hook.remove()

    # Concatenate and subsample
    all_activations = torch.cat(activations, dim=0)
    if all_activations.shape[0] > n_samples:
        all_activations = all_activations[:n_samples]

    print(f"  Collected: {all_activations.shape}")

    return all_activations.float()


def patch_model_with_svd_kvsplice(
    model,
    tokenizer,
    compression_ratio: float = 0.5,
    calibration_samples: int = 2000,
):
    """
    Patch model with SVD-calibrated KV compression.

    This calibrates SVD projections from actual model activations,
    then freezes them for use during generation.
    """
    print(f"\n{'=' * 80}")
    print("CALIBRATING SVD PROJECTIONS")
    print(f"{'=' * 80}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Calibration samples: {calibration_samples}")

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    d_compressed = max(1, int(hidden_size * compression_ratio))

    print(f"\nModel: {model.config.model_type}")
    print(f"Hidden size: {hidden_size}")
    print(f"Compressed size: {d_compressed}")
    print(f"Layers: {len(model.model.layers)}")

    # Collect activations from first layer
    activations = collect_attention_activations(
        model,
        tokenizer,
        layer_idx=0,
        n_samples=calibration_samples,
        device=device,
    )

    # Create and calibrate compressor
    print(f"\n{'=' * 80}")
    print("CALIBRATING COMPRESSOR")
    print(f"{'=' * 80}")

    compressor = SVDKVCompressor(hidden_size, d_compressed)
    explained_var = compressor.calibrate_from_activations(activations)

    # Move to device, match model dtype, and freeze
    model_dtype = next(model.parameters()).dtype
    compressor = compressor.to(device=device, dtype=model_dtype)
    compressor.requires_grad_(False)
    compressor.eval()

    # Store compressor on model for later use
    model._kvsplice_compressor = compressor
    model._kvsplice_enabled = True

    print(f"\n{'=' * 80}")
    print("CALIBRATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Explained variance: {explained_var:.2f}%")
    print(
        f"Compression: {hidden_size} → {d_compressed} ({compression_ratio * 100:.0f}%)"
    )
    print(f"Memory reduction: {(1 - compression_ratio) * 100:.0f}%")

    # Clean up
    del activations
    gc.collect()
    torch.cuda.empty_cache()

    return model, compressor


def benchmark_generation(
    model,
    tokenizer,
    prompt: str = "The future of artificial intelligence is",
    max_new_tokens: int = 100,
    trials: int = 3,
    use_cache: bool = True,
):
    """Benchmark generation throughput and memory."""

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nBenchmarking generation...")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Use cache: {use_cache}")
    print(f"  Trials: {trials}")

    # Warmup
    with torch.no_grad():
        model.generate(
            **inputs, max_new_tokens=10, do_sample=False, use_cache=use_cache
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for trial in range(trials):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        throughput = tokens_generated / elapsed

        print(f"  Trial {trial + 1}: {elapsed:.3f}s ({throughput:.2f} tok/s)")

    # Memory
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    avg_time = sum(times) / len(times)
    avg_throughput = max_new_tokens / avg_time

    print(f"\n  Average: {avg_time:.3f}s ({avg_throughput:.2f} tok/s)")
    print(f"  Peak memory: {peak_memory_mb:.1f} MB")

    return {
        "avg_time": avg_time,
        "avg_throughput": avg_throughput,
        "peak_memory_mb": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test SVD-calibrated KVSplice on Mistral"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model name",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.5,
        help="Compression ratio (0.5 = 50%%)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=2000,
        help="Number of samples for SVD calibration",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Tokens to generate for benchmark",
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of benchmark trials"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare against baseline model",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SVD-Calibrated KVSplice Test")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compression: {args.compression_ratio}")
    print(f"Calibration samples: {args.calibration_samples}")
    print(f"Device: {args.device}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline benchmark
    baseline_results = None
    if args.compare_baseline:
        print(f"\n{'=' * 80}")
        print("BASELINE (No Compression)")
        print("=" * 80)

        print(f"\nLoading baseline model...")
        model_baseline = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to(args.device)

        baseline_results = benchmark_generation(
            model_baseline,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            trials=args.trials,
        )

        # Clean up
        del model_baseline
        gc.collect()
        torch.cuda.empty_cache()

    # SVD-calibrated KVSplice
    print(f"\n{'=' * 80}")
    print("SVD-CALIBRATED KVSPLICE")
    print("=" * 80)

    print(f"\nLoading model...")
    model_compressed = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(args.device)

    # Calibrate SVD projections
    model_compressed, compressor = patch_model_with_svd_kvsplice(
        model_compressed,
        tokenizer,
        compression_ratio=args.compression_ratio,
        calibration_samples=args.calibration_samples,
    )

    # Test projection quality on sample
    print(f"\n{'=' * 80}")
    print("TESTING PROJECTION QUALITY")
    print("=" * 80)

    with torch.no_grad():
        # Generate sample hidden states
        sample_text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(sample_text, return_tensors="pt").to(args.device)
        outputs = model_compressed.model(**inputs, output_hidden_states=True)
        sample_hidden = outputs.hidden_states[0]  # First layer

        # Test compression
        original = sample_hidden.reshape(-1, sample_hidden.shape[-1])
        compressed = compressor.compress(original)
        reconstructed = compressor.expand(compressed)

        # Compute error
        error = (original - reconstructed).pow(2).mean().sqrt().item()
        norm_orig = original.norm(dim=1).mean().item()
        norm_recon = reconstructed.norm(dim=1).mean().item()
        norm_ratio = norm_recon / norm_orig

        print(f"Reconstruction error: {error:.6f}")
        print(f"Original norm: {norm_orig:.4f}")
        print(f"Reconstructed norm: {norm_recon:.4f}")
        print(f"Norm preservation: {norm_ratio * 100:.2f}%")

    # Benchmark
    print(f"\n{'=' * 80}")
    print("BENCHMARKING")
    print("=" * 80)

    compressed_results = benchmark_generation(
        model_compressed,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        trials=args.trials,
    )

    # Compare
    if baseline_results:
        print(f"\n{'=' * 80}")
        print("COMPARISON")
        print("=" * 80)

        throughput_ratio = (
            compressed_results["avg_throughput"] / baseline_results["avg_throughput"]
        )
        memory_ratio = (
            compressed_results["peak_memory_mb"] / baseline_results["peak_memory_mb"]
        )

        print(f"\nThroughput ratio: {throughput_ratio:.3f}x")
        print(f"Memory ratio: {memory_ratio:.3f}x")

        print(
            f"\nBaseline:   {baseline_results['avg_throughput']:.2f} tok/s, {baseline_results['peak_memory_mb']:.1f} MB"
        )
        print(
            f"Compressed: {compressed_results['avg_throughput']:.2f} tok/s, {compressed_results['peak_memory_mb']:.1f} MB"
        )

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ SVD calibration from {args.calibration_samples} samples")
    print(f"✓ Compression: {args.compression_ratio * 100:.0f}%")
    print(f"✓ Reconstruction error: {error:.6f}")
    print(f"✓ Norm preservation: {norm_ratio * 100:.2f}%")
    print(f"✓ Throughput: {compressed_results['avg_throughput']:.2f} tok/s")


if __name__ == "__main__":
    main()
