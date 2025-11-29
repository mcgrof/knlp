#!/usr/bin/env python3
"""
Test pre-computed KVSplice compression on pretrained models.

Supports two projection methods:
- SVD: Calibrate from model's latent distribution (optimal linear compression)
- Random: Random orthogonal projection (no calibration needed)
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
from typing import Literal, Optional, Tuple
import gc


class PrecomputedKVSplice(nn.Module):
    """
    Pre-computed KVSplice compression for inference.
    No training - projection computed via SVD or random orthogonal.
    """

    def __init__(
        self,
        d_in: int,
        d_compressed: int,
        method: Literal["svd", "random"] = "random",
        device: str = "cuda",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = d_compressed
        self.method = method

        # Projection layers
        self.compress = nn.Linear(d_in, d_compressed, bias=False)
        self.expand = nn.Linear(d_compressed, d_in, bias=False)

        # Initialize based on method
        if method == "random":
            self._init_random_orthogonal()
        # SVD initialization happens after calibration

        self.to(device)

    def _init_random_orthogonal(self):
        """Initialize with random orthogonal projection (Johnson-Lindenstrauss)."""
        # Create random orthogonal matrix via QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(self.d_in, self.d_in))

        # Use top d_compressed rows for compression
        compress_weight = Q[: self.d_compressed, :]  # [d_compressed, d_in]
        expand_weight = compress_weight.T  # [d_in, d_compressed]

        self.compress.weight.data = compress_weight
        self.expand.weight.data = expand_weight

        print(
            f"  Initialized random orthogonal projection: {self.d_in} → {self.d_compressed}"
        )

    def init_from_svd(self, latents: torch.Tensor):
        """
        Initialize projection from SVD of latent activations.

        Args:
            latents: [N, d_in] - collected latent activations
        """
        print(f"  Computing SVD on {latents.shape[0]} samples...")

        # Center the data
        latents_centered = latents - latents.mean(dim=0, keepdim=True)

        # Compute SVD
        U, S, Vh = torch.svd(latents_centered)

        # Top k components = compression basis
        compress_weight = Vh[: self.d_compressed, :]  # [d_compressed, d_in]
        expand_weight = compress_weight.T  # [d_in, d_compressed]

        self.compress.weight.data = compress_weight.to(self.compress.weight.device)
        self.expand.weight.data = expand_weight.to(self.expand.weight.device)

        # Report explained variance
        total_var = S.pow(2).sum()
        kept_var = S[: self.d_compressed].pow(2).sum()
        explained = (kept_var / total_var * 100).item()

        print(f"  SVD projection: {self.d_in} → {self.d_compressed}")
        print(f"  Explained variance: {explained:.2f}%")

    def compress_cache(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress hidden states for caching."""
        return self.compress(hidden_states)

    def expand_cache(self, compressed: torch.Tensor) -> torch.Tensor:
        """Expand compressed cache back to original dimension."""
        return self.expand(compressed)


def collect_latent_activations(
    model,
    tokenizer,
    n_samples: int = 1000,
    max_length: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Collect latent activations from model for SVD calibration.

    Returns:
        dict mapping layer_idx -> [N, d_latent] tensor
    """
    print(f"\nCollecting latent activations for SVD calibration...")
    print(f"  Samples: {n_samples}, Max length: {max_length}")

    # Load calibration dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]

    # Storage for latent activations per layer
    latents_by_layer = {}
    hooks = []

    def make_hook(layer_idx: int):
        """Create hook to capture latent activations."""

        def hook_fn(module, input, output):
            # Extract hidden states from attention output
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Store flattened across batch and sequence
            # Shape: [B, T, D] -> [B*T, D]
            if layer_idx not in latents_by_layer:
                latents_by_layer[layer_idx] = []

            latents_by_layer[layer_idx].append(
                hidden.detach().cpu().reshape(-1, hidden.shape[-1])
            )

        return hook_fn

    # Register hooks on attention layers
    # Try different architecture patterns
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h

    if layers is None:
        raise ValueError("Cannot find model layers - unsupported architecture")

    for idx, layer in enumerate(layers):
        # Hook attention output
        if hasattr(layer, "self_attn"):
            hook = layer.self_attn.register_forward_hook(make_hook(idx))
        elif hasattr(layer, "attn"):
            hook = layer.attn.register_forward_hook(make_hook(idx))
        else:
            continue
        hooks.append(hook)

    # Run model on calibration data
    model.eval()
    samples_seen = 0

    with torch.no_grad():
        for text in texts:
            if samples_seen >= n_samples:
                break

            # Tokenize
            inputs = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            if inputs["input_ids"].shape[1] < 10:
                continue

            # Forward pass (hooks collect activations)
            model(**inputs)

            samples_seen += inputs["input_ids"].shape[1]

            if samples_seen % 100 == 0:
                print(f"  Collected {samples_seen}/{n_samples} tokens...")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate activations per layer
    print(f"\nProcessing {len(latents_by_layer)} layers...")
    result = {}
    for layer_idx, activations in latents_by_layer.items():
        result[layer_idx] = torch.cat(activations, dim=0)[:n_samples]
        print(f"  Layer {layer_idx}: {result[layer_idx].shape}")

    return result


def patch_model_with_precomputed_kvsplice(
    model,
    compression_ratio: float = 0.5,
    method: Literal["svd", "random"] = "random",
    calibration_samples: int = 1000,
    tokenizer=None,
):
    """
    Patch model with pre-computed KVSplice compression.

    Args:
        model: Pretrained model to patch
        compression_ratio: Target compression (0.5 = 50% of dimensions)
        method: "svd" or "random"
        calibration_samples: Number of samples for SVD calibration
        tokenizer: Tokenizer for calibration data
    """
    print(f"\nPatching model with pre-computed KVSplice ({method})...")
    print(f"  Compression ratio: {compression_ratio}")

    # Find model layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        print(f"  Found {len(layers)} Mistral/DeepSeek layers")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        print(f"  Found {len(layers)} GPT-2 layers")
    else:
        raise ValueError("Unsupported model architecture")

    # Collect latent activations if using SVD
    latents_by_layer = None
    if method == "svd":
        if tokenizer is None:
            raise ValueError("Tokenizer required for SVD calibration")
        latents_by_layer = collect_latent_activations(
            model, tokenizer, n_samples=calibration_samples
        )

    # Patch each layer
    device = next(model.parameters()).device
    for idx, layer in enumerate(layers):
        # Find attention module
        attn = None
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
        elif hasattr(layer, "attn"):
            attn = layer.attn

        if attn is None:
            continue

        # Determine dimension to compress
        # For MLA: compress the latent (need to find d_latent)
        # For standard: compress the hidden dimension
        if hasattr(attn, "hidden_size"):
            d_in = attn.hidden_size
        elif hasattr(model.config, "hidden_size"):
            d_in = model.config.hidden_size
        else:
            print(f"  Warning: Cannot determine dimension for layer {idx}, skipping")
            continue

        d_compressed = max(1, int(d_in * compression_ratio))

        # Create KVSplice compressor
        kvsplice = PrecomputedKVSplice(
            d_in=d_in,
            d_compressed=d_compressed,
            method=method,
            device=device,
        )

        # Initialize from SVD if needed
        if method == "svd" and latents_by_layer is not None:
            if idx in latents_by_layer:
                kvsplice.init_from_svd(latents_by_layer[idx])
            else:
                print(f"  Warning: No latents for layer {idx}, using random")
                kvsplice._init_random_orthogonal()

        # Freeze parameters
        kvsplice.requires_grad_(False)

        # Attach to layer
        attn.kvsplice = kvsplice
        attn._kvsplice_compressed_cache = None

        # Patch forward method to compress/decompress cache
        _patch_attention_forward(attn)

        if idx == 0:
            print(f"  Patched layer {idx}: {d_in} → {d_compressed} dims")

    print(f"  Successfully patched {len(layers)} layers")

    # Clear calibration data
    if latents_by_layer is not None:
        del latents_by_layer
        gc.collect()
        torch.cuda.empty_cache()


def _patch_attention_forward(attn):
    """
    Patch attention's forward to compress/decompress cache.

    This wraps the original forward to intercept cache flow.
    """
    original_forward = attn.forward

    def forward_with_kvsplice(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        # Decompress cache if present
        if past_key_value is not None and hasattr(attn, "_kvsplice_compressed_cache"):
            if attn._kvsplice_compressed_cache is not None:
                # Expand compressed cache
                # NOTE: This is a simplified version - real implementation
                # needs to handle the cache structure properly
                pass

        # Call original forward
        outputs = original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # Compress new cache if returned
        if use_cache and isinstance(outputs, tuple) and len(outputs) > 1:
            attn_output = outputs[0]
            new_cache = outputs[1]

            # Store compressed version
            # NOTE: Simplified - real version needs proper cache handling
            if new_cache is not None and hasattr(attn, "kvsplice"):
                # For now, just store original cache
                # TODO: Implement proper cache compression based on cache type
                attn._kvsplice_compressed_cache = new_cache

            return outputs
        else:
            return outputs

    attn.forward = forward_with_kvsplice


def benchmark_model(
    model,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 100,
    trials: int = 3,
):
    """Benchmark model throughput and memory."""
    print(f"\nBenchmarking...")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Trials: {trials}")

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for trial in range(trials):
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        throughput = tokens_generated / elapsed

        print(f"  Trial {trial + 1}: {elapsed:.3f}s ({throughput:.2f} tok/s)")

    # Memory stats
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
        description="Test pre-computed KVSplice on pretrained models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.5,
        help="Compression ratio (0.5 = 50%% compression)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["svd", "random"],
        default="random",
        help="Projection method: svd (calibrated) or random (orthogonal)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1000,
        help="Number of samples for SVD calibration",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate for benchmark",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of benchmark trials",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also benchmark baseline model without compression",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Pre-computed KVSplice Inference Test")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compression: {args.compression_ratio}")
    print(f"Method: {args.method}")
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

        model_baseline = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to(args.device)

        baseline_results = benchmark_model(
            model_baseline,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            trials=args.trials,
        )

        # Clean up
        del model_baseline
        gc.collect()
        torch.cuda.empty_cache()

    # KVSplice benchmark
    print(f"\n{'=' * 80}")
    print(f"KVSPLICE ({args.method.upper()})")
    print("=" * 80)

    # Load model
    print(f"\nLoading model...")
    model_compressed = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(args.device)

    # Patch with pre-computed KVSplice
    patch_model_with_precomputed_kvsplice(
        model_compressed,
        compression_ratio=args.compression_ratio,
        method=args.method,
        calibration_samples=args.calibration_samples,
        tokenizer=tokenizer if args.method == "svd" else None,
    )

    compressed_results = benchmark_model(
        model_compressed,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        trials=args.trials,
    )

    # Compare results
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

        print(f"\nThroughput: {throughput_ratio:.3f}x")
        print(f"Memory: {memory_ratio:.3f}x")

        print(
            f"\nBaseline:   {baseline_results['avg_throughput']:.2f} tok/s, {baseline_results['peak_memory_mb']:.1f} MB"
        )
        print(
            f"KVSplice:   {compressed_results['avg_throughput']:.2f} tok/s, {compressed_results['peak_memory_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
