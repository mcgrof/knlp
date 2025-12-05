#!/usr/bin/env python3
"""
Post-Bugfix PPL Validation Script.

Validates that the fixed CompressedDynamicCache actually affects inference.
This script proves that compression is NOW on the hot path (v6 bugfix).

Tests:
1. Baseline (no cache modification)
2. Identity compressor (should match baseline)
3. Low-rank only (orthogonal projection)
4. Low-rank + int8 quantization
5. Low-rank + int4 quantization

Usage:
    python scripts/validate_ppl_postbugfix.py
    python scripts/validate_ppl_postbugfix.py --model Qwen/Qwen2.5-0.5B --quick
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


class OrthogonalCompressorSimple(nn.Module):
    """
    Simple orthogonal low-rank compressor.

    Uses SVD-initialized projection matrices for compression.
    """

    def __init__(
        self,
        d_input: int,
        d_compressed: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_compressed = d_compressed
        self.dtype = dtype

        # Initialize with orthogonal projection (SVD needs float32)
        # W: [d_input, d_compressed], U: [d_input, k], Vt: [k, d_compressed]
        W = torch.randn(d_input, d_compressed, dtype=torch.float32, device=device)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # compress_proj: [d_input, d_compressed] for x @ compress_proj
        # expand_proj: [d_compressed, d_input] for z @ expand_proj
        compress_proj = U[:, :d_compressed]  # [d_input, d_compressed]
        expand_proj = U[:, :d_compressed].T  # [d_compressed, d_input]
        self.register_buffer("compress_proj", compress_proj.contiguous().to(dtype))
        self.register_buffer("expand_proj", expand_proj.contiguous().to(dtype))

        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project to low-rank subspace: [*, d_input] -> [*, d_compressed]."""
        return x @ self.compress_proj

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from low-rank: [*, d_compressed] -> [*, d_input]."""
        return z @ self.expand_proj

    def calibrate(self, data: torch.Tensor) -> None:
        """Optional calibration (uses data to refine projection)."""
        pass


class QuantizedCompressor(nn.Module):
    """
    Low-rank compressor with quantization.

    Compresses to low-rank, then quantizes to int8 or int4.
    """

    def __init__(
        self,
        d_input: int,
        d_compressed: int,
        bits: int = 8,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_compressed = d_compressed
        self.bits = bits
        self.dtype = dtype

        # Initialize with orthogonal projection (SVD needs float32)
        W = torch.randn(d_input, d_compressed, dtype=torch.float32, device=device)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        compress_proj = U[:, :d_compressed]  # [d_input, d_compressed]
        expand_proj = U[:, :d_compressed].T  # [d_compressed, d_input]
        self.register_buffer("compress_proj", compress_proj.contiguous().to(dtype))
        self.register_buffer("expand_proj", expand_proj.contiguous().to(dtype))

        # Quantization range
        if bits == 8:
            self.qmin, self.qmax = -128, 127
        elif bits == 4:
            self.qmin, self.qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project to low-rank and quantize."""
        latent = x @ self.compress_proj

        # Quantize per-token
        absmax = latent.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / self.qmax
        quantized = (latent / scale).round().clamp(self.qmin, self.qmax)

        # Store scale in last dimension (simple approach)
        # Return dequantized for now (proper impl would store quantized)
        dequantized = quantized * scale
        return dequantized

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from low-rank representation."""
        return z @ self.expand_proj

    def calibrate(self, data: torch.Tensor) -> None:
        pass


def measure_ppl(
    model: nn.Module,
    tokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 1024,
    cache: Optional[CompressedDynamicCache] = None,
) -> Tuple[float, float]:
    """
    Measure perplexity on text.

    Returns:
        (perplexity, time_seconds)
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings.input_ids.to(device)

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    with torch.no_grad():
        if cache is not None:
            outputs = model(input_ids, labels=input_ids, past_key_values=cache)
        else:
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - start

    ppl = torch.exp(loss).item()
    return ppl, elapsed


def create_compressors(
    num_layers: int,
    d_model: int,
    compression_type: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> Tuple[List[nn.Module], List[nn.Module], str]:
    """
    Create compressors for K and V.

    Args:
        num_layers: Number of transformer layers
        d_model: Model dimension (head_dim)
        compression_type: One of 'identity', 'lowrank', 'int8', 'int4'
        device: Device to create on
        dtype: Data type

    Returns:
        (k_compressors, v_compressors, description)
    """
    if compression_type == "identity":
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        desc = "Identity (1x)"
    elif compression_type == "lowrank_conservative":
        # Very conservative: only drop 25% of dims
        d_compressed = (d_model * 3) // 4  # 48 for head_dim=64
        k_comp = [
            OrthogonalCompressorSimple(d_model, d_compressed, dtype, device)
            for _ in range(num_layers)
        ]
        v_comp = [
            OrthogonalCompressorSimple(d_model, d_compressed, dtype, device)
            for _ in range(num_layers)
        ]
        desc = f"Low-rank 1.33x ({d_model}->{d_compressed})"
    elif compression_type == "lowrank_2x":
        # Moderate: 2x compression
        d_compressed = d_model // 2  # 32 for head_dim=64
        k_comp = [
            OrthogonalCompressorSimple(d_model, d_compressed, dtype, device)
            for _ in range(num_layers)
        ]
        v_comp = [
            OrthogonalCompressorSimple(d_model, d_compressed, dtype, device)
            for _ in range(num_layers)
        ]
        desc = f"Low-rank 2x ({d_model}->{d_compressed})"
    elif compression_type == "lowrank":
        # Aggressive: 4x compression
        d_compressed = d_model // 4  # 16 for head_dim=64
        k_comp = [
            OrthogonalCompressorSimple(d_model, d_compressed, dtype, device)
            for _ in range(num_layers)
        ]
        v_comp = [
            OrthogonalCompressorSimple(d_model, d_compressed, dtype, device)
            for _ in range(num_layers)
        ]
        desc = f"Low-rank 4x ({d_model}->{d_compressed})"
    elif compression_type == "int8":
        d_compressed = d_model // 2  # Use 2x for int8 (more reasonable)
        k_comp = [
            QuantizedCompressor(
                d_model, d_compressed, bits=8, dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        v_comp = [
            QuantizedCompressor(
                d_model, d_compressed, bits=8, dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        desc = f"Low-rank 2x + int8 ({d_model}->{d_compressed})"
    elif compression_type == "int4":
        d_compressed = d_model // 2  # Use 2x for int4
        k_comp = [
            QuantizedCompressor(
                d_model, d_compressed, bits=4, dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        v_comp = [
            QuantizedCompressor(
                d_model, d_compressed, bits=4, dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        desc = f"Low-rank 2x + int4 ({d_model}->{d_compressed})"
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    return k_comp, v_comp, desc


def create_calibrated_compressors(
    calib_path: str,
    device: torch.device,
    target: str = "kv",
    quantize_bits: int = None,
) -> Tuple[List[nn.Module], List[nn.Module], str, Dict]:
    """
    Load calibrated compressors from file.

    Args:
        calib_path: Path to calibration file
        device: Device to load to
        target: Which to compress - "k", "v", or "kv" (both)
        quantize_bits: If set, apply int8/int4 quantization in latent space

    Returns:
        (k_compressors, v_compressors, description, metadata)
    """
    k_comp, v_comp, metadata = load_calibrated_compressors(
        calib_path, device=device, dtype=torch.float16, quantize_bits=quantize_bits
    )

    num_layers = metadata["n_layers"]
    ratio = metadata["compression_ratio"]
    rank = metadata["rank"]

    # Build description
    quant_str = f"+int{quantize_bits}" if quantize_bits else ""

    if target == "k":
        # K-only: use identity for V
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        desc = f"K-only {ratio:.2f}x (r={rank}){quant_str}"
    elif target == "v":
        # V-only: use identity for K
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        desc = f"V-only {ratio:.2f}x (r={rank}){quant_str}"
    else:
        desc = f"K+V {ratio:.2f}x (r={rank}){quant_str}"

    return k_comp, v_comp, desc, metadata


def run_validation(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cuda",
    quick: bool = False,
) -> Dict:
    """
    Run full PPL validation suite.

    Returns dict with results for each configuration.
    """
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    # For Qwen, head_dim is hidden_size / num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    dtype = torch.float16

    print(f"  Layers: {num_layers}")
    print(f"  Head dim: {head_dim}")

    # Evaluation text
    if quick:
        eval_text = (
            """
        Machine learning has transformed how we approach complex problems.
        Neural networks can learn patterns from data without explicit programming.
        """
            * 10
        )
    else:
        eval_text = (
            """
        Machine learning models have become increasingly powerful over the past decade.
        Large language models can generate coherent text, answer questions, and assist
        with various tasks. The computational requirements continue to grow.

        Transformer architectures have revolutionized natural language processing.
        Attention mechanisms allow models to focus on relevant parts of the input.
        Key-value caching enables efficient autoregressive generation by storing
        previously computed keys and values for reuse in subsequent tokens.

        Compression techniques for KV caches are essential for deploying large
        language models efficiently. Methods include low-rank approximation,
        quantization, and learned projections. The goal is to reduce memory
        usage while maintaining model quality.
        """
            * 5
        )

    results = {}

    # 1. Baseline (no cache modification)
    print("\n[1/5] Baseline (no compressed cache)...")
    baseline_ppl, baseline_time = measure_ppl(model, tokenizer, eval_text, device)
    results["baseline"] = {
        "ppl": baseline_ppl,
        "time": baseline_time,
        "description": "Baseline (no compression)",
    }
    print(f"  PPL: {baseline_ppl:.4f}, Time: {baseline_time:.3f}s")

    # Test configurations (progressive compression levels)
    configs = [
        "identity",
        "lowrank_conservative",
        "lowrank_2x",
        "lowrank",
        "int8",
        "int4",
    ]

    total_tests = len(configs) + 1  # +1 for baseline
    for i, comp_type in enumerate(configs, start=2):
        print(f"\n[{i}/{total_tests}] Testing {comp_type}...")

        # Create compressors
        k_comp, v_comp, desc = create_compressors(
            num_layers, head_dim, comp_type, torch.device(device), dtype
        )

        # Create cache
        cache = CompressedDynamicCache(k_comp, v_comp, num_layers)

        # Measure PPL
        try:
            ppl, elapsed = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100

            results[comp_type] = {
                "ppl": ppl,
                "time": elapsed,
                "delta_pct": delta,
                "description": desc,
            }
            print(f"  PPL: {ppl:.4f} ({delta:+.2f}%), Time: {elapsed:.3f}s")
        except Exception as e:
            results[comp_type] = {"error": str(e), "description": desc}
            print(f"  Error: {e}")

        # Clean up
        del cache, k_comp, v_comp
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("POST-BUGFIX PPL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print("-" * 70)
    print(f"{'Method':<30} {'PPL':>10} {'Delta':>10} {'Status':>15}")
    print("-" * 70)

    for name, data in results.items():
        if "error" in data:
            print(f"{data['description']:<30} {'ERROR':>10} {'N/A':>10} {'FAILED':>15}")
        else:
            ppl = data["ppl"]
            delta = data.get("delta_pct", 0)
            status = "PASS" if abs(delta) < 10 else "DEGRADED"
            if name == "identity" and abs(delta) > 0.1:
                status = "BUG!"
            print(
                f"{data['description']:<30} {ppl:>10.4f} {delta:>+9.2f}% {status:>15}"
            )

    print("-" * 70)

    # Critical checks
    print("\nCRITICAL VALIDATION CHECKS:")

    # Check 1: Identity should match baseline
    if "identity" in results and "error" not in results["identity"]:
        id_delta = abs(results["identity"]["delta_pct"])
        if id_delta < 0.1:
            print(f"  [PASS] Identity matches baseline (delta={id_delta:.4f}%)")
        else:
            print(f"  [FAIL] Identity differs from baseline (delta={id_delta:.4f}%)")
            print("         This indicates a BUG in cache integration!")

    # Check 2: Low-rank should show SOME degradation (proves compression is active)
    if "lowrank" in results and "error" not in results["lowrank"]:
        lr_delta = results["lowrank"]["delta_pct"]
        if lr_delta > 0.1:
            print(
                f"  [PASS] Low-rank shows degradation ({lr_delta:+.2f}%) - compression IS active"
            )
        else:
            print(f"  [WARN] Low-rank shows no degradation ({lr_delta:+.2f}%)")
            print("         This may indicate compression is still bypassed!")

    # Check 3: Quantization should show more degradation than low-rank alone
    if all(k in results and "error" not in results[k] for k in ["lowrank", "int8"]):
        lr_ppl = results["lowrank"]["ppl"]
        int8_ppl = results["int8"]["ppl"]
        if int8_ppl >= lr_ppl:
            print(
                f"  [PASS] int8 ({int8_ppl:.4f}) >= lowrank ({lr_ppl:.4f}) - quantization active"
            )
        else:
            print(
                f"  [INFO] int8 ({int8_ppl:.4f}) < lowrank ({lr_ppl:.4f}) - unexpected"
            )

    return results


def run_calibrated_validation(
    model_name: str,
    calib_files: List[str],
    device: str = "cuda",
    quick: bool = False,
) -> Dict:
    """
    Run PPL validation with calibrated compressors.

    Args:
        model_name: HuggingFace model name
        calib_files: List of calibration file paths
        device: Device to use
        quick: Use shorter text

    Returns:
        Dict with results
    """
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Calibration files: {len(calib_files)}")
    print("=" * 70)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    print(f"  Layers: {num_layers}")
    print(f"  Head dim: {head_dim}")

    # Evaluation text
    if quick:
        eval_text = (
            """
        Machine learning has transformed how we approach complex problems.
        Neural networks can learn patterns from data without explicit programming.
        """
            * 10
        )
    else:
        eval_text = (
            """
        Machine learning models have become increasingly powerful over the past decade.
        Large language models can generate coherent text, answer questions, and assist
        with various tasks. The computational requirements continue to grow.

        Transformer architectures have revolutionized natural language processing.
        Attention mechanisms allow models to focus on relevant parts of the input.
        Key-value caching enables efficient autoregressive generation by storing
        previously computed keys and values for reuse in subsequent tokens.

        Compression techniques for KV caches are essential for deploying large
        language models efficiently. Methods include low-rank approximation,
        quantization, and learned projections. The goal is to reduce memory
        usage while maintaining model quality.
        """
            * 5
        )

    results = {}

    # 1. Baseline
    print("\n[1] Baseline (no compressed cache)...")
    baseline_ppl, baseline_time = measure_ppl(model, tokenizer, eval_text, device)
    results["baseline"] = {
        "ppl": baseline_ppl,
        "time": baseline_time,
        "description": "Baseline",
    }
    print(f"  PPL: {baseline_ppl:.4f}, Time: {baseline_time:.3f}s")

    # 2. Identity
    print("\n[2] Identity compressor...")
    k_comp = [IdentityCompressor() for _ in range(num_layers)]
    v_comp = [IdentityCompressor() for _ in range(num_layers)]
    cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
    ppl, elapsed = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    results["identity"] = {
        "ppl": ppl,
        "time": elapsed,
        "delta_pct": delta,
        "description": "Identity (1x)",
    }
    print(f"  PPL: {ppl:.4f} ({delta:+.2f}%), Time: {elapsed:.3f}s")
    del cache

    # 3. Test each calibration file
    for i, calib_path in enumerate(calib_files, start=3):
        print(f"\n[{i}] Testing {Path(calib_path).name}...")
        try:
            k_comp, v_comp, desc, metadata = create_calibrated_compressors(
                calib_path, torch.device(device)
            )
            cache = CompressedDynamicCache(k_comp, v_comp, num_layers)

            ppl, elapsed = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100

            key = f"calibrated_{metadata['rank']}"
            results[key] = {
                "ppl": ppl,
                "time": elapsed,
                "delta_pct": delta,
                "description": desc,
                "rank": metadata["rank"],
                "compression_ratio": metadata["compression_ratio"],
            }
            print(f"  PPL: {ppl:.4f} ({delta:+.2f}%), Time: {elapsed:.3f}s")

            del cache, k_comp, v_comp
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            results[f"calibrated_{i}"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATED LOW-RANK PPL VALIDATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print("-" * 70)
    print(f"{'Method':<35} {'PPL':>10} {'Delta':>10} {'Status':>12}")
    print("-" * 70)

    for name, data in results.items():
        if "error" in data:
            print(f"{name:<35} {'ERROR':>10} {'N/A':>10} {'FAILED':>12}")
        else:
            ppl = data["ppl"]
            delta = data.get("delta_pct", 0)
            if delta < 1:
                status = "EXCELLENT"
            elif delta < 5:
                status = "GOOD"
            elif delta < 10:
                status = "OK"
            else:
                status = "DEGRADED"
            if name == "identity" and abs(delta) > 0.1:
                status = "BUG!"
            print(
                f"{data['description']:<35} {ppl:>10.4f} {delta:>+9.2f}% {status:>12}"
            )

    print("-" * 70)

    return results


def run_kv_ablation(
    model_name: str,
    calib_files: List[str],
    device: str = "cuda",
    quick: bool = False,
) -> Dict:
    """
    Run K vs V compression ablation.

    For each calibration file, tests:
    - K-only compression (V stays full)
    - V-only compression (K stays full)
    - K+V compression (both compressed)
    """
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("K vs V Compression Ablation")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Evaluation text
    if quick:
        eval_text = (
            """
        Machine learning has transformed how we approach complex problems.
        Neural networks can learn patterns from data without explicit programming.
        """
            * 10
        )
    else:
        eval_text = (
            """
        Machine learning models have become increasingly powerful over the past decade.
        Large language models can generate coherent text, answer questions, and assist
        with various tasks. The computational requirements continue to grow.

        Transformer architectures have revolutionized natural language processing.
        Attention mechanisms allow models to focus on relevant parts of the input.
        """
            * 5
        )

    results = {}

    # Baseline
    print("\n[1] Baseline...")
    baseline_ppl, _ = measure_ppl(model, tokenizer, eval_text, device)
    results["baseline"] = {"ppl": baseline_ppl, "description": "Baseline"}
    print(f"  PPL: {baseline_ppl:.4f}")

    # Test each calibration file with K-only, V-only, K+V
    test_num = 2
    for calib_path in calib_files:
        calib_name = Path(calib_path).stem

        for target in ["k", "v", "kv"]:
            print(f"\n[{test_num}] {calib_name} - {target.upper()}...")
            test_num += 1

            try:
                k_comp, v_comp, desc, metadata = create_calibrated_compressors(
                    calib_path, torch.device(device), target=target
                )
                cache = CompressedDynamicCache(k_comp, v_comp, num_layers)

                ppl, elapsed = measure_ppl(
                    model, tokenizer, eval_text, device, cache=cache
                )
                delta = ((ppl - baseline_ppl) / baseline_ppl) * 100

                key = f"r{metadata['rank']}_{target}"
                results[key] = {
                    "ppl": ppl,
                    "delta_pct": delta,
                    "description": desc,
                    "rank": metadata["rank"],
                    "target": target,
                }
                print(f"  PPL: {ppl:.4f} ({delta:+.2f}%)")

                del cache, k_comp, v_comp
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error: {e}")

    # Summary table
    print("\n" + "=" * 80)
    print("K vs V ABLATION SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print("-" * 80)
    print(f"{'Rank':<8} {'Target':<8} {'PPL':>10} {'Delta':>10} {'Status':>12}")
    print("-" * 80)

    for name, data in results.items():
        if name == "baseline":
            continue
        ppl = data["ppl"]
        delta = data["delta_pct"]
        target = data.get("target", "kv")
        rank = data.get("rank", "-")
        status = "GOOD" if delta < 5 else ("OK" if delta < 10 else "DEGRADED")
        print(
            f"{rank:<8} {target.upper():<8} {ppl:>10.4f} {delta:>+9.2f}% {status:>12}"
        )

    print("-" * 80)

    # Analysis
    print("\nANALYSIS:")
    for calib_path in calib_files:
        # Extract rank from filename
        import re

        match = re.search(r"r(\d+)", calib_path)
        if match:
            rank = int(match.group(1))
            k_key = f"r{rank}_k"
            v_key = f"r{rank}_v"
            kv_key = f"r{rank}_kv"

            if all(k in results for k in [k_key, v_key, kv_key]):
                k_delta = results[k_key]["delta_pct"]
                v_delta = results[v_key]["delta_pct"]
                kv_delta = results[kv_key]["delta_pct"]

                print(f"\n  Rank {rank}:")
                print(f"    K-only:  {k_delta:+.2f}%")
                print(f"    V-only:  {v_delta:+.2f}%")
                print(f"    K+V:     {kv_delta:+.2f}%")

                if k_delta < v_delta:
                    print(f"    -> K is MORE compressible than V")
                else:
                    print(f"    -> V is MORE compressible than K")

    return results


def run_layer_ablation(
    model_name: str,
    calib_path: str,
    device: str = "cuda",
    quick: bool = False,
    target: str = "kv",
) -> Dict:
    """
    Run per-layer sensitivity sweep.

    For a single calibration file, compress each layer one at a time
    while keeping all others at full precision.

    Args:
        model_name: HuggingFace model name
        calib_path: Path to calibration file
        device: Device to use
        quick: Use shorter text
        target: Which to compress - "k", "v", or "kv"

    Returns:
        Dict with per-layer PPL results
    """
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Calibration: {calib_path}")
    print(f"Target: {target.upper()}")
    print("Per-Layer Sensitivity Sweep")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Load calibration data
    calib_data = torch.load(calib_path, map_location=device, weights_only=False)
    rank = calib_data.get("rank", "?")
    print(f"  Layers: {num_layers}, Rank: {rank}")

    # Evaluation text
    if quick:
        eval_text = (
            """
        Machine learning has transformed how we approach complex problems.
        Neural networks can learn patterns from data without explicit programming.
        """
            * 10
        )
    else:
        eval_text = (
            """
        Machine learning models have become increasingly powerful over the past decade.
        Large language models can generate coherent text, answer questions, and assist
        with various tasks. The computational requirements continue to grow.

        Transformer architectures have revolutionized natural language processing.
        Attention mechanisms allow models to focus on relevant parts of the input.
        """
            * 5
        )

    results = {}

    # Baseline
    print("\n[1] Baseline...")
    baseline_ppl, _ = measure_ppl(model, tokenizer, eval_text, device)
    results["baseline"] = {"ppl": baseline_ppl}
    print(f"  PPL: {baseline_ppl:.4f}")

    # Load full calibrated compressors to get layer data
    from gpt2.compression.compressed_cache import CalibratedCompressor

    layers_data = calib_data["layers"]

    # Test each layer
    for layer_idx in range(num_layers):
        print(f"\n[{layer_idx + 2}] Layer {layer_idx} only...")

        # Create compressors - identity for all except target layer
        k_compressors = []
        v_compressors = []

        for i in range(num_layers):
            if i == layer_idx:
                # This layer gets compressed
                layer_data = layers_data[i]
                k_data = layer_data["K"]
                v_data = layer_data["V"]

                if target in ("k", "kv"):
                    k_comp = CalibratedCompressor(
                        U=k_data["U"].to(device).to(torch.float16),
                        mean=k_data["mean"].to(device).to(torch.float16),
                    )
                else:
                    k_comp = IdentityCompressor()

                if target in ("v", "kv"):
                    v_comp = CalibratedCompressor(
                        U=v_data["U"].to(device).to(torch.float16),
                        mean=v_data["mean"].to(device).to(torch.float16),
                    )
                else:
                    v_comp = IdentityCompressor()

                k_compressors.append(k_comp)
                v_compressors.append(v_comp)
            else:
                # Other layers stay full precision
                k_compressors.append(IdentityCompressor())
                v_compressors.append(IdentityCompressor())

        cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

        try:
            ppl, _ = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100

            results[f"layer_{layer_idx}"] = {
                "ppl": ppl,
                "delta_pct": delta,
                "layer": layer_idx,
            }
            print(f"  PPL: {ppl:.4f} ({delta:+.2f}%)")

        except Exception as e:
            print(f"  Error: {e}")
            results[f"layer_{layer_idx}"] = {"error": str(e), "layer": layer_idx}

        del cache, k_compressors, v_compressors
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("PER-LAYER SENSITIVITY SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Rank: {rank}, Target: {target.upper()}")
    print("-" * 80)
    print(f"{'Layer':<8} {'PPL':>10} {'Delta':>10} {'Sensitivity':>15}")
    print("-" * 80)

    # Collect layer results for sorting
    layer_results = []
    for i in range(num_layers):
        key = f"layer_{i}"
        if key in results and "error" not in results[key]:
            delta = results[key]["delta_pct"]
            layer_results.append((i, results[key]["ppl"], delta))

    # Sort by delta to find most/least sensitive
    layer_results.sort(key=lambda x: x[2])

    for layer_idx, ppl, delta in layer_results:
        if delta < 0.5:
            sensitivity = "VERY LOW"
        elif delta < 1.0:
            sensitivity = "LOW"
        elif delta < 2.0:
            sensitivity = "MODERATE"
        elif delta < 5.0:
            sensitivity = "HIGH"
        else:
            sensitivity = "VERY HIGH"
        print(f"{layer_idx:<8} {ppl:>10.4f} {delta:>+9.2f}% {sensitivity:>15}")

    print("-" * 80)

    # Analysis
    print("\nANALYSIS:")
    if layer_results:
        least_sensitive = layer_results[:3]
        most_sensitive = layer_results[-3:]

        print("\n  LEAST sensitive layers (best candidates for compression):")
        for layer_idx, ppl, delta in least_sensitive:
            print(f"    Layer {layer_idx}: {delta:+.2f}%")

        print("\n  MOST sensitive layers (avoid compressing):")
        for layer_idx, ppl, delta in reversed(most_sensitive):
            print(f"    Layer {layer_idx}: {delta:+.2f}%")

    return results


def run_quantize_ablation(
    model_name: str,
    calib_path: str,
    device: str = "cuda",
    quick: bool = False,
    target: str = "v",
) -> Dict:
    """
    Test int8 quantization on top of calibrated low-rank.

    Compares:
    - Low-rank only (V-only)
    - Low-rank + int8 (V-only)
    """
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Calibration: {calib_path}")
    print(f"Target: {target.upper()}")
    print("Quantization Ablation (int8 on calibrated low-rank)")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Load calibration metadata
    calib_data = torch.load(calib_path, map_location=device, weights_only=False)
    rank = calib_data.get("rank", "?")
    print(f"  Layers: {num_layers}, Rank: {rank}")

    # Evaluation text
    if quick:
        eval_text = (
            """
        Machine learning has transformed how we approach complex problems.
        Neural networks can learn patterns from data without explicit programming.
        """
            * 10
        )
    else:
        eval_text = (
            """
        Machine learning models have become increasingly powerful over the past decade.
        Large language models can generate coherent text, answer questions, and assist
        with various tasks. The computational requirements continue to grow.

        Transformer architectures have revolutionized natural language processing.
        Attention mechanisms allow models to focus on relevant parts of the input.
        """
            * 5
        )

    results = {}

    # Baseline
    print("\n[1] Baseline...")
    baseline_ppl, _ = measure_ppl(model, tokenizer, eval_text, device)
    results["baseline"] = {"ppl": baseline_ppl}
    print(f"  PPL: {baseline_ppl:.4f}")

    # Low-rank only
    print(f"\n[2] Low-rank only ({target.upper()})...")
    k_comp, v_comp, desc, metadata = create_calibrated_compressors(
        calib_path, torch.device(device), target=target, quantize_bits=None
    )
    cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
    ppl, _ = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    results["lowrank"] = {"ppl": ppl, "delta_pct": delta, "description": desc}
    print(f"  PPL: {ppl:.4f} ({delta:+.2f}%)")
    del cache, k_comp, v_comp
    torch.cuda.empty_cache()

    # Low-rank + int8
    print(f"\n[3] Low-rank + int8 ({target.upper()})...")
    k_comp, v_comp, desc, metadata = create_calibrated_compressors(
        calib_path, torch.device(device), target=target, quantize_bits=8
    )
    cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
    ppl, _ = measure_ppl(model, tokenizer, eval_text, device, cache=cache)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    results["lowrank_int8"] = {"ppl": ppl, "delta_pct": delta, "description": desc}
    print(f"  PPL: {ppl:.4f} ({delta:+.2f}%)")
    del cache, k_comp, v_comp
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("QUANTIZATION ABLATION SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Rank: {rank}, Target: {target.upper()}")
    print("-" * 80)
    print(f"{'Method':<30} {'PPL':>10} {'Delta':>10} {'Status':>12}")
    print("-" * 80)

    for name, data in results.items():
        if name == "baseline":
            continue
        ppl = data["ppl"]
        delta = data["delta_pct"]
        status = "GOOD" if delta < 5 else ("OK" if delta < 10 else "DEGRADED")
        print(f"{data['description']:<30} {ppl:>10.4f} {delta:>+9.2f}% {status:>12}")

    print("-" * 80)

    # Analysis
    lr_delta = results["lowrank"]["delta_pct"]
    lr_int8_delta = results["lowrank_int8"]["delta_pct"]
    quant_overhead = lr_int8_delta - lr_delta

    print(f"\nANALYSIS:")
    print(f"  Low-rank only: {lr_delta:+.2f}%")
    print(f"  Low-rank + int8: {lr_int8_delta:+.2f}%")
    print(f"  int8 overhead: {quant_overhead:+.2f}%")

    if abs(quant_overhead) < 1:
        print(
            f"\n  -> int8 quantization is MOSTLY FREE ({quant_overhead:+.2f}% overhead)"
        )
    elif quant_overhead < 5:
        print(f"\n  -> int8 quantization has SMALL overhead ({quant_overhead:+.2f}%)")
    else:
        print(
            f"\n  -> int8 quantization has SIGNIFICANT overhead ({quant_overhead:+.2f}%)"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Post-bugfix PPL validation")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with shorter text",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        nargs="*",
        default=None,
        help="Calibration file(s) to test",
    )
    parser.add_argument(
        "--kv-ablation",
        action="store_true",
        help="Run K vs V ablation (requires --calibration)",
    )
    parser.add_argument(
        "--layer-ablation",
        action="store_true",
        help="Run per-layer sensitivity sweep (requires --calibration)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="kv",
        choices=["k", "v", "kv"],
        help="Target for layer ablation: k, v, or kv (default: kv)",
    )
    parser.add_argument(
        "--quantize-ablation",
        action="store_true",
        help="Run int8 quantization on top of low-rank (requires --calibration)",
    )
    args = parser.parse_args()

    if args.quantize_ablation and args.calibration:
        # Run quantization ablation (int8 on calibrated low-rank)
        results = run_quantize_ablation(
            model_name=args.model,
            calib_path=args.calibration[0],  # Use first calibration file
            device=args.device,
            quick=args.quick,
            target=args.target,
        )
    elif args.layer_ablation and args.calibration:
        # Run per-layer sensitivity sweep
        results = run_layer_ablation(
            model_name=args.model,
            calib_path=args.calibration[0],  # Use first calibration file
            device=args.device,
            quick=args.quick,
            target=args.target,
        )
    elif args.kv_ablation and args.calibration:
        # Run K vs V ablation
        results = run_kv_ablation(
            model_name=args.model,
            calib_files=args.calibration,
            device=args.device,
            quick=args.quick,
        )
    elif args.calibration:
        # Run calibrated validation
        results = run_calibrated_validation(
            model_name=args.model,
            calib_files=args.calibration,
            device=args.device,
            quick=args.quick,
        )
    else:
        # Run standard validation (random projections)
        results = run_validation(
            model_name=args.model,
            device=args.device,
            quick=args.quick,
        )


if __name__ == "__main__":
    main()
