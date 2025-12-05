#!/usr/bin/env python3
"""
Cross-model KV Spectral Probe

Analyzes K and V singular value spectra across different models to determine
compressibility. This is the key tool for deciding whether low-rank KV compression
is viable for a given model architecture.

Usage:
    python scripts/kv_spectrum_probe.py --model openai-community/gpt2
    python scripts/kv_spectrum_probe.py --model gpt2-medium --layers 0,6,12,23
    python scripts/kv_spectrum_probe.py --model Qwen/Qwen2.5-0.5B
    python scripts/kv_spectrum_probe.py --checkpoint path/to/model.pt --model-type mla
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_calibration_data(tokenizer, num_samples=64, seq_len=256, seed=42):
    """Generate calibration data from simple text patterns."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    texts = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "In the beginning, there was nothing but darkness. " * 20,
        "Machine learning is a subset of artificial intelligence. " * 20,
        "The weather today is sunny with clear skies. " * 20,
    ]

    all_ids = []
    for i in range(num_samples):
        text = texts[i % len(texts)]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids * (seq_len // len(ids) + 1)
            ids = ids[:seq_len]
        all_ids.append(ids)

    return torch.tensor(all_ids, dtype=torch.long)


def compute_spectrum_stats(matrix, thresholds=(0.90, 0.95, 0.99), top_ks=(16, 32)):
    """
    Compute spectral statistics for a matrix.

    Args:
        matrix: [N, D] matrix (N samples, D features)
        thresholds: energy thresholds for effective rank
        top_ks: values of k for top-k energy computation

    Returns:
        dict with spectrum statistics
    """
    # Center the matrix
    matrix = matrix - matrix.mean(dim=0, keepdim=True)

    # Compute SVD
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # Convert to numpy for easier computation
    s = S.cpu().numpy()
    s_squared = s**2
    total_energy = s_squared.sum()

    if total_energy < 1e-10:
        return {
            "condition_number": float("inf"),
            "effective_rank": {str(t): 0 for t in thresholds},
            "energy_in_top_k": {str(k): 0.0 for k in top_ks},
            "top_5_sv": [0.0] * 5,
            "num_dims": len(s),
        }

    # Cumulative energy
    cumsum = np.cumsum(s_squared) / total_energy

    # Effective rank at each threshold
    effective_rank = {}
    for thresh in thresholds:
        rank = int(np.searchsorted(cumsum, thresh) + 1)
        effective_rank[str(thresh)] = min(rank, len(s))

    # Energy in top-k
    energy_top_k = {}
    for k in top_ks:
        if k <= len(s):
            energy_top_k[str(k)] = float(s_squared[:k].sum() / total_energy)
        else:
            energy_top_k[str(k)] = 1.0

    # Condition number
    cond = float(s[0] / s[-1]) if s[-1] > 1e-10 else float("inf")

    return {
        "condition_number": cond,
        "effective_rank": effective_rank,
        "energy_in_top_k": energy_top_k,
        "top_5_sv": [float(x) for x in s[:5]],
        "num_dims": len(s),
    }


def extract_kv_hf(model, input_ids, layers, device):
    """
    Extract K and V from HuggingFace model using native KV cache.

    Returns dict: {layer_idx: {"K": tensor, "V": tensor}}
    """
    model.eval()
    kv_cache = {}

    with torch.no_grad():
        try:
            outputs = model(input_ids.to(device), use_cache=True)
            past_kv = outputs.past_key_values

            if past_kv is not None:
                for layer_idx in layers:
                    if layer_idx < len(past_kv) and past_kv[layer_idx] is not None:
                        kv_tuple = past_kv[layer_idx]
                        if isinstance(kv_tuple, tuple) and len(kv_tuple) >= 2:
                            k, v = kv_tuple[0], kv_tuple[1]
                            kv_cache[layer_idx] = {"K": k.detach(), "V": v.detach()}
        except Exception as e:
            print(f"Warning: KV cache extraction failed: {e}")

    return kv_cache


def extract_kv_direct(model, input_ids, layers, device, model_config=None):
    """
    Directly compute K and V by running attention forward.
    Fallback when hook-based extraction doesn't work.
    """
    model.eval()
    kv_results = {}
    if model_config is None:
        model_config = model.config

    with torch.no_grad():
        input_ids = input_ids.to(device)

        # Get hidden states at each layer
        if hasattr(model, "transformer"):
            # GPT-2 style
            hidden = model.transformer.wte(input_ids)
            if hasattr(model.transformer, "wpe"):
                pos = torch.arange(input_ids.size(1), device=device)
                hidden = hidden + model.transformer.wpe(pos)

            for layer_idx, block in enumerate(model.transformer.h):
                if layer_idx in layers:
                    # Get K, V from attention
                    attn = block.attn
                    if hasattr(attn, "c_attn"):
                        # GPT-2 fused QKV
                        qkv = attn.c_attn(block.ln_1(hidden))
                        q, k, v = qkv.split(attn.split_size, dim=2)

                        # Reshape to [B, H, T, D]
                        B, T, _ = k.shape
                        n_head = attn.num_heads
                        d_head = k.size(-1) // n_head

                        k = k.view(B, T, n_head, d_head).transpose(1, 2)
                        v = v.view(B, T, n_head, d_head).transpose(1, 2)

                        kv_results[layer_idx] = {"K": k, "V": v}

                # Continue forward pass
                hidden = block(hidden)[0]

        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # Llama/Qwen style
            hidden = model.model.embed_tokens(input_ids)

            for layer_idx, layer in enumerate(model.model.layers):
                if layer_idx in layers:
                    attn = layer.self_attn

                    # Apply input layernorm
                    normed = layer.input_layernorm(hidden)

                    # Get Q, K, V projections
                    if hasattr(attn, "q_proj"):
                        q = attn.q_proj(normed)
                        k = attn.k_proj(normed)
                        v = attn.v_proj(normed)

                        # Handle GQA - k, v may have fewer heads
                        B, T, _ = k.shape
                        # Try various attribute paths for num_kv_heads
                        n_kv_heads = getattr(attn, "num_key_value_heads", None)
                        if n_kv_heads is None:
                            n_kv_heads = getattr(attn, "num_heads", None)
                        if n_kv_heads is None and hasattr(attn, "config"):
                            n_kv_heads = getattr(
                                attn.config, "num_key_value_heads", None
                            )
                        if n_kv_heads is None:
                            n_kv_heads = getattr(
                                model_config, "num_key_value_heads", None
                            )
                        if n_kv_heads is None:
                            n_kv_heads = model_config.num_attention_heads

                        d_head = getattr(attn, "head_dim", None)
                        if d_head is None:
                            d_head = (
                                model_config.hidden_size
                                // model_config.num_attention_heads
                            )

                        k = k.view(B, T, n_kv_heads, d_head).transpose(1, 2)
                        v = v.view(B, T, n_kv_heads, d_head).transpose(1, 2)

                        kv_results[layer_idx] = {"K": k, "V": v}

                # Continue forward
                hidden = layer(hidden)[0]

    return kv_results


def analyze_model(
    model,
    tokenizer,
    layers,
    device,
    num_samples=64,
    seq_len=256,
    seed=42,
):
    """
    Analyze KV spectra for a model.

    Returns:
        dict with per-layer, per-head analysis
    """
    # Get calibration data
    input_ids = get_calibration_data(tokenizer, num_samples, seq_len, seed)

    # Try native KV cache extraction first
    kv_data = extract_kv_hf(model, input_ids, layers, device)

    # Fall back to direct extraction only for GPT-2 style models
    if not kv_data and hasattr(model, "transformer"):
        print("KV cache extraction failed, trying direct extraction...")
        kv_data = extract_kv_direct(model, input_ids, layers, device)

    if not kv_data:
        print("Warning: Could not extract K/V from model")
        return {}

    results = {}

    for layer_idx in sorted(kv_data.keys()):
        K = kv_data[layer_idx]["K"]  # [B, H, T, D]
        V = kv_data[layer_idx]["V"]

        B, H, T, D = K.shape

        layer_results = {"K": {}, "V": {}, "n_heads": H, "d_head": D}

        # Analyze per head
        for h in range(H):
            # Flatten batch and sequence: [B*T, D]
            k_flat = K[:, h, :, :].reshape(-1, D).float()
            v_flat = V[:, h, :, :].reshape(-1, D).float()

            layer_results["K"][f"head_{h}"] = compute_spectrum_stats(k_flat)
            layer_results["V"][f"head_{h}"] = compute_spectrum_stats(v_flat)

        # Also compute aggregate (all heads concatenated)
        k_all = K.permute(0, 2, 1, 3).reshape(-1, H * D).float()  # [B*T, H*D]
        v_all = V.permute(0, 2, 1, 3).reshape(-1, H * D).float()

        layer_results["K"]["aggregate"] = compute_spectrum_stats(k_all)
        layer_results["V"]["aggregate"] = compute_spectrum_stats(v_all)

        results[f"layer_{layer_idx}"] = layer_results

    return results


def summarize_results(results, model_name):
    """Generate summary statistics across all layers."""
    summary = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layers_analyzed": len(results),
    }

    # Aggregate stats
    k_effective_ranks = []
    v_effective_ranks = []
    k_energy_32 = []
    v_energy_32 = []
    k_cond = []
    v_cond = []

    for layer_name, layer_data in results.items():
        # Use aggregate stats
        k_agg = layer_data["K"].get("aggregate", {})
        v_agg = layer_data["V"].get("aggregate", {})

        if k_agg:
            k_effective_ranks.append(k_agg.get("effective_rank", {}).get("0.99", 0))
            k_energy_32.append(k_agg.get("energy_in_top_k", {}).get("32", 0))
            k_cond.append(k_agg.get("condition_number", float("inf")))

        if v_agg:
            v_effective_ranks.append(v_agg.get("effective_rank", {}).get("0.99", 0))
            v_energy_32.append(v_agg.get("energy_in_top_k", {}).get("32", 0))
            v_cond.append(v_agg.get("condition_number", float("inf")))

    if k_effective_ranks:
        summary["K"] = {
            "avg_effective_rank_99": float(np.mean(k_effective_ranks)),
            "avg_energy_in_top_32": float(np.mean(k_energy_32)),
            "avg_condition_number": float(np.mean([c for c in k_cond if c < 1e10])),
            "min_energy_in_top_32": float(np.min(k_energy_32)),
            "max_energy_in_top_32": float(np.max(k_energy_32)),
        }

    if v_effective_ranks:
        summary["V"] = {
            "avg_effective_rank_99": float(np.mean(v_effective_ranks)),
            "avg_energy_in_top_32": float(np.mean(v_energy_32)),
            "avg_condition_number": float(np.mean([c for c in v_cond if c < 1e10])),
            "min_energy_in_top_32": float(np.min(v_energy_32)),
            "max_energy_in_top_32": float(np.max(v_energy_32)),
        }

    # Compressibility assessment
    k_compressible = summary.get("K", {}).get("avg_energy_in_top_32", 0) > 0.8
    v_compressible = summary.get("V", {}).get("avg_energy_in_top_32", 0) > 0.8

    summary["assessment"] = {
        "k_compressible_at_rank_32": k_compressible,
        "v_compressible_at_rank_32": v_compressible,
        "recommendation": (
            "Low-rank KV compression viable"
            if (k_compressible or v_compressible)
            else "Low-rank KV compression NOT recommended (near-identity)"
        ),
    }

    return summary


def plot_results(results, output_path, model_name):
    """Generate visualization of KV spectra."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_layers = len(results)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    for idx, (layer_name, layer_data) in enumerate(sorted(results.items())):
        k_agg = layer_data["K"].get("aggregate", {})
        v_agg = layer_data["V"].get("aggregate", {})

        # K spectrum
        ax_k = axes[0, idx]
        if k_agg and "top_5_sv" in k_agg:
            sv = k_agg["top_5_sv"]
            ax_k.bar(range(len(sv)), sv, color="tab:orange", alpha=0.7)
            ax_k.set_title(
                f"{layer_name}\nK (E@32={k_agg.get('energy_in_top_k', {}).get('32', 0):.1%})"
            )
            ax_k.set_xlabel("SV Index")
            ax_k.set_ylabel("Singular Value")

        # V spectrum
        ax_v = axes[1, idx]
        if v_agg and "top_5_sv" in v_agg:
            sv = v_agg["top_5_sv"]
            ax_v.bar(range(len(sv)), sv, color="tab:blue", alpha=0.7)
            ax_v.set_title(
                f"V (E@32={v_agg.get('energy_in_top_k', {}).get('32', 0):.1%})"
            )
            ax_v.set_xlabel("SV Index")
            ax_v.set_ylabel("Singular Value")

    fig.suptitle(f"KV Spectrum Analysis: {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-model KV Spectral Probe")
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint (for custom models)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "gpt2", "mla"],
        default="auto",
        help="Model architecture type",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: 0, mid, last)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=64, help="Number of calibration samples"
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Sequence length for calibration"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--plot", type=str, default=None, help="Output plot path")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Load model
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        # For custom checkpoints, need model-specific loading
        print("Custom checkpoint loading not yet implemented")
        sys.exit(1)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Determine number of layers
    n_layers = 0
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n_layers = len(model.transformer.h)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
    elif hasattr(model, "config"):
        n_layers = getattr(model.config, "num_hidden_layers", 0) or getattr(
            model.config, "n_layer", 0
        )

    print(f"Model has {n_layers} layers")

    # Determine layers to analyze
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        # Default: first, middle, last
        if n_layers <= 4:
            layers = list(range(n_layers))
        else:
            layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    print(f"Analyzing layers: {layers}")

    # Run analysis
    results = analyze_model(
        model,
        tokenizer,
        layers,
        args.device,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    # Generate summary
    summary = summarize_results(results, args.model)

    # Print summary
    print("\n" + "=" * 60)
    print(f"KV SPECTRUM ANALYSIS: {args.model}")
    print("=" * 60)

    if "K" in summary:
        print(f"\nK Statistics (aggregate):")
        print(
            f"  Avg effective rank (99%): {summary['K']['avg_effective_rank_99']:.1f}"
        )
        print(f"  Avg energy in top-32: {summary['K']['avg_energy_in_top_32']:.1%}")
        print(f"  Avg condition number: {summary['K']['avg_condition_number']:.2f}")

    if "V" in summary:
        print(f"\nV Statistics (aggregate):")
        print(
            f"  Avg effective rank (99%): {summary['V']['avg_effective_rank_99']:.1f}"
        )
        print(f"  Avg energy in top-32: {summary['V']['avg_energy_in_top_32']:.1%}")
        print(f"  Avg condition number: {summary['V']['avg_condition_number']:.2f}")

    print(f"\nAssessment:")
    print(
        f"  K compressible at rank-32: {summary['assessment']['k_compressible_at_rank_32']}"
    )
    print(
        f"  V compressible at rank-32: {summary['assessment']['v_compressible_at_rank_32']}"
    )
    print(f"  Recommendation: {summary['assessment']['recommendation']}")

    # Save results
    output_data = {"summary": summary, "per_layer": results}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to {args.output}")
    else:
        # Default output path
        model_slug = args.model.replace("/", "_").replace("-", "_")
        output_path = f"kv_spectrum_{model_slug}.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to {output_path}")

    # Generate plot
    if args.plot:
        plot_results(results, args.plot, args.model)
    else:
        plot_path = f"kv_spectrum_{model_slug}.png"
        plot_results(results, plot_path, args.model)


if __name__ == "__main__":
    main()
