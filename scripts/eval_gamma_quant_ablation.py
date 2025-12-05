#!/usr/bin/env python3
"""
γ-aware quantization ablation study.

Compares baseline vs γ-aware quantization across:
- Models: Qwen2.5-7B, Qwen2.5-0.5B
- Ranks: [full, 96, 80, 64]
- Bits: [8, 4]

Goal: Show γ-aware enables lower rank or fewer bits at same PPL.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    QuantizedCalibratedCompressor,
    GammaAwareQuantizedCompressor,
)


def calibrate_for_rank(
    model,
    tokenizer,
    rank: int,
    device: str = "cuda",
    num_samples: int = 40,
    max_length: int = 512,
) -> List[Dict]:
    """Quick calibration to get PCA basis for given rank."""
    from scripts.calibrate_kv_lowrank import compute_pca_basis, extract_kv_from_model

    # Get calibration texts
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t) > 100][:num_samples]

    all_kv_per_layer = None

    for text in texts:
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(device)

        kv_pairs = extract_kv_from_model(model, input_ids, device)

        if all_kv_per_layer is None:
            n_layers = len(kv_pairs)
            all_kv_per_layer = [{"K": [], "V": []} for _ in range(n_layers)]

        for layer_idx, (k, v) in enumerate(kv_pairs):
            B, n_heads, T, head_dim = k.shape
            k_flat = k.view(-1, head_dim)
            v_flat = v.view(-1, head_dim)
            all_kv_per_layer[layer_idx]["K"].append(k_flat)
            all_kv_per_layer[layer_idx]["V"].append(v_flat)

    # Compute PCA for each layer
    layers_data = []
    for layer_idx in range(len(all_kv_per_layer)):
        K_all = torch.cat(all_kv_per_layer[layer_idx]["K"], dim=0)
        V_all = torch.cat(all_kv_per_layer[layer_idx]["V"], dim=0)

        head_dim = K_all.shape[-1]
        effective_rank = min(rank, head_dim)

        K_stats = compute_pca_basis(K_all, effective_rank, compute_gamma_stats=True)
        V_stats = compute_pca_basis(V_all, effective_rank, compute_gamma_stats=True)

        layers_data.append(
            {
                "K": {
                    "U": K_stats["U"].half().to(device),
                    "mean": K_stats["mean"].half().to(device),
                    "latent_std": K_stats["latent_std"].half().to(device),
                    "latent_max_abs": K_stats["latent_max_abs"].half().to(device),
                },
                "V": {
                    "U": V_stats["U"].half().to(device),
                    "mean": V_stats["mean"].half().to(device),
                    "latent_std": V_stats["latent_std"].half().to(device),
                    "latent_max_abs": V_stats["latent_max_abs"].half().to(device),
                },
            }
        )

        del K_all, V_all
        torch.cuda.empty_cache()

    return layers_data


def eval_ppl(
    model,
    tokenizer,
    k_compressors: List,
    v_compressors: List,
    num_layers: int,
    device: str = "cuda",
    max_samples: int = 50,
) -> Tuple[float, float]:
    """Evaluate PPL with given compressors. Returns (ppl, eval_time)."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:max_samples]

    cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)

        if input_ids.shape[1] < 2:
            continue

        cache.reset()

        with torch.no_grad():
            outputs = model(input_ids, past_key_values=cache, use_cache=True)

            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

    eval_time = time.time() - start_time
    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl, eval_time


def create_compressors(
    calib_data: List[Dict],
    mode: str,
    bits: int,
    device: str = "cuda",
) -> Tuple[List, List]:
    """Create K and V compressors based on mode."""
    k_compressors = []
    v_compressors = []

    for layer_data in calib_data:
        # K: always identity (V-only compression)
        k_compressors.append(IdentityCompressor())

        # V: based on mode
        V_U = layer_data["V"]["U"]
        V_mean = layer_data["V"]["mean"]
        V_std = layer_data["V"]["latent_std"]

        if mode == "baseline":
            v_comp = QuantizedCalibratedCompressor(V_U, V_mean, bits=bits)
        elif mode == "gamma_aware":
            v_comp = GammaAwareQuantizedCompressor(V_U, V_mean, V_std, bits=bits)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        v_compressors.append(v_comp)

    return k_compressors, v_compressors


def run_ablation(
    model_name: str,
    ranks: List[int],
    bits_list: List[int],
    modes: List[str],
    device: str = "cuda",
    max_samples: int = 50,
) -> List[Dict]:
    """Run full ablation grid."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads

    print(f"Layers: {num_layers}, Head dim: {head_dim}")

    # Baseline PPL
    print("\nEvaluating baseline (no compression)...")
    k_id = [IdentityCompressor() for _ in range(num_layers)]
    v_id = [IdentityCompressor() for _ in range(num_layers)]
    ppl_baseline, _ = eval_ppl(
        model, tokenizer, k_id, v_id, num_layers, device, max_samples
    )
    print(f"Baseline PPL: {ppl_baseline:.4f}")

    results = []

    # Process "full" rank
    if "full" in ranks:
        ranks = [head_dim if r == "full" else r for r in ranks]

    for rank in ranks:
        effective_rank = min(rank, head_dim)
        compression_ratio = head_dim / effective_rank

        print(
            f"\n--- Rank {effective_rank} (compression: {compression_ratio:.2f}x) ---"
        )

        # Calibrate for this rank
        print("Calibrating...")
        calib_data = calibrate_for_rank(
            model, tokenizer, effective_rank, device, num_samples=40
        )

        for bits in bits_list:
            quant_compression = 16 / bits  # fp16 to int8/int4
            total_compression = compression_ratio * quant_compression

            for mode in modes:
                print(f"  {mode} @ {bits}-bit: ", end="", flush=True)

                k_comp, v_comp = create_compressors(calib_data, mode, bits, device)
                ppl, eval_time = eval_ppl(
                    model, tokenizer, k_comp, v_comp, num_layers, device, max_samples
                )

                ppl_delta = (ppl / ppl_baseline - 1) * 100
                print(f"PPL={ppl:.4f} (Δ={ppl_delta:+.2f}%)")

                results.append(
                    {
                        "model": model_name,
                        "rank": effective_rank,
                        "bits": bits,
                        "mode": mode,
                        "ppl": ppl,
                        "ppl_delta": ppl_delta,
                        "ppl_baseline": ppl_baseline,
                        "compression_v": total_compression,
                        "eval_time": eval_time,
                    }
                )

    return results


def print_results_table(results: List[Dict]):
    """Print results as markdown table."""
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)

    print("\n| Model | Rank | Bits | Mode | PPL | PPL Δ | V Compress |")
    print("|-------|------|------|------|-----|-------|------------|")

    for r in results:
        model_short = r["model"].split("/")[-1]
        print(
            f"| {model_short} | {r['rank']} | {r['bits']} | "
            f"{r['mode']} | {r['ppl']:.2f} | {r['ppl_delta']:+.2f}% | "
            f"{r['compression_v']:.2f}x |"
        )


def save_results(results: List[Dict], output_dir: Path):
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "gamma_quant_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown table
    with open(output_dir / "gamma_quant_ablation.md", "w") as f:
        f.write("# γ-Aware Quantization Ablation\n\n")
        f.write("| Model | Rank | Bits | Mode | PPL | PPL Δ | V Compress |\n")
        f.write("|-------|------|------|------|-----|-------|------------|\n")
        for r in results:
            model_short = r["model"].split("/")[-1]
            f.write(
                f"| {model_short} | {r['rank']} | {r['bits']} | "
                f"{r['mode']} | {r['ppl']:.2f} | {r['ppl_delta']:+.2f}% | "
                f"{r['compression_v']:.2f}x |\n"
            )

    print(f"\nResults saved to {output_dir}/")


def plot_results(results: List[Dict], output_dir: Path):
    """Generate ablation plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by model
    models = set(r["model"] for r in results)

    for model in models:
        model_results = [r for r in results if r["model"] == model]
        model_short = model.split("/")[-1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: PPL Δ vs Compression for int8
        ax1 = axes[0]
        int8_baseline = [
            r for r in model_results if r["bits"] == 8 and r["mode"] == "baseline"
        ]
        int8_gamma = [
            r for r in model_results if r["bits"] == 8 and r["mode"] == "gamma_aware"
        ]

        if int8_baseline:
            ax1.plot(
                [r["compression_v"] for r in int8_baseline],
                [r["ppl_delta"] for r in int8_baseline],
                "o-",
                label="baseline int8",
                markersize=8,
            )
        if int8_gamma:
            ax1.plot(
                [r["compression_v"] for r in int8_gamma],
                [r["ppl_delta"] for r in int8_gamma],
                "s-",
                label="γ-aware int8",
                markersize=8,
            )

        ax1.set_xlabel("Compression Ratio")
        ax1.set_ylabel("PPL Δ (%)")
        ax1.set_title(f"{model_short}: int8 Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Plot 2: PPL Δ vs Compression for int4
        ax2 = axes[1]
        int4_baseline = [
            r for r in model_results if r["bits"] == 4 and r["mode"] == "baseline"
        ]
        int4_gamma = [
            r for r in model_results if r["bits"] == 4 and r["mode"] == "gamma_aware"
        ]

        if int4_baseline:
            ax2.plot(
                [r["compression_v"] for r in int4_baseline],
                [r["ppl_delta"] for r in int4_baseline],
                "o-",
                label="baseline int4",
                markersize=8,
            )
        if int4_gamma:
            ax2.plot(
                [r["compression_v"] for r in int4_gamma],
                [r["ppl_delta"] for r in int4_gamma],
                "s-",
                label="γ-aware int4",
                markersize=8,
            )

        ax2.set_xlabel("Compression Ratio")
        ax2.set_ylabel("PPL Δ (%)")
        ax2.set_title(f"{model_short}: int4 Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f"gamma_vs_baseline_{model_short}.png", dpi=150)
        plt.close()

        print(f"Plot saved: {output_dir}/gamma_vs_baseline_{model_short}.png")


def main():
    parser = argparse.ArgumentParser(description="γ-aware quantization ablation")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["Qwen/Qwen2.5-7B"],
        help="Models to test",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        nargs="+",
        default=["full", "96", "80", "64"],
        help="Ranks to test (use 'full' for head_dim)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[8, 4],
        help="Bit widths to test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples for PPL eval",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/gamma_quant_ablation",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Parse ranks
    ranks = []
    for r in args.ranks:
        if r.lower() == "full":
            ranks.append("full")
        else:
            ranks.append(int(r))

    modes = ["baseline", "gamma_aware"]
    all_results = []

    for model in args.models:
        results = run_ablation(
            model,
            ranks,
            args.bits,
            modes,
            args.device,
            args.max_samples,
        )
        all_results.extend(results)

    # Print and save results
    print_results_table(all_results)
    output_dir = Path(args.output_dir)
    save_results(all_results, output_dir)
    plot_results(all_results, output_dir)


if __name__ == "__main__":
    main()
