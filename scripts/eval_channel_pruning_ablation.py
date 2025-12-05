#!/usr/bin/env python3
"""
Channel pruning ablation study.

Evaluates the trade-off between extra KV savings vs PPL hit from
pruning low-energy latent channels (V-only).

Sweeps:
- energy_threshold: [0.0, 0.005, 0.01, 0.02]
- max_prune_fraction: [0.0, 0.10, 0.20]
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
    GammaAwareQuantizedCompressor,
    ChannelPruningCompressor,
)


def calibrate_with_energy(
    model,
    tokenizer,
    rank: int,
    device: str = "cuda",
    num_samples: int = 40,
    max_length: int = 512,
) -> List[Dict]:
    """Quick calibration to get PCA basis with energy fraction."""
    from scripts.calibrate_kv_lowrank import compute_pca_basis, extract_kv_from_model

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
                    "energy_fraction": K_stats["energy_fraction"].half().to(device),
                },
                "V": {
                    "U": V_stats["U"].half().to(device),
                    "mean": V_stats["mean"].half().to(device),
                    "latent_std": V_stats["latent_std"].half().to(device),
                    "energy_fraction": V_stats["energy_fraction"].half().to(device),
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
) -> float:
    """Evaluate PPL with given compressors."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:max_samples]

    cache = CompressedDynamicCache(k_compressors, v_compressors, num_layers)

    total_loss = 0.0
    total_tokens = 0

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

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


class PruningGammaCompressor(torch.nn.Module):
    """Combined pruning + gamma-aware compressor for V."""

    def __init__(
        self,
        U: torch.Tensor,
        mean: torch.Tensor,
        latent_std: torch.Tensor,
        energy_fraction: torch.Tensor,
        energy_threshold: float = 0.01,
        max_prune_fraction: float = 0.25,
        bits: int = 8,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.d_input = U.shape[0]
        self.rank = U.shape[1]
        self.bits = bits
        self.dtype = dtype

        # Quantization range
        if bits == 8:
            self.qmin, self.qmax = -128, 127
        elif bits == 4:
            self.qmin, self.qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        self.register_buffer("U", U.to(dtype))
        self.register_buffer("mean", mean.to(dtype))
        self.register_buffer("scale", latent_std.to(dtype).clamp(min=1e-6))

        # Compute pruned mask using ChannelPruningCompressor logic
        self.pruner = ChannelPruningCompressor(
            rank=self.rank,
            energy_fraction=energy_fraction,
            energy_threshold=energy_threshold,
            max_prune_fraction=max_prune_fraction,
            target="v",
            dtype=dtype,
        )
        self._n_pruned = self.pruner._n_pruned
        self._pruned_energy = self.pruner._pruned_energy

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project, prune, normalize, quantize."""
        centered = x - self.mean
        latent = centered @ self.U

        # Prune low-energy channels
        latent = self.pruner.compress(latent)

        # Gamma-aware normalization
        normalized = latent / self.scale

        # Quantize
        absmax = normalized.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        qscale = absmax / self.qmax
        quantized = (normalized / qscale).round().clamp(self.qmin, self.qmax)

        # Dequantize and restore scale
        dequantized = quantized * qscale * self.scale
        return dequantized

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct."""
        return z @ self.U.T + self.mean

    def get_effective_rank(self) -> int:
        return self.rank - self._n_pruned


def create_pruning_compressors(
    calib_data: List[Dict],
    energy_threshold: float,
    max_prune_fraction: float,
    bits: int = 8,
) -> Tuple[List, List, Dict]:
    """Create compressors with pruning enabled."""
    k_compressors = []
    v_compressors = []
    pruning_stats = {"total_pruned": 0, "total_rank": 0, "total_pruned_energy": 0.0}

    for layer_data in calib_data:
        # K: identity
        k_compressors.append(IdentityCompressor())

        # V: pruning + gamma-aware
        V = layer_data["V"]
        v_comp = PruningGammaCompressor(
            U=V["U"],
            mean=V["mean"],
            latent_std=V["latent_std"],
            energy_fraction=V["energy_fraction"],
            energy_threshold=energy_threshold,
            max_prune_fraction=max_prune_fraction,
            bits=bits,
        )
        v_compressors.append(v_comp)

        pruning_stats["total_pruned"] += v_comp._n_pruned
        pruning_stats["total_rank"] += v_comp.rank
        pruning_stats["total_pruned_energy"] += v_comp._pruned_energy

    return k_compressors, v_compressors, pruning_stats


def run_ablation(
    model_name: str,
    rank: int,
    energy_thresholds: List[float],
    max_prune_fractions: List[float],
    bits: int = 8,
    device: str = "cuda",
    max_samples: int = 50,
) -> List[Dict]:
    """Run channel pruning ablation."""
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

    effective_rank = min(rank, head_dim)
    print(f"Layers: {num_layers}, Head dim: {head_dim}, Rank: {effective_rank}")

    # Calibrate
    print("\nCalibrating with energy fraction...")
    calib_data = calibrate_with_energy(model, tokenizer, effective_rank, device)

    # Baseline PPL (no compression)
    print("\nEvaluating baseline...")
    k_id = [IdentityCompressor() for _ in range(num_layers)]
    v_id = [IdentityCompressor() for _ in range(num_layers)]
    ppl_baseline = eval_ppl(
        model, tokenizer, k_id, v_id, num_layers, device, max_samples
    )
    print(f"Baseline PPL: {ppl_baseline:.4f}")

    results = []

    for energy_thr in energy_thresholds:
        for max_prune in max_prune_fractions:
            print(
                f"\n  energy_thr={energy_thr}, max_prune={max_prune}: ",
                end="",
                flush=True,
            )

            k_comp, v_comp, prune_stats = create_pruning_compressors(
                calib_data, energy_thr, max_prune, bits
            )

            ppl = eval_ppl(
                model, tokenizer, k_comp, v_comp, num_layers, device, max_samples
            )
            ppl_delta = (ppl / ppl_baseline - 1) * 100

            # Compute effective compression
            avg_pruned = prune_stats["total_pruned"] / num_layers
            effective_rank_avg = effective_rank - avg_pruned
            base_compression = head_dim / effective_rank  # from rank reduction
            prune_boost = (
                effective_rank / effective_rank_avg if effective_rank_avg > 0 else 1.0
            )
            quant_compression = 16 / bits
            total_compression = base_compression * prune_boost * quant_compression

            print(
                f"PPL={ppl:.4f} (Δ={ppl_delta:+.2f}%), compress={total_compression:.2f}x"
            )

            results.append(
                {
                    "model": model_name,
                    "rank": effective_rank,
                    "bits": bits,
                    "energy_threshold": energy_thr,
                    "max_prune_fraction": max_prune,
                    "avg_pruned_dims": avg_pruned,
                    "effective_rank": effective_rank_avg,
                    "compression_v": total_compression,
                    "ppl": ppl,
                    "ppl_delta": ppl_delta,
                    "ppl_baseline": ppl_baseline,
                }
            )

    return results


def print_results_table(results: List[Dict]):
    """Print results as markdown table."""
    print("\n" + "=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)

    print("\n| Model | EnergyThr | MaxPrune | Eff. Rank | Compress | ΔPPL |")
    print("|-------|-----------|----------|-----------|----------|------|")

    for r in results:
        model_short = r["model"].split("/")[-1]
        print(
            f"| {model_short} | {r['energy_threshold']:.3f} | {r['max_prune_fraction']:.2f} | "
            f"{r['effective_rank']:.1f} | {r['compression_v']:.2f}x | {r['ppl_delta']:+.2f}% |"
        )


def save_results(results: List[Dict], output_dir: Path):
    """Save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "channel_pruning_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown
    with open(output_dir / "channel_pruning_ablation.md", "w") as f:
        f.write("# Channel Pruning Ablation\n\n")
        f.write("| Model | EnergyThr | MaxPrune | Eff. Rank | Compress | ΔPPL |\n")
        f.write("|-------|-----------|----------|-----------|----------|------|\n")
        for r in results:
            model_short = r["model"].split("/")[-1]
            f.write(
                f"| {model_short} | {r['energy_threshold']:.3f} | {r['max_prune_fraction']:.2f} | "
                f"{r['effective_rank']:.1f} | {r['compression_v']:.2f}x | {r['ppl_delta']:+.2f}% |\n"
            )

    print(f"\nResults saved to {output_dir}/")


def plot_results(results: List[Dict], output_dir: Path):
    """Generate plots."""
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

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot compression vs PPL delta
        for max_prune in sorted(set(r["max_prune_fraction"] for r in model_results)):
            subset = [r for r in model_results if r["max_prune_fraction"] == max_prune]
            subset = sorted(subset, key=lambda x: x["compression_v"])
            ax.plot(
                [r["compression_v"] for r in subset],
                [r["ppl_delta"] for r in subset],
                "o-",
                label=f"max_prune={max_prune:.0%}",
                markersize=8,
            )

        ax.set_xlabel("Compression Ratio (V-only)")
        ax.set_ylabel("PPL Δ (%)")
        ax.set_title(f"{model_short}: Channel Pruning - Compression vs PPL")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{model_short}_pruning_compression_vs_ppl.png", dpi=150
        )
        plt.close()

        print(f"Plot saved: {output_dir}/{model_short}_pruning_compression_vs_ppl.png")


def main():
    parser = argparse.ArgumentParser(description="Channel pruning ablation")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["Qwen/Qwen2.5-0.5B"],
        help="Models to test",
    )
    parser.add_argument(
        "--rank", type=int, default=96, help="Rank for low-rank projection"
    )
    parser.add_argument(
        "--energy-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02],
        help="Energy thresholds to test",
    )
    parser.add_argument(
        "--max-prune-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.20],
        help="Max prune fractions to test",
    )
    parser.add_argument("--bits", type=int, default=8, help="Quantization bits")
    parser.add_argument("--max-samples", type=int, default=50, help="Max PPL samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/channel_pruning",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    all_results = []

    for model in args.models:
        results = run_ablation(
            model,
            args.rank,
            args.energy_thresholds,
            args.max_prune_fractions,
            args.bits,
            args.device,
            args.max_samples,
        )
        all_results.extend(results)

    print_results_table(all_results)
    output_dir = Path(args.output_dir)
    save_results(all_results, output_dir)
    plot_results(all_results, output_dir)


if __name__ == "__main__":
    main()
