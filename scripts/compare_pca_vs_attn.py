#!/usr/bin/env python3
"""
Compare PCA vs Attention-Trained Projectors.

Evaluates PPL with:
1. PCA-calibrated projector (baseline)
2. Attention-trained projector (logit/attn_kl/output loss)

For a single layer to see if attention-aware training improves compression.

Usage:
    python scripts/compare_pca_vs_attn.py \
        --pca kv_lowrank_calib_qwen-qwen2.5-0.5b_r56.pt \
        --attn attn_lowrank_qwen-qwen2.5-0.5b_l15_r56_logit_k.pt \
        --layer 15
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CalibratedCompressor,
    CompressedDynamicCache,
    IdentityCompressor,
)


class AttnTrainedCompressor(nn.Module):
    """
    Compressor using attention-trained W_c/W_e matrices.

    Note: Unlike PCA which uses U and U.T, attention training
    learns W_c and W_e separately (not necessarily transposes).
    """

    def __init__(
        self,
        W_c: torch.Tensor,
        W_e: torch.Tensor,
        mean: torch.Tensor,
    ):
        super().__init__()
        # W_c: [d_head, rank] for compression
        # W_e: [rank, d_head] for expansion
        self.register_buffer("W_c", W_c)
        self.register_buffer("W_e", W_e)
        self.register_buffer("mean", mean)
        self.d_compressed = W_c.shape[1]
        self.calibrated = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress: [*, d_head] -> [*, rank]"""
        return (x - self.mean) @ self.W_c

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Expand: [*, rank] -> [*, d_head]"""
        return z @ self.W_e + self.mean


def measure_ppl(
    model,
    tokenizer,
    text: str,
    device: str,
    cache=None,
) -> float:
    """Measure perplexity on text."""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        if cache is not None:
            outputs = model(input_ids, labels=input_ids, past_key_values=cache)
        else:
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def create_pca_compressors_single_layer(
    pca_path: str,
    layer_idx: int,
    target: str,
    device: str,
    num_layers: int,
) -> Tuple[List[nn.Module], List[nn.Module]]:
    """Create compressors with PCA only on one layer."""
    calib_data = torch.load(pca_path, map_location=device, weights_only=False)
    layers_data = calib_data["layers"]

    k_compressors = []
    v_compressors = []

    for i in range(num_layers):
        if i == layer_idx:
            layer_data = layers_data[i]

            if target in ("k", "kv"):
                k_comp = CalibratedCompressor(
                    U=layer_data["K"]["U"].to(device).to(torch.float16),
                    mean=layer_data["K"]["mean"].to(device).to(torch.float16),
                )
            else:
                k_comp = IdentityCompressor()

            if target in ("v", "kv"):
                v_comp = CalibratedCompressor(
                    U=layer_data["V"]["U"].to(device).to(torch.float16),
                    mean=layer_data["V"]["mean"].to(device).to(torch.float16),
                )
            else:
                v_comp = IdentityCompressor()

            k_compressors.append(k_comp)
            v_compressors.append(v_comp)
        else:
            k_compressors.append(IdentityCompressor())
            v_compressors.append(IdentityCompressor())

    return k_compressors, v_compressors


def create_attn_compressors_single_layer(
    attn_path: str,
    layer_idx: int,
    target: str,
    device: str,
    num_layers: int,
) -> Tuple[List[nn.Module], List[nn.Module]]:
    """Create compressors with attention-trained projector on one layer."""
    attn_data = torch.load(attn_path, map_location=device, weights_only=False)

    # Verify layer matches
    if attn_data["layer"] != layer_idx:
        print(
            f"Warning: attn file is for layer {attn_data['layer']}, using layer {layer_idx}"
        )

    k_compressors = []
    v_compressors = []

    for i in range(num_layers):
        if i == layer_idx:
            if target in ("k", "kv"):
                k_comp = AttnTrainedCompressor(
                    W_c=attn_data["K"]["U"].to(device).to(torch.float16),
                    W_e=attn_data["K"]["W_e"].to(device).to(torch.float16),
                    mean=attn_data["K"]["mean"].to(device).to(torch.float16),
                )
            else:
                k_comp = IdentityCompressor()

            if target in ("v", "kv"):
                v_comp = AttnTrainedCompressor(
                    W_c=attn_data["V"]["U"].to(device).to(torch.float16),
                    W_e=attn_data["V"]["W_e"].to(device).to(torch.float16),
                    mean=attn_data["V"]["mean"].to(device).to(torch.float16),
                )
            else:
                v_comp = IdentityCompressor()

            k_compressors.append(k_comp)
            v_compressors.append(v_comp)
        else:
            k_compressors.append(IdentityCompressor())
            v_compressors.append(IdentityCompressor())

    return k_compressors, v_compressors


def main():
    parser = argparse.ArgumentParser(
        description="Compare PCA vs attention-trained projectors"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to test",
    )
    parser.add_argument(
        "--pca",
        type=str,
        required=True,
        help="PCA calibration file",
    )
    parser.add_argument(
        "--attn",
        type=str,
        required=True,
        help="Attention-trained projector file",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to compare",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="k",
        choices=["k", "v", "kv"],
        help="What to compress",
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
        help="Quick mode",
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Target: {args.target}")
    print(f"PCA file: {args.pca}")
    print(f"Attn file: {args.attn}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Evaluation text
    if args.quick:
        eval_text = (
            "Machine learning has transformed how we approach complex problems. "
            "Neural networks can learn patterns from data without explicit programming. "
        ) * 10
    else:
        eval_text = (
            "Machine learning models have become increasingly powerful over the past decade. "
            "Large language models can generate coherent text, answer questions, and assist "
            "with various tasks. The computational requirements continue to grow. "
            "Transformer architectures have revolutionized natural language processing. "
        ) * 5

    # 1. Baseline
    print("\n[1] Baseline (no compression)...")
    baseline_ppl = measure_ppl(model, tokenizer, eval_text, args.device)
    print(f"  PPL: {baseline_ppl:.4f}")

    # 2. PCA on single layer
    print(f"\n[2] PCA on layer {args.layer}...")
    k_pca, v_pca = create_pca_compressors_single_layer(
        args.pca, args.layer, args.target, args.device, num_layers
    )
    cache_pca = CompressedDynamicCache(k_pca, v_pca, num_layers)
    pca_ppl = measure_ppl(model, tokenizer, eval_text, args.device, cache=cache_pca)
    pca_delta = ((pca_ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"  PPL: {pca_ppl:.4f} ({pca_delta:+.2f}%)")

    del cache_pca, k_pca, v_pca
    torch.cuda.empty_cache()

    # 3. Attention-trained on single layer
    print(f"\n[3] Attention-trained on layer {args.layer}...")
    k_attn, v_attn = create_attn_compressors_single_layer(
        args.attn, args.layer, args.target, args.device, num_layers
    )
    cache_attn = CompressedDynamicCache(k_attn, v_attn, num_layers)
    attn_ppl = measure_ppl(model, tokenizer, eval_text, args.device, cache=cache_attn)
    attn_delta = ((attn_ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"  PPL: {attn_ppl:.4f} ({attn_delta:+.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Layer {args.layer}, Target: {args.target.upper()}")
    print("-" * 60)
    print(f"{'Method':<30} {'PPL':>10} {'Delta':>10} {'Status':>12}")
    print("-" * 60)

    pca_status = "GOOD" if abs(pca_delta) < 5 else "DEGRADED"
    attn_status = "GOOD" if abs(attn_delta) < 5 else "DEGRADED"

    print(f"{'PCA':<30} {pca_ppl:>10.4f} {pca_delta:>+9.2f}% {pca_status:>12}")
    print(
        f"{'Attention-trained':<30} {attn_ppl:>10.4f} {attn_delta:>+9.2f}% {attn_status:>12}"
    )
    print("-" * 60)

    # Analysis
    improvement = pca_delta - attn_delta
    if improvement > 0.1:
        print(f"\nAttn-trained is BETTER by {improvement:.2f}% PPL delta")
    elif improvement < -0.1:
        print(f"\nPCA is BETTER by {-improvement:.2f}% PPL delta")
    else:
        print(f"\nBoth methods are SIMILAR (diff={improvement:.2f}%)")


if __name__ == "__main__":
    main()
