#!/usr/bin/env python3
"""
K vs V Compression Probe

Compares the singular value spectrum of K-only and V-only residual adapters
to determine which has more low-rank structure and is more amenable to
cache-mode compression.

Key question: Does M_K = I + s*W_k_out@W_k have more skewed singular values
than M_V = I + s*W_v_out@W_v?

If M_K is closer to low-rank (e.g., top-32 covers >90% energy), then K
compression may be more viable than V compression.

Usage:
    python scripts/k_vs_v_spectrum_probe.py --output ./k_vs_v_spectrum.json
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.compression.kv_compressor_plugin import (
    VResidualAdapter,
    KResidualAdapter,
    get_residual_operator_matrix,
    get_k_residual_operator_matrix,
    analyze_residual_operator,
    analyze_k_residual_operator,
)


RANK = 32
NUM_STEPS = 500
LR = 1e-4


def load_data(tokenizer, num_samples=500, max_length=256, seed=42):
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )
    torch.manual_seed(seed)
    samples = []
    for idx, item in enumerate(dataset):
        if idx >= num_samples * 2:
            break
        enc = tokenizer(
            item["text"], max_length=max_length, truncation=True, return_tensors="pt"
        )
        if enc["input_ids"].shape[1] >= 32:
            samples.append(enc["input_ids"])
        if len(samples) >= num_samples:
            break
    return samples


def evaluate_ppl(model, samples, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids in samples:
            input_ids = input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


class VWrapper(nn.Module):
    """Wrapper for V-only residual training."""
    def __init__(self, c_attn, adapter, embed_dim, num_heads, d_head):
        super().__init__()
        self.c_attn = c_attn
        self.adapter = adapter
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = d_head

    def forward(self, x):
        qkv = self.c_attn(x)
        q = qkv[:, :, :self.embed_dim]
        k = qkv[:, :, self.embed_dim:2*self.embed_dim]
        v = qkv[:, :, 2*self.embed_dim:]
        B, T, _ = v.shape
        v_heads = v.view(B, T, self.num_heads, self.d_head)
        v_latent = self.adapter.compress_v(v_heads)
        v_out = self.adapter.expand_v(v_latent, v_original=v_heads)
        v_flat = v_out.view(B, T, -1)
        return torch.cat([q, k, v_flat], dim=-1)


class KWrapper(nn.Module):
    """Wrapper for K-only residual training."""
    def __init__(self, c_attn, adapter, embed_dim, num_heads, d_head):
        super().__init__()
        self.c_attn = c_attn
        self.adapter = adapter
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = d_head

    def forward(self, x):
        qkv = self.c_attn(x)
        q = qkv[:, :, :self.embed_dim]
        k = qkv[:, :, self.embed_dim:2*self.embed_dim]
        v = qkv[:, :, 2*self.embed_dim:]
        B, T, _ = k.shape
        k_heads = k.view(B, T, self.num_heads, self.d_head)
        k_latent = self.adapter.compress_k(k_heads)
        k_out = self.adapter.expand_k(k_latent, k_original=k_heads)
        k_flat = k_out.view(B, T, -1)
        return torch.cat([q, k_flat, v], dim=-1)


def wrap_model_v(model, adapter, num_heads, d_head):
    embed_dim = num_heads * d_head
    for block in model.transformer.h:
        orig = block.attn.c_attn
        if isinstance(orig, (VWrapper, KWrapper)):
            orig = orig.c_attn
        block.attn.c_attn = VWrapper(orig, adapter, embed_dim, num_heads, d_head)
    return model


def wrap_model_k(model, adapter, num_heads, d_head):
    embed_dim = num_heads * d_head
    for block in model.transformer.h:
        orig = block.attn.c_attn
        if isinstance(orig, (VWrapper, KWrapper)):
            orig = orig.c_attn
        block.attn.c_attn = KWrapper(orig, adapter, embed_dim, num_heads, d_head)
    return model


def unwrap_model(model):
    for block in model.transformer.h:
        if isinstance(block.attn.c_attn, (VWrapper, KWrapper)):
            block.attn.c_attn = block.attn.c_attn.c_attn
    return model


def train_adapter(model, teacher, adapter, samples, num_steps, device):
    optimizer = torch.optim.Adam(adapter.parameters(), lr=LR)
    model.train()
    losses = []

    for step in tqdm(range(num_steps), desc="Training"):
        input_ids = samples[step % len(samples)].to(device)
        student_out = model(input_ids, output_hidden_states=True)
        with torch.no_grad():
            teacher_out = teacher(input_ids, output_hidden_states=True)

        T = 2.0
        loss_kl = F.kl_div(
            F.log_softmax(student_out.logits / T, dim=-1),
            F.softmax(teacher_out.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)
        loss_h = F.mse_loss(student_out.hidden_states[-1], teacher_out.hidden_states[-1])
        loss = loss_kl + 0.5 * loss_h

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def run_comparison(
    model_name: str = "openai-community/gpt2",
    num_steps: int = NUM_STEPS,
    device: str = "cuda",
    seed: int = 42,
    output_path: str = None,
    plot_path: str = None,
):
    print("=" * 70)
    print("K vs V COMPRESSION SPECTRUM PROBE")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Rank: {RANK}")
    print(f"Steps: {num_steps}")
    print()

    # Load teacher
    print("Loading teacher model...")
    teacher = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = teacher.config
    num_heads = config.num_attention_heads
    d_head = config.hidden_size // num_heads

    # Load data
    print("Loading data...")
    train_samples = load_data(tokenizer, 500, seed=seed)
    eval_samples = load_data(tokenizer, 50, seed=seed + 1000)

    # Teacher PPL
    print("\nEvaluating teacher...")
    teacher_ppl = evaluate_ppl(teacher, eval_samples, device)
    print(f"  Teacher PPL: {teacher_ppl:.2f}")

    # ========== V-only Residual ==========
    print("\n" + "=" * 50)
    print("V-ONLY RESIDUAL ADAPTER")
    print("=" * 50)

    v_adapter = VResidualAdapter(
        d_head=d_head, rank=RANK, init_std=0.01, init_scale=0.1,
        mode="residual", dtype=torch.float32, device=device,
    )

    v_student = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    v_student = wrap_model_v(v_student, v_adapter, num_heads, d_head)

    v_losses = train_adapter(v_student, teacher, v_adapter, train_samples, num_steps, device)
    v_ppl = evaluate_ppl(v_student, eval_samples, device)
    print(f"\n  V-only PPL: {v_ppl:.2f} (ΔPPL: {v_ppl - teacher_ppl:+.2f})")

    # Analyze V operator
    v_analysis = analyze_residual_operator(v_adapter)
    print(f"  V operator condition: {v_analysis['condition_number']:.4f}")
    print(f"  V energy in top-32: {v_analysis['energy_in_top_32']*100:.2f}%")
    print(f"  V effective rank (99%): {v_analysis['effective_rank_99']}")

    # ========== K-only Residual ==========
    print("\n" + "=" * 50)
    print("K-ONLY RESIDUAL ADAPTER")
    print("=" * 50)

    k_adapter = KResidualAdapter(
        d_head=d_head, rank=RANK, init_std=0.01, init_scale=0.1,
        mode="residual", dtype=torch.float32, device=device,
    )

    k_student = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    k_student = wrap_model_k(k_student, k_adapter, num_heads, d_head)

    k_losses = train_adapter(k_student, teacher, k_adapter, train_samples, num_steps, device)
    k_ppl = evaluate_ppl(k_student, eval_samples, device)
    print(f"\n  K-only PPL: {k_ppl:.2f} (ΔPPL: {k_ppl - teacher_ppl:+.2f})")

    # Analyze K operator
    k_analysis = analyze_k_residual_operator(k_adapter)
    print(f"  K operator condition: {k_analysis['condition_number']:.4f}")
    print(f"  K energy in top-32: {k_analysis['energy_in_top_32']*100:.2f}%")
    print(f"  K effective rank (99%): {k_analysis['effective_rank_99']}")

    # ========== Comparison Summary ==========
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'V-only':<15} {'K-only':<15} {'Better':<10}")
    print("-" * 70)
    print(f"{'PPL (lower=better)':<30} {v_ppl:<15.2f} {k_ppl:<15.2f} {'K' if k_ppl < v_ppl else 'V':<10}")
    print(f"{'ΔPPL vs teacher':<30} {v_ppl-teacher_ppl:<+15.2f} {k_ppl-teacher_ppl:<+15.2f} {'K' if abs(k_ppl-teacher_ppl) < abs(v_ppl-teacher_ppl) else 'V':<10}")
    print(f"{'Condition number':<30} {v_analysis['condition_number']:<15.4f} {k_analysis['condition_number']:<15.4f} {'K' if k_analysis['condition_number'] > v_analysis['condition_number'] else 'V':<10}")
    print(f"{'Energy in top-32 (%)':<30} {v_analysis['energy_in_top_32']*100:<15.2f} {k_analysis['energy_in_top_32']*100:<15.2f} {'K' if k_analysis['energy_in_top_32'] > v_analysis['energy_in_top_32'] else 'V':<10}")
    print(f"{'Effective rank (99%)':<30} {v_analysis['effective_rank_99']:<15} {k_analysis['effective_rank_99']:<15} {'K' if k_analysis['effective_rank_99'] < v_analysis['effective_rank_99'] else 'V':<10}")

    # Key conclusion
    k_more_compressible = k_analysis['energy_in_top_32'] > v_analysis['energy_in_top_32']
    print("\n" + "=" * 70)
    if k_more_compressible:
        print("CONCLUSION: K appears MORE compressible than V!")
        print(f"  K captures {k_analysis['energy_in_top_32']*100:.1f}% energy in top-32 vs V's {v_analysis['energy_in_top_32']*100:.1f}%")
        print("  Consider focusing compression efforts on K rather than V.")
    else:
        print("CONCLUSION: K appears LESS compressible than V (or similar)")
        print(f"  K captures {k_analysis['energy_in_top_32']*100:.1f}% energy in top-32 vs V's {v_analysis['energy_in_top_32']*100:.1f}%")
        print("  Both K and V have near-identity operators, limiting compression potential.")

    # Get singular values for plotting
    M_V = get_residual_operator_matrix(v_adapter)
    M_K = get_k_residual_operator_matrix(k_adapter)
    _, S_V, _ = torch.linalg.svd(M_V.float())
    _, S_K, _ = torch.linalg.svd(M_K.float())
    S_V = S_V.cpu().numpy()
    S_K = S_K.cpu().numpy()

    # Create plot
    if plot_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Singular values comparison
        ax1 = axes[0]
        x = np.arange(1, len(S_V) + 1)
        width = 0.35
        ax1.bar(x - width/2, S_V, width, label='V operator M_V', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, S_K, width, label='K operator M_K', color='coral', alpha=0.8)
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Identity (σ=1)')
        ax1.set_xlabel('Singular Value Index', fontsize=12)
        ax1.set_ylabel('Singular Value', fontsize=12)
        ax1.set_title('K vs V Operator Singular Values', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0.9, 1.1)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative energy
        ax2 = axes[1]
        cumsum_V = np.cumsum(S_V**2) / (S_V**2).sum() * 100
        cumsum_K = np.cumsum(S_K**2) / (S_K**2).sum() * 100
        ax2.plot(x, cumsum_V, 'b-', linewidth=2, marker='o', markersize=3, label='V operator')
        ax2.plot(x, cumsum_K, 'r-', linewidth=2, marker='s', markersize=3, label='K operator')
        ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% energy')
        ax2.axhline(y=99, color='green', linestyle='--', linewidth=1, label='99% energy')
        ax2.axvline(x=32, color='orange', linestyle=':', linewidth=2, label='Rank 32')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Energy (%)', fontsize=12)
        ax2.set_title('K vs V Energy Distribution', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add conclusion text
        conclusion = "K MORE compressible" if k_more_compressible else "K and V similar"
        ax2.text(0.95, 0.05, f'Energy at rank 32:\nV: {cumsum_V[31]:.1f}%\nK: {cumsum_K[31]:.1f}%\n\n{conclusion}',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {plot_path}")

    # Save results
    results = {
        "model": model_name,
        "rank": RANK,
        "num_steps": num_steps,
        "seed": seed,
        "teacher_ppl": teacher_ppl,
        "v_only": {
            "ppl": v_ppl,
            "delta_ppl": v_ppl - teacher_ppl,
            "condition_number": v_analysis["condition_number"],
            "energy_in_top_32": v_analysis["energy_in_top_32"],
            "effective_rank_99": v_analysis["effective_rank_99"],
            "top_5_sv": v_analysis["top_5_sv"],
        },
        "k_only": {
            "ppl": k_ppl,
            "delta_ppl": k_ppl - teacher_ppl,
            "condition_number": k_analysis["condition_number"],
            "energy_in_top_32": k_analysis["energy_in_top_32"],
            "effective_rank_99": k_analysis["effective_rank_99"],
            "top_5_sv": k_analysis["top_5_sv"],
        },
        "conclusion": {
            "k_more_compressible": k_more_compressible,
            "recommendation": "K compression" if k_more_compressible else "Both near-identity",
        },
        "timestamp": datetime.now().isoformat(),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="K vs V Compression Spectrum Probe")
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_comparison(
        model_name=args.model,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed,
        output_path=args.output,
        plot_path=args.plot,
    )


if __name__ == "__main__":
    main()
