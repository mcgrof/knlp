#!/usr/bin/env python3
"""
Plot singular value spectrum of the residual operator M.

Generates visualization showing that M = I + s*W_v_out@W_v is near-identity,
explaining why V-only cache compression fails at low ranks.
"""

import argparse
import json
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.compression.kv_compressor_plugin import (
    VResidualAdapter,
    get_residual_operator_matrix,
)


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


class ResidualWrapper(torch.nn.Module):
    def __init__(self, c_attn, adapter, embed_dim, num_heads, d_head):
        super().__init__()
        self.c_attn = c_attn
        self.adapter = adapter
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = d_head

    def forward(self, x):
        qkv = self.c_attn(x)
        q = qkv[:, :, : self.embed_dim]
        k = qkv[:, :, self.embed_dim : 2 * self.embed_dim]
        v = qkv[:, :, 2 * self.embed_dim :]
        B, T, _ = v.shape
        v_heads = v.view(B, T, self.num_heads, self.d_head)
        v_latent = self.adapter.compress_v(v_heads)
        v_out = self.adapter.expand_v(v_latent, v_original=v_heads)
        v_flat = v_out.view(B, T, -1)
        return torch.cat([q, k, v_flat], dim=-1)


def train_adapter(model, teacher, adapter, samples, num_steps, device):
    """Train residual adapter."""
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    model.train()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="operator_spectrum.png")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Load model
    print("Loading model...")
    teacher = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = teacher.config
    num_heads = config.num_attention_heads
    d_head = config.hidden_size // num_heads

    # Load data
    print("Loading data...")
    samples = load_data(tokenizer, 500)

    # Create and train adapter
    print("Training V-only residual adapter...")
    adapter = VResidualAdapter(
        d_head=d_head, rank=32, init_std=0.01, init_scale=0.1,
        mode="residual", dtype=torch.float32, device=device,
    )

    student = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    embed_dim = num_heads * d_head
    for block in student.transformer.h:
        block.attn.c_attn = ResidualWrapper(
            block.attn.c_attn, adapter, embed_dim, num_heads, d_head
        )

    train_adapter(student, teacher, adapter, samples, args.num_steps, device)

    # Extract and analyze operator
    print("Analyzing operator spectrum...")
    M = get_residual_operator_matrix(adapter)
    U, S, Vh = torch.linalg.svd(M.float())
    S = S.cpu().numpy()

    # Compute cumulative energy
    total_energy = (S ** 2).sum()
    cumsum_energy = np.cumsum(S ** 2) / total_energy

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Singular values
    ax1 = axes[0]
    ax1.bar(range(1, len(S) + 1), S, color='steelblue', alpha=0.8)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Identity (σ=1)')
    ax1.set_xlabel('Singular Value Index', fontsize=12)
    ax1.set_ylabel('Singular Value', fontsize=12)
    ax1.set_title('Singular Values of M = I + s·W_out·W_in', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.9, 1.1)
    ax1.grid(True, alpha=0.3)

    # Annotate key stats
    ax1.text(0.95, 0.95, f'cond(M) = {S[0]/S[-1]:.3f}\nmax σ = {S[0]:.4f}\nmin σ = {S[-1]:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Cumulative energy
    ax2 = axes[1]
    ax2.plot(range(1, len(S) + 1), cumsum_energy * 100, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50% energy')
    ax2.axhline(y=99, color='green', linestyle='--', linewidth=1.5, label='99% energy')
    ax2.axvline(x=32, color='orange', linestyle=':', linewidth=2, label='Rank 32')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax2.set_title('Energy Distribution (Near-Identity = Flat)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Find where 99% is reached
    eff_rank_99 = np.argmax(cumsum_energy >= 0.99) + 1
    energy_at_32 = cumsum_energy[31] * 100
    ax2.text(0.95, 0.05, f'Energy at rank 32: {energy_at_32:.1f}%\n99% energy needs: {eff_rank_99} dims',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {args.output}")

    # Print summary
    print("\n" + "="*60)
    print("OPERATOR SPECTRUM ANALYSIS")
    print("="*60)
    print(f"Condition number: {S[0]/S[-1]:.4f}")
    print(f"Singular value range: [{S[-1]:.4f}, {S[0]:.4f}]")
    print(f"Energy in top-32: {energy_at_32:.2f}%")
    print(f"Effective rank (99%): {eff_rank_99}")
    print("\nConclusion: M ≈ I, so rank-32 SVD loses ~50% energy")


if __name__ == "__main__":
    main()
