#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Double Attention: Evaluating Reciprocal Patterns

Research question: What does computing attention twice per layer add?

Standard attention: softmax(Q@K.T)@V
Double attention: softmax(Q@K.T)@V + softmax(K@Q.T)@V

This doubles FLOPs per attention layer. Fair comparison requires scaling down
model size to maintain total compute budget.

See docs/ra.md for experiment design and results.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleAttention(nn.Module):
    """
    Compute both forward (Q@K.T) and reciprocal (K@Q.T) attention.

    Standard: y = softmax(Q@K.T)@V
    Reciprocal: y = softmax(K@Q.T)@V
    Combined: y = y_forward + y_reciprocal

    This doubles attention FLOPs. Use with smaller models to maintain
    fair compute comparison vs baseline.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Single QKV projection (shared for both attention operations)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] input tensor
            mask: Optional attention mask

        Returns:
            [B, T, d_model] output tensor
        """
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3*d_model]
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape for multi-head attention: [B, H, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Forward attention: Q@K.T
        y_forward = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Reciprocal attention: K@Q.T
        y_reciprocal = F.scaled_dot_product_attention(
            k,
            q,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Combine: simple addition
        # Future: explore learned gating, concatenation, etc.
        y = y_forward + y_reciprocal  # [B, H, T, head_dim]

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class DoubleAttentionBlock(nn.Module):
    """
    Transformer block with double attention.

    Architecture:
        x = x + double_attention(LayerNorm(x))
        x = x + mlp(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = DoubleAttention(d_model, n_heads, dropout, bias)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] input

        Returns:
            [B, T, d_model] output
        """
        # Attention with residual
        x = x + self.attn(self.ln1(x))

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x


class GPT2TinyDoubleAttention(nn.Module):
    """
    GPT-2 Tiny with double attention.

    Configuration for fair comparison vs baseline:
    - 6 layers (half of GPT-2 124M's 12)
    - d_model=512 (vs 768 in GPT-2 124M)
    - n_heads=8 (vs 12 in GPT-2 124M)
    - 2 attention ops/layer (vs 1 in baseline)

    Total attention FLOPs approximately matches baseline GPT-2 Tiny with 1 attn/layer.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 1024,
        n_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.block_size = block_size
        self.n_layers = n_layers
        self.d_model = d_model

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DoubleAttentionBlock(d_model, n_heads, d_ff, dropout, bias)
                for _ in range(n_layers)
            ]
        )

        # Final layer norm and output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with standard GPT-2 scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: [B, T] token indices
            targets: Optional [B, T] target indices for loss

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Embeddings
        tok_emb = self.token_emb(idx)  # [B, T, d_model]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)  # [T, d_model]
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and logits
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: [B, T] conditioning tokens
            max_new_tokens: number of tokens to generate
            temperature: softmax temperature (1.0 = no change)
            top_k: if set, only sample from top k logits

        Returns:
            [B, T + max_new_tokens] generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Optionally filter with top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
