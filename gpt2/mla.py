#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Multi-head Latent Attention (MLA) implementations

This module contains MLA variants:
- MLA: Base multi-head latent attention with 6x cache compression
- MLA_KV2: Improved MLA with better K/V decomposition  
- MLA_KV2M: MLA_KV2 with MLP splice compression
- KVSplice: Learned compression for 12x total cache reduction

These are independent of Reciprocal Attention research.
See docs/kvsplice.md for details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



class MLA_Config:
    """
    Configuration for MLA (Multi-head Latent Attention).

    Features:
    - Latent compression: single latent decompresses to Q, K, V
    - KVSplice compatibility: learned compression for 12x cache reduction
    - RoPE positional embeddings

    Core dimensions
    ---------------
    d_model:      Model embedding dimension
    n_heads:      Number of attention heads
    head_dim:     Dimension per head (d_model // n_heads)
    d_latent:     Latent dimension for cache (compressed representation)
    block_size:   Maximum sequence length
    n_layers:     Total number of layers

    RoPE settings
    -------------
    rope_theta:   Base for rotary position embeddings (default: 10000.0)

    Other
    -----
    dropout:      Attention dropout probability
    """

    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64
    d_latent: int = 256  # Compressed latent dimension
    block_size: int = 1024
    n_layers: int = 12
    rope_theta: float = 10000.0
    dropout: float = 0.0




class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for attention."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for efficiency
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for positions up to seq_len."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]




def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor


class LearnedKVSplice(nn.Module):
    """
    Learned information bottleneck for QKV compression.

    Compresses in feature dimension (d_in → d_compressed) using:
    1. Monotonic transform: softplus(scale) * x + shift
       - Reorders/ranks dimensions before compression
    2. Low-rank projection: compress → expand linear layers
       - Forces information through smaller space

    This trains the model to produce representations that survive the
    bottleneck, acting as a regularizer that encourages structured,
    compressible QKV representations.

    Note: This is NOT approximating spline+PCA despite historical naming.
    It's a learned bottleneck with monotonic pre-transform.
    """

    def __init__(self, d_in: int, d_compressed: int):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = d_compressed

        # Learned monotonic transform
        self.transform_scale = nn.Parameter(torch.ones(d_in))
        self.transform_shift = nn.Parameter(torch.zeros(d_in))

        # Learned low-rank projection
        self.compress = nn.Linear(d_in, d_compressed, bias=False)
        self.expand = nn.Linear(d_compressed, d_in, bias=False)

        # Initialize as approximate inverse
        nn.init.orthogonal_(self.compress.weight)
        with torch.no_grad():
            self.expand.weight.copy_(self.compress.weight.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_transformed = x * F.softplus(self.transform_scale) + self.transform_shift
        compressed = self.compress(x_transformed)
        decompressed = self.expand(compressed)
        return (decompressed - self.transform_shift) / (
            F.softplus(self.transform_scale) + 1e-6
        )

    def compress_only(self, x: torch.Tensor) -> torch.Tensor:
        x_transformed = x * F.softplus(self.transform_scale) + self.transform_shift
        return self.compress(x_transformed)

    def decompress_only(self, compressed: torch.Tensor) -> torch.Tensor:
        decompressed = self.expand(compressed)
        return (decompressed - self.transform_shift) / (
            F.softplus(self.transform_scale) + 1e-6
        )

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.mse_loss(self.forward(x), x)

    def get_compression_stats(self) -> dict:
        return {
            "d_in": self.d_in,
            "d_compressed": self.d_compressed,
            "compression_ratio": self.d_compressed / self.d_in,
            "memory_reduction": 1.0 - (self.d_compressed / self.d_in),
        }




class MLA_Flash(nn.Module):
    """
    GPT-2 with Multi-head Latent Attention (DeepSeek-style MLA).

    Implements the MLA mechanism from DeepSeek-V2/V3 for KV cache compression.
    Q is computed directly (full dimension), while K and V share a compressed
    latent representation that gets cached and decompressed at inference time.

    Architecture:
    - Q path: x -> W_q -> Q (no compression, not cached)
    - KV path: x -> to_kv_latent -> d_latent -> from_kv_latent -> [K, V]

    The KV latent is what gets cached, reducing memory by compression_ratio.
    Flash attention compatible via PyTorch SDPA with RoPE.
    """

    def __init__(self, cfg: MLA_Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Q path - direct projection (no compression, not cached)
        q_dim = cfg.n_heads * cfg.head_dim
        self.W_q = nn.Linear(cfg.d_model, q_dim)

        # KV path - compressed latent (this is what gets cached)
        self.to_kv_latent = nn.Linear(cfg.d_model, cfg.d_latent)
        kv_dim = 2 * cfg.n_heads * cfg.head_dim
        self.from_kv_latent = nn.Linear(cfg.d_latent, kv_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with KV-latent cache, always using standard attention."""
        B, T, D = x.shape

        # Q computed directly from input (not cached)
        q = self.W_q(x)  # [B, T, n_heads * head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, H, T, head_dim]

        # KV from compressed latent (this is what we cache)
        kv_latent = self.to_kv_latent(x)  # [B, T, d_latent]

        # Handle cache
        if cache is not None:
            full_kv_latent = torch.cat([cache, kv_latent], dim=1)
            T_total = full_kv_latent.shape[1]
        else:
            full_kv_latent = kv_latent
            T_total = T

        # Decompress to K, V
        kv = self.from_kv_latent(full_kv_latent)  # [B, T_total, 2*H*head_dim]
        kv = kv.view(B, T_total, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, H, T_total, head_dim]
        k, v = kv[0], kv[1]

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # Standard attention: Q @ K.T @ V
        # Always use causal masking. Chunked decode (cache + T>1) would need a
        # custom mask accounting for prefix offset - is_causal alone won't work.
        use_causal = True
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=use_causal,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        new_cache = full_kv_latent if use_cache else None
        return out, new_cache




class MLA_Model(nn.Module):
    """
    Container for MLA_Flash layers (baseline without reciprocal alternation).

    Use for ablation testing:
    - GPT-2 baseline vs MLA: measures latent compression benefit
    - MLA vs RA_MLA: measures reciprocal alternation benefit
    """

    def __init__(self, cfg: MLA_Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([MLA_Flash(cfg, i) for i in range(cfg.n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward through all layers."""
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, layer_cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        return x, new_caches




class MLABlock(nn.Module):
    """Transformer block with MLA attention + MLP."""

    def __init__(self, cfg: MLA_Config, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_Flash(cfg, layer_idx)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # MLP: 4x expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        # Attention with residual
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache




class GPT2_MLA(nn.Module):
    """Full GPT-2 model with MLA attention."""

    def __init__(self, cfg: MLA_Config, vocab_size: int = 50257):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([MLABlock(cfg, i) for i in range(cfg.n_layers)])

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        assert self.wte.weight is self.lm_head.weight, "Weight tying failed"

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """
        Compute Fisher spectrum metrics for selected layers.

        Args:
            x: Input tensor [B, T]
            layer_indices: Which layers to analyze (default: [0, n_layers//2, -1])
            n_samples: Samples per head for eigenvalue computation
            topk: Number of top eigenvalues to log

        Returns:
            Dictionary of Fisher metrics for W&B logging
        """
        if layer_indices is None:
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass to capture attention probabilities
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h, _ = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "MLA_Flash", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from MLA_Flash layer."""
        B, T, _ = x.shape

        # Project to latent
        latent = attn.to_latent(x)

        # Decompress to Q, K, V
        qkv = attn.from_latent(latent)
        qkv = qkv.view(B, T, 3, attn.n_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores_masked = scores.masked_fill(causal_mask, float("-inf"))

        # Standard attention probs
        attn_probs = F.softmax(scores_masked, dim=-1)

        return attn_probs  # [B, H, T, T]




class MLA_KVSplice(nn.Module):
    """
    MLA with learned KVSplice compression but NO reciprocal alternation.

    Uses standard Q·K^T attention (not K·Q^T) while still applying the
    KVSplice compression bottleneck. This isolates the benefit of learned
    compression from the reciprocal attention mechanism.
    """

    def __init__(
        self,
        cfg: MLA_Config,
        layer_idx: int,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Learned KVSplice compression
        d_compressed = int(cfg.d_latent * compression_ratio)
        self.kvsplice = LearnedKVSplice(cfg.d_latent, d_compressed)
        self.d_compressed = d_compressed

        # Input projection to latent space
        self.to_latent = nn.Linear(cfg.d_model, cfg.d_latent)

        # Decompress latent to Q, K, V
        qkv_dim = 3 * cfg.n_heads * cfg.head_dim
        self.from_latent = nn.Linear(cfg.d_latent, qkv_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        # Track reconstruction error for metrics
        self._last_reconstruction_error = None

    def get_kvsplice_metrics(self) -> dict:
        """Get KVSplice metrics for this layer."""
        metrics = self.kvsplice.get_compression_stats()
        if self._last_reconstruction_error is not None:
            metrics["reconstruction_error"] = self._last_reconstruction_error
        return metrics

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with learned KVSplice compression.

        Always uses standard attention (Q·K^T), no alternation.
        Cache stores compressed latents (d_compressed instead of d_latent).
        """
        B, T, D = x.shape

        # Project to latent space
        latent_orig = self.to_latent(x)  # [B, T, d_latent]

        # Apply KVSplice bottleneck (learn compressible representations)
        latent = self.kvsplice(latent_orig)

        # Track reconstruction error (compute occasionally to avoid overhead)
        if self.training and torch.rand(1).item() < 0.01:  # 1% of steps
            self._last_reconstruction_error = self.kvsplice.get_reconstruction_error(
                latent_orig
            ).item()

        # Handle cache (stored in compressed form)
        if cache is not None:
            # Decompress cached latents
            cache_decompressed = self.kvsplice.decompress_only(cache)
            full_latent = torch.cat([cache_decompressed, latent], dim=1)
            T_total = full_latent.shape[1]
        else:
            full_latent = latent
            T_total = T

        # Decompress to Q, K, V
        qkv = self.from_latent(full_latent)
        qkv = qkv.view(B, T_total, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if cache is not None:
            q = q[:, :, -T:, :]

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # Standard attention (always Q·K^T, no alternation)
        # Causal masking: needed during training (cache=None) and when
        # processing multiple new tokens with cache (chunked decoding).
        # Safe to skip only when T=1 with cache (single token generation).
        # Always use causal masking. Chunked decode (cache + T>1) would need a
        # custom mask accounting for prefix offset - is_causal alone won't work.
        use_causal = True
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=use_causal,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # Store compressed cache
        if use_cache:
            new_cache = self.kvsplice.compress_only(full_latent)
        else:
            new_cache = None

        return out, new_cache




class MLAKVBlock(nn.Module):
    """Transformer block with MLA_KVSplice attention + MLP."""

    def __init__(
        self,
        cfg: MLA_Config,
        layer_idx: int,
        compression_ratio: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_KVSplice(cfg, layer_idx, compression_ratio)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # MLP: 4x expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache




class GPT2_MLA_KV(nn.Module):
    """Full GPT-2 model with MLA+KVSplice attention (no reciprocal alternation)."""

    def __init__(
        self,
        cfg: MLA_Config,
        vocab_size: int = 50257,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks (no alternation_logits needed)
        self.blocks = nn.ModuleList(
            [MLAKVBlock(cfg, i, compression_ratio) for i in range(cfg.n_layers)]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        assert self.wte.weight is self.lm_head.weight, "Weight tying failed"

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_compression_stats(self):
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        return {
            "d_latent": self.cfg.d_latent,
            "d_compressed": d_compressed,
            "compression_ratio": self.compression_ratio,
            "cache_reduction": f"{(1 - self.compression_ratio) * 100:.1f}%",
        }

    def get_kvsplice_metrics(self) -> dict:
        """Get comprehensive KVSplice metrics for logging to W&B."""
        metrics = {}

        # Overall compression stats
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        metrics["kvsplice/compression_ratio"] = self.compression_ratio
        metrics["kvsplice/d_latent"] = self.cfg.d_latent
        metrics["kvsplice/d_compressed"] = d_compressed
        metrics["kvsplice/memory_reduction_pct"] = (1 - self.compression_ratio) * 100

        # Collect per-layer reconstruction errors
        reconstruction_errors = []
        for i, block in enumerate(self.blocks):
            if hasattr(block.attn, "_last_reconstruction_error"):
                error = block.attn._last_reconstruction_error
                if error is not None:
                    reconstruction_errors.append(error)
                    metrics[f"kvsplice/layer_{i}_recon_error"] = error

        # Aggregate reconstruction error
        if reconstruction_errors:
            metrics["kvsplice/avg_reconstruction_error"] = sum(
                reconstruction_errors
            ) / len(reconstruction_errors)
            metrics["kvsplice/max_reconstruction_error"] = max(reconstruction_errors)
            metrics["kvsplice/min_reconstruction_error"] = min(reconstruction_errors)

        return metrics

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """Compute Fisher spectrum metrics for selected layers."""
        if layer_indices is None:
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass to capture attention probabilities
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h, _ = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "MLA_KVSplice", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from MLA_KVSplice layer."""
        B, T, _ = x.shape

        # Project to latent
        latent_orig = attn.to_latent(x)
        latent = attn.kvsplice(latent_orig)

        # Decompress to Q, K, V
        qkv = attn.from_latent(latent)
        qkv = qkv.view(B, T, 3, attn.n_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores_masked = scores.masked_fill(causal_mask, float("-inf"))

        # Standard attention probs
        attn_probs = F.softmax(scores_masked, dim=-1)

        return attn_probs  # [B, H, T, T]




class MLPSplice(nn.Module):
    """
    MLP operating in a compressed latent space.

    Instead of d_model -> 4*d_model -> d_model, we do:
    d_model -> d_latent -> 4*d_latent -> d_latent -> d_model

    This reduces MLP parameters by ~6-9x while maintaining expressiveness.
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        dropout: float = 0.1,
        tie_projections: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.tie = tie_projections

        # Project to latent
        self.proj_in = nn.Linear(d_model, d_latent, bias=False)

        # MLP in latent space
        d_hidden = 4 * d_latent
        self.fc1 = nn.Linear(d_latent, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_latent)
        self.act = nn.GELU()

        # Project back (optionally tied)
        if not tie_projections:
            self.proj_out = nn.Linear(d_latent, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compress
        h = self.proj_in(x)  # [B, T, d_latent]

        # MLP in latent
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)

        # Expand back
        if self.tie:
            out = F.linear(h, self.proj_in.weight.T)
        else:
            out = self.proj_out(h)

        return self.dropout(out)

    def get_compression_stats(self):
        # Standard MLP params: 2 * d_model * 4 * d_model
        standard = 2 * self.d_model * 4 * self.d_model
        # MLPSplice params: proj_in + fc1 + fc2 + proj_out (if not tied)
        mlpsplice = (
            self.d_model * self.d_latent  # proj_in
            + self.d_latent * 4 * self.d_latent  # fc1
            + 4 * self.d_latent * self.d_latent  # fc2
        )
        if not self.tie:
            mlpsplice += self.d_latent * self.d_model  # proj_out
        return {
            "d_latent": self.d_latent,
            "standard_params": standard,
            "mlpsplice_params": mlpsplice,
            "reduction": f"{(1 - mlpsplice/standard) * 100:.1f}%",
        }




class MLA_KV2_Attention(nn.Module):
    """
    MLA with 2 separate compressed latents for K and V only.

    Q is computed directly (no compression, not cached) for maximum efficiency.
    K and V have separate compressed latents that are learned and cached.

    This is more optimal than 3-latent (Q+K+V) because:
    - Q doesn't need caching (recomputed each forward pass)
    - Separate K/V latents allow independent optimization
    - Reduces cache size while maintaining quality
    """

    def __init__(
        self,
        cfg: MLA_Config,
        layer_idx: int,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Q path - direct projection (no compression, not cached)
        q_dim = cfg.n_heads * cfg.head_dim
        self.W_q = nn.Linear(cfg.d_model, q_dim)

        # K path - separate compressed latent
        d_k_compressed = int(cfg.d_latent * compression_ratio)
        self.to_k_latent = nn.Linear(cfg.d_model, cfg.d_latent)
        self.k_splice = LearnedKVSplice(cfg.d_latent, d_k_compressed)
        k_dim = cfg.n_heads * cfg.head_dim
        self.from_k_latent = nn.Linear(cfg.d_latent, k_dim)

        # V path - separate compressed latent
        d_v_compressed = int(cfg.d_latent * compression_ratio)
        self.to_v_latent = nn.Linear(cfg.d_model, cfg.d_latent)
        self.v_splice = LearnedKVSplice(cfg.d_latent, d_v_compressed)
        v_dim = cfg.n_heads * cfg.head_dim
        self.from_v_latent = nn.Linear(cfg.d_latent, v_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[tuple]]:
        """
        Forward with 2-latent caching (K and V only).

        Cache is tuple of (k_latent_compressed, v_latent_compressed).
        """
        B, T, D = x.shape

        # Q: Direct computation (no caching)
        q = self.W_q(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        # K: Compress latent
        k_latent_orig = self.to_k_latent(x)
        k_latent = self.k_splice(k_latent_orig)

        # V: Compress latent
        v_latent_orig = self.to_v_latent(x)
        v_latent = self.v_splice(v_latent_orig)

        # Handle cache
        if cache is not None:
            k_cache, v_cache = cache
            # Decompress and concatenate
            k_cache_decompressed = self.k_splice.decompress_only(k_cache)
            v_cache_decompressed = self.v_splice.decompress_only(v_cache)
            full_k_latent = torch.cat([k_cache_decompressed, k_latent], dim=1)
            full_v_latent = torch.cat([v_cache_decompressed, v_latent], dim=1)
            T_total = full_k_latent.shape[1]
        else:
            full_k_latent = k_latent
            full_v_latent = v_latent
            T_total = T

        # Decompress to K, V
        k = self.from_k_latent(full_k_latent)
        k = k.view(B, T_total, self.n_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = self.from_v_latent(full_v_latent)
        v = v.view(B, T_total, self.n_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # Standard attention
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )

        # Output
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # New cache (compressed K and V latents)
        new_cache = None
        if use_cache:
            # Compress for caching
            k_compressed = self.k_splice.compress_only(k_latent_orig)
            v_compressed = self.v_splice.compress_only(v_latent_orig)
            if cache is not None:
                k_cache, v_cache = cache
                k_compressed = torch.cat([k_cache, k_compressed], dim=1)
                v_compressed = torch.cat([v_cache, v_compressed], dim=1)
            new_cache = (k_compressed, v_compressed)

        return out, new_cache




class MLA_KV2_Block(nn.Module):
    """Transformer block with MLA_KV2_Attention + standard MLP."""

    def __init__(
        self,
        cfg: MLA_Config,
        layer_idx: int,
        compression_ratio: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_KV2_Attention(cfg, layer_idx, compression_ratio)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # Standard MLP
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache




class GPT2_MLA_KV2(nn.Module):
    """Full GPT-2 model with MLA 2-latent attention (K/V compressed, Q direct)."""

    def __init__(
        self,
        cfg: MLA_Config,
        vocab_size: int = 50257,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [MLA_KV2_Block(cfg, i, compression_ratio) for i in range(cfg.n_layers)]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        assert self.wte.weight is self.lm_head.weight, "Weight tying failed"

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_compression_stats(self):
        """Get compression statistics for all layers."""
        stats = {
            "k_compressed_dim": int(self.cfg.d_latent * self.compression_ratio),
            "v_compressed_dim": int(self.cfg.d_latent * self.compression_ratio),
            "compression_ratio": self.compression_ratio,
        }
        return stats

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """Compute Fisher spectrum metrics for selected layers."""
        if layer_indices is None:
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "MLA_KV2_Attention", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from MLA_KV2_Attention layer."""
        B, T, _ = x.shape

        # Q projection (direct, no compression)
        q = attn.to_q(x)
        q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        # K and V from separate compressed latents
        k_latent = attn.to_k_latent(x)
        v_latent = attn.to_v_latent(x)

        k = attn.from_k_latent(k_latent)
        v = attn.from_v_latent(v_latent)

        k = k.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        v = v.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Standard attention (no reciprocal for MLA_KV2)
        attn_probs = F.softmax(scores, dim=-1)

        return attn_probs




class MLA_KV2_MLPSPLICE_Block(nn.Module):
    """MLA 2-latent + MLPSplice (MLP compression)."""

    def __init__(
        self,
        cfg: MLA_Config,
        layer_idx: int,
        compression_ratio: float,
        mlp_d_latent: int,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_KV2_Attention(cfg, layer_idx, compression_ratio)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # MLPSplice - compressed MLP with latent bottleneck
        self.mlp = MLPSplice(cfg.d_model, mlp_d_latent)

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache




class GPT2_MLA_KV2M(nn.Module):
    """Full GPT-2 with MLA 2-latent + MLPSplice compression."""

    def __init__(
        self,
        cfg: MLA_Config,
        vocab_size: int = 50257,
        compression_ratio: float = 0.5,
        mlp_d_latent: int = 256,
    ):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio
        self.mlp_d_latent = mlp_d_latent

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks with MLP compression
        self.blocks = nn.ModuleList(
            [
                MLA_KV2_MLPSPLICE_Block(cfg, i, compression_ratio, mlp_d_latent)
                for i in range(cfg.n_layers)
            ]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        assert self.wte.weight is self.lm_head.weight, "Weight tying failed"

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_compression_stats(self):
        """Get compression statistics."""
        stats = {
            "k_compressed_dim": int(self.cfg.d_latent * self.compression_ratio),
            "v_compressed_dim": int(self.cfg.d_latent * self.compression_ratio),
            "mlp_latent_dim": self.mlp_d_latent,
            "compression_ratio": self.compression_ratio,
        }
        return stats

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """Compute Fisher spectrum metrics for selected layers."""
        if layer_indices is None:
            n = len(self.blocks)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "MLA_KV2_Attention", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from MLA_KV2_Attention layer."""
        B, T, _ = x.shape

        # Q projection (direct, no compression)
        q = attn.to_q(x)
        q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        # K and V from separate compressed latents
        k_latent = attn.to_k_latent(x)
        v_latent = attn.to_v_latent(x)

        k = attn.from_k_latent(k_latent)
        v = attn.from_v_latent(v_latent)

        k = k.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        v = v.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Standard attention (no reciprocal for MLA_KV2M)
        attn_probs = F.softmax(scores, dim=-1)

        return attn_probs


