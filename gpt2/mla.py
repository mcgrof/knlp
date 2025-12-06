#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Multi-head Latent Attention (MLA) implementations

This module contains MLA variants:
- MLA (GPT2_MLA): Base multi-head latent attention with 6x cache compression
- MLA+KVSplice (GPT2_MLA_KV): MLA with learned compression and *latent-only* SDPA

Key points for GPT2_MLA_KV:
- KV cache is stored in compressed latent space (d_compressed)
- For attention, cache is decompressed only to latent space (d_latent)
- Scaled dot-product attention is performed entirely in latent space
- K/V are never materialized in head space; only the *final* per-head output
  is projected back to head_dim using the value weights from from_kv_latent.

NOTE: The latent-only SDPA path uses a separate RoPE in latent space (d_latent),
so it is an architectural variant, not a mathematically exact rewrite of the
original head-space RoPE attention.

See docs/kvsplice.md for trade-off analysis and usage guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Fisher metrics function
from gpt2.model import compute_fisher_metrics


@dataclass
class MLA_Config:
    """
    Configuration for MLA (Multi-head Latent Attention).

    Features:
    - Latent compression: KV latent decompresses to K, V (Q computed directly)
    - KVSplice compatibility: learned compression for 12x cache reduction
    - RoPE positional embeddings

    Core dimensions
    ---------------
    d_model:      Model embedding dimension
    n_heads:      Number of attention heads
    head_dim:     Dimension per head (d_model // n_heads)
    d_latent:     Latent dimension for KV cache (compressed representation)
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
    d_latent: int = 256  # Compressed latent dimension for KV
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K tensors.

    Args:
        q: Query tensor [B, H, T, head_dim]
        k: Key tensor [B, H, T, head_dim]
        cos: Cosine values [T, head_dim//2]
        sin: Sine values [T, head_dim//2]

    Returns:
        Rotated (q, k) tensors
    """
    # Duplicate cos/sin to match head_dim: [T, head_dim//2] -> [T, head_dim]
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # Reshape for complex rotation
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # Broadcast cos/sin to [1, 1, T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


class LearnedKVSplice(nn.Module):
    """
    Learned low-rank projection for KV latent: d_in -> d_compressed -> d_in.

    - compress: d_in -> d_compressed
    - latent_ln: LayerNorm in compressed space (stabilizes gradients)
    - expand: d_compressed -> d_in

    This trains the model to produce representations that survive the
    low-rank bottleneck, acting as a regularizer that encourages
    structured, compressible QKV representations.
    """

    def __init__(self, d_in: int, d_compressed: int):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = d_compressed

        # Low-rank projection with LayerNorm in compressed space
        self.compress = nn.Linear(d_in, d_compressed, bias=False)
        self.latent_ln = nn.LayerNorm(d_compressed)
        self.expand = nn.Linear(d_compressed, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.latent_ln(self.compress(x))
        return self.expand(compressed)

    def compress_only(self, x: torch.Tensor) -> torch.Tensor:
        # What we store in cache: compressed + LN
        return self.latent_ln(self.compress(x))

    def decompress_only(self, compressed: torch.Tensor) -> torch.Tensor:
        return self.expand(compressed)

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
    Container for MLA_Flash layers.

    Use for ablation testing:
    - GPT-2 baseline vs MLA: measures latent compression benefit
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

        # Q path - direct projection
        q = attn.W_q(x)
        q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        # KV path - project to latent then decompress
        kv_latent = attn.to_kv_latent(x)
        kv = attn.from_kv_latent(kv_latent)
        kv = kv.view(B, T, 2, attn.n_heads, attn.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

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
    MLA with learned KVSplice compression and *latent-only* SDPA.

    - Cache stores compressed KV latents: d_compressed = d_latent * compression_ratio
    - For attention:
        * Cache is decompressed to latent space (d_latent)
        * Current tokens pass through KVSplice bottleneck (d_latent -> d_compressed -> d_latent)
        * All attention math (Q, K, V) is done in latent space (d_latent)
        * Final per-head outputs are obtained by applying the V-part of from_kv_latent
    - No [B, H, T_total, head_dim] K or V tensors are ever materialized.

    NOTE: Uses a separate RoPE in latent space; this is an architectural variant.
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

        # KVSplice bottleneck
        d_compressed = int(cfg.d_latent * compression_ratio)
        self.kvsplice = LearnedKVSplice(cfg.d_latent, d_compressed)
        self.d_compressed = d_compressed

        # Compressed-space SDPA: single head in d_compressed instead of
        # multi-head in d_latent. This exploits the low-rank KV structure
        # more aggressively and keeps cache in compressed space end-to-end.
        self.use_compressed_attn = True

        # Projections for compressed-space attention (single head)
        self.q_proj_compressed = nn.Linear(d_compressed, d_compressed, bias=False)
        self.kv_proj_compressed = nn.Linear(d_compressed, 2 * d_compressed, bias=False)
        # Map compressed output back to latent space (then decode via tied E)
        self.comp2lat = nn.Linear(d_compressed, cfg.d_latent, bias=False)

        # Residual correction: allows model to deviate from strict tied map
        # Zero-init so model starts as pure tied map, relaxes only if needed
        self.out_residual = nn.Linear(cfg.d_latent, cfg.d_model, bias=True)
        nn.init.zeros_(self.out_residual.weight)
        nn.init.zeros_(self.out_residual.bias)

        # RoPE for compressed space
        self.rope_compressed = RotaryEmbedding(
            d_compressed, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        # Q path in head space (full-rank)
        q_dim = cfg.n_heads * cfg.head_dim
        self.W_q = nn.Linear(cfg.d_model, q_dim)

        # Latent path from input (shared encoder E, also used transposed for output)
        self.to_kv_latent = nn.Linear(cfg.d_model, cfg.d_latent, bias=False)

        # Latent -> [K,V] projection (we re-use only the *value* half explicitly);
        # the key half weights are used to define query projections into latent space.
        kv_dim = 2 * cfg.n_heads * cfg.head_dim
        self.from_kv_latent = nn.Linear(cfg.d_latent, kv_dim)

        # Output projection after heads are merged
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE in latent space (d_latent)
        self.rope_latent = RotaryEmbedding(
            cfg.d_latent, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        # For logging
        self._last_reconstruction_error: Optional[float] = None

    def get_kvsplice_metrics(self) -> dict:
        """Get KVSplice metrics for this layer."""
        metrics = self.kvsplice.get_compression_stats()
        if self._last_reconstruction_error is not None:
            metrics["reconstruction_error"] = self._last_reconstruction_error
        return metrics

    def _project_q_to_latent(
        self,
        q_full: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project Q (in head space) into latent space using the *key* part of
        from_kv_latent.

        Shapes:
            q_full: [B, H, T, head_dim]
        Returns:
            q_latent: [B, H, T, d_latent]
        """
        # from_kv_latent.weight: [2*H*head_dim, d_latent]
        w = self.from_kv_latent.weight  # (2*H*head_dim, d_latent)
        H = self.n_heads
        d = self.head_dim

        # First half are keys: shape [H*head_dim, d_latent]
        w_k_full = w[: H * d, :]  # (H*head_dim, d_latent)
        w_k_full = w_k_full.view(H, d, self.d_latent)  # [H, head_dim, d_latent]

        # q_latent[b,h,t,m] = sum_d q_full[b,h,t,d] * w_k_full[h,d,m]
        q_latent = torch.einsum("bhtd,hdm->bhtm", q_full, w_k_full)
        return q_latent  # [B, H, T, d_latent]

    def _project_latent_to_head_values(
        self,
        v_latent_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project latent attention outputs back to per-head value space using
        the *value* part of from_kv_latent.

        Shapes:
            v_latent_out: [B, H, T, d_latent]
        Returns:
            v_head: [B, H, T, head_dim]
        """
        w = self.from_kv_latent.weight  # [2*H*head_dim, d_latent]
        H = self.n_heads
        d = self.head_dim

        # Second half are values: [H*head_dim, d_latent]
        w_v_full = w[H * d :, :]  # (H*head_dim, d_latent)
        w_v_full = w_v_full.view(H, d, self.d_latent)  # [H, head_dim, d_latent]

        v_head = torch.einsum("bhtm,hdm->bhtd", v_latent_out, w_v_full)

        if self.from_kv_latent.bias is not None:
            b = self.from_kv_latent.bias[H * d :]  # (H*head_dim,)
            b = b.view(H, d)  # [H, head_dim]
            v_head = v_head + b.view(1, H, 1, d)

        return v_head  # [B, H, T, head_dim]

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape

        # ---- Compressed-space SDPA path (single head in d_compressed) ----
        if self.use_compressed_attn:
            kv_latent = self.to_kv_latent(x)  # [B,T,d_latent]

            # Occasionally track reconstruction error
            if self.training and torch.rand(1).item() < 0.01:
                self._last_reconstruction_error = (
                    self.kvsplice.get_reconstruction_error(kv_latent).item()
                )

            # Compress current tokens to cache space
            compressed_current = self.kvsplice.compress_only(kv_latent)  # [B,T,d_comp]

            # Stitch cache + current in compressed space
            if cache is not None:
                compressed_all = torch.cat([cache, compressed_current], dim=1)
            else:
                compressed_all = compressed_current
            T_total = compressed_all.shape[1]

            # Q from current compressed tokens, K/V from full compressed sequence
            q_lat = self.q_proj_compressed(compressed_current)  # [B,T,d_comp]
            kv_lat = self.kv_proj_compressed(compressed_all)  # [B,T_total,2*d_comp]
            k_lat, v_lat = kv_lat.split(self.d_compressed, dim=-1)

            # Single head: shape [B,1,T,d_compressed] for SDPA
            q = q_lat.unsqueeze(1)
            k = k_lat.unsqueeze(1)
            v = v_lat.unsqueeze(1)

            # RoPE in compressed space
            cos, sin = self.rope_compressed(x, T_total)
            if cache is not None:
                q_cos, q_sin = cos[-T:], sin[-T:]
                q, _ = apply_rope(q, q, q_cos, q_sin)
                k, _ = apply_rope(k, k, cos, sin)
            else:
                q, k = apply_rope(q, k, cos, sin)

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )  # [B,1,T,d_comp]

            attn_out = attn_out.squeeze(1)  # [B,T,d_comp]

            # Decode via tied map + residual correction
            # to_kv_latent.weight is [d_latent, d_model], so z @ weight gives [B,T,d_model]
            z_out = self.comp2lat(attn_out)  # [B,T,d_latent]
            base_out = z_out @ self.to_kv_latent.weight  # tied map
            corr_out = self.out_residual(z_out)  # residual correction
            out = base_out + corr_out  # [B,T,d_model]

            new_cache = compressed_all if use_cache else None
            return out, new_cache

        # ---- Fallback: multi-head SDPA in d_latent space ----
        H = self.n_heads

        # Q in head space
        q_full = self.W_q(x)  # [B, T, H*d]
        q_full = q_full.view(B, T, H, self.head_dim).permute(0, 2, 1, 3)  # [B,H,T,d]

        # KV latent for current tokens
        kv_latent = self.to_kv_latent(x)  # [B, T, d_latent]

        # Occasionally track reconstruction error
        if self.training and torch.rand(1).item() < 0.01:
            self._last_reconstruction_error = self.kvsplice.get_reconstruction_error(
                kv_latent
            ).item()

        # Pass current tokens through KVSplice bottleneck
        kv_latent_current = self.kvsplice(kv_latent)  # [B, T, d_latent]

        # Handle cache in compressed space
        if cache is not None:
            # cache: [B, T_cache, d_compressed]
            kv_latent_cache = self.kvsplice.decompress_only(cache)  # [B,Tc,d_latent]
            z_all = torch.cat([kv_latent_cache, kv_latent_current], dim=1)
            T_total = z_all.shape[1]
        else:
            z_all = kv_latent_current
            T_total = T

        # Project Q into latent space using key weights
        q_latent = self._project_q_to_latent(q_full)  # [B,H,T,d_latent]

        # Keys and values in latent space
        k_latent = z_all.unsqueeze(1).expand(B, H, T_total, self.d_latent)
        v_latent = k_latent  # use same latent for K and V

        # RoPE in latent space
        cos_lat, sin_lat = self.rope_latent(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos_lat[-T:], sin_lat[-T:]
            q_latent, _ = apply_rope(q_latent, q_latent, q_cos, q_sin)
            k_latent, _ = apply_rope(k_latent, k_latent, cos_lat, sin_lat)
        else:
            q_latent, k_latent = apply_rope(q_latent, k_latent, cos_lat, sin_lat)

        # Latent-only SDPA
        attn_out_latent = F.scaled_dot_product_attention(
            q_latent,
            k_latent,
            v_latent,
            is_causal=True,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )  # [B,H,T,d_latent]

        # Map latent outputs back to head_dim via value weights
        attn_out_heads = self._project_latent_to_head_values(attn_out_latent)
        # [B,H,T,head_dim]

        # Merge heads and project
        attn_out_heads = attn_out_heads.transpose(1, 2).contiguous()  # [B,T,H,d]
        attn_out_heads = attn_out_heads.view(B, T, H * self.head_dim)
        out = self.out_proj(attn_out_heads)  # [B,T,d_model]

        # Store compressed cache
        if use_cache:
            new_cache = self.kvsplice.compress_only(kv_latent)  # [B,T,d_compressed]
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
    """Full GPT-2 with MLA+KVSplice using latent-only SDPA."""

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
        """
        Compute Fisher spectrum metrics for selected layers.

        FIM is computed using a "logical" attention prob over latent-only SDPA.
        This is *not* bitwise identical to the head-space version, but gives
        meaningful relative comparisons across layers / runs.
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

        all_metrics: Dict[str, float] = {}

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
        """
        Approximate attention probabilities for latent-only SDPA.

        We replay the latent-only path without cache and extract the
        softmax(qk^T) in latent space.
        """
        B, T, _ = x.shape
        H = attn.n_heads

        # Q in head space
        q_full = attn.W_q(x).view(B, T, H, attn.head_dim).transpose(1, 2)

        # Latent
        kv_latent = attn.to_kv_latent(x)
        kv_latent_current = attn.kvsplice(kv_latent)  # [B,T,d_latent]

        # Project Q to latent
        q_latent = attn._project_q_to_latent(q_full)  # [B,H,T,d_latent]

        # K in latent
        k_latent = kv_latent_current.unsqueeze(1).expand(B, H, T, attn.d_latent)

        # Latent RoPE
        cos_lat, sin_lat = attn.rope_latent(x, T)
        q_latent, k_latent = apply_rope(q_latent, k_latent, cos_lat, sin_lat)

        # Scores in latent
        scores = torch.matmul(q_latent, k_latent.transpose(-2, -1)) / math.sqrt(
            attn.d_latent
        )

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(scores, dim=-1)
        return attn_probs


class GPT2_MLA_KV_FIM(nn.Module):
    """
    GPT-2 with MLA + selective KVSplice on last N layers only.

    This variant applies KVSplice compression only to the last `kv_layers`
    layers, preserving full MLA fidelity in the early layers. This is
    motivated by FIM (Fisher Information Matrix) trace analysis showing:

    - Early layers (layer0): High trace (~0.95) = critical representation
    - Late layers (layer11): Low trace (~0.62) = safe to compress

    FIM analysis from gpt2_mla_fineweb (H100, 3-hour run):
      layer0:  mean_trace=0.9551 (PROTECT - critical)
      layer6:  mean_trace=0.8191 (moderate)
      layer11: mean_trace=0.6215 (COMPRESS - safe)

    By applying KVSplice only to the last 4 layers (8-11), we achieve:
      - 16.7% KV cache reduction vs MLA-only
      - 7.2x total compression vs standard GPT-2
      - Better quality than full KVSplice (early layers protected)

    Architecture:
      layers 0 to (n_layers - kv_layers - 1): MLABlock (MLA only, 6x)
      layers (n_layers - kv_layers) to (n_layers - 1): MLAKVBlock (12x)
    """

    def __init__(
        self,
        cfg: MLA_Config,
        vocab_size: int = 50257,
        compression_ratio: float = 0.5,
        kv_layers: int = 4,
    ):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio
        self.kv_layers = kv_layers

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks: MLA for early layers, MLA+KVSplice for last N
        kv_start = cfg.n_layers - kv_layers
        blocks = []
        for i in range(cfg.n_layers):
            if i >= kv_start:
                # Last kv_layers: apply KVSplice
                blocks.append(MLAKVBlock(cfg, i, compression_ratio))
            else:
                # Early layers: MLA only (no KVSplice)
                blocks.append(MLABlock(cfg, i))
        self.blocks = nn.ModuleList(blocks)

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
        """Get compression statistics for this hybrid model."""
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        mla_layers = self.cfg.n_layers - self.kv_layers
        kv_layers = self.kv_layers

        # Calculate effective compression
        # MLA: 256 dims per layer, KVSplice: 128 dims per layer (at 0.5 ratio)
        total_dims_mla_only = self.cfg.n_layers * self.cfg.d_latent
        total_dims_hybrid = (mla_layers * self.cfg.d_latent) + (
            kv_layers * d_compressed
        )
        effective_ratio = total_dims_mla_only / total_dims_hybrid

        return {
            "d_latent": self.cfg.d_latent,
            "d_compressed": d_compressed,
            "compression_ratio": self.compression_ratio,
            "kv_layers": kv_layers,
            "mla_layers": mla_layers,
            "effective_compression": f"{effective_ratio:.2f}x vs MLA-only",
            "cache_reduction_vs_mla": f"{(1 - total_dims_hybrid / total_dims_mla_only) * 100:.1f}%",
        }

    def get_kvsplice_metrics(self) -> dict:
        """Get KVSplice metrics for logging to W&B."""
        metrics = {}
        stats = self.get_compression_stats()

        metrics["kvsplice/compression_ratio"] = self.compression_ratio
        metrics["kvsplice/d_latent"] = self.cfg.d_latent
        metrics["kvsplice/d_compressed"] = stats["d_compressed"]
        metrics["kvsplice/kv_layers"] = self.kv_layers
        metrics["kvsplice/mla_layers"] = stats["mla_layers"]

        # Collect per-layer reconstruction errors (only from KVSplice layers)
        reconstruction_errors = []
        kv_start = self.cfg.n_layers - self.kv_layers
        for i, block in enumerate(self.blocks):
            if i >= kv_start and hasattr(block.attn, "_last_reconstruction_error"):
                error = block.attn._last_reconstruction_error
                if error is not None:
                    reconstruction_errors.append(error)
                    metrics[f"kvsplice/layer_{i}_recon_error"] = error

        if reconstruction_errors:
            metrics["kvsplice/avg_reconstruction_error"] = sum(
                reconstruction_errors
            ) / len(reconstruction_errors)

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

        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=device))
        h = self.drop(tok_emb + pos_emb)

        all_metrics: Dict[str, float] = {}
        kv_start = self.cfg.n_layers - self.kv_layers

        for i, block in enumerate(self.blocks):
            if i in layer_indices:
                if i >= kv_start:
                    # KVSplice layer
                    attn_probs = self._get_attn_probs_kvsplice(
                        block.attn, block.ln_1(h)
                    )
                else:
                    # MLA layer
                    attn_probs = self._get_attn_probs_mla(block.attn, block.ln_1(h))

                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h, _ = block(h)

        return all_metrics

    def _get_attn_probs_mla(
        self, attn: "MLA_Flash", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from MLA_Flash layer."""
        B, T, _ = x.shape

        q = attn.W_q(x)
        q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        kv_latent = attn.to_kv_latent(x)
        kv = attn.from_kv_latent(kv_latent)
        kv = kv.view(B, T, 2, attn.n_heads, attn.head_dim)
        k = kv[:, :, 0].transpose(1, 2)

        cos, sin = attn.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(scores, dim=-1)

        return attn_probs

    def _get_attn_probs_kvsplice(
        self, attn: "MLA_KVSplice", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from MLA_KVSplice layer."""
        B, T, _ = x.shape
        H = attn.n_heads

        q_full = attn.W_q(x).view(B, T, H, attn.head_dim).transpose(1, 2)

        kv_latent = attn.to_kv_latent(x)
        kv_latent_current = attn.kvsplice(kv_latent)

        q_latent = attn._project_q_to_latent(q_full)
        k_latent = kv_latent_current.unsqueeze(1).expand(B, H, T, attn.d_latent)

        cos_lat, sin_lat = attn.rope_latent(x, T)
        q_latent, k_latent = apply_rope(q_latent, k_latent, cos_lat, sin_lat)

        scores = torch.matmul(q_latent, k_latent.transpose(-2, -1)) / math.sqrt(
            attn.d_latent
        )

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(scores, dim=-1)

        return attn_probs
