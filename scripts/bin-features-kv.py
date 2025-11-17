# SPDX-License-Identifier: MIT
#
# Simple binary feature mask over K/V channels for attention.
#
# Intended usage:
#   - K, V: [batch, n_heads, seq_len, head_dim]
#   - mask: [n_heads, head_dim] (broadcasted over batch/seq)
#
# You can:
#   * use BinaryKVFeatureMask as a learnable mask module, or
#   * call apply_kv_feature_mask() with your own mask tensor.

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVMaskConfig:
    n_heads: int
    head_dim: int
    # Initial mask density (fraction of channels "on" at start)
    init_keep_prob: float = 0.5
    # Temperature for the relaxed / STE mask
    temperature: float = 1.0
    # If True: use straight-through estimator for hard {0,1} mask
    hard: bool = True


class BinaryKVFeatureMask(nn.Module):
    """
    Learnable binary feature mask over K/V channels.

    - Mask is parameterized as logits with a sigmoid -> probs
    - `hard=True` uses a straight-through estimator:
        hard_mask = (p > 0.5).float() + (p - p.detach())
      so gradients flow through the probabilities.

    Shapes:
      - K, V: [B, H, T, D]
      - internal logits: [H, D]
      - broadcast mask: [1, H, 1, D]
    """

    def __init__(self, cfg: KVMaskConfig):
        super().__init__()
        self.cfg = cfg

        # Initialize logits so that sigmoid(logits) ~= init_keep_prob
        p = cfg.init_keep_prob
        p = min(max(p, 1e-4), 1.0 - 1e-4)  # clamp
        init_logit = math.log(p / (1.0 - p))

        self.mask_logits = nn.Parameter(
            torch.full(
                (cfg.n_heads, cfg.head_dim),
                fill_value=init_logit,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the feature mask to K and V.

        Args:
          k: [B, H, T, D]
          v: [B, H, T, D]
          temperature: optional override for mask temperature

        Returns:
          k_masked: [B, H, T, D]
          v_masked: [B, H, T, D]
          mask:     [H, D] (post-sigmoid, STE if hard=True)
        """
        assert k.shape == v.shape, "K and V must have same shape"
        B, H, T, D = k.shape

        assert H == self.cfg.n_heads, f"Expected {self.cfg.n_heads} heads, got {H}"
        assert D == self.cfg.head_dim, f"Expected head_dim={self.cfg.head_dim}, got {D}"

        temp = temperature if temperature is not None else self.cfg.temperature

        # Relaxed probabilities in (0,1)
        probs = torch.sigmoid(self.mask_logits / temp)  # [H, D]

        if self.cfg.hard:
            # Straight-through binary mask: forward pass is {0,1}, gradients via probs
            hard_mask = (probs > 0.5).float()
            mask = hard_mask + (probs - probs.detach())
        else:
            # Soft mask (no hard thresholding)
            mask = probs

        # Broadcast mask over batch and seq_len: [1, H, 1, D]
        mask_b = mask.view(1, H, 1, D)

        k_masked = k * mask_b
        v_masked = v * mask_b
        return k_masked, v_masked, mask


def apply_kv_feature_mask(
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stateless helper if you already have a binary mask.

    Args:
      k:    [B, H, T, D]
      v:    [B, H, T, D]
      mask: [H, D] or [1, H, 1, D]  (binary {0,1} or real-valued)

    Returns:
      (k_masked, v_masked)
    """
    assert k.shape == v.shape, "K and V must have same shape"
    B, H, T, D = k.shape

    if mask.ndim == 2:
        assert mask.shape == (H, D), f"Expected mask [H,D]={H,D}, got {mask.shape}"
        mask_b = mask.view(1, H, 1, D)
    elif mask.ndim == 4:
        # Assume already broadcasted as [1,H,1,D] or [B,H,T,D]
        mask_b = mask
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape}")

    k_masked = k * mask_b
    v_masked = v * mask_b
    return k_masked, v_masked


# ---------------------------------------------------------------------------
# Example wiring into an attention block (pseudo-code)
# ---------------------------------------------------------------------------

class SimpleMaskedSelfAttention(nn.Module):
    """
    Minimal example of self-attention with a KV feature mask.

    This is NOT an optimized implementation; it's just a reference
    for how to hook BinaryKVFeatureMask into your existing code.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Plug in the learnable KV mask
        cfg = KVMaskConfig(
            n_heads=n_heads,
            head_dim=self.head_dim,
            init_keep_prob=0.5,
            temperature=1.0,
            hard=True,
        )
        self.kv_mask = BinaryKVFeatureMask(cfg)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D_model] -> [B, H, T, D_head]
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.head_dim
        return x.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D_model]
        B, T, D = x.shape

        q = self._shape(self.q_proj(x))  # [B,H,T,Dh]
        k = self._shape(self.k_proj(x))  # [B,H,T,Dh]
        v = self._shape(self.v_proj(x))  # [B,H,T,Dh]

        # === Apply KV feature mask here ===
        k, v, kv_feature_mask = self.kv_mask(k, v)  # kv_feature_mask: [H,Dh]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,T,T]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # e.g., additive -inf mask

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B,H,T,T]
        attn_out = torch.matmul(attn_weights, v)       # [B,H,T,Dh]

        # Merge heads back: [B,T,D_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out_proj(attn_out)
        return y
