"""
RA Value Tracker: compute inbound mass from surgical attention heads.

Computes chunk-level RA value scores using only attention weights already
produced in the forward pass. No second attention call, no extra QKV
projections. The inbound mass for position i is:

    in_mass[l,h,i] = sum_{t > i} A[l,h,t,i]

This is the column sum of the lower-triangular attention matrix,
measuring how much later tokens attended to position i.

Chunk-level aggregation:
    chunk_mass[c] = sum_{i in chunk c} in_mass[i]

Cross-head average across surgical set S:
    RA_value_chunk[c] = mean_{(l,h) in S} chunk_mass[l,h,c]
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SurgicalHead:
    """A single (layer, head) pair in the surgical set."""

    layer: int
    head: int


def load_surgical_heads(path: str) -> List[SurgicalHead]:
    """Load surgical head set from JSON config."""
    with open(path) as f:
        cfg = json.load(f)
    return [SurgicalHead(h["layer"], h["head"]) for h in cfg["heads"]]


class RAValueTracker:
    """Accumulates inbound mass stats for surgical heads only.

    Usage:
        tracker = RAValueTracker(surgical_heads, chunk_size=64)

        # During forward pass, after computing attention weights for layer l:
        tracker.accumulate(layer_idx, attn_weights)  # [B, H, T, T]

        # After full forward pass:
        chunk_values = tracker.get_chunk_values()  # [n_chunks]
        tracker.reset()
    """

    def __init__(
        self,
        surgical_heads: List[SurgicalHead],
        chunk_size: int = 64,
        ema_gamma: float = 0.0,
    ):
        self.surgical_heads = surgical_heads
        self.chunk_size = chunk_size
        self.ema_gamma = ema_gamma

        # Group heads by layer for fast lookup
        self.heads_by_layer: Dict[int, List[int]] = {}
        for sh in surgical_heads:
            self.heads_by_layer.setdefault(sh.layer, []).append(sh.head)

        # Accumulated chunk masses: list of [n_chunks] tensors
        self._chunk_masses: List[torch.Tensor] = []
        self._ema_values: Optional[torch.Tensor] = None
        self._seq_len: int = 0

    def reset(self):
        """Reset accumulated masses for a new forward pass."""
        self._chunk_masses = []
        self._seq_len = 0

    def accumulate(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
    ):
        """Accumulate inbound mass for surgical heads in this layer.

        Args:
            layer_idx: which transformer layer (0-indexed)
            attn_weights: [B, H, T, T] attention probabilities (post-softmax)
        """
        if layer_idx not in self.heads_by_layer:
            return

        heads = self.heads_by_layer[layer_idx]
        B, H, T, _ = attn_weights.shape
        self._seq_len = T
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size

        for h in heads:
            # Column sum: in_mass[i] = sum_{t>i} A[t,i]
            # attn_weights[:, h, :, :] is [B, T, T] with A[t,i]
            # Column sum = sum over dim=-2 (the query dimension)
            a_h = attn_weights[:, h, :, :]  # [B, T_q, T_k]
            in_mass = a_h.sum(dim=-2)  # [B, T_k]

            # Average across batch
            in_mass_avg = in_mass.mean(dim=0)  # [T]

            # Chunk aggregation
            chunk_mass = torch.zeros(n_chunks, device=in_mass_avg.device)
            for c in range(n_chunks):
                start = c * self.chunk_size
                end = min(start + self.chunk_size, T)
                chunk_mass[c] = in_mass_avg[start:end].sum()

            self._chunk_masses.append(chunk_mass)

    def get_chunk_values(self) -> torch.Tensor:
        """Get averaged chunk-level RA value scores.

        Returns:
            [n_chunks] tensor of RA value per chunk, averaged across
            all surgical heads.
        """
        if not self._chunk_masses:
            return torch.zeros(1)

        stacked = torch.stack(self._chunk_masses)  # [n_heads, n_chunks]
        values = stacked.mean(dim=0)  # [n_chunks]

        # Apply EMA if enabled
        if self.ema_gamma > 0 and self._ema_values is not None:
            n = min(len(values), len(self._ema_values))
            values[:n] = (1 - self.ema_gamma) * self._ema_values[
                :n
            ] + self.ema_gamma * values[:n]

        self._ema_values = values.clone()
        return values

    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics for logging."""
        values = self.get_chunk_values()
        if values.numel() <= 1:
            return {}

        # Normalize to probability distribution for entropy
        probs = values / (values.sum() + 1e-10)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        max_entropy = np.log(len(probs))

        return {
            "ra_value/chunk_mean": values.mean().item(),
            "ra_value/chunk_std": values.std().item(),
            "ra_value/chunk_max": values.max().item(),
            "ra_value/chunk_min": values.min().item(),
            "ra_value/entropy": entropy,
            "ra_value/normalized_entropy": entropy / (max_entropy + 1e-10),
            "ra_value/n_chunks": len(values),
            "ra_value/n_surgical_heads": len(self.surgical_heads),
        }


def compute_ra_value_forward(
    model,
    idx: torch.Tensor,
    tracker: RAValueTracker,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Manual forward pass that collects RA inbound mass for surgical heads.

    For surgical layers, computes explicit attention weights (no SDPA fusion).
    For non-surgical layers, uses fused SDPA for speed.

    Args:
        model: GPT2_BPA model
        idx: [B, T] input token ids
        tracker: RAValueTracker to accumulate into
        attn_mask: [T, T] bool, True = mask out. If None, full causal.

    Returns:
        logits: [B, T, vocab_size]
    """
    B, T = idx.shape
    device = idx.device
    cfg = model.config
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    head_dim = n_embd // n_head

    pos = torch.arange(0, T, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    )
    if attn_mask is not None:
        combined_mask = causal_mask | attn_mask
    else:
        combined_mask = causal_mask

    tracker.reset()

    for layer_idx, block in enumerate(model.transformer.h):
        h = block.ln_1(x)
        attn = block.attn

        qkv = attn.c_attn(h)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        # Always compute explicit attention weights (needed for RA value
        # tracking on surgical layers, and for consistent attention masking)
        scale = 1.0 / (head_dim**0.5)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(
            combined_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Collect inbound mass for surgical heads
        if layer_idx in tracker.heads_by_layer:
            tracker.accumulate(layer_idx, attn_weights)

        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.resid_dropout(attn.c_proj(attn_out))

        x = x + attn_out
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits
