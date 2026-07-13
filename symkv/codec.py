# SPDX-License-Identifier: GPL-2.0
"""SymKV encode/decode and honest byte accounting.

encode: Z = B^T X   (H x D -> m x D)      the stored payload
decode: X_hat = B Z  (m x D -> H x D)      the reconstruction actually attended to

Byte accounting is deliberately conservative and never credits a saving to a
materialized reference path. The stored bytes are the per-token payload (m*D at
the payload dtype) plus the amortized basis (H*m at the basis dtype, shared across
all tokens of a layer, so charged per token as H*m / n_tokens) plus a small
constant of metadata. The baseline is the uncompressed cache H*D at the same
payload dtype. compression_ratio = baseline / stored. A ratio > 1 is only claimed
from this stored-side model, not from any dense X_hat we compute for measuring
error.
"""

from __future__ import annotations

import torch


def encode(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """X: (..., H, D), B: (H, m) -> Z: (..., m, D)."""
    return torch.matmul(B.transpose(-1, -2).to(X.dtype), X)


def decode(Z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Z: (..., m, D), B: (H, m) -> X_hat: (..., H, D)."""
    return torch.matmul(B.to(Z.dtype), Z)


def reconstruct(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return decode(encode(X, B), B)


def recon_mse(X: torch.Tensor, B: torch.Tensor) -> float:
    Xh = reconstruct(X.to(torch.float64), B.to(torch.float64))
    return float(torch.mean((X.to(torch.float64) - Xh) ** 2))


def byte_accounting(n_heads: int, head_dim: int, n_modes: int,
                    n_tokens: int, payload_bytes: int = 2, basis_bytes: int = 2,
                    meta_bytes: int = 16) -> dict:
    """Per-token stored bytes and compression ratio vs the uncompressed H*D cache.
    The basis is shared across a layer's tokens, so its cost is amortized over
    n_tokens; meta_bytes is a small fixed per-token overhead (scales, flags)."""
    H, D, m = n_heads, head_dim, n_modes
    payload = m * D * payload_bytes
    basis_amort = (H * m * basis_bytes) / max(1, n_tokens)
    stored = payload + basis_amort + meta_bytes
    baseline = H * D * payload_bytes
    return {
        "payload_bytes": payload,
        "basis_amortized_bytes": basis_amort,
        "meta_bytes": meta_bytes,
        "stored_bytes_per_token": stored,
        "baseline_bytes_per_token": baseline,
        "compression_ratio": baseline / stored,
        "modes": m, "n_heads": H, "head_dim": D,
    }
