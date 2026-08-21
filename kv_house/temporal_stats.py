# SPDX-License-Identifier: GPL-2.0
"""Token-axis structure measurements for KV blocks.

Answers the first KV-House question: does a real KV cache have
exploitable temporal (cross-token) redundancy? Two distinct nulls
are needed, because the obvious one is blind to half the metrics:

- singular values are invariant under row permutation, so a
  within-block token shuffle CANNOT move any SVD-derived metric
  (energy concentration, effective rank) — it only breaks token
  ORDER, which adjacent-cosine and delta-variance see;
- the spectral null is a SCATTERED pseudo-block: B tokens drawn
  from random positions across the whole stream, which breaks
  LOCALITY (which tokens co-occur in a block) and therefore does
  move the temporal covariance and its spectrum.

block_stats() reports real vs shuffled for order metrics and real
vs scattered for spectral metrics. All functions take a block
matrix X of shape [B, d] and return plain floats for CSV/JSON.
"""

from __future__ import annotations

import torch


def adjacent_cosine(x):
    """Mean cosine similarity between consecutive token vectors."""
    a, b = x[:-1], x[1:]
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
    return float((num / den).mean())


def correlation_vs_distance(x, max_lag=None):
    """Mean token-vector cosine at each lag 1..max_lag."""
    b = x.shape[0]
    max_lag = max_lag or b - 1
    out = {}
    for lag in range(1, min(max_lag, b - 1) + 1):
        a, c = x[:-lag], x[lag:]
        num = (a * c).sum(-1)
        den = a.norm(dim=-1) * c.norm(dim=-1) + 1e-12
        out[lag] = float((num / den).mean())
    return out


def temporal_spectrum(x):
    """Singular values of X across the token axis, descending."""
    return torch.linalg.svdvals(x.to(torch.float32))


def effective_rank(s):
    """Entropy-based effective rank (Roy & Vetterli) of a singular
    value vector."""
    p = (s / (s.sum() + 1e-30)).clamp_min(1e-30)
    return float(torch.exp(-(p * p.log()).sum()))


def top_r_energy(s, r):
    """Cumulative energy fraction captured by the top r singular
    values."""
    e = s * s
    return float(e[:r].sum() / (e.sum() + 1e-30))


def delta_variance_ratio(x):
    """Variance of first differences over variance of tokens; below
    1.0 means neighboring tokens are closer than the block's overall
    spread (temporal locality a delta coder can use)."""
    dv = (x[1:] - x[:-1]).float().var()
    v = x.float().var()
    return float(dv / (v + 1e-30))


def quantized_symbol_entropy(x, bits=4):
    """Empirical entropy (bits/symbol) of the quantization indices
    at a given width, one absmax scale per row."""
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int64)
    q = q - q.min()
    counts = torch.bincount(q.flatten())
    p = counts[counts > 0].double()
    p = p / p.sum()
    return float(-(p * p.log2()).sum())


def shuffle_tokens(x, seed):
    """Deterministically permute the token axis: the ORDER null.
    Marginals and the singular spectrum are preserved by
    construction, so only order-sensitive metrics may move."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm]


def scattered_block(stream, block_size, seed):
    """The LOCALITY null: a pseudo-block of B tokens drawn without
    replacement from random positions across the whole [T, d]
    stream, in draw order. Same marginal token distribution, no
    contiguity — this null does move the temporal covariance."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(stream.shape[0], generator=gen)[:block_size]
    return stream[idx]


def _order_metrics(blk):
    return {
        "adjacent_cosine": adjacent_cosine(blk),
        "delta_variance_ratio": delta_variance_ratio(blk),
    }


def _spectral_metrics(blk):
    s = temporal_spectrum(blk)
    r = max(1, blk.shape[0] // 4)
    return {
        "effective_rank": effective_rank(s),
        "top_quarter_energy": top_r_energy(s, r),
        "top1_energy": top_r_energy(s, 1),
        "entropy4b": quantized_symbol_entropy(blk, 4),
    }


def block_stats(x, stream, name="", seed=0):
    """The milestone metric set for one contiguous block x taken
    from stream: real vs the order null (shuffled) for order
    metrics, real vs the locality null (scattered) for spectral
    metrics."""
    out = {}
    shuf = shuffle_tokens(x, seed)
    scat = scattered_block(stream, x.shape[0], seed)
    for k, v in _order_metrics(x).items():
        out[f"real_{k}"] = v
    for k, v in _order_metrics(shuf).items():
        out[f"shuf_{k}"] = v
    for k, v in _spectral_metrics(x).items():
        out[f"real_{k}"] = v
    for k, v in _spectral_metrics(scat).items():
        out[f"scat_{k}"] = v
    if name:
        out["name"] = name
    return out
