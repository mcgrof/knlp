"""
KV cache accounting utilities for BPA v3.

Provides analytic formulas for KV cache memory footprint,
bandwidth (read/write traffic), and FLOPs proxy computation.
These are geometry-only formulas that do not require running
the model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class KVAccountingResult:
    """Result of KV cache accounting for a single forward pass."""

    # Write cost: bytes written to KV cache per new token
    kv_bytes_written_per_token: float
    # Read cost: bytes read from KV cache per query token
    kv_bytes_read_per_token: float
    # Total KV traffic per token (write + read)
    kv_bytes_total_per_token: float
    # Peak KV allocation for the full sequence
    peak_kv_bytes: float
    # Effective tokens attended per query
    effective_kept_tokens: float
    # FLOPs proxy: proportional to attention computation
    flops_proxy: float
    # Metadata
    n_layers: int
    d_model: int
    seq_len: int
    bytes_per_elem: int


def kv_bytes_per_token(
    n_layers: int,
    d_model: int,
    bytes_per_elem: int = 2,
) -> float:
    """KV cache bytes stored per token (across all layers).

    Each token produces one K vector and one V vector per layer.
    K and V each have dimension d_model.

    Args:
        n_layers: number of transformer layers
        d_model: model dimension (n_embd)
        bytes_per_elem: bytes per element (2 for fp16/bf16, 4 for fp32)

    Returns:
        bytes per token stored in KV cache
    """
    return 2 * n_layers * d_model * bytes_per_elem


def kv_bytes_written_per_token(
    n_layers: int,
    d_model: int,
    bytes_per_elem: int = 2,
) -> float:
    """KV cache write traffic per new token.

    Same as kv_bytes_per_token: each new token writes one K,V
    pair per layer.
    """
    return kv_bytes_per_token(n_layers, d_model, bytes_per_elem)


def kv_bytes_read_per_token(
    attended_tokens: float,
    n_layers: int,
    d_model: int,
    bytes_per_elem: int = 2,
) -> float:
    """KV cache read traffic per query token.

    For each query, attention reads K and V for all attended
    positions across all layers.

    Args:
        attended_tokens: average number of tokens attended per query
        n_layers: number of transformer layers
        d_model: model dimension
        bytes_per_elem: bytes per element

    Returns:
        bytes read from KV cache per query token
    """
    return attended_tokens * 2 * n_layers * d_model * bytes_per_elem


def effective_kept_tokens_per_query(
    local_window: int,
    far_tokens_selected: float,
) -> float:
    """Effective tokens attended per query position.

    Args:
        local_window: always-attend local context size
        far_tokens_selected: additional far tokens selected by gate

    Returns:
        total tokens attended
    """
    return local_window + far_tokens_selected


def peak_kv_bytes(
    seq_len: int,
    n_layers: int,
    d_model: int,
    bytes_per_elem: int = 2,
) -> float:
    """Peak KV cache allocation for a full sequence.

    With dense cache layout, the full sequence is stored
    regardless of gating. With sparse cache, only kept tokens
    are stored.

    Args:
        seq_len: sequence length (number of tokens stored)
        n_layers: number of transformer layers
        d_model: model dimension
        bytes_per_elem: bytes per element

    Returns:
        peak KV cache bytes
    """
    return seq_len * kv_bytes_per_token(n_layers, d_model, bytes_per_elem)


def flops_proxy_attention(
    seq_len: int,
    attended_tokens_per_query: float,
    n_layers: int,
    d_model: int,
) -> float:
    """FLOPs proxy for attention computation.

    Attention FLOPs are proportional to:
      sum_over_layers(q_len * attended_len * d_head * n_heads)
    = q_len * attended_len * d_model * n_layers

    For training, q_len = seq_len. For inference, q_len = 1.
    We report per-token cost, so divide by seq_len.

    Args:
        seq_len: sequence length
        attended_tokens_per_query: average attended tokens
        n_layers: number of layers
        d_model: model dimension

    Returns:
        FLOPs proxy (proportional, not exact)
    """
    return attended_tokens_per_query * d_model * n_layers


def compute_kv_accounting(
    seq_len: int,
    n_layers: int,
    d_model: int,
    local_window: int,
    enabled_rate: float = 1.0,
    far_budget: Optional[int] = None,
    bytes_per_elem: int = 2,
) -> KVAccountingResult:
    """Compute full KV accounting for a BPA configuration.

    Args:
        seq_len: sequence length
        n_layers: number of transformer layers
        d_model: model dimension
        local_window: local attention window
        enabled_rate: fraction of positions where far-context is enabled
        far_budget: max far tokens when enabled (None = seq_len - local_window)
        bytes_per_elem: bytes per element (2 for fp16/bf16)

    Returns:
        KVAccountingResult with all metrics
    """
    if far_budget is None:
        far_budget = max(0, seq_len - local_window)

    # Average far tokens per query
    avg_far = enabled_rate * far_budget
    kept = effective_kept_tokens_per_query(local_window, avg_far)
    # Cap at seq_len
    kept = min(kept, seq_len)

    write_bytes = kv_bytes_written_per_token(n_layers, d_model, bytes_per_elem)
    read_bytes = kv_bytes_read_per_token(kept, n_layers, d_model, bytes_per_elem)
    total_bytes = write_bytes + read_bytes
    peak = peak_kv_bytes(seq_len, n_layers, d_model, bytes_per_elem)
    flops = flops_proxy_attention(seq_len, kept, n_layers, d_model)

    return KVAccountingResult(
        kv_bytes_written_per_token=write_bytes,
        kv_bytes_read_per_token=read_bytes,
        kv_bytes_total_per_token=total_bytes,
        peak_kv_bytes=peak,
        effective_kept_tokens=kept,
        flops_proxy=flops,
        n_layers=n_layers,
        d_model=d_model,
        seq_len=seq_len,
        bytes_per_elem=bytes_per_elem,
    )


def sanity_check_gpt2():
    """Sanity check KV accounting against known GPT-2 124M shapes."""
    n_layers = 12
    d_model = 768
    bytes_per_elem = 2  # bf16

    write = kv_bytes_written_per_token(n_layers, d_model, bytes_per_elem)
    print(f"GPT-2 124M (bf16):")
    print(f"  KV write per token: {write:,.0f} bytes ({write/1024:.1f} KB)")

    for seq_len in [512, 1024, 2048]:
        peak = peak_kv_bytes(seq_len, n_layers, d_model, bytes_per_elem)
        print(f"  Peak KV at L={seq_len}: {peak:,.0f} bytes ({peak/1024/1024:.2f} MB)")

    # Dense baseline: attend to all tokens
    dense_512 = compute_kv_accounting(512, n_layers, d_model, 512)
    print(f"\n  Dense L=512:")
    print(f"    Read/token:  {dense_512.kv_bytes_read_per_token:,.0f} bytes")
    print(f"    Total/token: {dense_512.kv_bytes_total_per_token:,.0f} bytes")
    print(f"    FLOPs proxy: {dense_512.flops_proxy:,.0f}")

    # BPA local_window=256, 50% enabled, L=512
    bpa_512 = compute_kv_accounting(512, n_layers, d_model, 256, enabled_rate=0.5)
    print(f"\n  BPA L=512 (local=256, rate=0.5):")
    print(f"    Kept tokens: {bpa_512.effective_kept_tokens:.0f}")
    print(f"    Read/token:  {bpa_512.kv_bytes_read_per_token:,.0f} bytes")
    print(
        f"    Read savings vs dense: {1 - bpa_512.kv_bytes_read_per_token / dense_512.kv_bytes_read_per_token:.1%}"
    )
    print(f"    FLOPs savings: {1 - bpa_512.flops_proxy / dense_512.flops_proxy:.1%}")

    # BPA local_window=256, 50% enabled, L=1024
    dense_1024 = compute_kv_accounting(1024, n_layers, d_model, 1024)
    bpa_1024 = compute_kv_accounting(1024, n_layers, d_model, 256, enabled_rate=0.5)
    print(f"\n  BPA L=1024 (local=256, rate=0.5):")
    print(f"    Kept tokens: {bpa_1024.effective_kept_tokens:.0f}")
    print(f"    Read/token:  {bpa_1024.kv_bytes_read_per_token:,.0f} bytes")
    print(
        f"    Read savings vs dense: {1 - bpa_1024.kv_bytes_read_per_token / dense_1024.kv_bytes_read_per_token:.1%}"
    )
    print(f"    FLOPs savings: {1 - bpa_1024.flops_proxy / dense_1024.flops_proxy:.1%}")


if __name__ == "__main__":
    sanity_check_gpt2()
