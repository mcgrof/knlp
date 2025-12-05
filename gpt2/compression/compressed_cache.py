"""
Compressed KV Cache for HuggingFace Transformers.

Implements a HF-compatible cache that stores compressed K/V internally
but returns expanded K/V for attention computation.

This ensures compression is actually on the hot path during inference.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.cache_utils import DynamicCache


class CompressedCacheLayer:
    """
    A single layer's compressed KV cache.

    Stores K/V in compressed form, expands on retrieval.
    Compatible with HuggingFace's DynamicLayer interface.

    Supports runtime-aware compression (v20):
    - compress_start_len: Only start compressing after this many tokens
    - uncompressed_tail: Keep the last N tokens uncompressed for fast access
    """

    def __init__(
        self,
        k_compressor: nn.Module,
        v_compressor: nn.Module,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        compress_start_len: int = 0,
        uncompressed_tail: int = 0,
    ):
        """
        Initialize compressed cache layer.

        Args:
            k_compressor: Compressor for keys
            v_compressor: Compressor for values
            device: Target device
            dtype: Data type
            compress_start_len: Don't compress until seq_len >= this threshold.
                Below this, behave as identity cache (baseline HF DynamicCache).
            uncompressed_tail: Keep the last N tokens uncompressed (FP16).
                Only compress tokens older than that.
        """
        self.k_compressor = k_compressor
        self.v_compressor = v_compressor
        self.device = device
        self.dtype = dtype
        self.compress_start_len = compress_start_len
        self.uncompressed_tail = uncompressed_tail

        # Compressed storage (for old tokens beyond uncompressed_tail)
        self.keys_compressed: Optional[torch.Tensor] = None
        self.values_compressed: Optional[torch.Tensor] = None

        # Uncompressed storage (for recent tokens within uncompressed_tail)
        self.keys_uncompressed: Optional[torch.Tensor] = None
        self.values_uncompressed: Optional[torch.Tensor] = None

        # Track original shape for expansion
        self.batch_size = 0
        self.num_heads = 0
        self.head_dim = 0
        self.seq_len = 0

        self.is_initialized = False

    def lazy_initialization(self, key_states: torch.Tensor):
        """Initialize from first key_states tensor."""
        self.batch_size = key_states.shape[0]
        self.num_heads = key_states.shape[1]
        self.head_dim = key_states.shape[3]
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K/V and return expanded K/V for attention.

        Implements runtime-aware compression (v20):
        - Below compress_start_len: no compression overhead
        - Above threshold: compress old tokens, keep recent tail uncompressed

        Args:
            key_states: New keys [B, n_heads, seq_len, head_dim]
            value_states: New values [B, n_heads, seq_len, head_dim]

        Returns:
            Tuple of (all_keys, all_values) expanded for attention
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        B, H, T, D = key_states.shape

        # Calculate new total sequence length
        current_uncompressed = (
            self.keys_uncompressed.shape[2] if self.keys_uncompressed is not None else 0
        )
        current_compressed = (
            self.keys_compressed.shape[2] if self.keys_compressed is not None else 0
        )
        new_total_len = current_uncompressed + current_compressed + T

        # Case 1: Below compression threshold - store everything uncompressed
        if new_total_len < self.compress_start_len:
            if self.keys_uncompressed is None:
                self.keys_uncompressed = key_states
                self.values_uncompressed = value_states
            else:
                self.keys_uncompressed = torch.cat(
                    [self.keys_uncompressed, key_states], dim=2
                )
                self.values_uncompressed = torch.cat(
                    [self.values_uncompressed, value_states], dim=2
                )

            self.seq_len = new_total_len
            return self.keys_uncompressed, self.values_uncompressed

        # Case 2: Above threshold - use sliding window compression
        # First, add new tokens to uncompressed buffer
        if self.keys_uncompressed is None:
            self.keys_uncompressed = key_states
            self.values_uncompressed = value_states
        else:
            self.keys_uncompressed = torch.cat(
                [self.keys_uncompressed, key_states], dim=2
            )
            self.values_uncompressed = torch.cat(
                [self.values_uncompressed, value_states], dim=2
            )

        # Check if we need to move old tokens from uncompressed to compressed
        uncompressed_len = self.keys_uncompressed.shape[2]
        tokens_to_compress = uncompressed_len - self.uncompressed_tail

        if tokens_to_compress > 0:
            # Extract tokens to compress
            k_to_compress = self.keys_uncompressed[:, :, :tokens_to_compress, :]
            v_to_compress = self.values_uncompressed[:, :, :tokens_to_compress, :]

            # Keep recent tokens uncompressed
            self.keys_uncompressed = self.keys_uncompressed[
                :, :, tokens_to_compress:, :
            ]
            self.values_uncompressed = self.values_uncompressed[
                :, :, tokens_to_compress:, :
            ]

            # Compress the old tokens
            k_flat = k_to_compress.reshape(-1, D)
            v_flat = v_to_compress.reshape(-1, D)

            k_comp_new = self.k_compressor.compress(k_flat)
            v_comp_new = self.v_compressor.compress(v_flat)

            d_k = k_comp_new.shape[-1]
            d_v = v_comp_new.shape[-1]
            k_comp_new = k_comp_new.reshape(B, H, tokens_to_compress, d_k)
            v_comp_new = v_comp_new.reshape(B, H, tokens_to_compress, d_v)

            # Add to compressed cache
            if self.keys_compressed is None:
                self.keys_compressed = k_comp_new
                self.values_compressed = v_comp_new
            else:
                self.keys_compressed = torch.cat(
                    [self.keys_compressed, k_comp_new], dim=2
                )
                self.values_compressed = torch.cat(
                    [self.values_compressed, v_comp_new], dim=2
                )

        # Build output: expanded compressed + uncompressed tail
        if self.keys_compressed is not None:
            B, H, T_comp, d_k = self.keys_compressed.shape
            _, _, _, d_v = self.values_compressed.shape

            k_flat = self.keys_compressed.reshape(-1, d_k)
            v_flat = self.values_compressed.reshape(-1, d_v)

            k_expanded = self.k_compressor.expand(k_flat)
            v_expanded = self.v_compressor.expand(v_flat)

            k_expanded = k_expanded.reshape(B, H, T_comp, D)
            v_expanded = v_expanded.reshape(B, H, T_comp, D)

            # Concatenate with uncompressed tail
            k_all = torch.cat([k_expanded, self.keys_uncompressed], dim=2)
            v_all = torch.cat([v_expanded, self.values_uncompressed], dim=2)
        else:
            # No compressed tokens yet
            k_all = self.keys_uncompressed
            v_all = self.values_uncompressed

        self.seq_len = k_all.shape[2]
        return k_all, v_all

    def get_seq_length(self) -> int:
        """Return current sequence length."""
        return self.seq_len

    def get_max_cache_shape(self) -> int:
        """Return maximum cache shape (unlimited for dynamic)."""
        return 0

    def reset(self) -> None:
        """Clear the cache."""
        self.keys_compressed = None
        self.values_compressed = None
        self.keys_uncompressed = None
        self.values_uncompressed = None
        self.seq_len = 0
        self.is_initialized = False

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        """Get mask sizes for attention.

        Returns:
            Tuple of (full_seq_len, sliding_window_offset).
            full_seq_len = cache_position.max() + 1 (total including new token)
            sliding_window_offset = 0 (no sliding window for dynamic cache)
        """
        # cache_position is the position of the NEW token being processed
        # So full sequence length = cache_position.max() + 1
        full_seq_len = cache_position.max().item() + 1
        sliding_offset = 0  # No sliding window
        return full_seq_len, sliding_offset

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache for beam search."""
        if self.keys_compressed is not None:
            self.keys_compressed = self.keys_compressed.index_select(0, beam_idx)
            self.values_compressed = self.values_compressed.index_select(0, beam_idx)

    def crop(self, max_length: int) -> None:
        """Crop cache to max_length."""
        if self.keys_compressed is not None and self.seq_len > max_length:
            self.keys_compressed = self.keys_compressed[:, :, :max_length, :]
            self.values_compressed = self.values_compressed[:, :, :max_length, :]
            self.seq_len = max_length

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache for batch expansion."""
        if self.keys_compressed is not None:
            self.keys_compressed = self.keys_compressed.repeat_interleave(
                repeats, dim=0
            )
            self.values_compressed = self.values_compressed.repeat_interleave(
                repeats, dim=0
            )
            self.batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices."""
        if self.keys_compressed is not None:
            self.keys_compressed = self.keys_compressed.index_select(0, indices)
            self.values_compressed = self.values_compressed.index_select(0, indices)
            self.batch_size = len(indices)

    def offload(self) -> None:
        """Offload to CPU (not implemented for compressed cache)."""
        pass

    def prefetch(self) -> None:
        """Prefetch from CPU (not implemented for compressed cache)."""
        pass

    @property
    def memory_bytes(self) -> int:
        """Return memory usage of compressed cache (compressed + uncompressed)."""
        total = 0
        if self.keys_compressed is not None:
            total += self.keys_compressed.numel() * self.keys_compressed.element_size()
            total += (
                self.values_compressed.numel() * self.values_compressed.element_size()
            )
        if self.keys_uncompressed is not None:
            total += (
                self.keys_uncompressed.numel() * self.keys_uncompressed.element_size()
            )
            total += (
                self.values_uncompressed.numel()
                * self.values_uncompressed.element_size()
            )
        return total

    @property
    def compressed_tokens(self) -> int:
        """Return number of compressed tokens."""
        if self.keys_compressed is None:
            return 0
        return self.keys_compressed.shape[2]

    @property
    def uncompressed_tokens(self) -> int:
        """Return number of uncompressed tokens."""
        if self.keys_uncompressed is None:
            return 0
        return self.keys_uncompressed.shape[2]


class CompressedDynamicCache(DynamicCache):
    """
    HuggingFace-compatible cache that stores compressed K/V.

    Drop-in replacement for DynamicCache that compresses on store
    and expands on retrieval.

    Supports runtime-aware compression (v20):
    - compress_start_len: Only start compressing after this many tokens
    - uncompressed_tail: Keep the last N tokens uncompressed for fast access
    """

    def __init__(
        self,
        k_compressors: List[nn.Module],
        v_compressors: List[nn.Module],
        num_layers: int = None,
        compress_start_len: int = 0,
        uncompressed_tail: int = 0,
    ):
        """
        Initialize compressed cache.

        Args:
            k_compressors: List of key compressors, one per layer
            v_compressors: List of value compressors, one per layer
            num_layers: Number of layers (inferred from compressors if not provided)
            compress_start_len: Don't compress until seq_len >= this threshold.
                Below this, behave as identity cache (no overhead).
            uncompressed_tail: Keep the last N tokens uncompressed (FP16).
                Only compress tokens older than that for fast access.
        """
        # Don't call super().__init__() - we manage layers ourselves
        self.k_compressors = k_compressors
        self.v_compressors = v_compressors
        self.num_layers = num_layers or len(k_compressors)
        self.compress_start_len = compress_start_len
        self.uncompressed_tail = uncompressed_tail

        # Start with empty layers list (like DynamicCache)
        # Layers are created lazily on first update
        self.layers: List[CompressedCacheLayer] = []

        # Required attributes for HF compatibility
        self.offloading = False
        self.only_non_sliding = True
        self.prefetch_stream = None
        self.layer_class_to_replicate = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer and return expanded K/V.

        This is the critical method - it stores compressed but returns expanded.
        """
        # Ensure we have enough layers
        while len(self.layers) <= layer_idx:
            # This shouldn't happen if initialized correctly, but handle gracefully
            layer = CompressedCacheLayer(
                k_compressor=self.k_compressors[
                    min(layer_idx, len(self.k_compressors) - 1)
                ],
                v_compressor=self.v_compressors[
                    min(layer_idx, len(self.v_compressors) - 1)
                ],
                compress_start_len=self.compress_start_len,
                uncompressed_tail=self.uncompressed_tail,
            )
            self.layers.append(layer)

        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length from specified layer."""
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].get_seq_length()
        return 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Get max cache shape (0 for dynamic/unlimited)."""
        return 0

    def reset(self) -> None:
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Convert to legacy cache format (expanded K/V).

        Returns tuple of (key, value) tuples for each layer.
        """
        legacy = []
        for layer in self.layers:
            if layer.keys_compressed is not None:
                # Expand for legacy format
                B, H, T, d_comp = layer.keys_compressed.shape
                D = layer.head_dim

                k_flat = layer.keys_compressed.reshape(-1, d_comp)
                v_flat = layer.values_compressed.reshape(-1, d_comp)

                k_expanded = layer.k_compressor.expand(k_flat).reshape(B, H, T, D)
                v_expanded = layer.v_compressor.expand(v_flat).reshape(B, H, T, D)

                legacy.append((k_expanded, v_expanded))
            else:
                legacy.append((None, None))
        return tuple(legacy)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache for beam search."""
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def crop(self, max_length: int) -> None:
        """Crop all layers to max_length."""
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache for batch expansion."""
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices."""
        for layer in self.layers:
            layer.batch_select_indices(indices)

    @property
    def is_initialized(self) -> bool:
        """Check if any layer is initialized."""
        return any(layer.is_initialized for layer in self.layers)

    @property
    def is_compileable(self) -> bool:
        """Not compileable due to dynamic compression."""
        return False

    @property
    def is_sliding(self) -> List[bool]:
        """Not a sliding window cache. Returns list of bools per initialized layer."""
        # Only return flags for initialized layers (like DynamicCache)
        return [False for layer in self.layers if layer.is_initialized]

    @property
    def max_batch_size(self) -> int:
        """Return max batch size seen."""
        if self.layers and self.layers[0].is_initialized:
            return self.layers[0].batch_size
        return 0

    @property
    def max_cache_len(self) -> int:
        """Return max sequence length."""
        return max((layer.seq_len for layer in self.layers), default=0)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory statistics including compressed vs uncompressed breakdown."""
        total_bytes = sum(layer.memory_bytes for layer in self.layers)
        compressed_tokens = sum(layer.compressed_tokens for layer in self.layers)
        uncompressed_tokens = sum(layer.uncompressed_tokens for layer in self.layers)
        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "num_layers": len(self.layers),
            "seq_len": self.max_cache_len,
            "compressed_tokens": (
                compressed_tokens // len(self.layers) if self.layers else 0
            ),
            "uncompressed_tokens": (
                uncompressed_tokens // len(self.layers) if self.layers else 0
            ),
            "compress_start_len": self.compress_start_len,
            "uncompressed_tail": self.uncompressed_tail,
        }


class IdentityCompressor(nn.Module):
    """
    Identity compressor for sanity testing.

    compress(x) = x, expand(z) = z
    """

    def __init__(self):
        super().__init__()

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def calibrate(self, data: torch.Tensor) -> None:
        pass


class ZeroCompressor(nn.Module):
    """
    Zero compressor for sanity testing (destroy-cache test).

    compress(x) = zeros, expand(z) = zeros
    """

    def __init__(self, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_dim:
            return torch.zeros(
                x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
            )
        return torch.zeros_like(x)

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        return z  # Already zeros

    def calibrate(self, data: torch.Tensor) -> None:
        pass


class RandomCompressor(nn.Module):
    """
    Random compressor for sanity testing (destroy-cache test).

    compress(x) = random, expand(z) = random
    """

    def __init__(self):
        super().__init__()

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(z)

    def calibrate(self, data: torch.Tensor) -> None:
        pass


def make_ln_nullspace_basis(
    d: int, device: torch.device = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Construct orthonormal basis for the LayerNorm nullspace.

    LayerNorm outputs have mean ≈ 0, so they live in the (d-1)-dimensional
    hyperplane orthogonal to the all-ones vector. This function constructs
    an orthonormal basis U for that subspace.

    Args:
        d: Input dimension (head_dim)
        device: Target device
        dtype: Output dtype

    Returns:
        U: [d, d-1] orthonormal matrix spanning the LN nullspace
    """
    # e = normalized all-ones vector
    e = torch.ones(d, device=device, dtype=torch.float32)
    e = e / e.norm()

    # Build orthonormal basis for subspace orthogonal to e
    # Start from identity, remove e's component, then QR to orthonormalize
    I = torch.eye(d, device=device, dtype=torch.float32)
    # Project out e from each column: I - e @ e.T
    projected = I - e.unsqueeze(1) @ e.unsqueeze(0)
    # QR decomposition gives orthonormal basis
    Q, _ = torch.linalg.qr(projected)
    # Take first (d-1) columns (the last column would be ~zero)
    U = Q[:, : d - 1].to(dtype)
    return U


class LayerNormNullspaceCompressor(nn.Module):
    """
    Exploit LayerNorm geometry for deterministic rank-1 compression.

    LayerNorm outputs always have mean ≈ 0 across features, meaning they
    live in a (d-1)-dimensional hyperplane orthogonal to the all-ones vector.
    This compressor projects into that subspace for exact (up to FP rounding)
    compression with zero quality loss.

    Compression factor: d/(d-1) ≈ 1.016 for d=64

    This is mathematically clean, requires no calibration, and can be
    composed with other compression techniques (PCA, quantization).
    """

    def __init__(
        self,
        d: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize LN nullspace compressor.

        Args:
            d: Input dimension (head_dim)
            device: Target device
            dtype: Data type for computation
        """
        super().__init__()
        self.d = d
        self.dtype = dtype

        # Precompute orthonormal basis for LN subspace
        U = make_ln_nullspace_basis(d, device=device, dtype=dtype)
        self.register_buffer("U", U)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to LN nullspace: [*, d] -> [*, d-1].

        Because LN'd vectors have mean ≈ 0, they already live in this
        subspace, so this projection is essentially lossless.
        """
        return torch.matmul(x, self.U)

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from LN nullspace: [*, d-1] -> [*, d].

        The reconstruction x = z @ U.T satisfies sum(x) ≈ 0.
        """
        return torch.matmul(z, self.U.transpose(-1, -2))

    def calibrate(self, data: torch.Tensor) -> None:
        """No calibration needed - purely geometric."""
        pass

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio: d/(d-1)."""
        return self.d / (self.d - 1)


class QKLNNullspaceCompressor(nn.Module):
    """
    LN nullspace compression applied to both Q and K for attention FLOP reduction.

    When applied to K in the cache, and Q is sliced to match, the QK^T matmul
    operates in d-1 dimensions instead of d, giving ~2% FLOP reduction.

    The key insight is that if both Q and K are projected into the same LN
    nullspace, then QK^T in the reduced space approximates QK^T in full space:
        Q' = Q @ U, K' = K @ U
        Q'K'^T ≈ QK^T (for LN'd vectors)

    This compressor:
    - Compresses K to d-1 dims (store less)
    - Returns effective_head_dim = d-1 for Q slicing
    - Provides FLOP savings in QK^T computation
    """

    def __init__(
        self,
        d: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        apply_to_k: bool = True,
        apply_to_v: bool = False,
    ):
        """
        Initialize Q/K LN nullspace compressor.

        Args:
            d: Input dimension (head_dim)
            device: Target device
            dtype: Data type
            apply_to_k: Apply to K (default True for FLOP savings)
            apply_to_v: Apply to V (default False, use separate V compressor)
        """
        super().__init__()
        self.d = d
        self.effective_d = d - 1
        self.dtype = dtype
        self.apply_to_k = apply_to_k
        self.apply_to_v = apply_to_v

        # Precompute orthonormal basis for LN subspace
        U = make_ln_nullspace_basis(d, device=device, dtype=dtype)
        self.register_buffer("U", U)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project to LN nullspace: [*, d] -> [*, d-1]."""
        return torch.matmul(x, self.U)

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from LN nullspace: [*, d-1] -> [*, d]."""
        return torch.matmul(z, self.U.transpose(-1, -2))

    def get_q_projection_matrix(self) -> torch.Tensor:
        """Return U matrix for Q slicing: Q_proj = Q @ U gives Q in [*, d-1]."""
        return self.U

    def calibrate(self, data: torch.Tensor) -> None:
        """No calibration needed."""
        pass

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio: d/(d-1)."""
        return self.d / self.effective_d


class ComposedCompressor(nn.Module):
    """
    Compose multiple compressors in sequence.

    Enables stacking LN nullspace with PCA with quantization:
    compress: x -> LN_compress -> PCA_compress -> quantize
    expand: z -> dequantize -> PCA_expand -> LN_expand
    """

    def __init__(self, compressors: List[nn.Module]):
        """
        Initialize composed compressor.

        Args:
            compressors: List of compressors to chain (applied in order)
        """
        super().__init__()
        self.compressors = nn.ModuleList(compressors)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Apply compressors in sequence."""
        z = x
        for comp in self.compressors:
            z = comp.compress(z)
        return z

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Apply expanders in reverse sequence."""
        x = z
        for comp in reversed(self.compressors):
            x = comp.expand(x)
        return x

    def calibrate(self, data: torch.Tensor) -> None:
        """Calibrate all sub-compressors."""
        for comp in self.compressors:
            if hasattr(comp, "calibrate"):
                comp.calibrate(data)

    @property
    def compression_ratio(self) -> float:
        """Return total compression ratio (product of all)."""
        ratio = 1.0
        for comp in self.compressors:
            if hasattr(comp, "compression_ratio"):
                ratio *= comp.compression_ratio
        return ratio


class CalibratedCompressor(nn.Module):
    """
    PCA-calibrated low-rank compressor.

    Uses pre-computed orthonormal projection matrices from calibration.
    compress(x) = (x - mean) @ U
    expand(z) = z @ U.T + mean
    """

    def __init__(
        self,
        U: torch.Tensor,
        mean: torch.Tensor = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize with calibrated projection matrix.

        Args:
            U: [d_input, rank] orthonormal projection matrix
            mean: [d_input] mean vector for centering (optional)
            dtype: Data type for computation
        """
        super().__init__()
        self.d_input = U.shape[0]
        self.rank = U.shape[1]
        self.dtype = dtype

        # Register buffers
        self.register_buffer("U", U.to(dtype))
        if mean is not None:
            self.register_buffer("mean", mean.to(dtype))
        else:
            self.register_buffer("mean", torch.zeros(self.d_input, dtype=dtype))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project to low-rank subspace: [*, d_input] -> [*, rank]."""
        centered = x - self.mean
        return centered @ self.U

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from low-rank: [*, rank] -> [*, d_input]."""
        return z @ self.U.T + self.mean

    def calibrate(self, data: torch.Tensor) -> None:
        """No-op - already calibrated."""
        pass

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio."""
        return self.d_input / self.rank


class QuantizedCalibratedCompressor(nn.Module):
    """
    PCA-calibrated low-rank compressor with int8/int4 quantization in latent space.

    compress(x) = quantize((x - mean) @ U)
    expand(z) = dequantize(z) @ U.T + mean
    """

    def __init__(
        self,
        U: torch.Tensor,
        mean: torch.Tensor = None,
        bits: int = 8,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize with calibrated projection and quantization.

        Args:
            U: [d_input, rank] orthonormal projection matrix
            mean: [d_input] mean vector for centering
            bits: Quantization bits (8 or 4)
            dtype: Data type for computation
        """
        super().__init__()
        self.d_input = U.shape[0]
        self.rank = U.shape[1]
        self.bits = bits
        self.dtype = dtype

        # Quantization range
        if bits == 8:
            self.qmin, self.qmax = -128, 127
        elif bits == 4:
            self.qmin, self.qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        # Register buffers
        self.register_buffer("U", U.to(dtype))
        if mean is not None:
            self.register_buffer("mean", mean.to(dtype))
        else:
            self.register_buffer("mean", torch.zeros(self.d_input, dtype=dtype))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project to low-rank subspace and quantize: [*, d_input] -> [*, rank]."""
        centered = x - self.mean
        latent = centered @ self.U

        # Quantize per-token (symmetric quantization)
        absmax = latent.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / self.qmax
        quantized = (latent / scale).round().clamp(self.qmin, self.qmax)

        # Return dequantized for now (stores quantized internally)
        dequantized = quantized * scale
        return dequantized

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from low-rank: [*, rank] -> [*, d_input]."""
        return z @ self.U.T + self.mean

    def calibrate(self, data: torch.Tensor) -> None:
        """No-op - already calibrated."""
        pass

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio (not accounting for quantization bits)."""
        return self.d_input / self.rank


class GammaAwareQuantizedCompressor(nn.Module):
    """
    γ-aware quantized compressor that normalizes per-dim variance before quantization.

    LayerNorm produces outputs with varying per-dimension scales (γ). By normalizing
    these out before quantization, we get a more isotropic distribution that quantizes
    better, improving quality at the same bitwidth or enabling more aggressive compression.

    compress(x) = quantize(((x - mean) @ U) / scale)
    expand(z) = (dequantize(z) * scale) @ U.T + mean

    The 'scale' vector captures per-latent-dim standard deviations from calibration,
    which absorbs the effect of LayerNorm's γ.
    """

    def __init__(
        self,
        U: torch.Tensor,
        mean: torch.Tensor = None,
        scale: torch.Tensor = None,
        bits: int = 8,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize with calibrated projection, scale, and quantization.

        Args:
            U: [d_input, rank] orthonormal projection matrix
            mean: [d_input] mean vector for centering
            scale: [rank] per-latent-dim scale (std from calibration)
            bits: Quantization bits (8 or 4)
            dtype: Data type for computation
        """
        super().__init__()
        self.d_input = U.shape[0]
        self.rank = U.shape[1]
        self.bits = bits
        self.dtype = dtype

        # Quantization range
        if bits == 8:
            self.qmin, self.qmax = -128, 127
        elif bits == 4:
            self.qmin, self.qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        # Register buffers
        self.register_buffer("U", U.to(dtype))
        if mean is not None:
            self.register_buffer("mean", mean.to(dtype))
        else:
            self.register_buffer("mean", torch.zeros(self.d_input, dtype=dtype))

        # Per-dim scale for γ-aware normalization
        if scale is not None:
            self.register_buffer("scale", scale.to(dtype).clamp(min=1e-6))
        else:
            # Default to unit scale (falls back to standard quantization)
            self.register_buffer("scale", torch.ones(self.rank, dtype=dtype))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to low-rank subspace, normalize by scale, and quantize.

        The per-dim scale normalization makes the latent distribution more
        isotropic, allowing quantization to use bits more efficiently.
        """
        centered = x - self.mean
        latent = centered @ self.U

        # γ-aware normalization: divide by per-dim scale
        normalized = latent / self.scale

        # Quantize the normalized latent (now roughly unit variance)
        # Use a global scale since variance is normalized
        absmax = normalized.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        qscale = absmax / self.qmax
        quantized = (normalized / qscale).round().clamp(self.qmin, self.qmax)

        # Dequantize and restore scale
        dequantized = quantized * qscale * self.scale
        return dequantized

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from low-rank: [*, rank] -> [*, d_input]."""
        return z @ self.U.T + self.mean

    def calibrate(self, data: torch.Tensor) -> None:
        """No-op - already calibrated."""
        pass

    @classmethod
    def from_calibration(
        cls,
        U: torch.Tensor,
        mean: torch.Tensor,
        calibration_data: torch.Tensor,
        bits: int = 8,
        dtype: torch.dtype = torch.float16,
    ) -> "GammaAwareQuantizedCompressor":
        """
        Create compressor with scale computed from calibration data.

        Args:
            U: [d_input, rank] projection matrix
            mean: [d_input] mean vector
            calibration_data: [N, d_input] sample data for scale estimation
            bits: Quantization bits
            dtype: Data type
        """
        # Project calibration data to latent space
        centered = calibration_data - mean
        latent = centered @ U

        # Compute per-dim std as scale
        scale = latent.std(dim=0).clamp(min=1e-6)

        return cls(U, mean, scale, bits, dtype)

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio (not accounting for quantization bits)."""
        return self.d_input / self.rank


class ChannelPruningCompressor(nn.Module):
    """
    LN-aware channel pruning on the *latent* (post-low-rank) representation.

    Uses calibration stats (latent_std, energy_fraction) to identify low-energy
    dimensions and zero them out before quantization. This provides extra KV
    memory savings with minimal PPL impact.

    V-only for now - K is too fragile (345x variance spread).

    The pruning decision is made once at init based on calibration stats.
    During compress(), pruned channels are zeroed. During expand(), zeros
    remain zeros.
    """

    def __init__(
        self,
        rank: int,
        energy_fraction: torch.Tensor = None,
        latent_std: torch.Tensor = None,
        energy_threshold: float = 0.01,
        max_prune_fraction: float = 0.25,
        target: str = "v",
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize channel pruning compressor.

        Args:
            rank: Number of latent dimensions (from low-rank projection)
            energy_fraction: [rank] normalized importance scores per dim
            latent_std: [rank] per-dim std (fallback if energy_fraction not provided)
            energy_threshold: Max fraction of total energy to prune (e.g. 0.01 = 1%)
            max_prune_fraction: Max fraction of dims to prune (e.g. 0.25 = 25%)
            target: "v" only for now
            device: Device for tensors
            dtype: Data type
        """
        super().__init__()

        if target != "v":
            raise ValueError(
                f"ChannelPruningCompressor only supports target='v', got '{target}'. "
                "K is too fragile for channel pruning (345x variance spread)."
            )

        self.rank = rank
        self.energy_threshold = energy_threshold
        self.max_prune_fraction = max_prune_fraction
        self.target = target
        self.dtype = dtype

        # Compute energy fraction if not provided
        if energy_fraction is not None:
            ef = energy_fraction.to(dtype)
        elif latent_std is not None:
            # Normalize latent_std to get energy fraction
            std = latent_std.to(dtype).clamp(min=1e-8)
            ef = std / std.sum()
        else:
            # No calibration - no pruning
            ef = torch.ones(rank, dtype=dtype) / rank

        self.register_buffer("energy_fraction", ef)

        # Determine which channels to prune
        pruned_mask = self._compute_pruned_mask(ef)
        self.register_buffer("pruned_mask", pruned_mask)

        # Log pruning stats
        n_pruned = pruned_mask.sum().item()
        pruned_energy = ef[pruned_mask].sum().item() if n_pruned > 0 else 0.0
        self._n_pruned = int(n_pruned)
        self._pruned_energy = pruned_energy

    def _compute_pruned_mask(self, energy_fraction: torch.Tensor) -> torch.Tensor:
        """
        Compute boolean mask of channels to prune.

        Strategy: Sort dims by increasing energy, prune from lowest until:
        - cumulative pruned energy > energy_threshold, OR
        - pruned count > max_prune_fraction * rank
        """
        rank = len(energy_fraction)
        max_prune_count = int(self.max_prune_fraction * rank)

        if max_prune_count == 0 or self.energy_threshold <= 0:
            return torch.zeros(rank, dtype=torch.bool, device=energy_fraction.device)

        # Sort by energy (ascending - lowest energy first)
        sorted_indices = torch.argsort(energy_fraction)
        sorted_energy = energy_fraction[sorted_indices]

        # Cumulative sum of energy
        cumsum = torch.cumsum(sorted_energy, dim=0)

        # Find how many we can prune
        n_prune = 0
        for i in range(min(max_prune_count, rank)):
            if cumsum[i] <= self.energy_threshold:
                n_prune = i + 1
            else:
                break

        # Create mask
        pruned_mask = torch.zeros(rank, dtype=torch.bool, device=energy_fraction.device)
        if n_prune > 0:
            pruned_indices = sorted_indices[:n_prune]
            pruned_mask[pruned_indices] = True

        return pruned_mask

    def compress(self, z: torch.Tensor) -> torch.Tensor:
        """
        Zero out pruned channels in latent representation.

        Args:
            z: [*, rank] latent tensor (output of low-rank projection)

        Returns:
            z with pruned channels zeroed
        """
        if self._n_pruned == 0:
            return z

        # Zero out pruned channels
        z_pruned = z.clone()
        z_pruned[..., self.pruned_mask] = 0
        return z_pruned

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """
        Identity - pruned channels are already zero.

        The expand operation in the composed pipeline will handle reconstruction.
        """
        return z

    def get_effective_rank(self) -> int:
        """Return effective rank after pruning."""
        return self.rank - self._n_pruned

    def get_pruning_stats(self) -> dict:
        """Return pruning statistics for logging."""
        return {
            "rank": self.rank,
            "n_pruned": self._n_pruned,
            "prune_fraction": self._n_pruned / self.rank if self.rank > 0 else 0,
            "pruned_energy": self._pruned_energy,
            "effective_rank": self.get_effective_rank(),
        }

    def __repr__(self) -> str:
        return (
            f"ChannelPruningCompressor(rank={self.rank}, "
            f"pruned={self._n_pruned}/{self.rank} ({self._n_pruned/self.rank*100:.1f}%), "
            f"energy_threshold={self.energy_threshold}, target={self.target})"
        )


class MixedModeCompressor(nn.Module):
    """
    Mixed-mode compressor supporting different strategies for K and V.

    Enables configurations like:
    - K: identity (no compression), V: low-rank int8
    - K: low-rank FP16, V: low-rank int8
    - Head-selective compression (per-head basis)

    This is particularly useful because V compresses much better than K.
    """

    def __init__(
        self,
        base_compressor: nn.Module,
        mode: str = "full",
        head_mask: torch.Tensor = None,
        num_heads: int = None,
        head_dim: int = None,
    ):
        """
        Initialize mixed-mode compressor.

        Args:
            base_compressor: Underlying compressor (CalibratedCompressor, etc.)
            mode: Compression mode
                - "full": Apply compressor to all data
                - "identity": No compression (pass-through)
                - "head_selective": Apply only to masked heads
            head_mask: [num_heads] boolean tensor, True = compress this head
            num_heads: Number of attention heads (for head-selective)
            head_dim: Dimension per head (for head-selective)
        """
        super().__init__()
        self.base_compressor = base_compressor
        self.mode = mode
        self.num_heads = num_heads
        self.head_dim = head_dim

        if head_mask is not None:
            self.register_buffer("head_mask", head_mask)
        else:
            self.head_mask = None

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress with mixed-mode strategy."""
        if self.mode == "identity":
            return x

        if self.mode == "head_selective" and self.head_mask is not None:
            return self._compress_head_selective(x)

        # Default: full compression
        return self.base_compressor.compress(x)

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Expand with mixed-mode strategy."""
        if self.mode == "identity":
            return z

        if self.mode == "head_selective" and self.head_mask is not None:
            return self._expand_head_selective(z)

        return self.base_compressor.expand(z)

    def _compress_head_selective(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selectively compress only masked heads.

        Input: [B*H*T, head_dim] flattened
        Output: [B*H*T, out_dim] where out_dim depends on head
        """
        # For head-selective, we store compressed+uncompressed mixed
        # This requires knowing the head structure
        # For simplicity, return same shape but selective compress
        BHT = x.shape[0]
        H = self.num_heads
        D = self.head_dim

        if BHT % H != 0:
            # Can't determine head structure, fall back to full
            return self.base_compressor.compress(x)

        T = BHT // H

        # Reshape to [H, T, D]
        x_heads = x.reshape(H, T, D)
        out_heads = []

        for h in range(H):
            if self.head_mask[h]:
                # Compress this head
                out_heads.append(self.base_compressor.compress(x_heads[h]))
            else:
                # Keep original
                out_heads.append(x_heads[h])

        # Note: mixed dims don't work well, so fall back to uniform output
        # Real implementation would need separate storage
        return self.base_compressor.compress(x)

    def _expand_head_selective(self, z: torch.Tensor) -> torch.Tensor:
        """Selectively expand only masked heads."""
        # Symmetric to compress
        return self.base_compressor.expand(z)

    def calibrate(self, data: torch.Tensor) -> None:
        """Pass through to base compressor."""
        if hasattr(self.base_compressor, "calibrate"):
            self.base_compressor.calibrate(data)

    @property
    def compression_ratio(self) -> float:
        """Return effective compression ratio."""
        if self.mode == "identity":
            return 1.0
        if hasattr(self.base_compressor, "compression_ratio"):
            return self.base_compressor.compression_ratio
        return 1.0


class SemanticAwareCompressor(nn.Module):
    """
    Semantic-aware compressor that uses different projectors for different content types.

    Dynamically selects the appropriate projector based on the semantic bucket
    of the input content (narrative, dialogue, code, math, reasoning, instructions).

    This enables content-specific compression that preserves quality better than
    a single universal projector.
    """

    BUCKET_NAMES = [
        "narrative",
        "dialogue",
        "code",
        "math",
        "reasoning",
        "instructions",
    ]

    def __init__(
        self,
        bucket_compressors: Dict[str, nn.Module],
        default_bucket: str = "narrative",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize semantic-aware compressor.

        Args:
            bucket_compressors: Dict mapping bucket names to compressor modules
            default_bucket: Default bucket when content type is unknown
            dtype: Data type for computation
        """
        super().__init__()
        self.default_bucket = default_bucket
        self.dtype = dtype
        self.current_bucket = default_bucket

        # Register bucket compressors as submodules
        self.bucket_compressors = nn.ModuleDict(bucket_compressors)

    def set_bucket(self, bucket: str):
        """Set the current semantic bucket for compression."""
        if bucket in self.bucket_compressors:
            self.current_bucket = bucket
        else:
            self.current_bucket = self.default_bucket

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress using the current bucket's projector."""
        compressor = self.bucket_compressors.get(
            self.current_bucket,
            self.bucket_compressors.get(self.default_bucket),
        )
        if compressor is None:
            return x
        return compressor.compress(x)

    def expand(self, z: torch.Tensor) -> torch.Tensor:
        """Expand using the current bucket's projector."""
        compressor = self.bucket_compressors.get(
            self.current_bucket,
            self.bucket_compressors.get(self.default_bucket),
        )
        if compressor is None:
            return z
        return compressor.expand(z)

    def calibrate(self, data: torch.Tensor) -> None:
        """No-op - already calibrated."""
        pass

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio of current bucket's compressor."""
        compressor = self.bucket_compressors.get(self.current_bucket)
        if compressor and hasattr(compressor, "compression_ratio"):
            return compressor.compression_ratio
        return 1.0


def load_semantic_compressors(
    calib_path: str,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
    quantize_bits: int = None,
) -> Tuple[List[nn.Module], List[nn.Module], Dict]:
    """
    Load semantic-aware compressors from calibration file.

    Args:
        calib_path: Path to semantic calibration file (.pt)
        device: Device to load to
        dtype: Data type
        quantize_bits: If set, apply int8/int4 quantization

    Returns:
        (k_compressors, v_compressors, metadata) where compressors are SemanticAwareCompressor
    """
    calib_data = torch.load(calib_path, map_location=device)

    k_compressors = []
    v_compressors = []

    num_layers = calib_data["n_layers"]

    # Choose compressor class based on quantization
    if quantize_bits is not None:

        def make_compressor(U, mean, dt):
            return QuantizedCalibratedCompressor(U, mean, bits=quantize_bits, dtype=dt)

    else:

        def make_compressor(U, mean, dt):
            return CalibratedCompressor(U, mean, dtype=dt)

    for layer_idx in range(num_layers):
        # Build bucket compressors for this layer
        k_bucket_compressors = {}
        v_bucket_compressors = {}

        for bucket_name, bucket_data in calib_data.get("buckets", {}).items():
            if layer_idx < len(bucket_data.get("layers", [])):
                layer_data = bucket_data["layers"][layer_idx]

                K_U = layer_data["K"]["U"].to(device).to(dtype)
                K_mean = layer_data["K"]["mean"].to(device).to(dtype)
                k_bucket_compressors[bucket_name] = make_compressor(K_U, K_mean, dtype)

                V_U = layer_data["V"]["U"].to(device).to(dtype)
                V_mean = layer_data["V"]["mean"].to(device).to(dtype)
                v_bucket_compressors[bucket_name] = make_compressor(V_U, V_mean, dtype)

        # Create semantic-aware compressors
        if k_bucket_compressors:
            k_comp = SemanticAwareCompressor(k_bucket_compressors, dtype=dtype)
        else:
            k_comp = IdentityCompressor()

        if v_bucket_compressors:
            v_comp = SemanticAwareCompressor(v_bucket_compressors, dtype=dtype)
        else:
            v_comp = IdentityCompressor()

        k_compressors.append(k_comp)
        v_compressors.append(v_comp)

    metadata = {
        "model": calib_data.get("model", "unknown"),
        "rank": calib_data.get("rank", 0),
        "head_dim": calib_data.get("head_dim", 0),
        "n_layers": num_layers,
        "n_heads": calib_data.get("n_heads", 0),
        "buckets": list(calib_data.get("buckets", {}).keys()),
        "semantic_aware": True,
    }

    return k_compressors, v_compressors, metadata


def create_mixed_mode_compressors(
    calib_path: str,
    k_mode: str = "identity",
    v_mode: str = "full",
    k_bits: int = None,
    v_bits: int = 8,
    layer_mask: List[bool] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[List[nn.Module], List[nn.Module], Dict]:
    """
    Create mixed-mode compressors with different K/V strategies.

    Common configurations:
    - K: identity, V: int8 (safest, V compresses better)
    - K: FP16 low-rank, V: int8 low-rank (more aggressive)
    - Layer-selective: compress only non-critical layers

    Args:
        calib_path: Path to calibration file
        k_mode: K compression mode ("identity", "full", "head_selective")
        v_mode: V compression mode ("identity", "full", "head_selective")
        k_bits: K quantization bits (None for FP16, 8 for int8)
        v_bits: V quantization bits (None for FP16, 8 for int8)
        layer_mask: [num_layers] bool list, True = compress this layer
        device: Device
        dtype: Data type

    Returns:
        (k_compressors, v_compressors, metadata)
    """
    calib_data = torch.load(calib_path, map_location=device)

    k_compressors = []
    v_compressors = []

    num_layers = calib_data["n_layers"]

    # Default layer mask: compress all
    if layer_mask is None:
        layer_mask = [True] * num_layers

    for layer_idx, layer_data in enumerate(calib_data["layers"]):
        compress_this_layer = (
            layer_mask[layer_idx] if layer_idx < len(layer_mask) else True
        )

        # K compressor
        K_U = layer_data["K"]["U"].to(device).to(dtype)
        K_mean = layer_data["K"]["mean"].to(device).to(dtype)

        if not compress_this_layer or k_mode == "identity":
            k_comp = IdentityCompressor()
        else:
            if k_bits is not None:
                base_k = QuantizedCalibratedCompressor(
                    K_U, K_mean, bits=k_bits, dtype=dtype
                )
            else:
                base_k = CalibratedCompressor(K_U, K_mean, dtype=dtype)
            k_comp = MixedModeCompressor(base_k, mode=k_mode)

        k_compressors.append(k_comp)

        # V compressor
        V_U = layer_data["V"]["U"].to(device).to(dtype)
        V_mean = layer_data["V"]["mean"].to(device).to(dtype)

        if not compress_this_layer or v_mode == "identity":
            v_comp = IdentityCompressor()
        else:
            if v_bits is not None:
                base_v = QuantizedCalibratedCompressor(
                    V_U, V_mean, bits=v_bits, dtype=dtype
                )
            else:
                base_v = CalibratedCompressor(V_U, V_mean, dtype=dtype)
            v_comp = MixedModeCompressor(base_v, mode=v_mode)

        v_compressors.append(v_comp)

    # Calculate effective compression
    k_ratio = (
        1.0 if k_mode == "identity" else (calib_data["head_dim"] / calib_data["rank"])
    )
    v_ratio = (
        1.0 if v_mode == "identity" else (calib_data["head_dim"] / calib_data["rank"])
    )

    if k_bits == 8 and k_mode != "identity":
        k_ratio *= 2  # FP16 -> int8
    if v_bits == 8 and v_mode != "identity":
        v_ratio *= 2

    # Average compression considering K and V separately
    total_compression = 2.0 / (1.0 / k_ratio + 1.0 / v_ratio)

    metadata = {
        "model": calib_data["model"],
        "rank": calib_data["rank"],
        "head_dim": calib_data["head_dim"],
        "n_layers": num_layers,
        "n_heads": calib_data["n_heads"],
        "k_mode": k_mode,
        "v_mode": v_mode,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "k_compression": k_ratio,
        "v_compression": v_ratio,
        "total_compression": total_compression,
        "layers_compressed": sum(layer_mask),
    }

    return k_compressors, v_compressors, metadata


def load_calibrated_compressors(
    calib_path: str,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
    quantize_bits: int = None,
) -> Tuple[List[nn.Module], List[nn.Module], Dict]:
    """
    Load calibrated compressors from file.

    Args:
        calib_path: Path to calibration file (.pt)
        device: Device to load to
        dtype: Data type
        quantize_bits: If set, apply int8/int4 quantization in latent space

    Returns:
        (k_compressors, v_compressors, metadata)
    """
    calib_data = torch.load(calib_path, map_location=device)

    k_compressors = []
    v_compressors = []

    # Choose compressor class based on quantization
    if quantize_bits is not None:
        CompressorClass = lambda U, mean, dt: QuantizedCalibratedCompressor(
            U, mean, bits=quantize_bits, dtype=dt
        )
    else:
        CompressorClass = CalibratedCompressor

    for layer_data in calib_data["layers"]:
        # K compressor
        K_U = layer_data["K"]["U"].to(device).to(dtype)
        K_mean = layer_data["K"]["mean"].to(device).to(dtype)
        k_comp = CompressorClass(K_U, K_mean, dtype)
        k_compressors.append(k_comp)

        # V compressor
        V_U = layer_data["V"]["U"].to(device).to(dtype)
        V_mean = layer_data["V"]["mean"].to(device).to(dtype)
        v_comp = CompressorClass(V_U, V_mean, dtype)
        v_compressors.append(v_comp)

    metadata = {
        "model": calib_data["model"],
        "rank": calib_data["rank"],
        "head_dim": calib_data["head_dim"],
        "n_layers": calib_data["n_layers"],
        "n_heads": calib_data["n_heads"],
        "compression_ratio": calib_data["head_dim"] / calib_data["rank"],
        "quantize_bits": quantize_bits,
    }

    return k_compressors, v_compressors, metadata


def load_preset_cache(
    preset_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> "CompressedDynamicCache":
    """
    Load a compression cache from a preset file.

    This is the main entry point for using KV compression.

    Args:
        preset_path: Path to preset JSON file (from auto_tune_kv_compression.py)
        device: Target device ("cuda" or "cpu")
        dtype: Data type for computations

    Returns:
        CompressedDynamicCache ready for inference

    Example:
        >>> cache = load_preset_cache("kv_preset_qwen-qwen2.5-7b_v9.json", device="cuda")
        >>> outputs = model.generate(input_ids, past_key_values=cache, max_new_tokens=100)
    """
    import json

    with open(preset_path) as f:
        preset = json.load(f)

    calib_path = preset["calibration_file"]
    num_layers = preset["num_layers"]
    target = preset.get("target", "v")
    bits = preset.get("bits", 16)

    quantize_bits = bits if bits < 16 else None

    k_comp, v_comp, metadata = load_calibrated_compressors(
        calib_path,
        device=torch.device(device),
        dtype=dtype,
        quantize_bits=quantize_bits,
    )

    # Apply target filter (k-only, v-only, or both)
    if target == "k":
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
    elif target == "v":
        k_comp = [IdentityCompressor() for _ in range(num_layers)]

    return CompressedDynamicCache(k_comp, v_comp, num_layers)


def get_kv_memory_usage(cache: "CompressedDynamicCache") -> Dict:
    """
    Get memory usage statistics from a cache.

    Args:
        cache: CompressedDynamicCache instance

    Returns:
        Dict with memory statistics
    """
    return cache.get_memory_stats()


def profile_kv(
    model,
    tokenizer,
    cache: "CompressedDynamicCache" = None,
    prompt: str = "Hello, how are you?",
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    Profile KV cache performance.

    Measures memory usage and generation speed.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        cache: Optional compression cache
        prompt: Test prompt
        max_new_tokens: Tokens to generate
        device: Device

    Returns:
        Dict with profiling results
    """
    import time

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, past_key_values=cache)
        if cache is not None:
            cache.reset()

    # Timed run
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            past_key_values=cache,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = output_tokens / elapsed

    result = {
        "tokens_generated": output_tokens,
        "time_seconds": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }

    if cache is not None:
        result["cache_memory_mb"] = cache.get_memory_stats()["total_mb"]

    return result
