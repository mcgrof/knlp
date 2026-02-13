"""
Tiering primitives for BPA v13 bitter methods.

Three storage tiers:
- Full KV: exact key/value storage
- MLA: low-rank projection (K' = K @ P, V' = V @ P)
- KVSplice: segment merging (average neighboring tokens)

All tiers operate on HF DynamicCache format.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers.cache_utils import DynamicCache


class MLAProjector:
    """Low-rank KV projector for MLA tier.

    Projects K,V from head_dim to latent_dim, then expands back.
    Uses random orthogonal projections (train-free baseline).
    """

    def __init__(self, head_dim, latent_dim, n_kv_heads, device, dtype):
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.n_kv_heads = n_kv_heads
        self.device = device
        self.dtype = dtype

        # Random orthogonal projection matrices
        # Shape: [head_dim, latent_dim]
        Pk = torch.randn(head_dim, latent_dim, device=device, dtype=torch.float32)
        Pv = torch.randn(head_dim, latent_dim, device=device, dtype=torch.float32)
        # Orthogonalize via QR
        self.Pk = torch.linalg.qr(Pk)[0][:, :latent_dim].to(dtype)
        self.Pv = torch.linalg.qr(Pv)[0][:, :latent_dim].to(dtype)

    def compress(self, k, v):
        """Compress K,V to latent space.

        Args:
            k: [B, n_kv_heads, T, head_dim]
            v: [B, n_kv_heads, T, head_dim]
        Returns:
            k_lat: [B, n_kv_heads, T, latent_dim]
            v_lat: [B, n_kv_heads, T, latent_dim]
        """
        k_lat = k.to(self.dtype) @ self.Pk
        v_lat = v.to(self.dtype) @ self.Pv
        return k_lat, v_lat

    def expand(self, k_lat, v_lat):
        """Expand from latent back to full dim.

        Args:
            k_lat: [B, n_kv_heads, T, latent_dim]
            v_lat: [B, n_kv_heads, T, latent_dim]
        Returns:
            k_hat: [B, n_kv_heads, T, head_dim]
            v_hat: [B, n_kv_heads, T, head_dim]
        """
        k_hat = k_lat @ self.Pk.T
        v_hat = v_lat @ self.Pv.T
        return k_hat, v_hat

    def bytes_per_token(self):
        """Bytes per token in compressed form."""
        elem_bytes = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        return 2 * self.n_kv_heads * self.latent_dim * elem_bytes


class KVSplicer:
    """Segment-level KV merging for KVSplice tier.

    Merges groups of consecutive tokens into segment representations
    by averaging their K,V vectors. Reduces sequence length by
    segment_size factor.
    """

    def __init__(self, segment_size=4):
        self.segment_size = segment_size

    def splice(self, k, v):
        """Merge consecutive tokens into segments.

        Args:
            k: [B, n_kv_heads, T, head_dim]
            v: [B, n_kv_heads, T, head_dim]
        Returns:
            k_seg: [B, n_kv_heads, T//seg, head_dim]
            v_seg: [B, n_kv_heads, T//seg, head_dim]
        """
        B, H, T, D = k.shape
        seg = self.segment_size
        # Trim to multiple of segment_size
        T_trim = (T // seg) * seg
        if T_trim == 0:
            return k, v  # too short to splice

        k_trim = k[:, :, :T_trim, :]
        v_trim = v[:, :, :T_trim, :]

        # Reshape and average
        k_seg = k_trim.reshape(B, H, T_trim // seg, seg, D).mean(dim=3)
        v_seg = v_trim.reshape(B, H, T_trim // seg, seg, D).mean(dim=3)

        return k_seg, v_seg

    def bytes_per_token(self, n_kv_heads, head_dim, dtype):
        """Effective bytes per original token after splicing."""
        elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        full_bpt = 2 * n_kv_heads * head_dim * elem_bytes
        return full_bpt / self.segment_size


class TieredCache:
    """Unified tiered KV cache.

    Manages tokens across three tiers:
    - full: exact K,V stored
    - mla: low-rank compressed K,V
    - splice: segment-averaged K,V

    Provides a unified retrieve() that combines all tiers.
    """

    def __init__(
        self,
        n_layers,
        n_kv_heads,
        head_dim,
        device,
        dtype,
        mla_latent_dim=None,
        splice_segment_size=4,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # MLA projector (shared across layers)
        if mla_latent_dim is None:
            mla_latent_dim = max(8, head_dim // 4)
        self.mla = MLAProjector(head_dim, mla_latent_dim, n_kv_heads, device, dtype)

        # KVSplice
        self.splicer = KVSplicer(splice_segment_size)

        # Per-layer storage
        # Each layer has: full_k, full_v, mla_k_lat, mla_v_lat,
        #                 splice_k, splice_v
        self.full_k = [None] * n_layers
        self.full_v = [None] * n_layers
        self.mla_k_lat = [None] * n_layers
        self.mla_v_lat = [None] * n_layers
        self.splice_k = [None] * n_layers
        self.splice_v = [None] * n_layers

        # Token counts per tier
        self.n_full = [0] * n_layers
        self.n_mla = [0] * n_layers
        self.n_splice = [0] * n_layers  # in original token count

    def store_full(self, layer_idx, k, v):
        """Store tokens at full fidelity.

        Args:
            k, v: [B, n_kv_heads, T, head_dim]
        """
        if self.full_k[layer_idx] is None:
            self.full_k[layer_idx] = k
            self.full_v[layer_idx] = v
        else:
            self.full_k[layer_idx] = torch.cat([self.full_k[layer_idx], k], dim=2)
            self.full_v[layer_idx] = torch.cat([self.full_v[layer_idx], v], dim=2)
        self.n_full[layer_idx] = self.full_k[layer_idx].shape[2]

    def store_mla(self, layer_idx, k, v):
        """Store tokens in MLA (low-rank compressed) form."""
        k_lat, v_lat = self.mla.compress(k, v)
        if self.mla_k_lat[layer_idx] is None:
            self.mla_k_lat[layer_idx] = k_lat
            self.mla_v_lat[layer_idx] = v_lat
        else:
            self.mla_k_lat[layer_idx] = torch.cat(
                [self.mla_k_lat[layer_idx], k_lat], dim=2
            )
            self.mla_v_lat[layer_idx] = torch.cat(
                [self.mla_v_lat[layer_idx], v_lat], dim=2
            )
        self.n_mla[layer_idx] = self.mla_k_lat[layer_idx].shape[2]

    def store_splice(self, layer_idx, k, v):
        """Store tokens in KVSplice (segment-averaged) form."""
        k_seg, v_seg = self.splicer.splice(k, v)
        if self.splice_k[layer_idx] is None:
            self.splice_k[layer_idx] = k_seg
            self.splice_v[layer_idx] = v_seg
        else:
            self.splice_k[layer_idx] = torch.cat(
                [self.splice_k[layer_idx], k_seg], dim=2
            )
            self.splice_v[layer_idx] = torch.cat(
                [self.splice_v[layer_idx], v_seg], dim=2
            )
        n_seg_tokens = self.splice_k[layer_idx].shape[2]
        self.n_splice[layer_idx] = n_seg_tokens * self.splicer.segment_size

    def retrieve_all(self, layer_idx):
        """Retrieve combined K,V from all tiers.

        Returns K,V that can be used as past_key_values for attention.
        MLA tokens are expanded; splice tokens are already in K,V form.
        Concatenated in order: splice (far) | mla (mid) | full (near).
        """
        parts_k = []
        parts_v = []

        # Splice (far past, lowest fidelity)
        if self.splice_k[layer_idx] is not None:
            parts_k.append(self.splice_k[layer_idx])
            parts_v.append(self.splice_v[layer_idx])

        # MLA (mid, medium fidelity)
        if self.mla_k_lat[layer_idx] is not None:
            k_hat, v_hat = self.mla.expand(
                self.mla_k_lat[layer_idx], self.mla_v_lat[layer_idx]
            )
            parts_k.append(k_hat)
            parts_v.append(v_hat)

        # Full (near, full fidelity)
        if self.full_k[layer_idx] is not None:
            parts_k.append(self.full_k[layer_idx])
            parts_v.append(self.full_v[layer_idx])

        if not parts_k:
            return None, None

        k_all = torch.cat(parts_k, dim=2)
        v_all = torch.cat(parts_v, dim=2)
        return k_all, v_all

    def total_seq_len(self, layer_idx):
        """Total effective sequence length across all tiers."""
        total = 0
        if self.full_k[layer_idx] is not None:
            total += self.full_k[layer_idx].shape[2]
        if self.mla_k_lat[layer_idx] is not None:
            total += self.mla_k_lat[layer_idx].shape[2]
        if self.splice_k[layer_idx] is not None:
            total += self.splice_k[layer_idx].shape[2]
        return total

    def kv_bytes_proxy(self, layer_idx):
        """Approximate bytes stored for this layer."""
        elem = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        full_bpt = 2 * self.n_kv_heads * self.head_dim * elem
        mla_bpt = self.mla.bytes_per_token()
        splice_bpt = self.splicer.bytes_per_token(
            self.n_kv_heads, self.head_dim, self.dtype
        )

        total = 0
        if self.full_k[layer_idx] is not None:
            total += self.full_k[layer_idx].shape[2] * full_bpt
        if self.mla_k_lat[layer_idx] is not None:
            total += self.mla_k_lat[layer_idx].shape[2] * mla_bpt
        if self.splice_k[layer_idx] is not None:
            # splice tokens count is n_segments, each covering segment_size originals
            total += self.splice_k[layer_idx].shape[2] * full_bpt
        return total

    def tier_counts(self, layer_idx=0):
        """Return token counts by tier for reporting."""
        n_full = (
            self.full_k[layer_idx].shape[2] if self.full_k[layer_idx] is not None else 0
        )
        n_mla = (
            self.mla_k_lat[layer_idx].shape[2]
            if self.mla_k_lat[layer_idx] is not None
            else 0
        )
        n_splice_segs = (
            self.splice_k[layer_idx].shape[2]
            if self.splice_k[layer_idx] is not None
            else 0
        )
        return {
            "full": n_full,
            "mla": n_mla,
            "splice_segs": n_splice_segs,
            "splice_orig_tokens": n_splice_segs * self.splicer.segment_size,
            "dropped": 0,
        }

    def clear(self):
        """Clear all stored data."""
        for i in range(self.n_layers):
            self.full_k[i] = self.full_v[i] = None
            self.mla_k_lat[i] = self.mla_v_lat[i] = None
            self.splice_k[i] = self.splice_v[i] = None
            self.n_full[i] = self.n_mla[i] = self.n_splice[i] = 0


def tiered_cache_to_hf(tiered_cache):
    """Convert TieredCache to HF DynamicCache for model forward pass.

    Combines all tiers into a single KV cache with uniform seq_len.
    """
    cache = DynamicCache()
    for layer_idx in range(tiered_cache.n_layers):
        k_all, v_all = tiered_cache.retrieve_all(layer_idx)
        if k_all is not None:
            cache.update(k_all, v_all, layer_idx)
        else:
            # Empty cache for this layer — shouldn't happen in practice
            B = 1
            k_empty = torch.zeros(
                B,
                tiered_cache.n_kv_heads,
                0,
                tiered_cache.head_dim,
                device=tiered_cache.device,
                dtype=tiered_cache.dtype,
            )
            cache.update(k_empty, k_empty.clone(), layer_idx)
    return cache
