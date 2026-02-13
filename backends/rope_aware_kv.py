"""
M1: RoPE-aware low-rank K compression + INT8 V.

Three approaches to preserve RoPE phase structure during K compression:
A) Pre-RoPE de-rotation: un-rotate K before SVD, re-rotate on retrieval
B) Complex-plane grouping: treat RoPE pairs as complex, compress magnitudes
C) Frequency-band splitting: separate high/low freq RoPE bands, compress less aggressively on high-freq

V is always kept in INT8 (proven lossless from v14b).
"""

import math
import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats
from backends.quant import quantize_int8_symmetric, dequantize_int8_symmetric


def build_rope_freqs(head_dim, max_pos, theta=10000.0, device="cpu"):
    """Build RoPE frequency tensor."""
    dim_pairs = head_dim // 2
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(max_pos, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)  # [max_pos, dim_pairs]
    return angles


def apply_rope(x, cos, sin):
    """Apply RoPE rotation to x. x: [..., head_dim], cos/sin: [T, dim_pairs]."""
    d = x.shape[-1]
    d2 = d // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:]
    # Reshape cos/sin to broadcast
    shape = [1] * (x.dim() - 2) + list(cos.shape)
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def apply_inverse_rope(x, cos, sin):
    """Apply inverse RoPE (de-rotation). Inverse of rotation by angle is rotation by -angle."""
    return apply_rope(x, cos, -sin)


class RoPEAwareKVBackend(CompressionBackend):
    """RoPE-aware low-rank K + INT8 V compression."""

    def __init__(self, mode="derotate", rank_frac=0.5):
        """
        Args:
            mode: 'derotate', 'complex', or 'freqband'
            rank_frac: fraction of head_dim for K rank (0.5 = d/2)
        """
        self.mode = mode
        self.rank_frac = rank_frac
        self.calibrated = False
        self.projections = None

    @property
    def name(self):
        return f"rope_{self.mode}"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)
        self.rope_theta = model_config.get("rope_theta", 10000.0)

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Calibrate K projections using RoPE-aware basis."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        rng = np.random.RandomState(42)
        idx = get_text_batch(token_data, 1, L, rng).to(device_str)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Build RoPE angles for de-rotation
        angles = build_rope_freqs(head_dim, L, self.rope_theta, device_str)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        target_rank = max(4, int(head_dim * self.rank_frac))

        if self.mode == "derotate":
            self.projections = self._calibrate_derotate(
                past,
                cos_angles,
                sin_angles,
                target_rank,
                n_layers,
                n_kv_heads,
                head_dim,
            )
        elif self.mode == "complex":
            self.projections = self._calibrate_complex(
                past, angles, target_rank, n_layers, n_kv_heads, head_dim
            )
        elif self.mode == "freqband":
            self.projections = self._calibrate_freqband(
                past, angles, target_rank, n_layers, n_kv_heads, head_dim
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.cos_angles = cos_angles
        self.sin_angles = sin_angles
        self.angles = angles

        del past, out
        torch.cuda.empty_cache()
        self.calibrated = True
        print(
            f"    rope_{self.mode} calibrated: rank={target_rank}/{head_dim}"
            f" rank_frac={self.rank_frac}"
        )

    def _calibrate_derotate(
        self, past, cos_angles, sin_angles, target_rank, n_layers, n_kv_heads, head_dim
    ):
        """A) Pre-RoPE de-rotation: un-rotate K, SVD in canonical space."""
        projections = []
        for li in range(n_layers):
            k, v = past[li]  # [B, n_kv_heads, T, head_dim]
            layer_projs = []

            for hi in range(n_kv_heads):
                k_h = k[0, hi, :, :].float()  # [T, head_dim]
                T = k_h.shape[0]

                # De-rotate K to canonical (pre-RoPE) space
                cos_T = cos_angles[:T]
                sin_T = sin_angles[:T]
                k_derot = (
                    apply_inverse_rope(k_h.unsqueeze(0).unsqueeze(0), cos_T, sin_T)
                    .squeeze(0)
                    .squeeze(0)
                )  # [T, head_dim]

                # SVD in de-rotated space (position-independent)
                U, S, Vh = torch.linalg.svd(k_derot, full_matrices=False)
                proj = Vh[:target_rank, :].to(k.dtype)  # [rank, head_dim]

                energy = (S[:target_rank] ** 2).sum() / (S**2).sum()
                layer_projs.append(
                    {
                        "proj_k": proj,
                        "rank_k": target_rank,
                        "energy": energy.item(),
                    }
                )

            projections.append(layer_projs)
        return projections

    def _calibrate_complex(
        self, past, angles, target_rank, n_layers, n_kv_heads, head_dim
    ):
        """B) Complex-plane grouping: compress magnitude and phase separately."""
        dim_pairs = head_dim // 2
        # For complex grouping, we pair dimensions that RoPE rotates together
        # RoPE pairs (i, i+dim/2) share the same rotation frequency
        projections = []
        for li in range(n_layers):
            k, v = past[li]
            layer_projs = []

            for hi in range(n_kv_heads):
                k_h = k[0, hi, :, :].float()  # [T, head_dim]
                T = k_h.shape[0]

                # Convert to complex: pair (x_i, x_{i+d/2}) -> complex z_i
                k_real = k_h[:, :dim_pairs]  # [T, d/2]
                k_imag = k_h[:, dim_pairs:]  # [T, d/2]
                k_complex = torch.complex(k_real, k_imag)  # [T, d/2]

                # SVD on magnitude (position-invariant under RoPE)
                k_mag = k_complex.abs()  # [T, d/2]
                U, S, Vh = torch.linalg.svd(k_mag, full_matrices=False)
                rank_mag = min(target_rank, dim_pairs)
                proj_mag = Vh[:rank_mag, :].to(k.dtype)  # [rank, d/2]

                energy = (S[:rank_mag] ** 2).sum() / (S**2).sum()
                layer_projs.append(
                    {
                        "proj_mag": proj_mag,
                        "rank_mag": rank_mag,
                        "energy": energy.item(),
                    }
                )

            projections.append(layer_projs)
        return projections

    def _calibrate_freqband(
        self, past, angles, target_rank, n_layers, n_kv_heads, head_dim
    ):
        """C) Frequency-band splitting: compress high-freq less aggressively."""
        dim_pairs = head_dim // 2
        # Split into low-freq (first half of pairs) and high-freq (second half)
        n_low = dim_pairs // 2
        n_high = dim_pairs - n_low

        # Different rank budgets: compress low-freq more, high-freq less
        rank_low = max(2, target_rank // 2)
        rank_high = max(2, target_rank - rank_low)

        projections = []
        for li in range(n_layers):
            k, v = past[li]
            layer_projs = []

            for hi in range(n_kv_heads):
                k_h = k[0, hi, :, :].float()  # [T, head_dim]

                # Low-freq band: dimensions [0:n_low] and [d/2:d/2+n_low]
                k_low = torch.cat(
                    [k_h[:, :n_low], k_h[:, dim_pairs : dim_pairs + n_low]], dim=-1
                )  # [T, 2*n_low]

                # High-freq band
                k_high = torch.cat(
                    [k_h[:, n_low:dim_pairs], k_h[:, dim_pairs + n_low :]], dim=-1
                )  # [T, 2*n_high]

                # SVD per band
                _, S_low, Vh_low = torch.linalg.svd(k_low, full_matrices=False)
                proj_low = Vh_low[:rank_low, :].to(k.dtype)

                _, S_high, Vh_high = torch.linalg.svd(k_high, full_matrices=False)
                proj_high = Vh_high[:rank_high, :].to(k.dtype)

                energy_low = ((S_low[:rank_low] ** 2).sum() / (S_low**2).sum()).item()
                energy_high = (
                    (S_high[:rank_high] ** 2).sum() / (S_high**2).sum()
                ).item()

                layer_projs.append(
                    {
                        "proj_low": proj_low,
                        "proj_high": proj_high,
                        "rank_low": rank_low,
                        "rank_high": rank_high,
                        "n_low": n_low,
                        "n_high": n_high,
                        "dim_pairs": dim_pairs,
                        "energy_low": energy_low,
                        "energy_high": energy_high,
                    }
                )

            projections.append(layer_projs)
        return projections

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        has_compressed = False

        if self.calibrated and self.projections:
            t0 = time.perf_counter()
            past, n_full, n_compressed = self._compress_far(past, device_str, dtype)
            compress_ms = (time.perf_counter() - t0) * 1000
            has_compressed = True
        else:
            n_full = past[0][0].shape[2]
            n_compressed = 0
            compress_ms = 0

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            pos_ids = None
            if has_compressed:
                pos_ids = torch.tensor(
                    [[actual_pos]], device=device_str, dtype=torch.long
                )

            out = model(
                next_token,
                past_key_values=past,
                use_cache=True,
                position_ids=pos_ids,
            )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            cache_len = past[0][0].shape[2]
            bpt = 2 * n_kv_heads * head_dim * elem

            # Bytes accounting: K compressed + V in INT8
            if self.calibrated:
                avg_rank = self._avg_rank()
                # K: rank/head_dim of full, V: INT8 = 0.5x
                k_ratio = avg_rank / head_dim
                v_ratio = 0.5  # INT8
                bytes_full = n_full * bpt * n_layers
                bytes_k_compressed = int(
                    n_compressed * n_kv_heads * avg_rank * elem * n_layers
                )
                bytes_v_compressed = int(
                    n_compressed * n_kv_heads * head_dim * 1 * n_layers  # 1 byte INT8
                )
                bytes_compressed = bytes_k_compressed + bytes_v_compressed
            else:
                bytes_full = cache_len * bpt * n_layers
                bytes_compressed = 0

            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=bytes_full,
                    kv_bytes_compressed=bytes_compressed,
                    kv_bytes_total=bytes_full + bytes_compressed,
                    compress_ms=compress_ms if step == 0 else 0,
                    n_full=n_full + step + 1,
                    n_compressed=n_compressed,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def _avg_rank(self):
        """Average K rank across layers and heads."""
        if self.mode == "derotate":
            ranks = [p["rank_k"] for lp in self.projections for p in lp]
        elif self.mode == "complex":
            ranks = [p["rank_mag"] for lp in self.projections for p in lp]
        elif self.mode == "freqband":
            ranks = [
                p["rank_low"] + p["rank_high"] for lp in self.projections for p in lp
            ]
        else:
            ranks = [self.mc["head_dim"]]
        return np.mean(ranks)

    def _compress_far(self, past, device_str, dtype):
        """Compress far K via RoPE-aware low-rank + V via INT8."""
        n_layers = len(past)
        cache_len = past[0][0].shape[2]

        if cache_len <= self.W_min + self.W_sink:
            return past, cache_len, 0

        far_end = cache_len - self.W_min
        n_far = far_end - self.W_sink
        if n_far <= 0:
            return past, cache_len, 0

        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]

        new_cache = DynamicCache()
        for li in range(n_layers):
            k, v = past[li]

            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            k_far = k[:, :, self.W_sink : far_end, :]
            v_far = v[:, :, self.W_sink : far_end, :]

            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            # V: INT8 quantize + dequantize
            v_q, v_s = quantize_int8_symmetric(v_far)
            v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)

            # K: RoPE-aware low-rank compression
            k_hat = self._compress_k_far(k_far, li, n_far, device_str, dtype)

            k_new = torch.cat([k_sink, k_hat, k_near], dim=2)
            v_new = torch.cat([v_sink, v_hat, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + self.W_min
        return new_cache, n_full, n_far

    def _compress_k_far(self, k_far, layer_idx, n_far, device_str, dtype):
        """Compress K far tokens using the chosen RoPE-aware method."""
        if self.mode == "derotate":
            return self._compress_k_derotate(k_far, layer_idx, n_far, device_str, dtype)
        elif self.mode == "complex":
            return self._compress_k_complex(k_far, layer_idx, n_far, device_str, dtype)
        elif self.mode == "freqband":
            return self._compress_k_freqband(k_far, layer_idx, n_far, device_str, dtype)
        else:
            return k_far

    def _compress_k_derotate(self, k_far, li, n_far, device_str, dtype):
        """A) De-rotate K, project in canonical space, re-rotate."""
        n_kv_heads = k_far.shape[1]
        head_dim = k_far.shape[3]
        T = k_far.shape[2]

        # Get RoPE angles for far token positions
        # Far tokens are at positions [W_sink, W_sink + n_far)
        cos_T = self.cos_angles[self.W_sink : self.W_sink + T]
        sin_T = self.sin_angles[self.W_sink : self.W_sink + T]

        k_parts = []
        for hi in range(n_kv_heads):
            k_h = k_far[:, hi : hi + 1, :, :]  # [B, 1, T, D]
            proj = self.projections[li][hi]
            proj_k = proj["proj_k"]  # [rank, head_dim]

            # De-rotate
            k_derot = apply_inverse_rope(k_h, cos_T, sin_T)

            # Low-rank project in canonical space
            k_proj = (k_derot @ proj_k.T) @ proj_k

            # Re-rotate
            k_rerot = apply_rope(k_proj, cos_T, sin_T)

            k_parts.append(k_rerot)

        return torch.cat(k_parts, dim=1).to(dtype)

    def _compress_k_complex(self, k_far, li, n_far, device_str, dtype):
        """B) Compress in complex-plane (magnitude only, preserve phase)."""
        n_kv_heads = k_far.shape[1]
        head_dim = k_far.shape[3]
        dim_pairs = head_dim // 2
        T = k_far.shape[2]

        k_parts = []
        for hi in range(n_kv_heads):
            k_h = k_far[0, hi, :, :].float()  # [T, D]
            proj = self.projections[li][hi]
            proj_mag = proj["proj_mag"]  # [rank, d/2]
            rank = proj["rank_mag"]

            # Convert to complex
            k_real = k_h[:, :dim_pairs]
            k_imag = k_h[:, dim_pairs:]
            k_complex = torch.complex(k_real, k_imag)

            # Compress magnitude, preserve phase
            k_mag = k_complex.abs()  # [T, d/2]
            k_phase = k_complex.angle()  # [T, d/2]

            # Low-rank on magnitude
            k_mag_proj = (k_mag @ proj_mag.float().T) @ proj_mag.float()

            # Reconstruct complex
            k_recon = torch.polar(k_mag_proj, k_phase)

            # Back to real
            k_out = torch.cat([k_recon.real, k_recon.imag], dim=-1)
            k_parts.append(k_out.unsqueeze(0).unsqueeze(0))

        return torch.cat(k_parts, dim=1).to(dtype)

    def _compress_k_freqband(self, k_far, li, n_far, device_str, dtype):
        """C) Split K by frequency band, compress each separately."""
        n_kv_heads = k_far.shape[1]
        head_dim = k_far.shape[3]
        T = k_far.shape[2]

        k_parts = []
        for hi in range(n_kv_heads):
            k_h = k_far[0, hi, :, :].float()  # [T, D]
            proj = self.projections[li][hi]
            n_low = proj["n_low"]
            n_high = proj["n_high"]
            dim_pairs = proj["dim_pairs"]
            proj_low = proj["proj_low"].float()
            proj_high = proj["proj_high"].float()

            # Extract bands
            k_low = torch.cat(
                [k_h[:, :n_low], k_h[:, dim_pairs : dim_pairs + n_low]], dim=-1
            )
            k_high = torch.cat(
                [k_h[:, n_low:dim_pairs], k_h[:, dim_pairs + n_low :]], dim=-1
            )

            # Low-rank per band
            k_low_proj = (k_low @ proj_low.T) @ proj_low
            k_high_proj = (k_high @ proj_high.T) @ proj_high

            # Reassemble
            k_out = torch.zeros_like(k_h)
            k_out[:, :n_low] = k_low_proj[:, :n_low]
            k_out[:, dim_pairs : dim_pairs + n_low] = k_low_proj[:, n_low:]
            k_out[:, n_low:dim_pairs] = k_high_proj[:, :n_high]
            k_out[:, dim_pairs + n_low :] = k_high_proj[:, n_high:]

            k_parts.append(k_out.unsqueeze(0).unsqueeze(0))

        return torch.cat(k_parts, dim=1).to(dtype)

    def compression_ratio(self):
        if self.calibrated:
            avg_rank = self._avg_rank()
            head_dim = self.mc["head_dim"]
            # K: rank/head_dim, V: 0.5 (INT8)
            return (avg_rank / head_dim + 0.5) / 2.0
        return 0.5

    def description(self):
        if self.calibrated:
            avg_rank = self._avg_rank()
            head_dim = self.mc["head_dim"]
            return (
                f"rope_{self.mode} (K_rank={avg_rank:.0f}/{head_dim}"
                f" V=INT8 frac={self.rank_frac})"
            )
        return f"rope_{self.mode} (frac={self.rank_frac})"
