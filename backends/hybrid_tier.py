"""
Phase 5: Hybrid tier schedule.

Combines the best primitives from v15 into a tiered compression:
- Tier 0: Sink tokens (W_sink=4) - full precision
- Tier 1: Near context (W_min) - full precision
- Tier 2: Mid context - INT8 quantization
- Tier 3: Far context - rope_complex (low-rank K magnitude + INT8 V)

The tier boundaries are defined by context distance from the
decode position.
"""

import math
import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats
from backends.quant import dequantize_int8_symmetric, quantize_int8_symmetric
from backends.rope_aware_kv import RoPEAwareKVBackend


class HybridTierBackend(CompressionBackend):
    """Multi-tier KV cache compression.

    Splits context into zones by distance from decode position:
    - sink (first W_sink=4 tokens): full precision
    - near (last W_min tokens): full precision
    - mid (between far/near boundary): INT8
    - far (oldest tokens beyond mid): rope_complex
    """

    def __init__(
        self,
        W_sink=4,
        W_min=1024,
        mid_frac=0.5,
        rank_frac=0.5,
        **kwargs,
    ):
        self.W_sink = W_sink
        self.W_min = W_min
        self.mid_frac = mid_frac
        self.rank_frac = rank_frac
        self.rope_backend = RoPEAwareKVBackend(mode="complex", rank_frac=rank_frac)
        self.calibrated = False

    @property
    def name(self):
        return "hybrid_tier"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.rope_backend.configure(
            L, model_config, W_min=self.W_min, W_sink=self.W_sink
        )

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Calibrate rope_complex for the far tier."""
        self.rope_backend.calibrate(model, token_data, L, device_str, model_config)
        self.calibrated = self.rope_backend.calibrated
        print(f"    hybrid_tier calibrated: mid_frac={self.mid_frac}")

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

        if self.calibrated:
            t0 = time.perf_counter()
            past, tier_info = self._compress_tiered(past, device_str, dtype)
            compress_ms = (time.perf_counter() - t0) * 1000
            has_compressed = True
        else:
            cache_len = past[0][0].shape[2]
            tier_info = {
                "n_sink": self.W_sink,
                "n_far": 0,
                "n_mid": 0,
                "n_near": cache_len - self.W_sink,
            }
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

            # Bytes accounting
            bytes_sink = tier_info["n_sink"] * bpt * n_layers
            bytes_near = (tier_info["n_near"] + step + 1) * bpt * n_layers
            bytes_mid = int(
                tier_info["n_mid"] * n_kv_heads * head_dim * 1 * n_layers * 2
            )
            avg_rank = self.rope_backend._avg_rank() if self.calibrated else head_dim
            bytes_far_k = int(
                tier_info["n_far"] * n_kv_heads * avg_rank * elem * n_layers
            )
            bytes_far_v = int(tier_info["n_far"] * n_kv_heads * head_dim * 1 * n_layers)
            bytes_total = (
                bytes_sink + bytes_near + bytes_mid + bytes_far_k + bytes_far_v
            )

            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=bytes_sink + bytes_near,
                    kv_bytes_compressed=bytes_mid + bytes_far_k + bytes_far_v,
                    kv_bytes_total=bytes_total,
                    compress_ms=compress_ms if step == 0 else 0,
                    n_full=tier_info["n_sink"] + tier_info["n_near"] + step + 1,
                    n_compressed=tier_info["n_far"] + tier_info["n_mid"],
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def _compress_tiered(self, past, device_str, dtype):
        """Apply tiered compression to KV cache."""
        n_layers = len(past)
        cache_len = past[0][0].shape[2]
        head_dim = self.mc["head_dim"]
        dim_pairs = head_dim // 2

        if cache_len <= self.W_min + self.W_sink:
            return past, {
                "n_sink": self.W_sink,
                "n_far": 0,
                "n_mid": 0,
                "n_near": cache_len - self.W_sink,
            }

        # Calculate tier boundaries
        n_near = self.W_min
        compressible = cache_len - self.W_sink - n_near
        if compressible <= 0:
            return past, {
                "n_sink": self.W_sink,
                "n_far": 0,
                "n_mid": 0,
                "n_near": cache_len - self.W_sink,
            }

        n_mid = int(compressible * self.mid_frac)
        n_far = compressible - n_mid

        # Boundaries
        far_start = self.W_sink
        far_end = far_start + n_far
        mid_start = far_end
        mid_end = mid_start + n_mid
        near_start = mid_end

        new_cache = DynamicCache()
        for li in range(n_layers):
            k, v = past[li]

            # Tier 0: Sink (full precision)
            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            # Tier 3: Far (rope_complex for K, INT8 for V)
            k_far = k[:, :, far_start:far_end, :]
            v_far = v[:, :, far_start:far_end, :]

            if n_far > 0 and self.rope_backend.projections:
                k_far_hat = self._compress_k_complex(k_far, li, device_str, dtype)
                v_q, v_s = quantize_int8_symmetric(v_far)
                v_far_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
            else:
                k_far_hat = k_far
                v_far_hat = v_far

            # Tier 2: Mid (INT8)
            k_mid = k[:, :, mid_start:mid_end, :]
            v_mid = v[:, :, mid_start:mid_end, :]

            if n_mid > 0:
                k_q, k_s = quantize_int8_symmetric(k_mid)
                k_mid_hat = dequantize_int8_symmetric(k_q, k_s).to(dtype)
                v_q, v_s = quantize_int8_symmetric(v_mid)
                v_mid_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
            else:
                k_mid_hat = k_mid
                v_mid_hat = v_mid

            # Tier 1: Near (full precision)
            k_near = k[:, :, near_start:, :]
            v_near = v[:, :, near_start:, :]

            k_new = torch.cat([k_sink, k_far_hat, k_mid_hat, k_near], dim=2)
            v_new = torch.cat([v_sink, v_far_hat, v_mid_hat, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        return new_cache, {
            "n_sink": self.W_sink,
            "n_far": n_far,
            "n_mid": n_mid,
            "n_near": n_near,
        }

    def _compress_k_complex(self, k_far, li, device_str, dtype):
        """Apply complex-plane K compression from rope_backend."""
        n_kv_heads = k_far.shape[1]
        head_dim = k_far.shape[3]
        dim_pairs = head_dim // 2

        k_parts = []
        for hi in range(n_kv_heads):
            k_h = k_far[0, hi, :, :].float()
            proj = self.rope_backend.projections[li][hi]
            proj_mag = proj["proj_mag"]

            k_real = k_h[:, :dim_pairs]
            k_imag = k_h[:, dim_pairs:]
            k_complex = torch.complex(k_real, k_imag)

            k_mag = k_complex.abs()
            k_phase = k_complex.angle()

            k_mag_proj = (k_mag @ proj_mag.float().T) @ proj_mag.float()
            k_recon = torch.polar(k_mag_proj, k_phase)

            k_out = torch.cat([k_recon.real, k_recon.imag], dim=-1)
            k_parts.append(k_out.unsqueeze(0).unsqueeze(0))

        return torch.cat(k_parts, dim=1).to(dtype)

    def compression_ratio(self):
        if not self.calibrated:
            return 1.0
        avg_rank = self.rope_backend._avg_rank()
        head_dim = self.mc["head_dim"]
        far_ratio = (avg_rank / head_dim + 0.5) / 2.0
        mid_ratio = 0.5
        total = self.mid_frac * mid_ratio + (1 - self.mid_frac) * far_ratio
        return total

    def description(self):
        return f"hybrid_tier (mid_frac={self.mid_frac}," f" rank_frac={self.rank_frac})"
