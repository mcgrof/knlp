"""
Phase 4: Layer sensitivity profiling.

For each layer, apply INT8 or INT4 quantization only to that layer
while keeping all others dense. This reveals which layers tolerate
compression error and which are sensitive.
"""

import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats
from backends.quant import (
    dequantize_int4_block,
    dequantize_int8_symmetric,
    quantize_int4_block,
    quantize_int8_symmetric,
)


class LayerSensitivityBackend(CompressionBackend):
    """Quantize only a single target layer, keep rest dense."""

    def __init__(self, target_layer=0, target_bits=8, block_size=32):
        self.target_layer = target_layer
        self.target_bits = target_bits
        self.block_size = block_size
        self.calibrated = True

    @property
    def name(self):
        return f"layer_sens_{self.target_bits}bit_L{self.target_layer}"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        pass

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

        t0 = time.perf_counter()
        past, n_full, n_compressed = self._quantize_target_layer(
            past, device_str, dtype
        )
        compress_ms = (time.perf_counter() - t0) * 1000
        has_compressed = n_compressed > 0

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
            bytes_full = cache_len * bpt * n_layers
            bytes_compressed = 0

            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=bytes_full,
                    kv_bytes_compressed=bytes_compressed,
                    kv_bytes_total=bytes_full,
                    compress_ms=compress_ms if step == 0 else 0,
                    n_full=cache_len,
                    n_compressed=n_compressed,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def _quantize_target_layer(self, past, device_str, dtype):
        """Quantize only the target layer's far KV."""
        n_layers = len(past)
        cache_len = past[0][0].shape[2]

        if cache_len <= self.W_min + self.W_sink:
            return past, cache_len, 0

        far_end = cache_len - self.W_min
        n_far = far_end - self.W_sink

        if n_far <= 0:
            return past, cache_len, 0

        new_cache = DynamicCache()
        for li in range(n_layers):
            k, v = past[li]

            if li != self.target_layer:
                new_cache.update(k, v, li)
                continue

            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            k_far = k[:, :, self.W_sink : far_end, :]
            v_far = v[:, :, self.W_sink : far_end, :]

            if self.target_bits == 8:
                k_q, k_s = quantize_int8_symmetric(k_far)
                k_hat = dequantize_int8_symmetric(k_q, k_s).to(dtype)
                v_q, v_s = quantize_int8_symmetric(v_far)
                v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
            else:
                k_q, k_s = quantize_int4_block(k_far, self.block_size)
                k_hat = dequantize_int4_block(k_q, k_s, self.block_size).to(dtype)
                v_q, v_s = quantize_int4_block(v_far, self.block_size)
                v_hat = dequantize_int4_block(v_q, v_s, self.block_size).to(dtype)

            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            k_new = torch.cat([k_sink, k_hat, k_near], dim=2)
            v_new = torch.cat([v_sink, v_hat, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        return new_cache, cache_len, n_far

    def compression_ratio(self):
        return 1.0

    def description(self):
        return (
            f"layer_sensitivity (layer={self.target_layer},"
            f" bits={self.target_bits})"
        )
