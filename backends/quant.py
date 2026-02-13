"""
Method B: KV cache quantization with per-layer bitwidth selection.

Implements:
- Per-channel INT8 symmetric quantization (2x compression)
- Simulated INT4 block quantization (4x compression)
- Per-layer aggressiveness: choose 4 or 8 bit per layer
  based on calibration sensitivity

During decode, far-past KV entries are quantized and dequantized
on the fly. Near window stays full precision.
"""

import time

import torch
import numpy as np
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats


def quantize_int8_symmetric(x):
    """Symmetric per-channel INT8 quantization.

    Args:
        x: [..., dim] float16/bfloat16 tensor
    Returns:
        x_q: [..., dim] int8 tensor
        scale: [..., 1] float tensor
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale


def dequantize_int8_symmetric(x_q, scale):
    """Dequantize INT8 symmetric."""
    return x_q.float() * scale


def quantize_int4_block(x, block_size=32):
    """Simulated INT4 block quantization (stored as int8).

    Per-block symmetric quantization with 4-bit range [-8, 7].
    Stored as int8 for simplicity (actual memory savings simulated).

    Args:
        x: [B, H, T, D] tensor
        block_size: number of elements per quantization block
    Returns:
        x_q: [B, H, T, D] int8 tensor (simulated 4-bit)
        scales: [B, H, T, D//block_size] float tensor
    """
    B, H, T, D = x.shape
    n_blocks = (D + block_size - 1) // block_size
    # Pad D to multiple of block_size
    pad = n_blocks * block_size - D
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))

    x_blocks = x.reshape(B, H, T, n_blocks, block_size)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0  # 4-bit range: [-8, 7]
    x_q = (x_blocks / scale).round().clamp(-8, 7).to(torch.int8)

    # Flatten back
    x_q = x_q.reshape(B, H, T, -1)[:, :, :, :D]
    scale = scale.squeeze(-1)  # [B, H, T, n_blocks]

    return x_q, scale


def dequantize_int4_block(x_q, scale, block_size=32):
    """Dequantize simulated INT4 block."""
    B, H, T, D = x_q.shape
    n_blocks = scale.shape[-1]
    pad = n_blocks * block_size - D
    if pad > 0:
        x_q = torch.nn.functional.pad(x_q, (0, pad))

    x_blocks = x_q.reshape(B, H, T, n_blocks, block_size).float()
    x_deq = x_blocks * scale.unsqueeze(-1)
    x_deq = x_deq.reshape(B, H, T, -1)[:, :, :, :D]
    return x_deq


class QuantBackend(CompressionBackend):
    """Per-layer KV quantization backend."""

    @property
    def name(self):
        return "quant"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)
        self.default_bits = kwargs.get("default_bits", 8)
        self.block_size = kwargs.get("block_size", 32)
        # Preserve calibration across configure calls
        if not hasattr(self, "calibrated"):
            self.calibrated = False
            self.layer_bits = None

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Calibrate per-layer bitwidth based on sensitivity."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        rng = np.random.RandomState(42)
        cal_len = min(L, 4096)
        idx = get_text_batch(token_data, 1, cal_len, rng).to(device_str)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Measure quantization error per layer
        errors_int8 = []
        errors_int4 = []

        for li in range(n_layers):
            k, v = past[li]  # [B, n_kv_heads, T, head_dim]

            # INT8 error
            k_q8, k_s8 = quantize_int8_symmetric(k)
            k_hat8 = dequantize_int8_symmetric(k_q8, k_s8).to(k.dtype)
            v_q8, v_s8 = quantize_int8_symmetric(v)
            v_hat8 = dequantize_int8_symmetric(v_q8, v_s8).to(v.dtype)
            err8 = ((k - k_hat8) ** 2).mean().item() + ((v - v_hat8) ** 2).mean().item()
            errors_int8.append(err8)

            # INT4 error
            k_q4, k_s4 = quantize_int4_block(k, self.block_size)
            k_hat4 = dequantize_int4_block(k_q4, k_s4, self.block_size).to(k.dtype)
            v_q4, v_s4 = quantize_int4_block(v, self.block_size)
            v_hat4 = dequantize_int4_block(v_q4, v_s4, self.block_size).to(v.dtype)
            err4 = ((k - k_hat4) ** 2).mean().item() + ((v - v_hat4) ** 2).mean().item()
            errors_int4.append(err4)

        # Assign bitwidth: use INT4 for layers where error is tolerable
        # Heuristic: if INT4 error < 2x median INT8 error, use INT4
        med_err8 = np.median(errors_int8)
        self.layer_bits = []
        for li in range(n_layers):
            if errors_int4[li] < 2.0 * med_err8:
                self.layer_bits.append(4)
            else:
                self.layer_bits.append(8)

        n_int4 = sum(1 for b in self.layer_bits if b == 4)
        n_int8 = sum(1 for b in self.layer_bits if b == 8)
        print(f"    quant calibrated: {n_int4} layers@INT4, {n_int8} layers@INT8")

        del past, out
        torch.cuda.empty_cache()
        self.calibrated = True

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

        # Quantize far tokens
        if self.calibrated:
            t0 = time.perf_counter()
            past, n_full, n_compressed = self._quantize_far(past, device_str, dtype)
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

            # Bytes accounting
            if self.calibrated:
                avg_bits = np.mean(self.layer_bits) if self.layer_bits else 8
                bytes_full = n_full * bpt * n_layers
                bytes_per_compressed_tok = 2 * n_kv_heads * head_dim * (avg_bits / 8)
                # Add scale overhead
                if avg_bits == 4:
                    scale_overhead = 2 * n_kv_heads * (head_dim / self.block_size) * 2
                else:
                    scale_overhead = 2 * n_kv_heads * 2  # one scale per channel
                bytes_compressed = int(
                    n_compressed
                    * (bytes_per_compressed_tok + scale_overhead)
                    * n_layers
                )
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

    def _quantize_far(self, past, device_str, dtype):
        """Quantize and immediately dequantize far tokens.

        Since HF DynamicCache requires float tensors, we quantize
        and dequantize in place. The memory savings are simulated
        (bytes_proxy reflects actual compression), but the quality
        impact is real.
        """
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
            bits = self.layer_bits[li] if self.layer_bits else 8

            # Sink (full)
            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            # Far (quantize + dequantize)
            k_far = k[:, :, self.W_sink : far_end, :]
            v_far = v[:, :, self.W_sink : far_end, :]

            if bits == 8:
                k_q, k_s = quantize_int8_symmetric(k_far)
                k_hat = dequantize_int8_symmetric(k_q, k_s).to(dtype)
                v_q, v_s = quantize_int8_symmetric(v_far)
                v_hat = dequantize_int8_symmetric(v_q, v_s).to(dtype)
            else:  # 4-bit
                k_q, k_s = quantize_int4_block(k_far, self.block_size)
                k_hat = dequantize_int4_block(k_q, k_s, self.block_size).to(dtype)
                v_q, v_s = quantize_int4_block(v_far, self.block_size)
                v_hat = dequantize_int4_block(v_q, v_s, self.block_size).to(dtype)

            # Near (full)
            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            k_new = torch.cat([k_sink, k_hat, k_near], dim=2)
            v_new = torch.cat([v_sink, v_hat, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + self.W_min
        return new_cache, n_full, n_far

    def compression_ratio(self):
        if self.calibrated and self.layer_bits:
            avg_bits = np.mean(self.layer_bits)
            return avg_bits / 16.0  # vs fp16
        return self.default_bits / 16.0

    def description(self):
        if self.calibrated and self.layer_bits:
            n4 = sum(1 for b in self.layer_bits if b == 4)
            n8 = sum(1 for b in self.layer_bits if b == 8)
            return f"quant ({n4}xINT4 + {n8}xINT8)"
        return f"quant (default={self.default_bits}bit)"
