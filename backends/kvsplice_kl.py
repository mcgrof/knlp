"""
Method C: KVSplice trained to preserve attention logits (KL divergence).

Compresses far-past KV entries into M segment representations by
learning a merge operator that minimizes KL divergence between
teacher (dense) and student (compressed) attention distributions.

Key design:
- Teacher: full KV cache attention
- Student: compressed segment KV via learned linear projection
- Training: minimize KL(softmax(Q@K_full/sqrt(d)) || softmax(Q@K_seg/sqrt(d)))
- Plus value reconstruction loss
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats


class SegmentCompressor(nn.Module):
    """Learned segment compression for KV entries.

    Groups consecutive tokens into segments and projects them
    to compressed representations using a learned linear layer.
    """

    def __init__(self, head_dim, segment_size, n_kv_heads, device, dtype):
        super().__init__()
        self.head_dim = head_dim
        self.segment_size = segment_size

        # Learned projection: [segment_size * head_dim] -> [head_dim]
        # Shared across KV heads
        self.k_proj = nn.Linear(segment_size * head_dim, head_dim, bias=False).to(
            device=device, dtype=torch.float32
        )
        self.v_proj = nn.Linear(segment_size * head_dim, head_dim, bias=False).to(
            device=device, dtype=torch.float32
        )

        # Initialize to average (like KVSplice baseline)
        with torch.no_grad():
            avg_init = torch.zeros(head_dim, segment_size * head_dim)
            for i in range(segment_size):
                avg_init[:, i * head_dim : (i + 1) * head_dim] = (
                    torch.eye(head_dim) / segment_size
                )
            self.k_proj.weight.copy_(avg_init)
            self.v_proj.weight.copy_(avg_init)

    def forward(self, k, v):
        """Compress K,V into segment representations.

        Args:
            k: [B, H, T, D] (T must be multiple of segment_size)
            v: [B, H, T, D]
        Returns:
            k_seg: [B, H, T//seg, D]
            v_seg: [B, H, T//seg, D]
        """
        B, H, T, D = k.shape
        seg = self.segment_size
        n_segs = T // seg

        if n_segs == 0:
            return k, v

        # Reshape to [B, H, n_segs, seg*D]
        k_blocks = k[:, :, : n_segs * seg, :].reshape(B, H, n_segs, seg * D).float()
        v_blocks = v[:, :, : n_segs * seg, :].reshape(B, H, n_segs, seg * D).float()

        # Project
        k_seg = self.k_proj(k_blocks)  # [B, H, n_segs, D]
        v_seg = self.v_proj(v_blocks)

        return k_seg.to(k.dtype), v_seg.to(v.dtype)


class KVSpliceKLBackend(CompressionBackend):
    """KVSplice trained with attention KL divergence loss."""

    @property
    def name(self):
        return "kvsplice_kl"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)
        self.segment_size = kwargs.get("segment_size", 4)
        self.train_steps = kwargs.get("train_steps", 2000)
        self.lr = kwargs.get("lr", 1e-3)
        self.temperature = kwargs.get("temperature", 1.0)
        # Preserve calibration across configure calls
        if not hasattr(self, "calibrated"):
            self.calibrated = False
            self.compressor = None

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Train segment compressor with attention KL loss."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]
        n_heads = model_config["n_heads"]

        # Create compressor
        dtype = torch.float16
        self.compressor = SegmentCompressor(
            head_dim, self.segment_size, n_kv_heads, device_str, dtype
        )

        # Get calibration data
        rng = np.random.RandomState(42)
        cal_len = min(L, 4096)
        idx = get_text_batch(token_data, 1, cal_len, rng).to(device_str)

        # Get dense KV cache
        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Extract K,V from a few layers for training
        train_layers = list(range(0, n_layers, max(1, n_layers // 4)))[:4]

        # Collect training data: (Q, K_full, V_full) for far region
        # Use last layer's attention Q as representative
        train_data = []
        for li in train_layers:
            k_full, v_full = past[li]  # [1, n_kv_heads, T, head_dim]

            # Simulate queries from random positions in near window
            near_start = max(self.W_sink, cal_len - self.W_min)
            far_end = near_start
            n_far = far_end - self.W_sink

            if n_far < self.segment_size:
                continue

            # Far K,V (what we compress)
            k_far = k_full[:, :, self.W_sink : far_end, :].detach()
            v_far = v_full[:, :, self.W_sink : far_end, :].detach()

            # Trim to segment boundary
            n_segs = n_far // self.segment_size
            k_far = k_far[:, :, : n_segs * self.segment_size, :]
            v_far = v_far[:, :, : n_segs * self.segment_size, :]

            # Use K from near window as pseudo-queries (GQA: repeat for heads)
            k_near = k_full[:, :, far_end:, :].detach()
            # For GQA, K has n_kv_heads but Q has n_heads
            # Use K as proxy for Q (they share similar distribution)
            q_samples = k_near[:, :, : min(32, k_near.shape[2]), :]

            train_data.append((q_samples, k_far, v_far, li))

        if not train_data:
            print("    kvsplice_kl: no training data (context too short)")
            self.calibrated = False
            return

        # Train compressor
        optimizer = torch.optim.Adam(self.compressor.parameters(), lr=self.lr)
        best_loss = float("inf")

        for step in range(self.train_steps):
            total_loss = 0.0
            n_batches = 0

            for q, k_far, v_far, li in train_data:
                # Compress far K,V
                k_seg, v_seg = self.compressor(k_far, v_far)

                # Teacher attention logits: Q @ K_far^T / sqrt(d)
                # q: [1, H, n_q, D], k_far: [1, H, T_far, D]
                q_f = q.float()
                scale = head_dim**0.5
                logits_teacher = (
                    torch.matmul(q_f, k_far.float().transpose(-2, -1)) / scale
                )
                # [1, H, n_q, T_far]

                # Student logits: Q @ K_seg^T / sqrt(d)
                logits_student = (
                    torch.matmul(q_f, k_seg.float().transpose(-2, -1)) / scale
                )
                # [1, H, n_q, n_segs]

                # KL divergence on attention distributions
                # Need to handle different sequence lengths
                # Teacher: softmax over T_far positions
                # Student: softmax over n_segs positions
                # These have different lengths, so we compute cross-entropy-like loss

                # Approach: compute attention outputs and compare
                attn_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)
                out_teacher = torch.matmul(attn_teacher, v_far.float())
                # [1, H, n_q, D]

                attn_student = F.softmax(logits_student / self.temperature, dim=-1)
                out_student = torch.matmul(attn_student, v_seg.float())

                # Value reconstruction loss
                value_loss = F.mse_loss(out_student, out_teacher.detach())

                # Attention distribution loss: compare attention mass
                # allocated to each segment vs sum of teacher mass for
                # tokens in that segment
                T_far = k_far.shape[2]
                n_segs = k_seg.shape[2]
                seg = self.segment_size

                # Sum teacher attention per segment
                attn_t_segs = (
                    attn_teacher[:, :, :, : n_segs * seg]
                    .reshape(1, q.shape[1], q.shape[2], n_segs, seg)
                    .sum(dim=-1)
                )
                # [1, H, n_q, n_segs]

                # KL divergence between segment-level distributions
                # Normalize both
                attn_t_norm = attn_t_segs / (
                    attn_t_segs.sum(dim=-1, keepdim=True) + 1e-8
                )
                attn_s_norm = attn_student / (
                    attn_student.sum(dim=-1, keepdim=True) + 1e-8
                )

                kl_loss = F.kl_div(
                    (attn_s_norm + 1e-8).log(),
                    attn_t_norm,
                    reduction="batchmean",
                )

                loss = value_loss + 0.1 * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if step % 500 == 0 or step == self.train_steps - 1:
                print(f"    kvsplice_kl step {step}: loss={avg_loss:.6f}")
            best_loss = min(best_loss, avg_loss)

        self.compressor.eval()
        self.calibrated = True
        print(f"    kvsplice_kl trained: final_loss={avg_loss:.6f}")

        del past, out
        torch.cuda.empty_cache()

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

        # Compress far tokens
        if self.calibrated and self.compressor is not None:
            t0 = time.perf_counter()
            past, n_full, n_compressed, n_segs = self._compress_far(
                past, device_str, dtype
            )
            compress_ms = (time.perf_counter() - t0) * 1000
            has_compressed = True
        else:
            n_full = past[0][0].shape[2]
            n_compressed = 0
            n_segs = 0
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

            bytes_full = n_full * bpt * n_layers
            bytes_compressed = n_segs * bpt * n_layers  # segments are full-dim
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

    def _compress_far(self, past, device_str, dtype):
        """Compress far tokens using trained segment compressor."""
        n_layers = len(past)
        cache_len = past[0][0].shape[2]

        if cache_len <= self.W_min + self.W_sink:
            return past, cache_len, 0, 0

        far_end = cache_len - self.W_min
        n_far = far_end - self.W_sink

        if n_far < self.segment_size:
            return past, cache_len, 0, 0

        n_segs = n_far // self.segment_size
        usable_far = n_segs * self.segment_size

        new_cache = DynamicCache()
        for li in range(n_layers):
            k, v = past[li]

            # Sink
            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            # Far (compress)
            k_far = k[:, :, self.W_sink : self.W_sink + usable_far, :]
            v_far = v[:, :, self.W_sink : self.W_sink + usable_far, :]

            k_seg, v_seg = self.compressor(k_far, v_far)

            # Remainder (not a full segment)
            k_rem = k[:, :, self.W_sink + usable_far : far_end, :]
            v_rem = v[:, :, self.W_sink + usable_far : far_end, :]

            # Near
            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            k_new = torch.cat([k_sink, k_seg, k_rem, k_near], dim=2)
            v_new = torch.cat([v_sink, v_seg, v_rem, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + (n_far - usable_far) + self.W_min
        return new_cache, n_full, usable_far, n_segs

    def compression_ratio(self):
        if self.calibrated:
            return 1.0 / self.segment_size
        return 1.0

    def description(self):
        return f"kvsplice_kl (seg={self.segment_size}, trained={self.calibrated})"
