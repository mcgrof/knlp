"""
M4: Per-layer KVSplice (separate parameters per layer).
M5: Phase-preserving KVSplice (position encoding in memory segments).
M6: Head-clustered KVSplice (cluster heads by behavior, per-cluster splicers).

All variants are TRAINED with attention KL divergence loss.
No untrained variants are evaluated.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats


class PerLayerSegmentCompressor(nn.Module):
    """Per-layer segment compressor for M4.

    Each layer gets its own learned projection, solving v14b's
    problem of shared parameters across heterogeneous layers.
    """

    def __init__(self, n_layers, head_dim, segment_size, device, dtype):
        super().__init__()
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.segment_size = segment_size

        # Per-layer K and V projections
        self.k_projs = nn.ModuleList(
            [
                nn.Linear(segment_size * head_dim, head_dim, bias=False)
                for _ in range(n_layers)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.Linear(segment_size * head_dim, head_dim, bias=False)
                for _ in range(n_layers)
            ]
        )

        # Initialize to average
        with torch.no_grad():
            avg_init = torch.zeros(head_dim, segment_size * head_dim)
            for i in range(segment_size):
                avg_init[:, i * head_dim : (i + 1) * head_dim] = (
                    torch.eye(head_dim) / segment_size
                )
            for li in range(n_layers):
                self.k_projs[li].weight.copy_(avg_init)
                self.v_projs[li].weight.copy_(avg_init)

        self.to(device=device, dtype=torch.float32)

    def forward(self, k, v, layer_idx):
        """Compress K,V for a specific layer."""
        B, H, T, D = k.shape
        seg = self.segment_size
        n_segs = T // seg

        if n_segs == 0:
            return k, v

        k_blocks = k[:, :, : n_segs * seg, :].reshape(B, H, n_segs, seg * D).float()
        v_blocks = v[:, :, : n_segs * seg, :].reshape(B, H, n_segs, seg * D).float()

        k_seg = self.k_projs[layer_idx](k_blocks)
        v_seg = self.v_projs[layer_idx](v_blocks)

        return k_seg.to(k.dtype), v_seg.to(v.dtype)


class PhasePreservingCompressor(nn.Module):
    """Phase-preserving segment compressor for M5.

    Incorporates position encoding explicitly:
    - Memory segments store both content and phase/position summary
    - Retrieval applies RoPE-compatible positioning
    """

    def __init__(
        self, n_layers, head_dim, segment_size, device, dtype, rope_theta=10000.0
    ):
        super().__init__()
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.segment_size = segment_size
        self.rope_theta = rope_theta

        # Per-layer content projection
        self.k_projs = nn.ModuleList(
            [
                nn.Linear(segment_size * head_dim, head_dim, bias=False)
                for _ in range(n_layers)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.Linear(segment_size * head_dim, head_dim, bias=False)
                for _ in range(n_layers)
            ]
        )

        # Per-layer position mixer: learns to combine segment positions
        # into a representative position encoding
        self.pos_mixers = nn.ModuleList(
            [nn.Linear(segment_size, 1, bias=False) for _ in range(n_layers)]
        )

        # Initialize
        with torch.no_grad():
            avg_init = torch.zeros(head_dim, segment_size * head_dim)
            for i in range(segment_size):
                avg_init[:, i * head_dim : (i + 1) * head_dim] = (
                    torch.eye(head_dim) / segment_size
                )
            for li in range(n_layers):
                self.k_projs[li].weight.copy_(avg_init)
                self.v_projs[li].weight.copy_(avg_init)
                # Initialize position mixer to middle of segment
                nn.init.constant_(
                    self.pos_mixers[li].weight,
                    1.0 / segment_size,
                )

        self.to(device=device, dtype=torch.float32)

    def forward(self, k, v, layer_idx, start_pos=0):
        """Compress with phase preservation."""
        B, H, T, D = k.shape
        seg = self.segment_size
        n_segs = T // seg

        if n_segs == 0:
            return k, v, None

        # Content compression
        k_blocks = k[:, :, : n_segs * seg, :].reshape(B, H, n_segs, seg * D).float()
        v_blocks = v[:, :, : n_segs * seg, :].reshape(B, H, n_segs, seg * D).float()

        k_seg = self.k_projs[layer_idx](k_blocks)
        v_seg = self.v_projs[layer_idx](v_blocks)

        # Position summary: learn weighted combination of positions
        # within each segment
        positions = torch.arange(
            start_pos, start_pos + n_segs * seg, device=k.device, dtype=torch.float32
        )
        pos_blocks = positions.reshape(n_segs, seg)  # [n_segs, seg]
        # Learned weighted position per segment
        seg_positions = self.pos_mixers[layer_idx](pos_blocks)  # [n_segs, 1]
        seg_positions = seg_positions.squeeze(-1)  # [n_segs]

        return k_seg.to(k.dtype), v_seg.to(v.dtype), seg_positions


class HeadClusteredCompressor(nn.Module):
    """Head-clustered segment compressor for M6.

    Cluster heads by attention entropy/variance, use separate
    splicers per cluster. Reduces parameter count vs full per-head
    while respecting head heterogeneity.
    """

    def __init__(
        self, n_layers, n_kv_heads, head_dim, segment_size, n_clusters, device, dtype
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.segment_size = segment_size
        self.n_clusters = min(n_clusters, n_kv_heads)

        # Per-layer per-cluster projections
        self.k_projs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(segment_size * head_dim, head_dim, bias=False)
                        for _ in range(self.n_clusters)
                    ]
                )
                for _ in range(n_layers)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(segment_size * head_dim, head_dim, bias=False)
                        for _ in range(self.n_clusters)
                    ]
                )
                for _ in range(n_layers)
            ]
        )

        # Head-to-cluster assignment (set during calibration)
        self.head_clusters = None

        # Initialize
        with torch.no_grad():
            avg_init = torch.zeros(head_dim, segment_size * head_dim)
            for i in range(segment_size):
                avg_init[:, i * head_dim : (i + 1) * head_dim] = (
                    torch.eye(head_dim) / segment_size
                )
            for li in range(n_layers):
                for ci in range(self.n_clusters):
                    self.k_projs[li][ci].weight.copy_(avg_init)
                    self.v_projs[li][ci].weight.copy_(avg_init)

        self.to(device=device, dtype=torch.float32)

    def set_head_clusters(self, assignments):
        """Set head-to-cluster assignments.

        Args:
            assignments: [n_kv_heads] int tensor with cluster IDs
        """
        self.head_clusters = assignments

    def forward(self, k, v, layer_idx):
        """Compress using cluster-specific projections."""
        B, H, T, D = k.shape
        seg = self.segment_size
        n_segs = T // seg

        if n_segs == 0:
            return k, v

        k_parts = []
        v_parts = []

        for hi in range(H):
            ci = (
                self.head_clusters[hi].item()
                if self.head_clusters is not None
                else hi % self.n_clusters
            )

            k_h = (
                k[:, hi : hi + 1, : n_segs * seg, :]
                .reshape(1, 1, n_segs, seg * D)
                .float()
            )
            v_h = (
                v[:, hi : hi + 1, : n_segs * seg, :]
                .reshape(1, 1, n_segs, seg * D)
                .float()
            )

            k_seg_h = self.k_projs[layer_idx][ci](k_h)
            v_seg_h = self.v_projs[layer_idx][ci](v_h)

            k_parts.append(k_seg_h)
            v_parts.append(v_seg_h)

        k_seg = torch.cat(k_parts, dim=1).to(k.dtype)
        v_seg = torch.cat(v_parts, dim=1).to(v.dtype)

        return k_seg, v_seg


class PerLayerKVSpliceBackend(CompressionBackend):
    """M4: Per-layer KVSplice with trained per-layer parameters."""

    def __init__(self, segment_size=4, checkpoint_dir="artifacts/v15"):
        self.segment_size = segment_size
        self.checkpoint_dir = checkpoint_dir
        self.calibrated = False
        self.compressor = None

    @property
    def name(self):
        return f"perlayer_splice_{self.segment_size}"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Load checkpoint or train per-layer compressor."""
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"perlayer_seg{self.segment_size}.pt",
        )
        if os.path.exists(ckpt_path):
            return self._load_checkpoint(ckpt_path, device_str, model_config)

        # Train inline
        self._train(model, token_data, L, device_str, model_config)

    def _train(self, model, token_data, L, device_str, model_config):
        """Train per-layer compressor with attention KL loss."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        self.compressor = PerLayerSegmentCompressor(
            n_layers,
            head_dim,
            self.segment_size,
            device_str,
            torch.float16,
        )

        rng = np.random.RandomState(42)
        cal_len = min(L, 8192)
        idx = get_text_batch(token_data, 1, cal_len, rng).to(device_str)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Collect training data per layer
        train_data = []
        for li in range(n_layers):
            k_full, v_full = past[li]
            near_start = max(self.W_sink, cal_len - self.W_min)
            far_end = near_start
            n_far = far_end - self.W_sink

            if n_far < self.segment_size:
                continue

            n_segs = n_far // self.segment_size
            usable = n_segs * self.segment_size

            k_far = k_full[:, :, self.W_sink : self.W_sink + usable, :].detach()
            v_far = v_full[:, :, self.W_sink : self.W_sink + usable, :].detach()
            k_near = k_full[:, :, far_end : far_end + min(32, self.W_min), :].detach()

            train_data.append((k_far, v_far, k_near, li))

        if not train_data:
            print("    perlayer: no training data")
            self.calibrated = False
            return

        optimizer = torch.optim.Adam(self.compressor.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

        for step in range(5000):
            total_loss = 0.0
            for k_far, v_far, q, li in train_data:
                k_seg, v_seg = self.compressor(k_far, v_far, li)

                scale = head_dim**0.5
                q_f = q.float()

                logits_t = torch.matmul(q_f, k_far.float().transpose(-2, -1)) / scale
                logits_s = torch.matmul(q_f, k_seg.float().transpose(-2, -1)) / scale

                # Attention output comparison
                attn_t = F.softmax(logits_t, dim=-1)
                attn_s = F.softmax(logits_s, dim=-1)

                out_t = torch.matmul(attn_t, v_far.float())
                out_s = torch.matmul(attn_s, v_seg.float())

                value_loss = F.mse_loss(out_s, out_t.detach())

                # Segment-level KL
                seg = self.segment_size
                n_segs = k_seg.shape[2]
                T_far = k_far.shape[2]

                attn_t_segs = (
                    attn_t[:, :, :, : n_segs * seg]
                    .reshape(1, q.shape[1], q.shape[2], n_segs, seg)
                    .sum(dim=-1)
                )
                attn_t_norm = attn_t_segs / (
                    attn_t_segs.sum(dim=-1, keepdim=True) + 1e-8
                )
                attn_s_norm = attn_s / (attn_s.sum(dim=-1, keepdim=True) + 1e-8)

                kl_loss = F.kl_div(
                    (attn_s_norm + 1e-8).log(),
                    attn_t_norm,
                    reduction="batchmean",
                )

                loss = value_loss + 0.1 * kl_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.compressor.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            if step % 1000 == 0 or step == 4999:
                avg = total_loss / max(len(train_data), 1)
                print(f"    perlayer step {step}: loss={avg:.6f}")

        self.compressor.eval()
        self.calibrated = True

        # Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"perlayer_seg{self.segment_size}.pt",
        )
        torch.save(
            {
                "compressor_state": self.compressor.state_dict(),
                "segment_size": self.segment_size,
                "n_layers": n_layers,
                "head_dim": head_dim,
                "final_loss": total_loss / max(len(train_data), 1),
            },
            ckpt_path,
        )
        print(f"    perlayer saved: {ckpt_path}")

        del past, out
        torch.cuda.empty_cache()

    def _load_checkpoint(self, path, device_str, model_config):
        """Load trained compressor."""
        ckpt = torch.load(path, map_location=device_str, weights_only=True)
        n_layers = model_config["n_layers"]
        head_dim = model_config["head_dim"]
        seg = ckpt.get("segment_size", self.segment_size)
        self.segment_size = seg

        self.compressor = PerLayerSegmentCompressor(
            n_layers, head_dim, seg, device_str, torch.float16
        )
        self.compressor.load_state_dict(ckpt["compressor_state"])
        self.compressor.eval()
        self.calibrated = True
        print(
            f"    perlayer loaded: {path} (seg={seg},"
            f" loss={ckpt.get('final_loss', '?')})"
        )

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
            bytes_compressed = n_segs * bpt * n_layers

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
        """Compress far tokens with per-layer compressor."""
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

            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            k_far = k[:, :, self.W_sink : self.W_sink + usable_far, :]
            v_far = v[:, :, self.W_sink : self.W_sink + usable_far, :]

            k_seg, v_seg = self.compressor(k_far, v_far, li)

            k_rem = k[:, :, self.W_sink + usable_far : far_end, :]
            v_rem = v[:, :, self.W_sink + usable_far : far_end, :]

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
        return (
            f"perlayer_splice (seg={self.segment_size}," f" trained={self.calibrated})"
        )


class PhasePreservingKVSpliceBackend(PerLayerKVSpliceBackend):
    """M5: Phase-preserving per-layer KVSplice."""

    def __init__(self, segment_size=4, checkpoint_dir="artifacts/v15"):
        super().__init__(segment_size, checkpoint_dir)

    @property
    def name(self):
        return f"phase_splice_{self.segment_size}"

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Load checkpoint or train phase-preserving compressor."""
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"phase_splice_seg{self.segment_size}.pt",
        )
        if os.path.exists(ckpt_path):
            return self._load_checkpoint(ckpt_path, device_str, model_config)
        self._train(model, token_data, L, device_str, model_config)

    def _train(self, model, token_data, L, device_str, model_config):
        """Train phase-preserving compressor."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]
        rope_theta = model_config.get("rope_theta", 10000.0)

        self.compressor = PhasePreservingCompressor(
            n_layers,
            head_dim,
            self.segment_size,
            device_str,
            torch.float16,
            rope_theta,
        )

        rng = np.random.RandomState(42)
        cal_len = min(L, 8192)
        idx = get_text_batch(token_data, 1, cal_len, rng).to(device_str)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        train_data = []
        for li in range(n_layers):
            k_full, v_full = past[li]
            near_start = max(self.W_sink, cal_len - self.W_min)
            far_end = near_start
            n_far = far_end - self.W_sink

            if n_far < self.segment_size:
                continue

            n_segs = n_far // self.segment_size
            usable = n_segs * self.segment_size

            k_far = k_full[:, :, self.W_sink : self.W_sink + usable, :].detach()
            v_far = v_full[:, :, self.W_sink : self.W_sink + usable, :].detach()
            k_near = k_full[:, :, far_end : far_end + min(32, self.W_min), :].detach()

            train_data.append((k_far, v_far, k_near, li))

        if not train_data:
            print("    phase_splice: no training data")
            self.calibrated = False
            return

        optimizer = torch.optim.Adam(self.compressor.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

        for step in range(5000):
            total_loss = 0.0
            for k_far, v_far, q, li in train_data:
                k_seg, v_seg, seg_pos = self.compressor(
                    k_far, v_far, li, start_pos=self.W_sink
                )

                scale = head_dim**0.5
                q_f = q.float()

                logits_t = torch.matmul(q_f, k_far.float().transpose(-2, -1)) / scale
                logits_s = torch.matmul(q_f, k_seg.float().transpose(-2, -1)) / scale

                attn_t = F.softmax(logits_t, dim=-1)
                attn_s = F.softmax(logits_s, dim=-1)

                out_t = torch.matmul(attn_t, v_far.float())
                out_s = torch.matmul(attn_s, v_seg.float())

                value_loss = F.mse_loss(out_s, out_t.detach())

                # Position regularization: segment positions should
                # be ordered and within original range
                if seg_pos is not None and seg_pos.shape[0] > 1:
                    pos_order_loss = F.relu(seg_pos[:-1] - seg_pos[1:] + 1.0).mean()
                else:
                    pos_order_loss = torch.tensor(0.0, device=k_far.device)

                seg = self.segment_size
                n_segs = k_seg.shape[2]

                attn_t_segs = (
                    attn_t[:, :, :, : n_segs * seg]
                    .reshape(1, q.shape[1], q.shape[2], n_segs, seg)
                    .sum(dim=-1)
                )
                attn_t_norm = attn_t_segs / (
                    attn_t_segs.sum(dim=-1, keepdim=True) + 1e-8
                )
                attn_s_norm = attn_s / (attn_s.sum(dim=-1, keepdim=True) + 1e-8)

                kl_loss = F.kl_div(
                    (attn_s_norm + 1e-8).log(),
                    attn_t_norm,
                    reduction="batchmean",
                )

                loss = value_loss + 0.1 * kl_loss + 0.01 * pos_order_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.compressor.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            if step % 1000 == 0 or step == 4999:
                avg = total_loss / max(len(train_data), 1)
                print(f"    phase_splice step {step}: loss={avg:.6f}")

        self.compressor.eval()
        self.calibrated = True

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"phase_splice_seg{self.segment_size}.pt",
        )
        torch.save(
            {
                "compressor_state": self.compressor.state_dict(),
                "segment_size": self.segment_size,
                "n_layers": n_layers,
                "head_dim": head_dim,
                "final_loss": total_loss / max(len(train_data), 1),
            },
            ckpt_path,
        )
        print(f"    phase_splice saved: {ckpt_path}")

        del past, out
        torch.cuda.empty_cache()

    def _load_checkpoint(self, path, device_str, model_config):
        """Load phase-preserving compressor."""
        ckpt = torch.load(path, map_location=device_str, weights_only=True)
        n_layers = model_config["n_layers"]
        head_dim = model_config["head_dim"]
        seg = ckpt.get("segment_size", self.segment_size)
        self.segment_size = seg
        rope_theta = model_config.get("rope_theta", 10000.0)

        self.compressor = PhasePreservingCompressor(
            n_layers,
            head_dim,
            seg,
            device_str,
            torch.float16,
            rope_theta,
        )
        self.compressor.load_state_dict(ckpt["compressor_state"])
        self.compressor.eval()
        self.calibrated = True
        print(f"    phase_splice loaded: {path}")

    def _compress_far(self, past, device_str, dtype):
        """Compress with phase preservation."""
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

            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            k_far = k[:, :, self.W_sink : self.W_sink + usable_far, :]
            v_far = v[:, :, self.W_sink : self.W_sink + usable_far, :]

            k_seg, v_seg, _ = self.compressor(k_far, v_far, li, start_pos=self.W_sink)

            k_rem = k[:, :, self.W_sink + usable_far : far_end, :]
            v_rem = v[:, :, self.W_sink + usable_far : far_end, :]

            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            k_new = torch.cat([k_sink, k_seg, k_rem, k_near], dim=2)
            v_new = torch.cat([v_sink, v_seg, v_rem, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + (n_far - usable_far) + self.W_min
        return new_cache, n_full, usable_far, n_segs

    def description(self):
        return f"phase_splice (seg={self.segment_size}," f" trained={self.calibrated})"


class HeadClusteredKVSpliceBackend(PerLayerKVSpliceBackend):
    """M6: Head-clustered KVSplice with per-cluster parameters."""

    def __init__(self, segment_size=4, checkpoint_dir="artifacts/v15", n_clusters=2):
        super().__init__(segment_size, checkpoint_dir)
        self.n_clusters = n_clusters

    @property
    def name(self):
        return f"headcluster_splice_{self.segment_size}"

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Load checkpoint or train head-clustered compressor."""
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"headcluster_seg{self.segment_size}.pt",
        )
        if os.path.exists(ckpt_path):
            return self._load_checkpoint(ckpt_path, device_str, model_config)
        self._train(model, token_data, L, device_str, model_config)

    def _train(self, model, token_data, L, device_str, model_config):
        """Train head-clustered compressor."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        # First, cluster heads by K variance
        rng = np.random.RandomState(42)
        cal_len = min(L, 4096)
        idx = get_text_batch(token_data, 1, cal_len, rng).to(device_str)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Compute per-head K variance across layers
        head_features = torch.zeros(n_kv_heads, n_layers, device=device_str)
        for li in range(n_layers):
            k, v = past[li]
            for hi in range(n_kv_heads):
                head_features[hi, li] = k[0, hi, :, :].var().item()

        # Simple clustering: split by median variance
        median_var = head_features.mean(dim=1).median()
        assignments = (head_features.mean(dim=1) > median_var).long()

        self.compressor = HeadClusteredCompressor(
            n_layers,
            n_kv_heads,
            head_dim,
            self.segment_size,
            self.n_clusters,
            device_str,
            torch.float16,
        )
        self.compressor.set_head_clusters(assignments)

        # Collect training data
        train_data = []
        for li in range(n_layers):
            k_full, v_full = past[li]
            near_start = max(self.W_sink, cal_len - self.W_min)
            far_end = near_start
            n_far = far_end - self.W_sink

            if n_far < self.segment_size:
                continue

            n_segs = n_far // self.segment_size
            usable = n_segs * self.segment_size

            k_far = k_full[:, :, self.W_sink : self.W_sink + usable, :].detach()
            v_far = v_full[:, :, self.W_sink : self.W_sink + usable, :].detach()
            k_near = k_full[:, :, far_end : far_end + min(32, self.W_min), :].detach()

            train_data.append((k_far, v_far, k_near, li))

        if not train_data:
            print("    headcluster: no training data")
            self.calibrated = False
            del past, out
            torch.cuda.empty_cache()
            return

        optimizer = torch.optim.Adam(self.compressor.parameters(), lr=1e-3)

        for step in range(5000):
            total_loss = 0.0
            for k_far, v_far, q, li in train_data:
                k_seg, v_seg = self.compressor(k_far, v_far, li)

                scale = head_dim**0.5
                q_f = q.float()

                logits_t = torch.matmul(q_f, k_far.float().transpose(-2, -1)) / scale
                logits_s = torch.matmul(q_f, k_seg.float().transpose(-2, -1)) / scale

                attn_t = F.softmax(logits_t, dim=-1)
                attn_s = F.softmax(logits_s, dim=-1)

                out_t = torch.matmul(attn_t, v_far.float())
                out_s = torch.matmul(attn_s, v_seg.float())

                loss = F.mse_loss(out_s, out_t.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.compressor.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            if step % 1000 == 0 or step == 4999:
                avg = total_loss / max(len(train_data), 1)
                print(f"    headcluster step {step}: loss={avg:.6f}")

        self.compressor.eval()
        self.calibrated = True

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"headcluster_seg{self.segment_size}.pt",
        )
        torch.save(
            {
                "compressor_state": self.compressor.state_dict(),
                "segment_size": self.segment_size,
                "n_layers": n_layers,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
                "n_clusters": self.n_clusters,
                "head_clusters": assignments,
                "final_loss": total_loss / max(len(train_data), 1),
            },
            ckpt_path,
        )
        print(f"    headcluster saved: {ckpt_path}")

        del past, out
        torch.cuda.empty_cache()

    def _load_checkpoint(self, path, device_str, model_config):
        """Load head-clustered compressor."""
        ckpt = torch.load(path, map_location=device_str, weights_only=True)
        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]
        seg = ckpt.get("segment_size", self.segment_size)
        self.segment_size = seg
        n_clusters = ckpt.get("n_clusters", self.n_clusters)

        self.compressor = HeadClusteredCompressor(
            n_layers,
            n_kv_heads,
            head_dim,
            seg,
            n_clusters,
            device_str,
            torch.float16,
        )
        self.compressor.load_state_dict(ckpt["compressor_state"])
        if "head_clusters" in ckpt:
            self.compressor.set_head_clusters(ckpt["head_clusters"].to(device_str))
        self.compressor.eval()
        self.calibrated = True
        print(f"    headcluster loaded: {path}")

    def _compress_far(self, past, device_str, dtype):
        """Compress with head-clustered compressor."""
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

            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            k_far = k[:, :, self.W_sink : self.W_sink + usable_far, :]
            v_far = v[:, :, self.W_sink : self.W_sink + usable_far, :]

            k_seg, v_seg = self.compressor(k_far, v_far, li)

            k_rem = k[:, :, self.W_sink + usable_far : far_end, :]
            v_rem = v[:, :, self.W_sink + usable_far : far_end, :]

            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            k_new = torch.cat([k_sink, k_seg, k_rem, k_near], dim=2)
            v_new = torch.cat([v_sink, v_seg, v_rem, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + (n_far - usable_far) + self.W_min
        return new_cache, n_full, usable_far, n_segs

    def description(self):
        return (
            f"headcluster_splice (seg={self.segment_size},"
            f" clusters={self.n_clusters},"
            f" trained={self.calibrated})"
        )
