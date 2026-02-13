"""
M2: Logit-space Nystrom / landmark approximation.
M3: Head-wise far-context logit sketch.

Instead of approximating KV directly, approximate the attention logits
contribution from far context. The model cares about QK^T, so compress
a representation that directly approximates logit contributions.

M2 (Nystrom): Choose m landmark tokens, approximate attention to all
far tokens via low-rank in logit space.

M3 (Logit sketch): Maintain per-head sketch that approximates K_far
distribution using random features for dot-product attention.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats
from backends.quant import quantize_int8_symmetric, dequantize_int8_symmetric


class NystromBackend(CompressionBackend):
    """M2: Nystrom / landmark approximation for far context.

    Instead of keeping all far KV, keep only m landmark tokens.
    Attention to far tokens is approximated via Nystrom extension:
    - Select m landmarks from far tokens (k-means or uniform)
    - For each query, approximate attention to all far tokens using
      only the landmark keys and values.

    Since HF DynamicCache requires actual KV tensors, we implement
    this by replacing far KV with the landmark subset. This is
    equivalent to dropping non-landmark tokens but informed by
    importance (attention mass to landmarks approximates full far).
    """

    def __init__(self, n_landmarks=128, selection="uniform"):
        self.n_landmarks = n_landmarks
        self.selection = selection
        self.calibrated = False

    @property
    def name(self):
        return f"nystrom_{self.n_landmarks}"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Calibrate landmark selection using KV cache statistics.

        Uses K norms as importance proxy rather than full attention
        matrices (which OOM at L=32K).
        """
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        rng = np.random.RandomState(42)
        idx = get_text_batch(token_data, 1, L, rng).to(device_str)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Use K norms as importance proxy (higher norm = more likely
        # to receive attention mass via dot product)
        importance = torch.zeros(L, device=device_str)
        for li in range(n_layers):
            k, v = past[li]  # [B, n_kv_heads, T, head_dim]
            # Average K norm across KV heads
            k_norms = k[0].norm(dim=-1).mean(dim=0)  # [T]
            importance += k_norms

        importance = importance / n_layers
        self.importance = importance.detach()
        self.calibrated = True

        del past, out
        torch.cuda.empty_cache()
        print(
            f"    nystrom calibrated: m={self.n_landmarks}"
            f" selection={self.selection}"
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

        # Select landmarks and replace far KV
        t0 = time.perf_counter()
        past, n_full, n_compressed, n_landmarks = self._select_landmarks(
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
            bytes_full = n_full * bpt * n_layers
            bytes_landmarks = n_landmarks * bpt * n_layers

            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=bytes_full,
                    kv_bytes_compressed=bytes_landmarks,
                    kv_bytes_total=bytes_full + bytes_landmarks,
                    compress_ms=compress_ms if step == 0 else 0,
                    n_full=n_full + step + 1,
                    n_compressed=n_compressed,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def _select_landmarks(self, past, device_str, dtype):
        """Select m landmark tokens from far context."""
        n_layers = len(past)
        cache_len = past[0][0].shape[2]

        if cache_len <= self.W_min + self.W_sink:
            return past, cache_len, 0, 0

        far_end = cache_len - self.W_min
        n_far = far_end - self.W_sink
        if n_far <= 0:
            return past, cache_len, 0, 0

        m = min(self.n_landmarks, n_far)
        if m >= n_far:
            return past, cache_len, 0, 0

        # Select landmark indices
        if self.calibrated and self.selection == "importance":
            # Use calibrated importance scores
            imp = self.importance[self.W_sink : far_end]
            if len(imp) > m:
                _, indices = torch.topk(imp, m)
                indices = indices.sort().values
            else:
                indices = torch.arange(n_far, device=device_str)
        else:
            # Uniform selection
            indices = torch.linspace(0, n_far - 1, m, device=device_str).long()

        # Replace far region with landmarks
        new_cache = DynamicCache()
        for li in range(n_layers):
            k, v = past[li]

            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            k_far = k[:, :, self.W_sink : far_end, :]
            v_far = v[:, :, self.W_sink : far_end, :]

            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            # Select landmarks
            k_landmarks = k_far[:, :, indices, :]
            v_landmarks = v_far[:, :, indices, :]

            k_new = torch.cat([k_sink, k_landmarks, k_near], dim=2)
            v_new = torch.cat([v_sink, v_landmarks, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + self.W_min
        return new_cache, n_full, n_far, m

    def compression_ratio(self):
        return self.n_landmarks / max(self.L - self.W_min, 1)

    def description(self):
        return f"nystrom (m={self.n_landmarks}, sel={self.selection})"


class LogitSketchBackend(CompressionBackend):
    """M3: Head-wise logit sketch using random features.

    Approximate exp(QK^T/sqrt(d)) using random Fourier features:
    phi(x) = [cos(Wx), sin(Wx)] / sqrt(D)

    Then: attention(Q, K) ≈ phi(Q) @ phi(K)^T

    Store phi(K_far) as the compressed representation instead of
    full K. For V, use weighted combination based on approximate
    attention weights.

    Since HF DynamicCache requires standard KV format, we implement
    this as a learned subspace projection that preserves attention
    logit structure.
    """

    def __init__(self, n_features=128, seed=42):
        self.n_features = n_features
        self.seed = seed
        self.calibrated = False

    @property
    def name(self):
        return "logit_sketch"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Calibrate logit sketch using attention logit distribution."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        rng = np.random.RandomState(42)
        idx = get_text_batch(token_data, 1, L, rng).to(device_str)

        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # For each layer/head, find the principal directions in K
        # that best preserve QK^T structure
        self.projections = []
        for li in range(n_layers):
            k, v = past[li]
            layer_projs = []

            for hi in range(n_kv_heads):
                k_h = k[0, hi, :, :].float()  # [T, D]

                # Use attention-weighted SVD: weight K vectors by their
                # contribution to attention logits
                # Approximate: use K norms as proxy for attention importance
                k_norms = k_h.norm(dim=-1, keepdim=True)  # [T, 1]
                k_weighted = k_h * k_norms  # upweight important K vectors

                _, S, Vh = torch.linalg.svd(k_weighted, full_matrices=False)
                n_proj = min(self.n_features, head_dim)
                proj = Vh[:n_proj, :].to(k.dtype)

                energy = (S[:n_proj] ** 2).sum() / (S**2).sum()
                layer_projs.append(
                    {
                        "proj_k": proj,
                        "n_proj": n_proj,
                        "energy": energy.item(),
                    }
                )

            self.projections.append(layer_projs)

        self.calibrated = True
        del past, out
        torch.cuda.empty_cache()
        print(f"    logit_sketch calibrated: n_features={self.n_features}")

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

            if self.calibrated:
                avg_proj = np.mean([p["n_proj"] for lp in self.projections for p in lp])
                bytes_full = n_full * bpt * n_layers
                bytes_compressed = int(
                    n_compressed * n_kv_heads * (avg_proj + head_dim) * elem * n_layers
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

    def _compress_far(self, past, device_str, dtype):
        """Compress far K via logit-preserving projection, keep V full."""
        n_layers = len(past)
        cache_len = past[0][0].shape[2]

        if cache_len <= self.W_min + self.W_sink:
            return past, cache_len, 0

        far_end = cache_len - self.W_min
        n_far = far_end - self.W_sink
        if n_far <= 0:
            return past, cache_len, 0

        n_kv_heads = self.mc["n_kv_heads"]

        new_cache = DynamicCache()
        for li in range(n_layers):
            k, v = past[li]

            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            k_far = k[:, :, self.W_sink : far_end, :]
            v_far = v[:, :, self.W_sink : far_end, :]

            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            # Project K to logit-preserving subspace
            k_parts = []
            for hi in range(n_kv_heads):
                proj = self.projections[li][hi]
                proj_k = proj["proj_k"]  # [n_proj, head_dim]
                k_h = k_far[:, hi : hi + 1, :, :]
                k_hat = (k_h @ proj_k.T) @ proj_k
                k_parts.append(k_hat)

            k_far_recon = torch.cat(k_parts, dim=1)

            # V: keep full (not compressed in logit sketch)
            k_new = torch.cat([k_sink, k_far_recon, k_near], dim=2)
            v_new = torch.cat([v_sink, v_far, v_near], dim=2)
            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + self.W_min
        return new_cache, n_full, n_far

    def compression_ratio(self):
        if self.calibrated:
            avg_proj = np.mean([p["n_proj"] for lp in self.projections for p in lp])
            head_dim = self.mc["head_dim"]
            # K compressed, V full
            return (avg_proj / head_dim + 1.0) / 2.0
        return 0.75

    def description(self):
        return f"logit_sketch (n_features={self.n_features})"
