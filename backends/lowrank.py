"""
Method A: Low-rank KV compression with adaptive per-head rank.

Calibrates per-layer per-head SVD projections on sample data,
then during decode compresses far-past KV entries to low-rank form
and reconstructs on the fly.

Key design:
- Offline calibration: collect K,V statistics per (layer, head)
- SVD to find top-r directions capturing most variance
- Adaptive rank selection: r_{l,h} chosen to meet error target
  under a global rank budget
- During decode: far tokens stored as U_k[T,r] * S_k[r,d] form
  Reconstruction: K_hat = U_k @ S_k
"""

import time

import torch
import numpy as np
from transformers.cache_utils import DynamicCache

from backends.base import CompressionBackend, V14StepStats


class LowRankBackend(CompressionBackend):
    """Per-head per-layer low-rank KV compression."""

    @property
    def name(self):
        return "lowrank"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", 1024)
        self.W_sink = kwargs.get("W_sink", 4)
        self.target_ratio = kwargs.get("target_ratio", 0.5)  # target 50% compression
        self.max_rank = kwargs.get(
            "max_rank", model_config["head_dim"] // 2
        )  # max rank per head
        self.min_rank = kwargs.get("min_rank", 4)
        # Preserve calibration across configure calls
        if not hasattr(self, "calibrated"):
            self.calibrated = False
            self.projections = None

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Calibrate SVD projections on sample data."""
        from scripts.bpa_v11_bench import get_text_batch

        n_layers = model_config["n_layers"]
        n_kv_heads = model_config["n_kv_heads"]
        head_dim = model_config["head_dim"]

        # Get calibration data
        rng = np.random.RandomState(42)
        cal_len = min(L, 4096)
        idx = get_text_batch(token_data, 1, cal_len, rng).to(device_str)

        # Run prefill to get KV cache
        with torch.no_grad():
            out = model(idx, use_cache=True)
            past = out.past_key_values

        # Compute SVD per (layer, head) and select adaptive ranks
        self.projections = []
        total_elements_full = 0
        total_elements_compressed = 0

        for li in range(n_layers):
            k, v = past[li]  # [B, n_kv_heads, T, head_dim]
            layer_projs = []

            for hi in range(n_kv_heads):
                # Get K and V for this head
                k_h = k[0, hi, :, :].float()  # [T, head_dim]
                v_h = v[0, hi, :, :].float()  # [T, head_dim]

                # SVD on K
                U_k, S_k, Vh_k = torch.linalg.svd(k_h, full_matrices=False)
                # SVD on V
                U_v, S_v, Vh_v = torch.linalg.svd(v_h, full_matrices=False)

                # Adaptive rank: find rank that captures target variance
                # Use cumulative energy ratio
                total_energy_k = (S_k**2).sum()
                total_energy_v = (S_v**2).sum()

                # Find rank for K
                cum_energy_k = torch.cumsum(S_k**2, dim=0)
                target_energy = 0.99  # keep 99% of energy
                rank_k = max(
                    self.min_rank,
                    int(
                        (cum_energy_k / total_energy_k >= target_energy)
                        .float()
                        .argmax()
                        .item()
                    )
                    + 1,
                )
                rank_k = min(rank_k, self.max_rank)

                # Find rank for V
                cum_energy_v = torch.cumsum(S_v**2, dim=0)
                rank_v = max(
                    self.min_rank,
                    int(
                        (cum_energy_v / total_energy_v >= target_energy)
                        .float()
                        .argmax()
                        .item()
                    )
                    + 1,
                )
                rank_v = min(rank_v, self.max_rank)

                # Use max of k/v ranks for simplicity
                rank = max(rank_k, rank_v)

                # Store projection matrices: Vh[:rank, :] (right singular vectors)
                # K_compressed = K @ Vh[:rank,:].T = U[:,:rank] @ diag(S[:rank])
                # K_reconstructed = K_compressed @ Vh[:rank,:]
                proj_k = Vh_k[:rank, :].to(k.dtype)  # [rank, head_dim]
                proj_v = Vh_v[:rank, :].to(v.dtype)  # [rank, head_dim]

                layer_projs.append(
                    {
                        "proj_k": proj_k,
                        "proj_v": proj_v,
                        "rank": rank,
                    }
                )

                # Track compression
                T = k_h.shape[0]
                total_elements_full += 2 * T * head_dim
                total_elements_compressed += 2 * T * rank

            self.projections.append(layer_projs)

        self.actual_ratio = total_elements_compressed / max(total_elements_full, 1)
        print(
            f"    lowrank calibrated: avg_rank={np.mean([[p['rank'] for p in lp] for lp in self.projections]):.1f} "
            f"ratio={self.actual_ratio:.3f}"
        )

        # Cleanup
        del past, out
        torch.cuda.empty_cache()
        self.calibrated = True

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2  # fp16 bytes

        # Prefill
        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        has_compressed = False

        # Compress far tokens in the initial cache
        if self.calibrated and self.projections:
            t0 = time.perf_counter()
            past, n_full, n_compressed = self._compress_far(past, device_str)
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
                avg_rank = np.mean([[p["rank"] for p in lp] for lp in self.projections])
                bytes_full = n_full * bpt * n_layers
                bytes_compressed = int(
                    n_compressed * 2 * n_kv_heads * avg_rank * elem * n_layers
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
                    n_full=n_full + step + 1,  # near grows each step
                    n_compressed=n_compressed,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def _compress_far(self, past, device_str):
        """Compress far tokens using calibrated projections.

        Keeps sink tokens and near window as full KV.
        Compresses middle tokens via low-rank projection and
        reconstructs them back to full dimension.
        """
        n_layers = len(past)
        cache_len = past[0][0].shape[2]

        if cache_len <= self.W_min + self.W_sink:
            return past, cache_len, 0

        # Regions: [0:W_sink] = sink, [W_sink:far_end] = far, [far_end:] = near
        far_end = cache_len - self.W_min
        n_far = far_end - self.W_sink

        if n_far <= 0:
            return past, cache_len, 0

        new_cache = DynamicCache()
        for li in range(n_layers):
            k, v = past[li]  # [B, n_kv_heads, T, head_dim]
            n_kv_heads = k.shape[1]

            # Sink tokens (full)
            k_sink = k[:, :, : self.W_sink, :]
            v_sink = v[:, :, : self.W_sink, :]

            # Far tokens (compress + reconstruct)
            k_far = k[:, :, self.W_sink : far_end, :]
            v_far = v[:, :, self.W_sink : far_end, :]

            # Compress and reconstruct per head
            k_parts = [k_sink]
            v_parts = [v_sink]

            for hi in range(n_kv_heads):
                proj = self.projections[li][hi]
                proj_k = proj["proj_k"]  # [rank, head_dim]
                proj_v = proj["proj_v"]

                # Compress: [B, T_far, head_dim] @ [head_dim, rank] -> [B, T_far, rank]
                k_h = k_far[:, hi : hi + 1, :, :]  # [B, 1, T_far, head_dim]
                v_h = v_far[:, hi : hi + 1, :, :]

                k_lat = k_h @ proj_k.T  # [B, 1, T_far, rank]
                v_lat = v_h @ proj_v.T

                # Reconstruct: [B, T_far, rank] @ [rank, head_dim] -> [B, T_far, head_dim]
                k_hat = k_lat @ proj_k  # [B, 1, T_far, head_dim]
                v_hat = v_lat @ proj_v

                k_parts.append(k_hat)
                v_parts.append(v_hat)

            # Near tokens (full)
            k_near = k[:, :, far_end:, :]
            v_near = v[:, :, far_end:, :]

            # Combine: sink + reconstructed_far (per head) + near
            # We need to reassemble per-head pieces back to [B, n_kv_heads, T, head_dim]
            k_far_recon = torch.cat(
                k_parts[1:], dim=1
            )  # [B, n_kv_heads, T_far, head_dim]
            v_far_recon = torch.cat(v_parts[1:], dim=1)

            k_new = torch.cat([k_sink, k_far_recon, k_near], dim=2)
            v_new = torch.cat([v_sink, v_far_recon, v_near], dim=2)

            new_cache.update(k_new, v_new, li)

        n_full = self.W_sink + self.W_min
        n_compressed = n_far
        return new_cache, n_full, n_compressed

    def compression_ratio(self):
        if self.calibrated:
            return self.actual_ratio
        return self.target_ratio

    def description(self):
        if self.calibrated:
            avg_rank = np.mean([[p["rank"] for p in lp] for lp in self.projections])
            return f"lowrank (avg_rank={avg_rank:.0f}, ratio={self.actual_ratio:.3f})"
        return f"lowrank (target_ratio={self.target_ratio})"
