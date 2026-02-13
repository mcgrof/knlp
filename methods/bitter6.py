"""
bitter6: Distance-tier compression.

Partition context by distance bands:
- near: full KV (last W_near tokens)
- mid: MLA low-rank compression
- far: KVSplice segment averaging

No eviction unless over hard cap. This is the "don't delete,
reduce fidelity by distance" approach.

Distance bands scale with L:
- near = min(1024, L//16)
- mid = next L//4
- far = rest
"""

import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from methods.base import BitterMethod, StepStats
from methods.tiering import MLAProjector, KVSplicer


class Bitter6(BitterMethod):
    @property
    def name(self):
        return "bitter6"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config

        # Distance bands scale with L
        self.W_near = kwargs.get("W_near", min(1024, max(64, L // 16)))
        self.W_sink = kwargs.get("W_sink", 4)
        self.W_mid = kwargs.get("W_mid", max(128, L // 4))
        # far = everything older than W_near + W_mid

        self.mla_latent_dim = kwargs.get(
            "mla_latent_dim", max(8, model_config["head_dim"] // 4)
        )
        self.segment_size = kwargs.get("segment_size", 4)
        self.gate_every = kwargs.get("gate_every", 8)
        self.hard_cap_factor = kwargs.get("hard_cap_factor", 2.0)

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2
        dtype = torch.float16

        # Initialize compression operators
        mla = MLAProjector(head_dim, self.mla_latent_dim, n_kv_heads, device_str, dtype)
        splicer = KVSplicer(self.segment_size)

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        actual_pos = prefix_ids.shape[1]
        has_restructured = False
        n_near = past[0][0].shape[2]
        n_mid = 0
        n_far_segs = 0
        n_dropped = 0

        # Storage for compressed tiers (accumulated)
        # These get prepended to the HF cache at gate time
        mla_k_lat = [None] * n_layers  # mid tier
        mla_v_lat = [None] * n_layers
        splice_k = [None] * n_layers  # far tier
        splice_v = [None] * n_layers

        # Reinitialize MLA projector with actual dtype
        mla = MLAProjector(head_dim, self.mla_latent_dim, n_kv_heads, device_str, dtype)

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]

            t_gate = 0.0
            if step > 0 and step % self.gate_every == 0:
                t0 = time.perf_counter()
                cache_len = past[0][0].shape[2]

                # Restructure: move old tokens to compressed tiers
                # Boundary: near = last W_near, mid = next W_mid, far = rest
                # Protect sink tokens from compression
                far_start = self.W_sink
                far_end = max(far_start, cache_len - self.W_near - self.W_mid)
                mid_end = max(far_end, cache_len - self.W_near)

                if far_end - far_start > self.segment_size:
                    # Move far tokens to splice tier
                    for li in range(n_layers):
                        k, v = past[li]

                        # Sink: [0, W_sink) — always full fidelity
                        k_sink = k[:, :, :far_start, :]
                        v_sink = v[:, :, :far_start, :]

                        # Far region: [W_sink, far_end)
                        k_far = k[:, :, far_start:far_end, :]
                        v_far = v[:, :, far_start:far_end, :]
                        k_seg, v_seg = splicer.splice(k_far, v_far)

                        # Mid region: [far_end, mid_end)
                        k_mid = k[:, :, far_end:mid_end, :]
                        v_mid = v[:, :, far_end:mid_end, :]
                        k_lat, v_lat = mla.compress(k_mid, v_mid)

                        # Near region: [mid_end, cache_len)
                        k_near = k[:, :, mid_end:, :]
                        v_near = v[:, :, mid_end:, :]

                        # Reconstruct: sink + splice + mla_expanded + near
                        k_mla_hat, v_mla_hat = mla.expand(k_lat, v_lat)

                        k_combined = torch.cat(
                            [k_sink, k_seg, k_mla_hat, k_near], dim=2
                        )
                        v_combined = torch.cat(
                            [v_sink, v_seg, v_mla_hat, v_near], dim=2
                        )

                        if li == 0:
                            new_cache = DynamicCache()
                        new_cache.update(k_combined, v_combined, li)

                    past = new_cache
                    n_far_segs = k_seg.shape[2]
                    n_mid = k_mla_hat.shape[2]
                    n_near = k_near.shape[2]
                    has_restructured = True

                t_gate = (time.perf_counter() - t0) * 1000

            pos_ids = None
            if has_restructured:
                pos_ids = torch.tensor(
                    [[actual_pos]], device=device_str, dtype=torch.long
                )
            out = model(
                next_token, past_key_values=past, use_cache=True, position_ids=pos_ids
            )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            cache_len = past[0][0].shape[2]
            bpt_full = 2 * n_kv_heads * head_dim * elem
            bpt_splice = bpt_full / self.segment_size
            # Approximate bytes: splice segs at reduced rate, mid at MLA rate,
            # near at full rate
            mla_bpt = mla.bytes_per_token()
            near_bytes = n_near * bpt_full * n_layers
            mid_bytes = n_mid * mla_bpt * n_layers
            far_bytes = n_far_segs * bpt_full * n_layers  # segs at full dim

            step_stats.append(
                StepStats(
                    kv_kept=cache_len,
                    tier_full=n_near,
                    tier_mla=n_mid,
                    tier_splice=n_far_segs * self.segment_size,
                    tier_dropped=n_dropped,
                    gate_ms=t_gate,
                    kv_bytes_proxy=near_bytes + mid_bytes + far_bytes,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def n_thresholds(self):
        return 4  # W_near, W_mid, segment_size, gate_every

    def learned_fraction(self):
        return 0.0
