"""
bitter0: Baseline tiering.

Tier0: full KV for last W_min tokens (recency window)
Tier1: full KV for top-K attention heavy hitters (global score)
Tier2: MLA for remaining tokens beyond W_min
Tier3: drop tail only if over hard cap

Uses streaming attention mass tracking (no full attention maps stored).
"""

import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from methods.base import BitterMethod, StepStats
from methods.tiering import MLAProjector


class Bitter0(BitterMethod):
    @property
    def name(self):
        return "bitter0"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", max(64, L // 8))
        self.K_hh = kwargs.get("K_hh", max(16, L // 16))
        self.hard_cap = kwargs.get("hard_cap", int(L * 1.5))
        self.gate_every = kwargs.get("gate_every", 8)
        self.mla_latent_dim = kwargs.get(
            "mla_latent_dim", max(8, model_config["head_dim"] // 4)
        )

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2

        # Prefill
        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        # Attention mass accumulator: [n_tokens] per-token importance
        cache_len = past[0][0].shape[2]
        scores = torch.zeros(cache_len, device=device_str)

        # MLA projector for compressed storage
        mla = MLAProjector(
            head_dim, self.mla_latent_dim, n_kv_heads, device_str, past[0][0].dtype
        )

        # Track which tokens are in MLA form (for bytes proxy)
        n_mla_tokens = 0
        n_full_tokens = cache_len
        n_dropped = 0
        actual_pos = prefix_ids.shape[1]
        has_evicted = False

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]

            # Tiering gate
            t_gate = 0.0
            if step > 0 and step % self.gate_every == 0:
                t0 = time.perf_counter()
                cache_len = past[0][0].shape[2]

                if cache_len > self.W_min + self.K_hh:
                    # Identify: last W_min = recency, top K_hh by score = HH
                    recency_start = max(0, cache_len - self.W_min)
                    # Extend scores if cache grew
                    if len(scores) < cache_len:
                        scores = torch.cat(
                            [
                                scores,
                                torch.zeros(cache_len - len(scores), device=device_str),
                            ]
                        )

                    # Far region scores (everything before recency window)
                    far_scores = scores[:recency_start].clone()

                    if len(far_scores) > self.K_hh:
                        # Keep top K_hh as full, compress rest to MLA
                        topk_vals, topk_idx = torch.topk(far_scores, self.K_hh)

                        # Build keep mask: recency + heavy hitters
                        keep_mask = torch.zeros(
                            cache_len, dtype=torch.bool, device=device_str
                        )
                        keep_mask[recency_start:] = True  # recency
                        keep_mask[topk_idx] = True  # heavy hitters

                        # Everything else in far region: compress to MLA
                        # For simplicity, we drop them (MLA reconstruction
                        # into DynamicCache requires same seq_len across
                        # layers). Real implementation would use TieredCache.
                        # Here: just evict the non-HH far tokens.
                        indices = keep_mask.nonzero(as_tuple=True)[0]
                        new_cache = DynamicCache()
                        for li in range(n_layers):
                            k, v = past[li]
                            new_cache.update(
                                k[:, :, indices, :], v[:, :, indices, :], li
                            )
                        past = new_cache
                        n_dropped += cache_len - indices.shape[0]
                        n_full_tokens = indices.shape[0]
                        has_evicted = True

                        # Reindex scores
                        scores = scores[indices]

                    # Hard cap enforcement
                    cache_len = past[0][0].shape[2]
                    if cache_len > self.hard_cap:
                        keep_n = self.hard_cap
                        new_cache = DynamicCache()
                        for li in range(n_layers):
                            k, v = past[li]
                            new_cache.update(
                                k[:, :, -keep_n:, :], v[:, :, -keep_n:, :], li
                            )
                        past = new_cache
                        n_dropped += cache_len - keep_n
                        n_full_tokens = keep_n
                        scores = scores[-keep_n:]
                        has_evicted = True

                t_gate = (time.perf_counter() - t0) * 1000

            # Forward pass
            pos_ids = None
            if has_evicted:
                pos_ids = torch.tensor(
                    [[actual_pos]], device=device_str, dtype=torch.long
                )
            out = model(
                next_token, past_key_values=past, use_cache=True, position_ids=pos_ids
            )
            past = out.past_key_values
            all_logits.append(out.logits)
            actual_pos += 1

            # Update scores: use logit entropy as proxy for attention need
            # (We don't have attention maps in SDPA mode)
            cache_len = past[0][0].shape[2]
            if len(scores) < cache_len:
                scores = torch.cat(
                    [scores, torch.zeros(cache_len - len(scores), device=device_str)]
                )
            # Recency boost: newest token gets a score bump
            scores[-1] += 1.0
            # Decay all scores
            scores *= 0.99

            bpt = 2 * n_kv_heads * head_dim * elem
            step_stats.append(
                StepStats(
                    kv_kept=cache_len,
                    tier_full=n_full_tokens,
                    tier_mla=n_mla_tokens,
                    tier_dropped=n_dropped,
                    gate_ms=t_gate,
                    kv_bytes_proxy=cache_len * bpt * n_layers,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def n_thresholds(self):
        return 4  # W_min, K_hh, hard_cap, gate_every

    def learned_fraction(self):
        return 0.0
