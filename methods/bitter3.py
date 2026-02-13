"""
bitter3: Two-signal routing.

score_i = attn_mass_received_i * value_proxy_i
value_proxy = ||V|| (value vector norm as proxy for information content).

Combines recency-decayed importance with value magnitude.
"""

import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from methods.base import BitterMethod, StepStats


class Bitter3(BitterMethod):
    @property
    def name(self):
        return "bitter3"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", max(64, L // 8))
        self.W_sink = kwargs.get("W_sink", 4)
        self.budget = kwargs.get("budget", int(L * 0.9))
        self.decay = kwargs.get("decay", 0.95)
        self.gate_every = kwargs.get("gate_every", 8)

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        cache_len = past[0][0].shape[2]
        # Initialize recency scores
        recency_scores = torch.linspace(0.1, 1.0, cache_len, device=device_str)
        actual_pos = prefix_ids.shape[1]
        has_evicted = False
        n_dropped = 0

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]

            t_gate = 0.0
            if step > 0 and step % self.gate_every == 0:
                t0 = time.perf_counter()
                cache_len = past[0][0].shape[2]

                # Extend scores
                if len(recency_scores) < cache_len:
                    recency_scores = torch.cat(
                        [
                            recency_scores,
                            torch.ones(
                                cache_len - len(recency_scores), device=device_str
                            ),
                        ]
                    )

                # Decay recency
                recency_scores *= self.decay
                recency_scores[-1] = 1.0  # newest always important

                # Compute value proxy: mean ||V|| across last 4 layers
                v_proxy = torch.zeros(cache_len, device=device_str)
                n_scoring_layers = min(4, n_layers)
                for li in range(n_layers - n_scoring_layers, n_layers):
                    _, v = past[li]
                    v_norms = v[0].norm(dim=-1).mean(dim=0)  # [T]
                    v_proxy[: len(v_norms)] += v_norms[:cache_len]
                v_proxy /= n_scoring_layers

                # Two-signal score
                scores = recency_scores[:cache_len] * v_proxy

                if cache_len > self.budget:
                    recency_start = max(0, cache_len - self.W_min)
                    n_far_keep = self.budget - self.W_min

                    if n_far_keep > 0 and recency_start > 0:
                        far_scores = scores[:recency_start]
                        if len(far_scores) > n_far_keep:
                            _, topk_idx = torch.topk(far_scores, n_far_keep)
                            keep_mask = torch.zeros(
                                cache_len, dtype=torch.bool, device=device_str
                            )
                            keep_mask[: self.W_sink] = True
                            keep_mask[recency_start:] = True
                            keep_mask[topk_idx] = True
                        else:
                            keep_mask = torch.ones(
                                cache_len, dtype=torch.bool, device=device_str
                            )
                    else:
                        keep_mask = torch.zeros(
                            cache_len, dtype=torch.bool, device=device_str
                        )
                        keep_mask[-self.budget :] = True

                    indices = keep_mask.nonzero(as_tuple=True)[0]
                    new_cache = DynamicCache()
                    for li in range(n_layers):
                        k, v = past[li]
                        new_cache.update(k[:, :, indices, :], v[:, :, indices, :], li)
                    past = new_cache
                    n_dropped += cache_len - indices.shape[0]
                    recency_scores = recency_scores[indices]
                    has_evicted = True

                t_gate = (time.perf_counter() - t0) * 1000

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

            cache_len = past[0][0].shape[2]
            if len(recency_scores) < cache_len:
                recency_scores = torch.cat(
                    [
                        recency_scores,
                        torch.ones(cache_len - len(recency_scores), device=device_str),
                    ]
                )

            bpt = 2 * n_kv_heads * head_dim * elem
            step_stats.append(
                StepStats(
                    kv_kept=cache_len,
                    tier_full=cache_len,
                    tier_dropped=n_dropped,
                    gate_ms=t_gate,
                    kv_bytes_proxy=cache_len * bpt * n_layers,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def n_thresholds(self):
        return 4  # W_min, budget, decay, gate_every

    def learned_fraction(self):
        return 0.0
