"""
bitter2: Layer-weighted importance scoring.

score_i += sum_l w_l * attn_mass_l(i)
Default w_l increases with depth (later layers more important).

Since SDPA doesn't provide attention maps, we approximate:
- Use value norm ||V_l[i]|| as proxy for importance at layer l
- Weight by layer depth: w_l = l / n_layers
- Periodically extract V norms from cache for scoring
"""

import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from methods.base import BitterMethod, StepStats


class Bitter2(BitterMethod):
    @property
    def name(self):
        return "bitter2"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", max(64, L // 8))
        self.budget = kwargs.get("budget", int(L * 0.9))
        self.gate_every = kwargs.get("gate_every", 8)

        # Layer weights: increase with depth
        n_layers = model_config["n_layers"]
        self.layer_weights = torch.linspace(0.2, 1.0, n_layers)
        self.layer_weights /= self.layer_weights.sum()

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2
        layer_w = self.layer_weights.to(device_str)

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        cache_len = past[0][0].shape[2]
        scores = torch.zeros(cache_len, device=device_str)
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
                if len(scores) < cache_len:
                    scores = torch.cat(
                        [
                            scores,
                            torch.zeros(cache_len - len(scores), device=device_str),
                        ]
                    )

                # Compute layer-weighted V-norm importance
                importance = torch.zeros(cache_len, device=device_str)
                for li in range(n_layers):
                    _, v = past[li]  # [B, n_kv_heads, T, head_dim]
                    # Average V norm across batch and heads
                    v_norms = v[0].norm(dim=-1).mean(dim=0)  # [T]
                    if len(v_norms) < cache_len:
                        v_norms = torch.cat(
                            [
                                v_norms,
                                torch.zeros(
                                    cache_len - len(v_norms), device=device_str
                                ),
                            ]
                        )
                    importance += layer_w[li] * v_norms[:cache_len]

                scores = 0.9 * scores[:cache_len] + 0.1 * importance

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
                    scores = scores[indices]
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
            if len(scores) < cache_len:
                scores = torch.cat(
                    [scores, torch.zeros(cache_len - len(scores), device=device_str)]
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
        return 3  # W_min, budget, gate_every

    def learned_fraction(self):
        return 0.0
