"""
bitter5: KVSplice-first.

Tier0: small recency full KV (W_min)
Tier1: KVSplice latent memory ALWAYS maintained for far past
Tier2: optional small set of heavy hitters remain full KV temporarily
Tier3: drop full KV aggressively (but latent always exists)

Key: latent memory (segment-averaged KV) is never evicted.
Far past tokens are always represented, just at lower fidelity.
"""

import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from methods.base import BitterMethod, StepStats
from methods.tiering import KVSplicer


class Bitter5(BitterMethod):
    @property
    def name(self):
        return "bitter5"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", max(64, L // 8))
        self.W_sink = kwargs.get("W_sink", 4)
        self.K_hh = kwargs.get("K_hh", max(8, L // 32))
        self.segment_size = kwargs.get("segment_size", 4)
        self.gate_every = kwargs.get("gate_every", 8)
        self.splicer = KVSplicer(self.segment_size)

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

        # Splice storage: accumulated segment KV per layer
        splice_k = [None] * n_layers
        splice_v = [None] * n_layers
        n_splice_segs = 0

        actual_pos = prefix_ids.shape[1]
        has_evicted = False
        n_dropped = 0
        n_full_tokens = past[0][0].shape[2]

        # V-norm scores for heavy hitter selection
        scores = torch.linspace(0.1, 1.0, n_full_tokens, device=device_str)

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]

            t_gate = 0.0
            if step > 0 and step % self.gate_every == 0:
                t0 = time.perf_counter()
                cache_len = past[0][0].shape[2]

                if cache_len > self.W_min + self.K_hh:
                    recency_start = max(0, cache_len - self.W_min)

                    # Extend scores
                    if len(scores) < cache_len:
                        scores = torch.cat(
                            [
                                scores,
                                torch.ones(cache_len - len(scores), device=device_str),
                            ]
                        )
                    scores *= 0.95

                    # Far region: everything before recency window
                    if recency_start > self.K_hh:
                        far_scores = scores[:recency_start]
                        _, topk_idx = torch.topk(
                            far_scores, min(self.K_hh, len(far_scores))
                        )

                        # Tokens to splice: far region minus heavy hitters and sinks
                        hh_mask = torch.zeros(
                            recency_start, dtype=torch.bool, device=device_str
                        )
                        hh_mask[: self.W_sink] = True
                        hh_mask[topk_idx] = True
                        splice_mask = ~hh_mask

                        splice_indices = splice_mask.nonzero(as_tuple=True)[0]

                        if len(splice_indices) >= self.segment_size:
                            # Extract tokens to splice
                            for li in range(n_layers):
                                k, v = past[li]
                                k_splice = k[:, :, splice_indices, :]
                                v_splice = v[:, :, splice_indices, :]

                                # Average into segments
                                k_seg, v_seg = self.splicer.splice(k_splice, v_splice)

                                if splice_k[li] is None:
                                    splice_k[li] = k_seg
                                    splice_v[li] = v_seg
                                else:
                                    splice_k[li] = torch.cat(
                                        [splice_k[li], k_seg], dim=2
                                    )
                                    splice_v[li] = torch.cat(
                                        [splice_v[li], v_seg], dim=2
                                    )
                            n_splice_segs = splice_k[0].shape[2]

                            # Keep: heavy hitters + recency window
                            keep_mask = torch.zeros(
                                cache_len, dtype=torch.bool, device=device_str
                            )
                            keep_mask[: self.W_sink] = True
                            keep_mask[topk_idx] = True
                            keep_mask[recency_start:] = True

                            indices = keep_mask.nonzero(as_tuple=True)[0]
                            new_cache = DynamicCache()
                            for li in range(n_layers):
                                k, v = past[li]
                                # Prepend splice segments + kept tokens
                                k_combined = torch.cat(
                                    [splice_k[li], k[:, :, indices, :]], dim=2
                                )
                                v_combined = torch.cat(
                                    [splice_v[li], v[:, :, indices, :]], dim=2
                                )
                                new_cache.update(k_combined, v_combined, li)
                            past = new_cache
                            n_dropped += len(splice_indices)
                            n_full_tokens = indices.shape[0]
                            has_evicted = True

                            # Reset splice storage (already in cache)
                            splice_k = [None] * n_layers
                            splice_v = [None] * n_layers

                            # Rebuild scores for new cache layout
                            new_len = past[0][0].shape[2]
                            scores = torch.linspace(
                                0.1, 1.0, new_len, device=device_str
                            )

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
                    [scores, torch.ones(cache_len - len(scores), device=device_str)]
                )

            bpt_full = 2 * n_kv_heads * head_dim * elem
            bpt_splice = bpt_full / self.segment_size
            splice_bytes = n_splice_segs * bpt_splice * n_layers
            full_bytes = n_full_tokens * bpt_full * n_layers
            step_stats.append(
                StepStats(
                    kv_kept=cache_len,
                    tier_full=n_full_tokens,
                    tier_splice=n_splice_segs * self.segment_size,
                    tier_dropped=n_dropped,
                    gate_ms=t_gate,
                    kv_bytes_proxy=full_bytes + splice_bytes,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def n_thresholds(self):
        return 4  # W_min, K_hh, segment_size, gate_every

    def learned_fraction(self):
        return 0.0
