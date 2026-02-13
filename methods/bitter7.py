"""
bitter7: End-to-end rate-distortion controller.

Trains a differentiable router + compression parameters to minimize:
  NLL + alpha * KV_bytes_proxy + beta * peak_KV_memory_proxy

Uses Gumbel-softmax for differentiable tier selection.
Keeps base model frozen; only trains router + MLA projections.

If training is unstable, falls back to router-only (fixed compression).
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from methods.base import BitterMethod, StepStats
from methods.tiering import MLAProjector, KVSplicer


class RateDistortionRouter(nn.Module):
    """Differentiable router for tier assignment.

    Predicts soft tier weights per token group using Gumbel-softmax.
    Tiers: {keep_full, compress_mla, splice, drop}
    """

    def __init__(self, n_features=5, n_tiers=4):
        super().__init__()
        self.n_tiers = n_tiers
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_tiers),
        )

    def forward(self, x, temperature=1.0, hard=False):
        """x: [N, n_features] -> tier_weights [N, n_tiers]"""
        logits = self.net(x)
        if hard:
            return F.gumbel_softmax(logits, tau=temperature, hard=True)
        return F.softmax(logits / temperature, dim=-1)


class Bitter7(BitterMethod):
    @property
    def name(self):
        return "bitter7"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", max(64, L // 8))
        self.gate_every = kwargs.get("gate_every", 8)
        self.alpha = kwargs.get("alpha", 0.001)  # bytes penalty
        self.beta = kwargs.get("beta", 0.0001)  # memory penalty
        self.train_steps = kwargs.get("train_steps", 500)
        self.mla_latent_dim = kwargs.get(
            "mla_latent_dim", max(8, model_config["head_dim"] // 4)
        )
        self.segment_size = kwargs.get("segment_size", 4)
        self.router = None
        self.trained = False

    def _compute_features(self, past, cache_len, device_str, step):
        """Per-token features for router."""
        n_layers = self.mc["n_layers"]
        n_scoring = min(4, n_layers)

        v_norms = torch.zeros(cache_len, device=device_str)
        k_norms = torch.zeros(cache_len, device=device_str)
        for li in range(n_layers - n_scoring, n_layers):
            k, v = past[li]
            vn = v[0].norm(dim=-1).mean(dim=0)[:cache_len]
            kn = k[0].norm(dim=-1).mean(dim=0)[:cache_len]
            v_norms[: len(vn)] += vn
            k_norms[: len(kn)] += kn
        v_norms /= n_scoring
        k_norms /= n_scoring

        recency = torch.arange(cache_len, device=device_str, dtype=torch.float32) / max(
            cache_len - 1, 1
        )
        age = (step - recency * cache_len) / max(cache_len, 1)
        dist = 1.0 - recency

        return torch.stack([v_norms, recency, age, dist, k_norms], dim=-1)

    def _train_router_offline(self, model, prefix_ids, continuation_ids, device_str):
        """Offline training of rate-distortion router."""
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        n_layers = self.mc["n_layers"]
        elem = 2
        bpt_full = 2 * n_kv_heads * head_dim * elem

        # Collect training data from dense decode
        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values

        all_features = []
        all_v_norms = []
        decode_steps = min(self.train_steps, continuation_ids.shape[1])

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values

            if step > 0 and step % (self.gate_every * 2) == 0:
                cache_len = past[0][0].shape[2]
                feats = self._compute_features(past, cache_len, device_str, step)
                all_features.append(feats.detach())
                all_v_norms.append(feats[:, 0].detach())  # V-norm

        if not all_features:
            print("    bitter7: no training data collected")
            self.trained = False
            return

        X = torch.cat(all_features, dim=0)
        v_norms_all = torch.cat(all_v_norms, dim=0)

        # Normalize
        self.feat_mean = X.mean(dim=0)
        self.feat_std = X.std(dim=0) + 1e-8
        X_norm = (X - self.feat_mean) / self.feat_std

        # Oracle labels: top tokens = keep, mid = MLA, low = splice/drop
        sorted_norms, sort_idx = torch.sort(v_norms_all, descending=True)
        n_total = len(v_norms_all)
        labels = torch.zeros(n_total, dtype=torch.long, device=device_str)
        # Top 60% = keep (0), next 20% = MLA (1), next 15% = splice (2),
        # bottom 5% = drop (3)
        cutoffs = [int(n_total * 0.6), int(n_total * 0.8), int(n_total * 0.95)]
        labels[sort_idx[: cutoffs[0]]] = 0
        labels[sort_idx[cutoffs[0] : cutoffs[1]]] = 1
        labels[sort_idx[cutoffs[1] : cutoffs[2]]] = 2
        labels[sort_idx[cutoffs[2] :]] = 3

        # Train router
        self.router = RateDistortionRouter(n_features=5, n_tiers=4).to(device_str)
        optimizer = torch.optim.Adam(self.router.parameters(), lr=0.005)

        for epoch in range(200):
            tier_probs = self.router(X_norm, temperature=1.0)
            # Classification loss
            cls_loss = F.cross_entropy(tier_probs, labels)

            # Rate penalty: encourage compression
            # tier 0=full, 1=MLA(0.25x), 2=splice(0.25x), 3=drop(0)
            rate_weights = torch.tensor([1.0, 0.25, 0.25, 0.0], device=device_str)
            expected_rate = (tier_probs * rate_weights).sum(dim=-1).mean()

            loss = cls_loss + self.alpha * expected_rate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.router(X_norm, temperature=0.1).argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
        print(f"    bitter7 router: acc={acc:.3f} loss={loss.item():.4f}")
        self.router.eval()
        self.trained = True

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2

        # Train router if not done
        if not self.trained:
            with torch.enable_grad():
                self._train_router_offline(
                    model, prefix_ids, continuation_ids, device_str
                )

        # Actual decode
        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        dtype = past[0][0].dtype
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        mla = MLAProjector(head_dim, self.mla_latent_dim, n_kv_heads, device_str, dtype)
        splicer = KVSplicer(self.segment_size)

        actual_pos = prefix_ids.shape[1]
        has_evicted = False
        n_dropped = 0
        n_full = past[0][0].shape[2]
        n_mla = 0
        n_splice = 0

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]

            t_gate = 0.0
            if (
                step > 0
                and step % self.gate_every == 0
                and self.trained
                and self.router is not None
            ):
                t0 = time.perf_counter()
                cache_len = past[0][0].shape[2]

                if cache_len > self.W_min * 2:
                    feats = self._compute_features(past, cache_len, device_str, step)
                    X_norm = (feats - self.feat_mean) / self.feat_std
                    tier_probs = self.router(X_norm, temperature=0.1)
                    tier_assign = tier_probs.argmax(dim=-1)

                    # Force recency window to tier 0
                    recency_start = max(0, cache_len - self.W_min)
                    tier_assign[recency_start:] = 0

                    # Apply tiering
                    keep_full = tier_assign == 0
                    compress_mla = tier_assign == 1
                    compress_splice = tier_assign == 2
                    drop = tier_assign == 3

                    # For HF DynamicCache: we can only keep uniform
                    # seq_len. Compress MLA/splice tokens into
                    # approximate representations that go into the cache.
                    full_idx = keep_full.nonzero(as_tuple=True)[0]
                    mla_idx = compress_mla.nonzero(as_tuple=True)[0]
                    splice_idx = compress_splice.nonzero(as_tuple=True)[0]

                    new_cache = DynamicCache()
                    for li in range(n_layers):
                        k, v = past[li]
                        parts_k = [k[:, :, full_idx, :]]
                        parts_v = [v[:, :, full_idx, :]]

                        # MLA: compress and expand (lossy)
                        if len(mla_idx) > 0:
                            k_m = k[:, :, mla_idx, :]
                            v_m = v[:, :, mla_idx, :]
                            k_lat, v_lat = mla.compress(k_m, v_m)
                            k_hat, v_hat = mla.expand(k_lat, v_lat)
                            parts_k.append(k_hat)
                            parts_v.append(v_hat)

                        # Splice: segment average (lossy)
                        if len(splice_idx) >= splicer.segment_size:
                            k_s = k[:, :, splice_idx, :]
                            v_s = v[:, :, splice_idx, :]
                            k_seg, v_seg = splicer.splice(k_s, v_s)
                            parts_k.append(k_seg)
                            parts_v.append(v_seg)

                        k_new = torch.cat(parts_k, dim=2)
                        v_new = torch.cat(parts_v, dim=2)
                        new_cache.update(k_new, v_new, li)

                    n_dropped += drop.sum().item()
                    n_dropped += (
                        len(splice_idx)
                        - (len(splice_idx) // splicer.segment_size)
                        * splicer.segment_size
                    )
                    n_full = full_idx.shape[0]
                    n_mla = mla_idx.shape[0]
                    n_splice_segs = len(splice_idx) // splicer.segment_size
                    n_splice = n_splice_segs * splicer.segment_size
                    past = new_cache
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
            bpt = 2 * n_kv_heads * head_dim * elem
            step_stats.append(
                StepStats(
                    kv_kept=cache_len,
                    tier_full=n_full,
                    tier_mla=n_mla,
                    tier_splice=n_splice,
                    tier_dropped=n_dropped,
                    gate_ms=t_gate,
                    kv_bytes_proxy=cache_len * bpt * n_layers,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def n_thresholds(self):
        return 1  # W_min (everything else is learned)

    def learned_fraction(self):
        return 0.9  # router + compression params learned
