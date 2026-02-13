"""
bitter4: Learned routing scorer.

Replaces heuristics with a small learned MLP router.
Inputs: decayed_score, recency_bucket, token_age, ||V|| stats
Output: keep probability (binary: keep full or drop)

Training: oracle labeling on calibration set — keep tokens with
highest total V-norm (oracle-lite). Train router to match.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from methods.base import BitterMethod, StepStats


class TierRouter(nn.Module):
    """Tiny MLP that predicts keep/drop for each token."""

    def __init__(self, n_features=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        """x: [N, n_features] -> logits [N, 1]"""
        return self.net(x)


class Bitter4(BitterMethod):
    @property
    def name(self):
        return "bitter4"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config
        self.W_min = kwargs.get("W_min", max(64, L // 8))
        self.budget = kwargs.get("budget", int(L * 0.9))
        self.gate_every = kwargs.get("gate_every", 8)
        self.calibration_steps = kwargs.get("calibration_steps", 200)
        self.router = None

    def _compute_features(self, past, cache_len, device_str, step):
        """Compute per-token features for the router.

        Returns: [cache_len, 5] tensor of features.
        """
        n_layers = self.mc["n_layers"]

        # Feature 1: V-norm (mean across last 4 layers)
        v_norms = torch.zeros(cache_len, device=device_str)
        n_scoring = min(4, n_layers)
        for li in range(n_layers - n_scoring, n_layers):
            _, v = past[li]
            vn = v[0].norm(dim=-1).mean(dim=0)[:cache_len]
            v_norms[: len(vn)] += vn
        v_norms /= n_scoring

        # Feature 2: recency (normalized position)
        recency = torch.arange(cache_len, device=device_str, dtype=torch.float32)
        recency = recency / max(cache_len - 1, 1)

        # Feature 3: token age (how old relative to decode step)
        age = (step - recency * cache_len) / max(cache_len, 1)

        # Feature 4: distance from end (normalized)
        dist = 1.0 - recency

        # Feature 5: K-norm (mean across last 4 layers)
        k_norms = torch.zeros(cache_len, device=device_str)
        for li in range(n_layers - n_scoring, n_layers):
            k, _ = past[li]
            kn = k[0].norm(dim=-1).mean(dim=0)[:cache_len]
            k_norms[: len(kn)] += kn
        k_norms /= n_scoring

        features = torch.stack([v_norms, recency, age, dist, k_norms], dim=-1)
        return features

    def _train_router(self, model, prefix_ids, continuation_ids, device_str):
        """Train router on oracle labels from a calibration run."""
        # Run dense decode to collect oracle labels
        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values

        # Collect features and oracle labels
        all_features = []
        all_labels = []
        decode_steps = min(self.calibration_steps, continuation_ids.shape[1])

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values

            if step > 0 and step % self.gate_every == 0:
                cache_len = past[0][0].shape[2]
                feats = self._compute_features(past, cache_len, device_str, step)

                # Oracle: keep tokens with highest V-norm (budget fraction)
                v_norms = feats[:, 0]
                threshold_idx = max(1, int(cache_len * (1 - self.budget / self.L)))
                if threshold_idx < cache_len:
                    threshold = torch.kthvalue(v_norms, threshold_idx).values
                    labels = (v_norms >= threshold).float()
                else:
                    labels = torch.ones(cache_len, device=device_str)

                all_features.append(feats.detach())
                all_labels.append(labels.detach())

        if not all_features:
            return

        X = torch.cat(all_features, dim=0)
        y = torch.cat(all_labels, dim=0)

        # Normalize features
        self.feat_mean = X.mean(dim=0)
        self.feat_std = X.std(dim=0) + 1e-8
        X_norm = (X - self.feat_mean) / self.feat_std

        # Train router
        self.router = TierRouter(n_features=5).to(device_str)
        optimizer = torch.optim.Adam(self.router.parameters(), lr=0.01)

        for epoch in range(100):
            logits = self.router(X_norm).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = (self.router(X_norm).squeeze(-1) > 0).float()
            acc = (preds == y).float().mean().item()
        print(f"    Router trained: acc={acc:.3f} loss={loss.item():.4f}")
        self.router.eval()

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2

        # Train router on first portion of data
        if self.router is None:
            with torch.enable_grad():
                self._train_router(model, prefix_ids, continuation_ids, device_str)

        # Now run actual decode
        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []
        actual_pos = prefix_ids.shape[1]
        has_evicted = False
        n_dropped = 0

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]

            t_gate = 0.0
            if step > 0 and step % self.gate_every == 0:
                t0 = time.perf_counter()
                cache_len = past[0][0].shape[2]

                if cache_len > self.budget and self.router is not None:
                    feats = self._compute_features(past, cache_len, device_str, step)
                    X_norm = (feats - self.feat_mean) / self.feat_std
                    keep_probs = torch.sigmoid(self.router(X_norm).squeeze(-1))

                    # Always keep recency window
                    recency_start = max(0, cache_len - self.W_min)
                    keep_probs[recency_start:] = 1.0

                    # Keep tokens above threshold
                    # Adjust threshold to hit budget
                    n_keep = self.budget
                    if keep_probs.sum() > n_keep:
                        sorted_probs, sorted_idx = torch.sort(
                            keep_probs, descending=True
                        )
                        threshold = sorted_probs[min(n_keep, len(sorted_probs) - 1)]
                        keep_mask = keep_probs >= threshold
                    else:
                        keep_mask = keep_probs > 0.5

                    # Enforce recency
                    keep_mask[recency_start:] = True

                    indices = keep_mask.nonzero(as_tuple=True)[0]
                    if len(indices) < cache_len:
                        new_cache = DynamicCache()
                        for li in range(n_layers):
                            k, v = past[li]
                            new_cache.update(
                                k[:, :, indices, :], v[:, :, indices, :], li
                            )
                        past = new_cache
                        n_dropped += cache_len - indices.shape[0]
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
                    tier_full=cache_len,
                    tier_dropped=n_dropped,
                    gate_ms=t_gate,
                    kv_bytes_proxy=cache_len * bpt * n_layers,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def n_thresholds(self):
        return 2  # W_min, budget (gate_every is structural)

    def learned_fraction(self):
        return 0.8  # router decisions are learned
