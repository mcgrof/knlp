#!/usr/bin/env python
"""
BPA v2 Phase 0: Oracle Label + Local Feature Collector

Runs a BPA model on validation data and collects:
- Local-only features (cheap, no far-context computation)
- Oracle boundary_pressure labels (requires full attention, OK offline)

Saves dataset shards to bpa_v2_gate_dataset/ for training the
learned gate in Phase 1.

Feature set (all local-only, per token per layer per head):
  F_local_logits:
    - entropy of local-only logits (per layer output entropy)
  F_local_attn:
    - attention entropy within local window
    - max attention weight within local window (top1 mass)
    - boundary-band concentration (last k positions of local window)
  F_query_state:
    - ||q|| norm per head
    - q norm ratio to EMA (spike detection)
  F_position:
    - normalized position t/T
    - position bucket (0=early, 1=mid, 2=late)

Oracle label:
  y = 1 if boundary_pressure(x,t,l,h) > threshold
  threshold = 75th percentile of boundary_pressure distribution

Usage:
    python scripts/bpa_v2_collect.py [--n-samples 500] [--seq-len 256]
        [--seed 1] [--output-dir bpa_v2_gate_dataset]
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.bpa import BPAConfig, GPT2_BPA


# Feature names in the order they appear in the feature vector
FEATURE_NAMES = [
    "local_attn_entropy",  # attention entropy within local window
    "local_attn_max",  # max attention weight in local window
    "boundary_band_mass",  # attention mass in boundary band
    "query_norm",  # ||q|| per head
    "query_spike",  # q norm / EMA ratio
    "pos_normalized",  # t / T
    "pos_bucket",  # 0=early, 1=mid, 2=late
]
N_FEATURES = len(FEATURE_NAMES)


class FeatureCollector:
    """Collects local-only features and oracle boundary_pressure labels."""

    def __init__(self, model: GPT2_BPA, local_window: int = 64):
        self.model = model
        self.cfg = model.config
        self.local_window = local_window
        self.n_layer = self.cfg.n_layer
        self.n_head = self.cfg.n_head
        self.n_embd = self.cfg.n_embd
        self.head_dim = self.n_embd // self.n_head

        # EMA for query norm spike detection (per layer, per head)
        self.query_norm_ema = torch.ones(self.n_layer, self.n_head)
        self.ema_alpha = 0.1

    @torch.no_grad()
    def collect_batch(
        self,
        idx: torch.Tensor,
        positions: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Collect features and labels for one batch.

        Args:
            idx: [B, T] input tokens
            positions: token positions to sample

        Returns:
            features: [B * n_pos * n_layer * n_head, N_FEATURES]
            labels: [B * n_pos * n_layer * n_head] boundary_pressure values
            metadata: dict with position/layer/head indices
        """
        B, T = idx.shape
        n_pos = len(positions)
        device = idx.device

        # Allocate feature and label tensors
        # Shape: [B, n_pos, n_layer, n_head, N_FEATURES]
        features = torch.zeros(B, n_pos, self.n_layer, self.n_head, N_FEATURES)
        # Shape: [B, n_pos, n_layer, n_head]
        bp_labels = torch.zeros(B, n_pos, self.n_layer, self.n_head)

        # Get embeddings
        tok_emb = self.model.transformer.wte(idx)
        pos_arange = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.model.transformer.wpe(pos_arange)
        x = self.model.transformer.drop(tok_emb + pos_emb)

        # Precompute position features (same for all layers/heads)
        for pi, pos in enumerate(positions):
            pos_norm = pos / max(T - 1, 1)
            # Bucket: early (0-33%), mid (33-66%), late (66-100%)
            if pos_norm < 0.33:
                bucket = 0
            elif pos_norm < 0.66:
                bucket = 1
            else:
                bucket = 2
            features[:, pi, :, :, 5] = pos_norm
            features[:, pi, :, :, 6] = bucket

        # Run through each layer
        for layer_idx, block in enumerate(self.model.transformer.h):
            h = block.ln_1(x)
            attn = block.attn

            # Compute Q, K, V
            qkv = attn.c_attn(h)
            q, k, v = qkv.split(self.n_embd, dim=2)

            # Reshape: [B, H, T, D]
            q_heads = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k_heads = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            # Compute full attention scores for oracle + local features
            scale = 1.0 / (self.head_dim**0.5)
            scores = (q_heads @ k_heads.transpose(-2, -1)) * scale  # [B,H,T,T]

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            attn_probs = F.softmax(scores, dim=-1)  # [B, H, T, T]

            for pi, pos in enumerate(positions):
                if pos < self.local_window:
                    continue

                local_start = max(0, pos - self.local_window + 1)
                local_end = pos + 1
                boundary_band = min(8, self.local_window // 4)

                for hi in range(self.n_head):
                    # Attention weights for this (pos, head)
                    w = attn_probs[:, hi, pos, : pos + 1]  # [B, pos+1]

                    # --- F_local_attn ---

                    # Local attention weights (renormalized)
                    local_w = w[:, local_start:local_end]
                    local_w = local_w / (local_w.sum(dim=-1, keepdim=True) + 1e-10)

                    # 0: local attention entropy
                    ent = -(local_w * (local_w + 1e-10).log()).sum(dim=-1)
                    max_ent = np.log(local_end - local_start + 1e-10)
                    features[:, pi, layer_idx, hi, 0] = ent / (max_ent + 1e-10)

                    # 1: max attention weight in local window
                    features[:, pi, layer_idx, hi, 1] = local_w.max(dim=-1)[0]

                    # 2: boundary band mass (attention at oldest local positions)
                    band_end = local_start + boundary_band
                    band_mass = w[:, local_start:band_end].sum(dim=-1)
                    features[:, pi, layer_idx, hi, 2] = band_mass

                    # --- F_query_state ---

                    # 3: query norm
                    q_norm = q_heads[:, hi, pos, :].norm(dim=-1)  # [B]
                    features[:, pi, layer_idx, hi, 3] = q_norm

                    # 4: query spike (ratio to EMA)
                    ema_val = self.query_norm_ema[layer_idx, hi]
                    features[:, pi, layer_idx, hi, 4] = q_norm / (ema_val + 1e-8)

                    # Update EMA
                    self.query_norm_ema[layer_idx, hi] = (
                        self.ema_alpha * q_norm.mean().item()
                        + (1 - self.ema_alpha) * ema_val
                    )

                    # --- Oracle label: boundary_pressure ---
                    far_end = pos - self.local_window
                    if far_end > 0:
                        bp = attn_probs[:, hi, pos, :far_end].sum(dim=-1)
                        bp_labels[:, pi, layer_idx, hi] = bp

            # Forward through block for next layer
            x, _ = block(x)

        # Flatten: [B * n_pos * n_layer * n_head, N_FEATURES]
        features_flat = features.reshape(-1, N_FEATURES).numpy()
        labels_flat = bp_labels.reshape(-1).numpy()

        # Metadata for later analysis
        meta = {
            "positions": positions,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "batch_size": B,
            "seq_len": T,
        }

        return features_flat, labels_flat, meta


def run_collection(
    n_samples: int = 500,
    batch_size: int = 4,
    seq_len: int = 256,
    n_positions: int = 8,
    local_window: int = 64,
    seed: int = 1,
    output_dir: str = "bpa_v2_gate_dataset",
    shard_size: int = 100000,
) -> Dict:
    """
    Run the full collection pipeline.

    Args:
        n_samples: Number of batches to process
        batch_size: Batch size per sample
        seq_len: Sequence length
        n_positions: Positions to sample per sequence
        local_window: Local attention window size
        seed: Random seed
        output_dir: Output directory for shards
        shard_size: Max examples per shard file
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = BPAConfig(
        block_size=seq_len,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        local_window=local_window,
        chunk_size=32,
        top_b=4,
    )

    print(f"Creating BPA model (124M, seq_len={seq_len})...")
    model = GPT2_BPA(cfg)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.1f}M")

    collector = FeatureCollector(model, local_window=local_window)

    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_labels = []
    shard_idx = 0
    total_examples = 0
    t0 = time.time()

    examples_per_batch = batch_size * n_positions * cfg.n_layer * cfg.n_head

    print(f"\nCollection plan:")
    print(f"  Batches: {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Positions/seq: {n_positions}")
    print(f"  Examples/batch: {examples_per_batch}")
    print(f"  Target total: ~{n_samples * examples_per_batch:,}")
    print(f"  Features: {N_FEATURES} ({', '.join(FEATURE_NAMES)})")
    print()

    for sample_idx in range(n_samples):
        # Generate random input (untrained model, random data is fine
        # for collecting feature/label distributions)
        idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        # Sample positions past local_window
        valid_range = range(local_window, seq_len)
        k = min(n_positions, len(valid_range))
        positions = sorted(
            np.random.choice(list(valid_range), size=k, replace=False).tolist()
        )

        features, labels, meta = collector.collect_batch(idx, positions)

        all_features.append(features)
        all_labels.append(labels)
        total_examples += len(labels)

        # Write shard if buffer is large enough
        if total_examples >= (shard_idx + 1) * shard_size:
            _write_shard(output_dir, shard_idx, all_features, all_labels)
            all_features = []
            all_labels = []
            shard_idx += 1

        if (sample_idx + 1) % 50 == 0 or sample_idx == 0:
            elapsed = time.time() - t0
            rate = (sample_idx + 1) / elapsed
            print(
                f"  [{sample_idx+1}/{n_samples}] "
                f"{total_examples:,} examples, "
                f"{rate:.1f} batches/s"
            )

    # Write remaining data
    if all_features:
        _write_shard(output_dir, shard_idx, all_features, all_labels)
        shard_idx += 1

    elapsed = time.time() - t0
    print(f"\nCollection complete: {total_examples:,} examples in {elapsed:.1f}s")
    print(f"  Shards written: {shard_idx}")

    # Analysis
    report = analyze_dataset(output_dir, shard_idx, cfg, n_positions)
    report["collection"] = {
        "n_samples": n_samples,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_positions": n_positions,
        "local_window": local_window,
        "seed": seed,
        "total_examples": total_examples,
        "n_shards": shard_idx,
        "elapsed_s": round(elapsed, 1),
    }

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")

    def _json_default(x):
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.integer, np.int32, np.int64)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(f"Not JSON serializable: {type(x)}")

    with open(manifest_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\nManifest saved to {manifest_path}")

    return report


def _write_shard(
    output_dir: str,
    shard_idx: int,
    features_list: List[np.ndarray],
    labels_list: List[np.ndarray],
):
    """Write a shard of features and labels to disk."""
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.npz")
    np.savez_compressed(shard_path, features=features, labels=labels)
    print(f"    Shard {shard_idx}: {len(labels):,} examples -> {shard_path}")


def analyze_dataset(
    output_dir: str,
    n_shards: int,
    cfg: BPAConfig,
    n_positions: int,
) -> Dict:
    """Analyze collected dataset for sanity checks."""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    # Load all labels and features
    all_labels = []
    all_features = []
    for i in range(n_shards):
        shard_path = os.path.join(output_dir, f"shard_{i:04d}.npz")
        data = np.load(shard_path)
        all_labels.append(data["labels"])
        all_features.append(data["features"])

    labels = np.concatenate(all_labels)
    features = np.concatenate(all_features)

    report = {}

    # Label statistics
    print(f"\nLabel (boundary_pressure) statistics:")
    print(f"  Total examples: {len(labels):,}")
    print(f"  Mean: {labels.mean():.6f}")
    print(f"  Std:  {labels.std():.6f}")
    print(f"  Min:  {labels.min():.6f}")
    print(f"  Max:  {labels.max():.6f}")

    percentiles = [25, 50, 75, 90, 95]
    pvals = np.percentile(labels, percentiles)
    print(f"  Percentiles:")
    for p, v in zip(percentiles, pvals):
        print(f"    p{p}: {v:.6f}")

    report["label_stats"] = {
        "mean": float(labels.mean()),
        "std": float(labels.std()),
        "min": float(labels.min()),
        "max": float(labels.max()),
        "percentiles": {str(p): float(v) for p, v in zip(percentiles, pvals)},
    }

    # Binary label at p75 threshold
    thr = np.percentile(labels, 75)
    binary = (labels > thr).astype(int)
    pos_rate = binary.mean()
    print(f"\n  Binary labels (threshold=p75={thr:.6f}):")
    print(f"    Positive rate: {pos_rate:.4f}")
    print(f"    Class ratio (neg/pos): {(1-pos_rate)/(pos_rate+1e-10):.2f}")

    report["binary_stats"] = {
        "threshold": float(thr),
        "positive_rate": float(pos_rate),
        "class_ratio": float((1 - pos_rate) / (pos_rate + 1e-10)),
    }

    # Degeneracy check
    is_degenerate = pos_rate < 0.01 or pos_rate > 0.99
    print(f"    Degenerate: {'YES (problem!)' if is_degenerate else 'No (good)'}")
    report["degenerate"] = is_degenerate

    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  {'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*65}")

    feature_stats = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        fvals = features[:, fi]
        fmean = fvals.mean()
        fstd = fvals.std()
        fmin = fvals.min()
        fmax = fvals.max()
        print(f"  {fname:<25} {fmean:>10.4f} {fstd:>10.4f} {fmin:>10.4f} {fmax:>10.4f}")
        feature_stats[fname] = {
            "mean": float(fmean),
            "std": float(fstd),
            "min": float(fmin),
            "max": float(fmax),
        }

    report["feature_stats"] = feature_stats

    # Per-layer label rate (reshape to recover structure)
    # Structure: examples are [B * n_pos * n_layer * n_head]
    # We need to recover per-layer info
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    total = len(labels)
    per_lh = n_layer * n_head
    n_tokens = total // per_lh  # B * n_pos

    if total % per_lh == 0:
        labels_structured = labels.reshape(n_tokens, n_layer, n_head)
        print(f"\nPer-layer label rate (binary, thr=p75):")
        print(f"  {'Layer':<8} {'Mean BP':>10} {'Pos Rate':>10}")
        print(f"  {'-'*30}")

        layer_stats = {}
        for li in range(n_layer):
            layer_labels = labels_structured[:, li, :]
            layer_mean = layer_labels.mean()
            layer_binary = (layer_labels > thr).mean()
            print(f"  L{li:<6} {layer_mean:>10.6f} {layer_binary:>10.4f}")
            layer_stats[f"layer_{li}"] = {
                "mean_bp": float(layer_mean),
                "pos_rate": float(layer_binary),
            }

        report["per_layer"] = layer_stats

        # Per-head label rate (average across layers)
        print(f"\nPer-head label rate (averaged across layers):")
        print(f"  {'Head':<8} {'Mean BP':>10} {'Pos Rate':>10}")
        print(f"  {'-'*30}")

        head_stats = {}
        for hi in range(n_head):
            head_labels = labels_structured[:, :, hi]
            head_mean = head_labels.mean()
            head_binary = (head_labels > thr).mean()
            print(f"  H{hi:<6} {head_mean:>10.6f} {head_binary:>10.4f}")
            head_stats[f"head_{hi}"] = {
                "mean_bp": float(head_mean),
                "pos_rate": float(head_binary),
            }

        report["per_head"] = head_stats

    # Position bucket analysis
    print(f"\nPosition bucket label rate:")
    # bucket is feature index 6
    bucket_feat = features[:, 6]
    print(f"  {'Bucket':<10} {'Count':>10} {'Mean BP':>10} {'Pos Rate':>10}")
    print(f"  {'-'*42}")

    bucket_stats = {}
    for b_val, b_name in [(0, "early"), (1, "mid"), (2, "late")]:
        mask = bucket_feat == b_val
        if mask.sum() > 0:
            b_labels = labels[mask]
            b_mean = b_labels.mean()
            b_pos = (b_labels > thr).mean()
            n_count = mask.sum()
            print(f"  {b_name:<10} {n_count:>10} {b_mean:>10.6f} {b_pos:>10.4f}")
            bucket_stats[b_name] = {
                "count": int(n_count),
                "mean_bp": float(b_mean),
                "pos_rate": float(b_pos),
            }

    report["position_buckets"] = bucket_stats

    # Sanity checks
    print(f"\n{'='*60}")
    print("SANITY CHECKS")
    print(f"{'='*60}")

    checks = {}

    # Check 1: Labels not degenerate
    check1 = not is_degenerate
    print(f"  Labels non-degenerate: {'PASS' if check1 else 'FAIL'}")
    checks["labels_non_degenerate"] = check1

    # Check 2: Feature distributions reasonable (no NaN/Inf)
    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()
    check2 = not has_nan and not has_inf
    print(f"  Features finite: {'PASS' if check2 else 'FAIL'}")
    checks["features_finite"] = check2

    # Check 3: Some variance in labels (not constant)
    check3 = labels.std() > 1e-8
    print(f"  Label variance > 0: {'PASS' if check3 else 'FAIL'}")
    checks["label_variance"] = check3

    # Check 4: Feature variance (each feature has some variance)
    all_varied = True
    for fi, fname in enumerate(FEATURE_NAMES):
        if features[:, fi].std() < 1e-8:
            print(f"  WARNING: {fname} has near-zero variance")
            all_varied = False
    check4 = all_varied
    print(f"  All features varied: {'PASS' if check4 else 'FAIL (see warnings)'}")
    checks["all_features_varied"] = check4

    all_pass = all(checks.values())
    print(f"\n  OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")

    report["sanity_checks"] = checks
    report["all_checks_pass"] = all_pass

    return report


def main():
    parser = argparse.ArgumentParser(
        description="BPA v2 Phase 0: Collect gate training data"
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Number of batches")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument(
        "--n-positions", type=int, default=8, help="Positions per sequence"
    )
    parser.add_argument(
        "--local-window", type=int, default=64, help="Local attention window"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v2_gate_dataset",
        help="Output directory",
    )
    parser.add_argument(
        "--shard-size", type=int, default=100000, help="Examples per shard"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BPA v2 Phase 0: Oracle Label + Local Feature Collector")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  n_samples:    {args.n_samples}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  seq_len:      {args.seq_len}")
    print(f"  n_positions:  {args.n_positions}")
    print(f"  local_window: {args.local_window}")
    print(f"  seed:         {args.seed}")
    print(f"  output_dir:   {args.output_dir}")

    report = run_collection(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_positions=args.n_positions,
        local_window=args.local_window,
        seed=args.seed,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
    )

    if report["all_checks_pass"]:
        print("\nPhase 0 COMPLETE: Dataset ready for Phase 1 gate training.")
    else:
        print("\nPhase 0 INCOMPLETE: Fix issues before proceeding.")


if __name__ == "__main__":
    main()
