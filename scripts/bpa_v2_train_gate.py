#!/usr/bin/env python
"""
BPA v2 Phase 1: Train Learned Gate (Supervised Imitation of Oracle)

Trains a small gate model to predict y_oracle (boundary_pressure > threshold)
from local-only features collected in Phase 0.

Models:
  - Logistic regression baseline (mandatory)
  - MLP with 1-2 hidden layers, width 64-256

Loss: BCE with class-imbalance weighting
Metrics: AUC ROC (primary), PR-AUC, ECE, stability across seeds

Acceptance: AUC >= 0.75 overall, no severe position-bucket collapse.
Stop if AUC < 0.65.

Usage:
    python scripts/bpa_v2_train_gate.py [--data-dir bpa_v2_gate_dataset]
        [--seeds 1,2,3] [--epochs 20] [--hidden 128]
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Feature names (must match bpa_v2_collect.py)
FEATURE_NAMES = [
    "local_attn_entropy",
    "local_attn_max",
    "boundary_band_mass",
    "query_norm",
    "query_spike",
    "pos_normalized",
    "pos_bucket",
]
N_FEATURES = len(FEATURE_NAMES)


class LogisticGate(nn.Module):
    """Logistic regression baseline gate."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPGate(nn.Module):
    """MLP gate with 1-2 hidden layers."""

    def __init__(self, n_features: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all shards from a Phase 0 dataset directory."""
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.json found in {data_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_shards = manifest["collection"]["n_shards"]
    all_features = []
    all_labels = []

    for i in range(n_shards):
        shard_path = os.path.join(data_dir, f"shard_{i:04d}.npz")
        data = np.load(shard_path)
        all_features.append(data["features"])
        all_labels.append(data["labels"])

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Loaded {len(labels):,} examples from {n_shards} shards")
    return features, labels


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        bin_weight = mask.sum() / total
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_gate(
    model: nn.Module,
    features: np.ndarray,
    labels_binary: np.ndarray,
    labels_continuous: np.ndarray,
    feature_bucket_idx: int = 6,
) -> Dict:
    """Evaluate a gate model on held-out data."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    from scipy import stats as scipy_stats

    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model(x).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    results = {}

    # ROC AUC
    if len(np.unique(labels_binary)) >= 2:
        results["roc_auc"] = float(roc_auc_score(labels_binary, probs))
    else:
        results["roc_auc"] = 0.5

    # PR AUC
    if len(np.unique(labels_binary)) >= 2:
        precision, recall, _ = precision_recall_curve(labels_binary, probs)
        results["pr_auc"] = float(auc(recall, precision))
    else:
        results["pr_auc"] = 0.5

    # Spearman correlation with continuous labels
    corr, _ = scipy_stats.spearmanr(probs, labels_continuous)
    results["spearman_r"] = float(corr) if not np.isnan(corr) else 0.0

    # ECE
    results["ece"] = compute_ece(probs, labels_binary)

    # Per-bucket AUC
    buckets = features[:, int(feature_bucket_idx)]
    bucket_aucs = {}
    for b_val, b_name in [(0, "early"), (1, "mid"), (2, "late")]:
        mask = buckets == b_val
        if mask.sum() > 100 and len(np.unique(labels_binary[mask])) >= 2:
            bucket_aucs[b_name] = float(roc_auc_score(labels_binary[mask], probs[mask]))
        else:
            bucket_aucs[b_name] = None
    results["bucket_auc"] = bucket_aucs

    return results


def train_gate(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_labels_continuous: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 1024,
    pos_weight: float = 1.0,
) -> Tuple[Dict, List[Dict]]:
    """Train a gate model and return final + per-epoch metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Create DataLoader
    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate
        eval_result = evaluate_gate(
            model, val_features, val_labels, val_labels_continuous
        )
        eval_result["epoch"] = epoch
        eval_result["train_loss"] = avg_loss
        history.append(eval_result)

        if eval_result["roc_auc"] > best_auc:
            best_auc = eval_result["roc_auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                f"AUC={eval_result['roc_auc']:.4f} "
                f"PR-AUC={eval_result['pr_auc']:.4f} "
                f"ECE={eval_result['ece']:.4f}"
            )

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    final = evaluate_gate(model, val_features, val_labels, val_labels_continuous)
    return final, history


def run_single_seed(
    features: np.ndarray,
    labels_continuous: np.ndarray,
    seed: int,
    hidden: int = 128,
    n_layers: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    threshold_percentile: float = 75,
) -> Dict:
    """Run gate training for a single seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Binarize labels
    thr = np.percentile(labels_continuous, threshold_percentile)
    labels_binary = (labels_continuous > thr).astype(np.float32)
    pos_rate = labels_binary.mean()
    pos_weight = (1 - pos_rate) / (pos_rate + 1e-10)

    print(f"\n  Threshold (p{threshold_percentile:.0f}): {thr:.6f}")
    print(f"  Positive rate: {pos_rate:.4f}, pos_weight: {pos_weight:.2f}")

    # Standardize features
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-8
    features_norm = (features - feat_mean) / feat_std

    # Train/val split (80/20)
    n = len(features_norm)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = perm[:split]
    val_idx = perm[split:]

    train_f = features_norm[train_idx]
    train_l = labels_binary[train_idx]
    val_f = features_norm[val_idx]
    val_l = labels_binary[val_idx]
    val_lc = labels_continuous[val_idx]

    results = {"seed": seed}

    # 1. Logistic regression baseline
    print(f"\n  --- Logistic Regression ---")
    lr_model = LogisticGate(N_FEATURES)
    lr_final, lr_hist = train_gate(
        lr_model,
        train_f,
        train_l,
        val_f,
        val_l,
        val_lc,
        epochs=epochs,
        lr=lr,
        pos_weight=pos_weight,
    )
    results["logistic"] = lr_final
    print(
        f"  Final: AUC={lr_final['roc_auc']:.4f} "
        f"PR-AUC={lr_final['pr_auc']:.4f} "
        f"ECE={lr_final['ece']:.4f}"
    )

    # 2. MLP gate
    print(f"\n  --- MLP Gate (hidden={hidden}, layers={n_layers}) ---")
    mlp_model = MLPGate(N_FEATURES, hidden=hidden, n_layers=n_layers)
    mlp_final, mlp_hist = train_gate(
        mlp_model,
        train_f,
        train_l,
        val_f,
        val_l,
        val_lc,
        epochs=epochs,
        lr=lr,
        pos_weight=pos_weight,
    )
    results["mlp"] = mlp_final
    print(
        f"  Final: AUC={mlp_final['roc_auc']:.4f} "
        f"PR-AUC={mlp_final['pr_auc']:.4f} "
        f"ECE={mlp_final['ece']:.4f}"
    )

    # 3. Larger MLP (if small one didn't hit 0.75)
    if mlp_final["roc_auc"] < 0.75 and hidden < 256:
        print(f"\n  --- MLP Gate Large (hidden=256, layers=2) ---")
        mlp_large = MLPGate(N_FEATURES, hidden=256, n_layers=2)
        large_final, large_hist = train_gate(
            mlp_large,
            train_f,
            train_l,
            val_f,
            val_l,
            val_lc,
            epochs=epochs * 2,
            lr=lr * 0.5,
            pos_weight=pos_weight,
        )
        results["mlp_large"] = large_final
        print(
            f"  Final: AUC={large_final['roc_auc']:.4f} "
            f"PR-AUC={large_final['pr_auc']:.4f}"
        )

    # Save best model checkpoint
    best_name = max(
        [k for k in results if k != "seed"],
        key=lambda k: results[k]["roc_auc"],
    )
    best_auc = results[best_name]["roc_auc"]
    results["best_model"] = best_name
    results["best_auc"] = best_auc

    # Save normalization stats for deployment
    results["normalization"] = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
    }

    return results


def run_multi_seed(
    data_dir: str,
    seeds: List[int],
    hidden: int = 128,
    n_layers: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    output_dir: str = "bpa_v2_gate_results",
) -> Dict:
    """Run gate training across multiple seeds."""
    # First collect data if not already done
    features, labels = load_dataset(data_dir)

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        result = run_single_seed(
            features,
            labels,
            seed=seed,
            hidden=hidden,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
        )
        all_results.append(result)

    # Aggregate
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")

    model_types = set()
    for r in all_results:
        model_types.update(
            k for k in r if k not in ("seed", "best_model", "best_auc", "normalization")
        )

    aggregated = {"seeds": seeds, "n_seeds": len(seeds), "models": {}}

    print(
        f"\n{'Model':<20} {'Mean AUC':>10} {'Std AUC':>10} {'Mean PR-AUC':>12} {'Mean ECE':>10}"
    )
    print("-" * 65)

    for mtype in sorted(model_types):
        aucs = [r[mtype]["roc_auc"] for r in all_results if mtype in r]
        pr_aucs = [r[mtype]["pr_auc"] for r in all_results if mtype in r]
        eces = [r[mtype]["ece"] for r in all_results if mtype in r]

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        mean_pr = float(np.mean(pr_aucs))
        mean_ece = float(np.mean(eces))

        aggregated["models"][mtype] = {
            "mean_roc_auc": mean_auc,
            "std_roc_auc": std_auc,
            "mean_pr_auc": mean_pr,
            "mean_ece": mean_ece,
            "per_seed_auc": aucs,
        }

        print(
            f"{mtype:<20} {mean_auc:>10.4f} {std_auc:>10.4f} {mean_pr:>12.4f} {mean_ece:>10.4f}"
        )

    print("-" * 65)

    # Best model
    best_type = max(
        aggregated["models"],
        key=lambda k: aggregated["models"][k]["mean_roc_auc"],
    )
    best_auc = aggregated["models"][best_type]["mean_roc_auc"]
    aggregated["best_model"] = best_type
    aggregated["best_mean_auc"] = best_auc

    # Per-bucket analysis for best model
    print(f"\nBest model: {best_type} (mean AUC = {best_auc:.4f})")

    bucket_analysis = {}
    for r in all_results:
        if best_type in r and "bucket_auc" in r[best_type]:
            for bucket, val in r[best_type]["bucket_auc"].items():
                if val is not None:
                    bucket_analysis.setdefault(bucket, []).append(val)

    if bucket_analysis:
        print(f"\nPer-bucket AUC for {best_type}:")
        for bucket in ["early", "mid", "late"]:
            if bucket in bucket_analysis:
                vals = bucket_analysis[bucket]
                print(f"  {bucket}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
                aggregated["bucket_auc_" + bucket] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if best_auc >= 0.75:
        print(f"PASS: Best AUC ({best_auc:.4f}) >= 0.75")
        print("Proceed to Phase 2 (speculative local-first pipeline)")
        aggregated["verdict"] = "PASS"
    elif best_auc >= 0.65:
        print(f"MARGINAL: Best AUC ({best_auc:.4f}) in [0.65, 0.75)")
        print("Gate has some predictive power but below target.")
        print("Consider: more features, longer training, or accept marginal gate.")
        aggregated["verdict"] = "MARGINAL"
    else:
        print(f"FAIL: Best AUC ({best_auc:.4f}) < 0.65")
        print("Gate cannot predict boundary_pressure from local features.")
        print("BPA v2 learned gate approach is not viable.")
        aggregated["verdict"] = "FAIL"

    # Check for seed instability
    if best_type in aggregated["models"]:
        std = aggregated["models"][best_type]["std_roc_auc"]
        if std > 0.05:
            print(f"\nWARNING: High seed variance (std={std:.4f}). Results unstable.")

    # Save results
    result_path = os.path.join(output_dir, "phase1_results.json")

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

    with open(result_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=_json_default)
    print(f"\nResults saved to {result_path}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="BPA v2 Phase 1: Train Learned Gate")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="bpa_v2_gate_dataset",
        help="Phase 0 dataset directory",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds for training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--hidden", type=int, default=128, help="MLP hidden dim")
    parser.add_argument("--n-layers", type=int, default=2, help="MLP hidden layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v2_gate_results",
        help="Output directory",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("BPA v2 Phase 1: Train Learned Gate")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  data_dir:  {args.data_dir}")
    print(f"  seeds:     {seeds}")
    print(f"  epochs:    {args.epochs}")
    print(f"  hidden:    {args.hidden}")
    print(f"  n_layers:  {args.n_layers}")
    print(f"  lr:        {args.lr}")

    run_multi_seed(
        data_dir=args.data_dir,
        seeds=seeds,
        hidden=args.hidden,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
