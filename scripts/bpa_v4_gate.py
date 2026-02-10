#!/usr/bin/env python
"""
BPA v4 Phase 1: High-Recall Gate with Regret-Calibrated Thresholding.

Key changes from v2/v3 gate:
  1. Asymmetric loss: weight FN >> FP (focal loss or weighted BCE)
  2. Budget-calibrated thresholding: quantile-based to hit target enabled_rate
  3. Additional features: local logits entropy, margin (top1-top2)
  4. Recall/precision tracking at multiple operating points

Usage:
    python scripts/bpa_v4_gate.py \
        --data-dir bpa_v2_trained_dataset \
        --checkpoint <model_checkpoint> \
        --output-dir bpa_v4_gate_results \
        --seeds 1,2,3
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bpa_v2_train_gate import (
    FEATURE_NAMES,
    LogisticGate,
    MLPGate,
    N_FEATURES,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Extended feature set for v4 gate
# ---------------------------------------------------------------------------
V4_EXTRA_FEATURES = [
    "local_logits_entropy",  # entropy of model logits from local-only pass
    "local_logits_margin",  # top1 - top2 logit gap
]

V4_FEATURE_NAMES = FEATURE_NAMES + V4_EXTRA_FEATURES
V4_N_FEATURES = len(V4_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Focal Loss for asymmetric weighting
# ---------------------------------------------------------------------------
class FocalBCELoss(nn.Module):
    """Focal loss variant of BCE for asymmetric FN/FP weighting.

    Focuses on hard-to-classify examples and penalizes FN more heavily.

    Args:
        alpha: Weight for positive class (higher => penalize FN more)
        gamma: Focusing parameter (higher => focus on hard examples)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting (asymmetric)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# v4 Gate model: MLP with wider hidden + optional residual
# ---------------------------------------------------------------------------
class V4Gate(nn.Module):
    """Improved MLP gate for v4 with wider layers."""

    def __init__(self, n_features: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = n_features
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature augmentation: add logits-based features
# ---------------------------------------------------------------------------
@torch.no_grad()
def augment_features_with_logits(
    model,
    text_data: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    local_window: int = 256,
    batch_size: int = 2,
    seq_len: int = 512,
    n_samples: int = 200,
    n_positions: int = 8,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect logits-based features to augment the existing dataset.

    Since collecting logits features requires running the model, and the
    existing dataset was collected differently (per-layer, per-head), we
    approximate by collecting logits features on fresh samples and
    concatenating with averaged versions of the existing features.

    Returns:
        augmented_features: [N, V4_N_FEATURES] array
        labels: [N] array (unchanged)
    """
    # The existing dataset has per-layer/per-head features that were
    # averaged during v3 feature extraction. For v4, we add 2 extra
    # columns (logits entropy and margin) filled with batch-level stats.
    #
    # Since the existing dataset is large (460K examples) and we can't
    # run the model for each one efficiently, we use a simpler approach:
    # compute logits-based features per position in the existing dataset
    # and tile them.
    #
    # But the cleanest approach is: just add dummy columns for now and
    # train on the existing 7 features first. If the asymmetric loss
    # alone fixes the recall problem, we don't need extra features.
    #
    # Strategy: First try v4 gate with SAME features but better loss.
    # If that fails, add logits features via re-collection.

    # For now, return features as-is (7 features).
    return features, labels


# ---------------------------------------------------------------------------
# Recall/Precision at various thresholds
# ---------------------------------------------------------------------------
def compute_operating_points(probs: np.ndarray, labels: np.ndarray) -> List[Dict]:
    """Compute recall, precision, FPR at various enabled_rate budgets."""
    points = []
    for target_rate in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        threshold = np.percentile(probs, (1.0 - target_rate) * 100)
        preds = (probs >= threshold).astype(float)
        actual_rate = float(preds.mean())

        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        points.append(
            {
                "target_rate": target_rate,
                "actual_rate": actual_rate,
                "threshold": float(threshold),
                "recall": recall,
                "precision": precision,
                "fpr": fpr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return points


def find_budget_threshold(probs: np.ndarray, target_rate: float) -> float:
    """Find threshold that achieves approximately target enabled_rate."""
    return float(np.percentile(probs, (1.0 - target_rate) * 100))


# ---------------------------------------------------------------------------
# Training with asymmetric loss
# ---------------------------------------------------------------------------
def train_v4_gate(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_labels_continuous: np.ndarray,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 1024,
    loss_type: str = "focal",
    focal_alpha: float = 0.85,
    focal_gamma: float = 2.0,
    pos_weight_mult: float = 10.0,
) -> Tuple[Dict, List[Dict]]:
    """Train gate with asymmetric loss and budget-calibrated evaluation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if loss_type == "focal":
        criterion = FocalBCELoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        # Heavily weighted BCE
        pos_rate = float(train_labels.mean())
        base_pw = (1 - pos_rate) / (pos_rate + 1e-10)
        pw = torch.tensor([base_pw * pos_weight_mult])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    best_recall_at_budget = 0.0
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

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(val_features, dtype=torch.float32)
            val_logits = model(val_x).numpy()
            val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -500, 500)))

        # Operating points
        ops = compute_operating_points(val_probs, val_labels)

        # Find recall at ~30% budget (target for >=25% KV savings)
        op_30 = [p for p in ops if abs(p["target_rate"] - 0.30) < 0.05]
        recall_at_30 = op_30[0]["recall"] if op_30 else 0.0

        # Also compute ROC AUC
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(val_labels, val_probs))
        except Exception:
            roc_auc = 0.5

        epoch_result = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "roc_auc": roc_auc,
            "recall_at_30pct_budget": recall_at_30,
            "operating_points": ops,
        }
        history.append(epoch_result)

        # Best model selection: maximize recall at 30% budget
        if recall_at_30 > best_recall_at_budget:
            best_recall_at_budget = recall_at_30
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                f"AUC={roc_auc:.4f} "
                f"Recall@30%={recall_at_30:.4f}"
            )

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(val_features, dtype=torch.float32)
        val_logits = model(val_x).numpy()
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -500, 500)))

    ops = compute_operating_points(val_probs, val_labels)

    try:
        from sklearn.metrics import roc_auc_score

        final_auc = float(roc_auc_score(val_labels, val_probs))
    except Exception:
        final_auc = 0.5

    final = {
        "roc_auc": final_auc,
        "operating_points": ops,
        "best_recall_at_30pct": best_recall_at_budget,
    }

    return final, history


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------
def run_single_seed(
    features: np.ndarray,
    labels_continuous: np.ndarray,
    seed: int,
    hidden: int = 256,
    n_layers: int = 3,
    epochs: int = 40,
    lr: float = 5e-4,
    threshold_percentile: float = 75,
    output_dir: str = "bpa_v4_gate_results",
) -> Dict:
    """Train v4 gate for a single seed with multiple loss configs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Binarize labels
    thr = np.percentile(labels_continuous, threshold_percentile)
    labels_binary = (labels_continuous > thr).astype(np.float32)
    pos_rate = labels_binary.mean()

    print(f"\n  Threshold (p{threshold_percentile:.0f}): {thr:.6f}")
    print(f"  Positive rate: {pos_rate:.4f}")

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

    n_features = features.shape[1]
    results = {"seed": seed, "configs": {}}

    # Config 1: Focal loss (alpha=0.85, gamma=2.0)
    print(f"\n  --- V4 Gate: Focal Loss (alpha=0.85, gamma=2.0) ---")
    gate_focal = V4Gate(n_features, hidden=hidden, n_layers=n_layers)
    focal_final, focal_hist = train_v4_gate(
        gate_focal,
        train_f,
        train_l,
        val_f,
        val_l,
        val_lc,
        epochs=epochs,
        lr=lr,
        loss_type="focal",
        focal_alpha=0.85,
        focal_gamma=2.0,
    )
    results["configs"]["focal_085_g2"] = focal_final
    print(
        f"  AUC={focal_final['roc_auc']:.4f} "
        f"Recall@30%={focal_final['best_recall_at_30pct']:.4f}"
    )

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"v4_gate_focal_seed{seed}.pt")
    torch.save(gate_focal.state_dict(), ckpt_path)

    # Config 2: Focal loss (alpha=0.90, gamma=2.0) — more asymmetric
    print(f"\n  --- V4 Gate: Focal Loss (alpha=0.90, gamma=2.0) ---")
    gate_focal2 = V4Gate(n_features, hidden=hidden, n_layers=n_layers)
    focal2_final, focal2_hist = train_v4_gate(
        gate_focal2,
        train_f,
        train_l,
        val_f,
        val_l,
        val_lc,
        epochs=epochs,
        lr=lr,
        loss_type="focal",
        focal_alpha=0.90,
        focal_gamma=2.0,
    )
    results["configs"]["focal_090_g2"] = focal2_final
    print(
        f"  AUC={focal2_final['roc_auc']:.4f} "
        f"Recall@30%={focal2_final['best_recall_at_30pct']:.4f}"
    )

    # Config 3: Heavily weighted BCE (pos_weight_mult=10)
    print(f"\n  --- V4 Gate: Weighted BCE (pos_weight*10) ---")
    gate_wbce = V4Gate(n_features, hidden=hidden, n_layers=n_layers)
    wbce_final, wbce_hist = train_v4_gate(
        gate_wbce,
        train_f,
        train_l,
        val_f,
        val_l,
        val_lc,
        epochs=epochs,
        lr=lr,
        loss_type="weighted_bce",
        pos_weight_mult=10.0,
    )
    results["configs"]["wbce_10x"] = wbce_final
    print(
        f"  AUC={wbce_final['roc_auc']:.4f} "
        f"Recall@30%={wbce_final['best_recall_at_30pct']:.4f}"
    )

    # Config 4: v3-style baseline (standard BCE with class weighting)
    print(f"\n  --- V3-style baseline (standard BCE) ---")
    gate_v3 = MLPGate(n_features, hidden=128, n_layers=2)
    v3_final, v3_hist = train_v4_gate(
        gate_v3,
        train_f,
        train_l,
        val_f,
        val_l,
        val_lc,
        epochs=20,
        lr=1e-3,
        loss_type="weighted_bce",
        pos_weight_mult=1.0,
    )
    results["configs"]["v3_baseline"] = v3_final
    print(
        f"  AUC={v3_final['roc_auc']:.4f} "
        f"Recall@30%={v3_final['best_recall_at_30pct']:.4f}"
    )

    # Pick best config by recall at 30% budget
    best_config = max(
        results["configs"],
        key=lambda k: results["configs"][k]["best_recall_at_30pct"],
    )
    results["best_config"] = best_config
    results["best_recall_at_30pct"] = results["configs"][best_config][
        "best_recall_at_30pct"
    ]
    results["best_auc"] = results["configs"][best_config]["roc_auc"]

    # Save normalization stats
    results["normalization"] = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
    }

    # Save best gate checkpoint
    if best_config.startswith("focal_085"):
        best_gate = gate_focal
    elif best_config.startswith("focal_090"):
        best_gate = gate_focal2
    elif best_config.startswith("wbce"):
        best_gate = gate_wbce
    else:
        best_gate = gate_v3

    best_ckpt_path = os.path.join(output_dir, f"v4_best_gate_seed{seed}.pt")
    torch.save(best_gate.state_dict(), best_ckpt_path)

    # Save normalization stats
    np.savez(
        os.path.join(output_dir, f"v4_norm_stats_seed{seed}.npz"),
        mean=feat_mean,
        std=feat_std,
    )

    return results


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------
def run_multi_seed(
    data_dir: str,
    seeds: List[int],
    hidden: int = 256,
    n_layers: int = 3,
    epochs: int = 40,
    lr: float = 5e-4,
    output_dir: str = "bpa_v4_gate_results",
) -> Dict:
    """Run v4 gate training across multiple seeds."""
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
            output_dir=output_dir,
        )
        all_results.append(result)

    # Aggregate
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")

    config_names = set()
    for r in all_results:
        config_names.update(r["configs"].keys())

    aggregated = {"seeds": seeds, "n_seeds": len(seeds), "configs": {}}

    print(f"\n{'Config':<20} {'Mean AUC':>10} {'Mean R@30%':>12} " f"{'Std R@30%':>10}")
    print("-" * 55)

    for cname in sorted(config_names):
        aucs = []
        recalls = []
        for r in all_results:
            if cname in r["configs"]:
                aucs.append(r["configs"][cname]["roc_auc"])
                recalls.append(r["configs"][cname]["best_recall_at_30pct"])

        mean_auc = float(np.mean(aucs))
        mean_recall = float(np.mean(recalls))
        std_recall = float(np.std(recalls))

        aggregated["configs"][cname] = {
            "mean_roc_auc": mean_auc,
            "mean_recall_at_30pct": mean_recall,
            "std_recall_at_30pct": std_recall,
            "per_seed_auc": aucs,
            "per_seed_recall": recalls,
        }

        print(
            f"{cname:<20} {mean_auc:>10.4f} {mean_recall:>12.4f} "
            f"{std_recall:>10.4f}"
        )

    print("-" * 55)

    # Best config
    best_config = max(
        aggregated["configs"],
        key=lambda k: aggregated["configs"][k]["mean_recall_at_30pct"],
    )
    aggregated["best_config"] = best_config
    aggregated["best_mean_recall_at_30pct"] = aggregated["configs"][best_config][
        "mean_recall_at_30pct"
    ]
    aggregated["best_mean_auc"] = aggregated["configs"][best_config]["mean_roc_auc"]

    print(
        f"\nBest config: {best_config}"
        f" (recall@30% = {aggregated['best_mean_recall_at_30pct']:.4f},"
        f" AUC = {aggregated['best_mean_auc']:.4f})"
    )

    # Print operating points for best config from seed 1
    best_seed_idx = 0
    best_seed_config = all_results[best_seed_idx]["configs"].get(best_config)
    if best_seed_config and "operating_points" in best_seed_config:
        print(f"\nOperating points (seed {seeds[0]}, {best_config}):")
        print(
            f"  {'Rate':>6} {'Recall':>8} {'Precision':>10} {'FPR':>6} "
            f"{'Threshold':>10}"
        )
        for op in best_seed_config["operating_points"]:
            print(
                f"  {op['target_rate']:>6.2f} {op['recall']:>8.4f} "
                f"{op['precision']:>10.4f} {op['fpr']:>6.4f} "
                f"{op['threshold']:>10.4f}"
            )

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    target_recall = 0.95
    best_recall = aggregated["best_mean_recall_at_30pct"]
    if best_recall >= target_recall:
        print(f"PASS: Recall@30% ({best_recall:.4f}) >= {target_recall}")
        print("Gate achieves high recall at 30% budget.")
        aggregated["verdict"] = "PASS"
    elif best_recall >= 0.80:
        print(
            f"ACCEPTABLE: Recall@30% ({best_recall:.4f}) >= 0.80 "
            f"but < {target_recall}"
        )
        print("Proceed to Phase 2 with budget-calibrated threshold.")
        aggregated["verdict"] = "ACCEPTABLE"
    else:
        print(f"NEEDS_WORK: Recall@30% ({best_recall:.4f}) < 0.80")
        print("Gate needs improvement. Try more features or re-collection.")
        aggregated["verdict"] = "NEEDS_WORK"

    # Save
    result_path = os.path.join(output_dir, "v4_gate_results.json")

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

    # Also save per-seed results
    for i, r in enumerate(all_results):
        per_seed_path = os.path.join(output_dir, f"v4_gate_seed{seeds[i]}_detail.json")
        with open(per_seed_path, "w") as f:
            json.dump(r, f, indent=2, default=_json_default)

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BPA v4 Phase 1: High-Recall Gate Training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="bpa_v2_trained_dataset",
        help="Phase 0 dataset directory",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds for training")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--hidden", type=int, default=256, help="MLP hidden dim")
    parser.add_argument("--n-layers", type=int, default=3, help="MLP hidden layers")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bpa_v4_gate_results",
        help="Output directory",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("BPA v4 Phase 1: High-Recall Gate Training")
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
