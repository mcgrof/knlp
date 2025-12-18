#!/usr/bin/env python3
"""
FIM-Guided Fraud Detection Benchmark (v2 - Backward Hooks)

Uses backward hooks for zero-overhead importance tracking instead of
expensive autograd.grad() calls. Throughput should match baseline.

Key changes from v1:
- Uses backward hooks (zero extra compute)
- Step-based replication updates (not epoch boundaries)
- Class-conditional weighting (fraud-focused)

Usage:
    python benchmark_fim.py --time 300
    python benchmark_fim.py --time 300 --ablation
    python benchmark_fim.py --time 3600 --only fim_importance
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from tqdm import tqdm

# Local imports
from fim_importance import FIMConfig, NodeImportanceTracker, create_importance_tracker
from fim_sampler import FIMCppPageBatchSampler
from page_batch_sampler import CppPageBatchSampler

# Optional W&B
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

PAGE_SIZE = 4096
FEATURE_BYTES = 68
NODES_PER_PAGE = PAGE_SIZE // FEATURE_BYTES  # ~60 nodes per 4KB page


# =============================================================================
# Read Amplification Computation (consistent across all samplers)
# =============================================================================


@dataclass
class RAMetrics:
    """Read amplification metrics for a batch."""

    pages_touched: int  # Unique pages accessed
    loaded_nodes: int  # Total nodes whose features were gathered
    supervised_nodes: int  # Nodes contributing to loss
    bytes_read: int  # pages_touched * PAGE_SIZE
    ra_fetch: float  # bytes_read / (loaded_nodes * FEATURE_BYTES) - locality metric
    ra_signal: float  # bytes_read / (supervised_nodes * FEATURE_BYTES) - training efficiency


def compute_ra_metrics(
    loaded_node_ids: np.ndarray,
    supervised_node_count: int,
    node_page_id: np.ndarray,
) -> RAMetrics:
    """
    Compute read amplification metrics from loaded node IDs.

    Args:
        loaded_node_ids: Global node IDs whose features were gathered (n_id for
                        NeighborLoader, batch.nodes for page-batch)
        supervised_node_count: Number of nodes contributing to loss
        node_page_id: Precomputed page ID for each node (node_order * FEATURE_BYTES // PAGE_SIZE)

    Returns:
        RAMetrics with both RA_fetch and RA_signal
    """
    # Compute pages touched
    pages = node_page_id[loaded_node_ids]
    unique_pages = np.unique(pages)
    pages_touched = len(unique_pages)

    # Compute bytes
    bytes_read = pages_touched * PAGE_SIZE
    loaded_nodes = len(loaded_node_ids)

    # RA_fetch: locality metric (primary)
    # How scattered are feature accesses? Perfect locality = 1.0x
    bytes_requested_fetch = loaded_nodes * FEATURE_BYTES
    ra_fetch = bytes_read / bytes_requested_fetch if bytes_requested_fetch > 0 else 0.0

    # RA_signal: training efficiency metric (secondary)
    # How much I/O per supervised signal? Inflated by label sparsity
    bytes_requested_signal = supervised_node_count * FEATURE_BYTES
    ra_signal = bytes_read / bytes_requested_signal if bytes_requested_signal > 0 else 0.0

    return RAMetrics(
        pages_touched=pages_touched,
        loaded_nodes=loaded_nodes,
        supervised_nodes=supervised_node_count,
        bytes_read=bytes_read,
        ra_fetch=ra_fetch,
        ra_signal=ra_signal,
    )


def validate_labels(y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray):
    """
    Validate that masks contain only labels {0, 1} for binary fraud detection.

    Raises AssertionError if masks contain unexpected labels.
    """
    train_labels = set(np.unique(y[train_mask]))
    val_labels = set(np.unique(y[val_mask]))
    test_labels = set(np.unique(y[test_mask]))

    assert train_labels <= {0, 1}, f"Train mask contains unexpected labels: {train_labels}"
    assert val_labels <= {0, 1}, f"Val mask contains unexpected labels: {val_labels}"
    assert test_labels <= {0, 1}, f"Test mask contains unexpected labels: {test_labels}"

    print(f"  Label validation: train={train_labels}, val={val_labels}, test={test_labels} ✓")


@dataclass
class FIMBenchmarkConfig:
    """Configuration for FIM benchmark."""

    # Training
    time_limit: float = 300.0
    lr: float = 0.003
    weight_decay: float = 5e-7
    hidden_channels: int = 128
    num_classes: int = 2  # Binary: 0=normal, 1=fraud (classes 2,3 not in train/val/test)

    # Class imbalance handling
    use_weighted_loss: bool = False
    pos_weight_scale: float = 1.0  # Scale factor for fraud class weight
    pos_oversample_prob: float = 0.0  # Probability of injecting positive page per batch

    # Evaluation checkpoints (for learning curves)
    eval_interval_steps: int = 2000  # Evaluate every N steps (0 to disable)

    # Page-aware settings
    pages_per_batch: int = 32
    boundary_budget: float = 0.2
    boundary_hops: int = 1  # 1 for 1-layer GNN, 2 for 2-layer GNN

    # FIM settings (v2)
    beta: float = 0.01
    lambda_pagepair: float = 0.5
    replication_budget: float = 0.005
    replication_interval_steps: int = 5000
    importance_root: str = "fourth_root"  # raw, sqrt, fourth_root

    # Paths
    data_dir: str = ".."
    layout_path: str = "../layout_metis_bfs.npz"

    # W&B
    wandb_project: str = "gnn-fraud-detection"
    use_wandb: bool = True

    # Variants
    variants: List[str] = field(
        default_factory=lambda: [
            "baseline",
            "fim_importance",
            "fim_replication",
        ]
    )


@dataclass
class FraudMetrics:
    """Fraud detection metrics."""

    f1: float = 0.0
    pr_auc: float = 0.0  # Same as Average Precision (AP)
    precision_at_05pct: float = 0.0
    recall_at_1pct: float = 0.0
    recall_at_k: float = 0.0  # Recall when K = num_positives
    threshold: float = 0.5
    num_positives: int = 0
    num_total: int = 0
    lift_over_random: float = 0.0  # PR-AUC / prevalence


# =============================================================================
# Data Loading
# =============================================================================


def load_dgraphfin(
    data_dir: str,
) -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load DGraphFin dataset."""
    npz_path = os.path.join(data_dir, "dgraphfin.npz")
    print(f"Loading DGraphFin from {npz_path}...")

    npz_data = np.load(npz_path)

    x = torch.from_numpy(npz_data["x"]).float()
    y = torch.from_numpy(npz_data["y"]).long()
    edge_index = torch.from_numpy(npz_data["edge_index"].T).long()

    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    num_nodes = x.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[npz_data["train_mask"]] = True
    val_mask[npz_data["valid_mask"]] = True
    test_mask[npz_data["test_mask"]] = True

    data = Data(x=x, y=y, edge_index=edge_index)

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Train/Val/Test: {train_mask.sum():,}/{val_mask.sum():,}/{test_mask.sum():,}")

    # Detailed positive rate analysis per split
    y_np = y.numpy()
    fraud_total = (y_np == 1).sum()
    print(f"\n  Class distribution (fraud = class 1):")
    print(f"    Total fraud: {fraud_total:,} ({100*fraud_total/num_nodes:.3f}%)")

    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        n_split = mask.sum()
        n_pos = (y_np[mask] == 1).sum()
        pos_rate = n_pos / n_split if n_split > 0 else 0
        print(f"    {name:5s}: {n_pos:,} positives / {n_split:,} = {100*pos_rate:.3f}%")

    # Random baseline PR-AUC = prevalence
    train_pos_rate = (y_np[train_mask] == 1).sum() / train_mask.sum()
    print(f"\n  Random baseline PR-AUC (=prevalence): {train_pos_rate:.4f}")

    # Validate that masks contain only {0, 1} (binary fraud detection)
    validate_labels(y_np, train_mask, val_mask, test_mask)

    return data, train_mask, val_mask, test_mask, 2  # Binary classification


def compute_class_weights(
    y: torch.Tensor, train_mask: np.ndarray, num_classes: int, pos_weight_scale: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights for weighted loss based on inverse frequency.

    For fraud detection: class 0 (normal) vs class 1 (fraud).
    Uses sqrt dampening to avoid extreme weights.

    Args:
        y: Labels tensor
        train_mask: Boolean mask for training nodes
        num_classes: Number of classes (should be 2 for DGraphFin)
        pos_weight_scale: Additional scale factor for fraud class (class 1)

    Returns:
        Class weights tensor [num_classes]
    """
    y_train = y.numpy()[train_mask]
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)

    n_total = len(y_train)

    # Compute weights using sqrt dampening (avoids extreme weights)
    # weight_i = sqrt(n_total / count_i), then normalize so mean = 1
    weights = np.sqrt(n_total / (counts + 1e-8))
    weights = weights / weights.mean()

    # Apply additional scale to fraud class (class 1) if requested
    if pos_weight_scale != 1.0:
        weights[1] *= pos_weight_scale

    print(f"\n  Class weights (sqrt dampened):")
    for c in range(num_classes):
        print(f"    Class {c}: count={int(counts[c]):,}, weight={weights[c]:.3f}")

    return torch.from_numpy(weights).float()


def load_layout(layout_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-computed page layout.

    Returns:
        page_id: METIS partition ID for each node
        node_order: Storage position for each node (node_order[node_id] = storage_pos)
    """
    print(f"Loading layout from {layout_path}...")
    layout_data = np.load(layout_path)

    if "page_id" in layout_data:
        page_id = layout_data["page_id"]
    elif "node_to_page" in layout_data:
        page_id = layout_data["node_to_page"]
    else:
        raise ValueError(f"Layout file missing page_id: {layout_path}")

    if "node_order" in layout_data:
        # node_order[node_id] = storage_position (direct mapping)
        node_order = layout_data["node_order"]
    else:
        # Fallback: compute storage order from METIS partitions
        sorted_nodes = np.argsort(page_id, kind="stable")
        node_order = np.empty_like(sorted_nodes)
        node_order[sorted_nodes] = np.arange(len(sorted_nodes))

    print(f"  METIS partitions: {len(np.unique(page_id)):,}")

    return page_id, node_order


# =============================================================================
# Model
# =============================================================================


class GraphSAGEWithHidden(torch.nn.Module):
    """GraphSAGE that exposes intermediate hidden states for FIM hooks."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self._hidden = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor:
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        self._hidden = h1

        h2 = self.conv2(h1, edge_index)
        out = F.log_softmax(h2, dim=-1)

        if return_hidden:
            return out, h1
        return out

    def get_hidden(self) -> Optional[torch.Tensor]:
        """Get last computed hidden state."""
        return self._hidden


# =============================================================================
# Evaluation
# =============================================================================


def compute_fraud_metrics(
    probs: np.ndarray, y_true: np.ndarray, threshold: Optional[float] = None
) -> FraudMetrics:
    """Compute fraud detection metrics."""
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)

    f1_scores = (
        2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)
    )

    if threshold is None:
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    y_pred = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = auc(recall_curve, precision_curve)

    n = len(probs)
    total_pos = int(y_true.sum())
    prevalence = total_pos / n if n > 0 else 0
    sorted_idx = np.argsort(probs)[::-1]
    sorted_y = y_true[sorted_idx]

    # Precision@0.5%
    k_05 = max(1, int(n * 0.005))
    tp_at_05 = sorted_y[:k_05].sum()
    p_at_05 = tp_at_05 / k_05

    # Recall@1%
    k_1 = max(1, int(n * 0.01))
    tp_at_1 = sorted_y[:k_1].sum()
    r_at_1 = tp_at_1 / total_pos if total_pos > 0 else 0

    # Recall@K where K = number of positives (ideal = 1.0 if perfect ranking)
    k_pos = max(1, total_pos)
    tp_at_k = sorted_y[:k_pos].sum()
    recall_at_k = tp_at_k / total_pos if total_pos > 0 else 0

    # Lift over random baseline
    lift = pr_auc / prevalence if prevalence > 0 else 0

    return FraudMetrics(
        f1=f1,
        pr_auc=pr_auc,
        precision_at_05pct=p_at_05,
        recall_at_1pct=r_at_1,
        recall_at_k=recall_at_k,
        threshold=threshold,
        num_positives=total_pos,
        num_total=n,
        lift_over_random=lift,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Data,
    mask: np.ndarray,
    device: torch.device,
    threshold: Optional[float] = None,
) -> FraudMetrics:
    """Evaluate model on masked nodes."""
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    out = model(x, edge_index)
    probs = torch.exp(out[mask, 1]).cpu().numpy()
    y_true = (data.y[mask].numpy() == 1).astype(int)

    return compute_fraud_metrics(probs, y_true, threshold)


# =============================================================================
# W&B Logger
# =============================================================================


class FIMWandBLogger:
    """W&B logger with FIM-specific metrics."""

    def __init__(self, config: FIMBenchmarkConfig, variant: str):
        self.config = config
        self.variant = variant
        self.enabled = WANDB_AVAILABLE and config.use_wandb
        self.step = 0

    def __enter__(self):
        if self.enabled:
            # Include weighted loss in run name if enabled
            run_suffix = "_weighted" if self.config.use_weighted_loss else ""
            wandb.init(
                project=self.config.wandb_project,
                name=f"{self.variant}{run_suffix}-{self.config.time_limit:.0f}s-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "variant": self.variant,
                    "time_limit_s": self.config.time_limit,
                    "lr": self.config.lr,
                    "hidden_channels": self.config.hidden_channels,
                    "num_classes": self.config.num_classes,
                    "pages_per_batch": self.config.pages_per_batch,
                    "boundary_budget": self.config.boundary_budget,
                    "boundary_hops": self.config.boundary_hops,
                    "use_weighted_loss": self.config.use_weighted_loss,
                    "pos_weight_scale": self.config.pos_weight_scale,
                    "pos_oversample_prob": self.config.pos_oversample_prob,
                    "eval_interval_steps": self.config.eval_interval_steps,
                    "beta": self.config.beta,
                    "lambda_pagepair": self.config.lambda_pagepair,
                    "replication_budget": self.config.replication_budget,
                    "replication_interval_steps": self.config.replication_interval_steps,
                    "importance_root": self.config.importance_root,
                },
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            wandb.finish()

    def log_batch(self, metrics: Dict):
        """Log batch metrics."""
        if not self.enabled:
            return
        self.step += 1
        wandb.log(metrics, step=self.step)

    def log_eval(self, val_metrics: "FraudMetrics", test_metrics: "FraudMetrics"):
        """Log evaluation checkpoint metrics for learning curves."""
        if not self.enabled:
            return
        wandb.log(
            {
                "eval/val_pr_auc": val_metrics.pr_auc,
                "eval/val_f1": val_metrics.f1,
                "eval/val_recall_at_k": val_metrics.recall_at_k,
                "eval/val_lift": val_metrics.lift_over_random,
                "eval/test_pr_auc": test_metrics.pr_auc,
                "eval/test_f1": test_metrics.f1,
                "eval/test_recall_at_k": test_metrics.recall_at_k,
                "eval/test_lift": test_metrics.lift_over_random,
            },
            step=self.step,
        )

    def log_final(self, results: Dict):
        """Log final results."""
        if not self.enabled:
            return

        final_metrics = {
            "test/f1": results.get("f1", 0),
            "test/pr_auc": results.get("pr_auc", 0),
            "test/p_at_0.5%": results.get("p_at_0.5%", 0),
            "test/r_at_1%": results.get("r_at_1%", 0),
            "test/recall_at_k": results.get("recall_at_k", 0),
            "test/lift_over_random": results.get("lift_over_random", 0),
            "final/read_amplification": results.get("read_amplification", 0),
            "final/epochs": results.get("epochs", 0),
            "final/batches": results.get("batches", 0),
            "final/weighted_loss": results.get("weighted_loss", False),
        }

        # Batch composition stats
        if "batch_stats" in results:
            bs = results["batch_stats"]
            final_metrics.update(
                {
                    "batch_stats/avg_train_nodes": bs.get("avg_train_nodes", 0),
                    "batch_stats/avg_pos_nodes": bs.get("avg_pos_nodes", 0),
                    "batch_stats/frac_with_positives": bs.get("frac_with_positives", 0),
                }
            )

        if "fim_stats" in results:
            fim = results["fim_stats"]
            final_metrics.update(
                {
                    "final/fim_updates": fim.get("total_updates", 0),
                    "final/batches_with_fraud": fim.get("batches_with_fraud", 0),
                    "final/node_importance_coverage": fim.get("node_coverage", 0),
                    "final/replicated_nodes": fim.get("replicated_nodes", 0),
                    "final/saved_reads_total": fim.get("saved_reads_total", 0),
                }
            )

        wandb.log(final_metrics)
        for key, value in final_metrics.items():
            wandb.run.summary[key] = value


# =============================================================================
# Training Variants
# =============================================================================


def train_neighbor_sampler(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    node_order: np.ndarray,
    config: FIMBenchmarkConfig,
    device: torch.device,
    wandb_logger: Optional[FIMWandBLogger] = None,
) -> Dict:
    """Train with standard NeighborLoader (PyG baseline)."""
    print("\n" + "=" * 70)
    variant_name = "NEIGHBOR-SAMPLER: PyG NeighborLoader [10, 5]"
    if config.use_weighted_loss:
        variant_name += " + Weighted Loss"
    print(variant_name)
    print("=" * 70)

    # Create boolean mask for NeighborLoader
    train_mask_bool = np.zeros(data.x.shape[0], dtype=bool)
    train_mask_bool[train_mask] = True
    train_mask_tensor = torch.from_numpy(train_mask_bool)

    # Precompute page IDs for all nodes (for page-based RA calculation)
    # node_order[i] = storage position of node i
    # page_id = storage_position * feature_bytes / page_size
    node_page_id = (node_order * FEATURE_BYTES // PAGE_SIZE).astype(np.int32)

    # NeighborLoader with [10, 5] like DGraphFin baseline
    loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=1024,
        input_nodes=train_mask_tensor,
        shuffle=True,
        num_workers=4,
    )

    model = GraphSAGEWithHidden(
        data.x.shape[1], config.hidden_channels, config.num_classes
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Compute class weights for weighted loss
    class_weights = None
    if config.use_weighted_loss:
        class_weights = compute_class_weights(
            data.y, train_mask, config.num_classes, config.pos_weight_scale
        ).to(device)

    # Pre-compute fraud mask for batch composition tracking
    y_np = data.y.numpy()
    fraud_mask = y_np == 1

    stats = {"bytes_requested": 0, "bytes_read": 0, "batches": 0}
    batch_composition = {"total_train": 0, "total_pos": 0, "batches_with_pos": 0}

    start_time = time.time()
    epoch = 0
    global_step = 0
    last_eval_step = 0

    print(f"\nTraining for {config.time_limit:.0f} seconds...")
    if config.use_weighted_loss:
        print(f"  Using weighted loss (fraud weight ~{class_weights[1].item():.2f}x)")
    if config.eval_interval_steps > 0:
        print(f"  Evaluation checkpoints every {config.eval_interval_steps} steps")

    while True:
        model.train()
        epoch_loss = 0
        epoch_samples = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            elapsed = time.time() - start_time
            if elapsed >= config.time_limit:
                pbar.close()
                break

            global_step += 1
            stats["batches"] += 1

            # Move batch to device
            batch = batch.to(device)

            # Training nodes are the first batch_size nodes (input nodes)
            num_train = batch.batch_size
            train_nodes_global = batch.n_id[:num_train].cpu().numpy()
            num_pos = fraud_mask[train_nodes_global].sum()

            batch_composition["total_train"] += num_train
            batch_composition["total_pos"] += num_pos
            if num_pos > 0:
                batch_composition["batches_with_pos"] += 1

            # Compute RA metrics using consistent helper
            # loaded_node_ids = all sampled nodes (seeds + neighbors)
            loaded_node_ids = batch.n_id.cpu().numpy()
            ra = compute_ra_metrics(loaded_node_ids, num_train, node_page_id)
            stats["bytes_read"] += ra.bytes_read
            stats["bytes_requested"] += ra.loaded_nodes * FEATURE_BYTES

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            pred = out[:num_train]  # Only first batch_size nodes are targets
            target = batch.y[:num_train]

            loss = F.nll_loss(pred, target, weight=class_weights)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * num_train
            epoch_samples += num_train

            if wandb_logger:
                wandb_logger.log_batch({
                    "loss": loss.item(),
                    "ra_fetch": ra.ra_fetch,
                    "ra_signal": ra.ra_signal,
                    "pages_touched": ra.pages_touched,
                    "loaded_nodes": ra.loaded_nodes,
                })

            pbar.set_postfix(loss=f"{loss.item():.4f}", ra=f"{ra.ra_fetch:.2f}x")

            # Periodic evaluation checkpoint for learning curves
            if (
                config.eval_interval_steps > 0
                and global_step - last_eval_step >= config.eval_interval_steps
            ):
                last_eval_step = global_step
                val_m = evaluate(model, data, val_mask, device)
                test_m = evaluate(model, data, test_mask, device, val_m.threshold)
                model.train()
                if wandb_logger:
                    wandb_logger.log_eval(val_m, test_m)
                print(
                    f"  [Step {global_step}] Val PR-AUC={val_m.pr_auc:.4f}, "
                    f"Test PR-AUC={test_m.pr_auc:.4f}, Lift={test_m.lift_over_random:.1f}x"
                )

        pbar.close()

        elapsed = time.time() - start_time
        if elapsed >= config.time_limit:
            break

        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            avg_ra = stats["bytes_read"] / stats["bytes_requested"] if stats["bytes_requested"] > 0 else 0
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, RA={avg_ra:.2f}x")

        epoch += 1

    # Final evaluation
    val_metrics = evaluate(model, data, val_mask, device)
    test_metrics = evaluate(model, data, test_mask, device, val_metrics.threshold)

    avg_ra = stats["bytes_read"] / stats["bytes_requested"] if stats["bytes_requested"] > 0 else 0
    avg_train = batch_composition["total_train"] / stats["batches"] if stats["batches"] > 0 else 0
    avg_pos = batch_composition["total_pos"] / stats["batches"] if stats["batches"] > 0 else 0
    frac_with_pos = batch_composition["batches_with_pos"] / stats["batches"] if stats["batches"] > 0 else 0

    results = {
        "variant": "neighbor_sampler",
        "f1": test_metrics.f1,
        "pr_auc": test_metrics.pr_auc,
        "p_at_0.5%": test_metrics.precision_at_05pct,
        "r_at_1%": test_metrics.recall_at_1pct,
        "recall_at_k": test_metrics.recall_at_k,
        "lift_over_random": test_metrics.lift_over_random,
        "read_amplification": avg_ra,
        "batches": stats["batches"],
        "batch_stats": {
            "avg_train_nodes": avg_train,
            "avg_pos_nodes": avg_pos,
            "frac_with_positives": frac_with_pos,
        },
    }

    if wandb_logger:
        wandb_logger.log_final(results)

    print(f"\nResults: F1={test_metrics.f1:.4f}, PR-AUC={test_metrics.pr_auc:.4f}, RA={avg_ra:.2f}x")
    print(f"  Recall@K={test_metrics.recall_at_k:.4f}, Lift={test_metrics.lift_over_random:.1f}x over random")
    print(f"  Batch composition: avg {avg_train:.0f} train nodes, {avg_pos:.1f} positives")
    print(f"  Batches with ≥1 positive: {100*frac_with_pos:.1f}% ({batch_composition['batches_with_pos']}/{stats['batches']})")

    return results


def train_page_batch(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    page_id: np.ndarray,
    node_order: np.ndarray,
    config: FIMBenchmarkConfig,
    device: torch.device,
    wandb_logger: Optional[FIMWandBLogger] = None,
) -> Dict:
    """Train with page-aware batching (no FIM)."""
    print("\n" + "=" * 70)
    variant_name = "PAGE-BATCH: Page-Aware (Random Boundary)"
    if config.use_weighted_loss:
        variant_name += " + Weighted Loss"
    print(variant_name)
    print("=" * 70)

    edge_index_undirected = to_undirected(data.edge_index)
    page_id_4kb = (node_order * FEATURE_BYTES // PAGE_SIZE).astype(np.int32)

    # Convert labels for positive oversampling (fraud = class 1)
    pos_labels = (data.y.numpy() == 1).astype(np.int32) if config.pos_oversample_prob > 0 else None

    sampler = CppPageBatchSampler(
        edge_index=edge_index_undirected,
        page_id=page_id_4kb,
        train_mask=train_mask,
        num_nodes=data.x.shape[0],
        pages_per_batch=config.pages_per_batch,
        boundary_budget=config.boundary_budget,
        boundary_hops=config.boundary_hops,
        use_inter_page_edges=False,
        shuffle=True,
        seed=42,
        pos_labels=pos_labels,
        pos_oversample_prob=config.pos_oversample_prob,
    )

    model = GraphSAGEWithHidden(
        data.x.shape[1], config.hidden_channels, config.num_classes
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    x_device = data.x.to(device)
    y_device = data.y.to(device)

    # Compute class weights for weighted loss
    class_weights = None
    if config.use_weighted_loss:
        class_weights = compute_class_weights(
            data.y, train_mask, config.num_classes, config.pos_weight_scale
        ).to(device)

    # Pre-compute fraud mask for batch composition tracking
    y_np = data.y.numpy()
    fraud_mask = y_np == 1

    stats = {"bytes_requested": 0, "bytes_read": 0, "batches": 0}
    batch_composition = {"total_train": 0, "total_pos": 0, "batches_with_pos": 0}
    best_loss = float("inf")
    best_state = None

    start_time = time.time()
    epoch = 0
    global_step = 0
    last_eval_step = 0

    print(f"\nTraining for {config.time_limit:.0f} seconds...")
    if config.use_weighted_loss:
        print(f"  Using weighted loss (fraud weight ~{class_weights[1].item():.2f}x)")
    if config.eval_interval_steps > 0:
        print(f"  Evaluation checkpoints every {config.eval_interval_steps} steps")

    while True:
        model.train()
        epoch_loss = 0
        epoch_samples = 0
        epoch_bytes_req = 0
        epoch_bytes_read = 0

        pbar = tqdm(sampler, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            elapsed = time.time() - start_time
            if elapsed >= config.time_limit:
                pbar.close()
                break

            if batch.num_train_nodes == 0:
                continue

            global_step += 1

            # Compute RA metrics using consistent helper
            # loaded_node_ids = all nodes in batch (core + boundary)
            loaded_node_ids = batch.nodes.numpy()
            ra = compute_ra_metrics(loaded_node_ids, batch.num_train_nodes, page_id_4kb)
            stats["bytes_read"] += ra.bytes_read
            stats["bytes_requested"] += ra.loaded_nodes * FEATURE_BYTES
            stats["batches"] += 1
            epoch_bytes_req += ra.loaded_nodes * FEATURE_BYTES
            epoch_bytes_read += ra.bytes_read

            # Track batch composition
            train_nodes = loaded_node_ids[batch.train_mask.numpy()]
            num_pos = fraud_mask[train_nodes].sum()
            batch_composition["total_train"] += len(train_nodes)
            batch_composition["total_pos"] += num_pos
            if num_pos > 0:
                batch_composition["batches_with_pos"] += 1

            batch_x = x_device[batch.nodes]
            batch_y = y_device[batch.nodes[batch.train_mask]]
            batch_edge_index = batch.edge_index.to(device)

            optimizer.zero_grad()
            out = model(batch_x, batch_edge_index)
            pred = out[batch.train_mask]

            loss = F.nll_loss(pred, batch_y, weight=class_weights)
            loss.backward()
            optimizer.step()

            train_count = batch.num_train_nodes
            epoch_loss += loss.item() * train_count
            epoch_samples += train_count

            if wandb_logger:
                wandb_logger.log_batch({
                    "loss": loss.item(),
                    "ra_fetch": ra.ra_fetch,
                    "ra_signal": ra.ra_signal,
                    "pages_touched": ra.pages_touched,
                    "loaded_nodes": ra.loaded_nodes,
                })

            pbar.set_postfix(loss=f"{loss.item():.4f}", ra=f"{ra.ra_fetch:.2f}x")

            # Periodic evaluation checkpoint for learning curves
            if (
                config.eval_interval_steps > 0
                and global_step - last_eval_step >= config.eval_interval_steps
            ):
                last_eval_step = global_step
                val_m = evaluate(model, data, val_mask, device)
                test_m = evaluate(model, data, test_mask, device, val_m.threshold)
                model.train()  # Back to training mode
                if wandb_logger:
                    wandb_logger.log_eval(val_m, test_m)
                print(
                    f"  [Step {global_step}] Val PR-AUC={val_m.pr_auc:.4f}, "
                    f"Test PR-AUC={test_m.pr_auc:.4f}, Lift={test_m.lift_over_random:.1f}x"
                )

        pbar.close()

        elapsed = time.time() - start_time
        if elapsed >= config.time_limit:
            break

        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            epoch_ra = epoch_bytes_read / epoch_bytes_req if epoch_bytes_req > 0 else 0
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, RA={epoch_ra:.2f}x")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch += 1

    elapsed_time = time.time() - start_time
    read_amp = (
        stats["bytes_read"] / stats["bytes_requested"]
        if stats["bytes_requested"] > 0
        else 0
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, data, val_mask, device)
    test_metrics = evaluate(model, data, test_mask, device, val_metrics.threshold)

    # Compute batch composition stats
    n_batches = stats["batches"]
    avg_train = batch_composition["total_train"] / n_batches if n_batches > 0 else 0
    avg_pos = batch_composition["total_pos"] / n_batches if n_batches > 0 else 0
    frac_with_pos = batch_composition["batches_with_pos"] / n_batches if n_batches > 0 else 0

    variant_str = "page_batch_weighted" if config.use_weighted_loss else "page_batch"
    results = {
        "variant": variant_str,
        "f1": test_metrics.f1,
        "pr_auc": test_metrics.pr_auc,
        "p_at_0.5%": test_metrics.precision_at_05pct,
        "r_at_1%": test_metrics.recall_at_1pct,
        "recall_at_k": test_metrics.recall_at_k,
        "lift_over_random": test_metrics.lift_over_random,
        "read_amplification": read_amp,
        "epochs": epoch,
        "batches": stats["batches"],
        "elapsed_time": elapsed_time,
        "weighted_loss": config.use_weighted_loss,
        "batch_stats": {
            "avg_train_nodes": avg_train,
            "avg_pos_nodes": avg_pos,
            "frac_with_positives": frac_with_pos,
        },
    }

    if wandb_logger:
        wandb_logger.log_final(results)

    print(f"\nResults: F1={test_metrics.f1:.4f}, PR-AUC={test_metrics.pr_auc:.4f}, RA={read_amp:.2f}x")
    print(f"  Recall@K={test_metrics.recall_at_k:.4f}, Lift={test_metrics.lift_over_random:.1f}x over random")
    print(f"  Batch composition: avg {avg_train:.0f} train nodes, {avg_pos:.1f} positives")
    print(f"  Batches with ≥1 positive: {frac_with_pos:.1%} ({batch_composition['batches_with_pos']}/{n_batches})")

    return results


def train_fim_guided(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    page_id: np.ndarray,
    node_order: np.ndarray,
    config: FIMBenchmarkConfig,
    device: torch.device,
    use_replication: bool = False,
    wandb_logger: Optional[FIMWandBLogger] = None,
) -> Dict:
    """Train with FIM-guided importance sampling (backward hooks)."""
    variant = "fim_replication" if use_replication else "fim_importance"
    print("\n" + "=" * 70)
    variant_name = f"FIM-GUIDED: {'+ Replication' if use_replication else 'Importance Sampling'}"
    if config.use_weighted_loss:
        variant_name += " + Weighted Loss"
    print(variant_name)
    print("=" * 70)

    edge_index_undirected = to_undirected(data.edge_index)
    page_id_4kb = (node_order * FEATURE_BYTES // PAGE_SIZE).astype(np.int32)
    num_nodes = data.x.shape[0]

    # Create FIM tracker (uses backward hooks)
    fim_config = FIMConfig(
        beta=config.beta,
        lambda_pagepair=config.lambda_pagepair,
        replication_budget=config.replication_budget,
        replication_interval_steps=config.replication_interval_steps,
        importance_root=config.importance_root,
    )
    tracker = create_importance_tracker(num_nodes, page_id_4kb, fim_config)

    # Convert labels for positive oversampling (fraud = class 1)
    pos_labels = (data.y.numpy() == 1).astype(np.int32) if config.pos_oversample_prob > 0 else None

    # Create FIM-guided sampler (uses C++ sampler core for speed)
    sampler = FIMCppPageBatchSampler(
        edge_index=edge_index_undirected,
        page_id=page_id_4kb,
        train_mask=train_mask,
        num_nodes=num_nodes,
        importance_tracker=tracker,
        pages_per_batch=config.pages_per_batch,
        boundary_budget=config.boundary_budget,
        boundary_hops=config.boundary_hops,
        shuffle=True,
        seed=42,
        pos_labels=pos_labels,
        pos_oversample_prob=config.pos_oversample_prob,
    )

    model = GraphSAGEWithHidden(
        data.x.shape[1], config.hidden_channels, config.num_classes
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    x_device = data.x.to(device)
    y_device = data.y.to(device)

    # Compute class weights for weighted loss
    class_weights = None
    if config.use_weighted_loss:
        class_weights = compute_class_weights(
            data.y, train_mask, config.num_classes, config.pos_weight_scale
        ).to(device)

    # Pre-compute fraud mask for batch composition tracking
    y_np = data.y.numpy()
    fraud_mask = y_np == 1

    stats = {
        "bytes_requested": 0,
        "bytes_read": 0,
        "batches": 0,
    }
    batch_composition = {"total_train": 0, "total_pos": 0, "batches_with_pos": 0}
    best_loss = float("inf")
    best_state = None

    start_time = time.time()
    epoch = 0
    global_step = 0
    last_eval_step = 0

    print(f"\nTraining for {config.time_limit:.0f} seconds...")
    print(f"  Importance: backward hooks (zero overhead)")
    if use_replication:
        print(f"  Replication interval: {config.replication_interval_steps} steps")
    if config.use_weighted_loss:
        print(f"  Using weighted loss (fraud weight ~{class_weights[1].item():.2f}x)")
    if config.eval_interval_steps > 0:
        print(f"  Evaluation checkpoints every {config.eval_interval_steps} steps")

    while True:
        model.train()
        epoch_loss = 0
        epoch_samples = 0
        epoch_bytes_req = 0
        epoch_bytes_read = 0

        pbar = tqdm(sampler, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            elapsed = time.time() - start_time
            if elapsed >= config.time_limit:
                pbar.close()
                break

            if batch.num_train_nodes == 0:
                continue

            global_step += 1

            # RA calculation
            # Compute RA metrics using consistent helper
            # loaded_node_ids = all nodes in batch (core + boundary)
            loaded_node_ids = batch.nodes.numpy()
            ra = compute_ra_metrics(loaded_node_ids, batch.num_train_nodes, page_id_4kb)
            stats["bytes_read"] += ra.bytes_read
            stats["bytes_requested"] += ra.loaded_nodes * FEATURE_BYTES
            stats["batches"] += 1
            epoch_bytes_req += ra.loaded_nodes * FEATURE_BYTES
            epoch_bytes_read += ra.bytes_read

            # Track batch composition
            train_nodes = loaded_node_ids[batch.train_mask.numpy()]
            num_pos = fraud_mask[train_nodes].sum()
            batch_composition["total_train"] += len(train_nodes)
            batch_composition["total_pos"] += num_pos
            if num_pos > 0:
                batch_composition["batches_with_pos"] += 1

            # Forward pass with hidden state capture
            batch_x = x_device[batch.nodes]
            batch_y = y_device[batch.nodes]
            batch_edge_index = batch.edge_index.to(device)

            optimizer.zero_grad()

            # Get hidden states for importance tracking
            out, hidden = model(batch_x, batch_edge_index, return_hidden=True)

            # Set batch info for persistent hook (minimal overhead)
            # Pass tensor directly (no .numpy() conversion)
            tracker.register_hook(
                hidden_states=hidden,
                batch_nodes=batch.nodes,  # Keep as tensor
            )

            # Compute loss on training nodes only
            pred = out[batch.train_mask]
            train_y = batch_y[batch.train_mask]
            loss = F.nll_loss(pred, train_y, weight=class_weights)

            # Backward - hook fires automatically here
            loss.backward()
            optimizer.step()

            # Clear hook
            tracker.clear_hook()

            # Update replication set periodically (step-based)
            if use_replication and tracker.should_update_replication(global_step):
                tracker.update_replication_set(global_step)

            train_count = batch.num_train_nodes
            epoch_loss += loss.item() * train_count
            epoch_samples += train_count

            # Log to W&B every batch (same as baseline)
            if wandb_logger:
                wandb_logger.log_batch({
                    "loss": loss.item(),
                    "ra_fetch": ra.ra_fetch,
                    "ra_signal": ra.ra_signal,
                    "pages_touched": ra.pages_touched,
                    "loaded_nodes": ra.loaded_nodes,
                })

            # Update tqdm and log FIM stats every 100 batches (expensive stats)
            if global_step % 100 == 0:
                coverage = tracker.node_table.stats()["coverage"]
                if wandb_logger:
                    wandb_logger.log_batch({"fim/coverage": coverage})
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ra=f"{ra.ra_fetch:.2f}x",
                    cov=f"{coverage:.1%}",
                )

            # Periodic evaluation checkpoint for learning curves
            if (
                config.eval_interval_steps > 0
                and global_step - last_eval_step >= config.eval_interval_steps
            ):
                last_eval_step = global_step
                val_m = evaluate(model, data, val_mask, device)
                test_m = evaluate(model, data, test_mask, device, val_m.threshold)
                model.train()  # Back to training mode
                if wandb_logger:
                    wandb_logger.log_eval(val_m, test_m)
                print(
                    f"  [Step {global_step}] Val PR-AUC={val_m.pr_auc:.4f}, "
                    f"Test PR-AUC={test_m.pr_auc:.4f}, Lift={test_m.lift_over_random:.1f}x"
                )

        pbar.close()

        elapsed = time.time() - start_time
        if elapsed >= config.time_limit:
            break

        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            epoch_ra = epoch_bytes_read / epoch_bytes_req if epoch_bytes_req > 0 else 0

            tracker_stats = tracker.stats()
            print(
                f"Epoch {epoch:3d}: loss={avg_loss:.4f}, RA={epoch_ra:.2f}x, "
                f"coverage={tracker_stats['node_coverage']:.1%}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch += 1

    # Drain remaining GPU data and remove hook
    tracker.finish()

    elapsed_time = time.time() - start_time
    read_amp = (
        stats["bytes_read"] / stats["bytes_requested"]
        if stats["bytes_requested"] > 0
        else 0
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, data, val_mask, device)
    test_metrics = evaluate(model, data, test_mask, device, val_metrics.threshold)

    # Compute batch composition stats
    n_batches = stats["batches"]
    avg_train = batch_composition["total_train"] / n_batches if n_batches > 0 else 0
    avg_pos = batch_composition["total_pos"] / n_batches if n_batches > 0 else 0
    frac_with_pos = batch_composition["batches_with_pos"] / n_batches if n_batches > 0 else 0

    tracker_stats = tracker.stats()
    variant_str = f"{variant}_weighted" if config.use_weighted_loss else variant
    results = {
        "variant": variant_str,
        "f1": test_metrics.f1,
        "pr_auc": test_metrics.pr_auc,
        "p_at_0.5%": test_metrics.precision_at_05pct,
        "r_at_1%": test_metrics.recall_at_1pct,
        "recall_at_k": test_metrics.recall_at_k,
        "lift_over_random": test_metrics.lift_over_random,
        "read_amplification": read_amp,
        "epochs": epoch,
        "batches": stats["batches"],
        "elapsed_time": elapsed_time,
        "weighted_loss": config.use_weighted_loss,
        "batch_stats": {
            "avg_train_nodes": avg_train,
            "avg_pos_nodes": avg_pos,
            "frac_with_positives": frac_with_pos,
        },
        "fim_stats": {
            "total_updates": tracker_stats["total_updates"],
            "batches_with_fraud": tracker_stats["batches_with_fraud"],
            "node_coverage": tracker_stats["node_coverage"],
            "replicated_nodes": tracker_stats["replicated_nodes"],
            "saved_reads_total": 0,  # Computed from replication set
        },
    }

    if wandb_logger:
        wandb_logger.log_final(results)

    print(f"\nResults: F1={test_metrics.f1:.4f}, PR-AUC={test_metrics.pr_auc:.4f}, RA={read_amp:.2f}x")
    print(f"  Recall@K={test_metrics.recall_at_k:.4f}, Lift={test_metrics.lift_over_random:.1f}x over random")
    print(f"  Batch composition: avg {avg_train:.0f} train nodes, {avg_pos:.1f} positives")
    print(f"  Batches with ≥1 positive: {frac_with_pos:.1%} ({batch_composition['batches_with_pos']}/{n_batches})")
    print(f"  Coverage: {tracker_stats['node_coverage']:.1%}, FIM updates: {tracker_stats['total_updates']}")
    if use_replication:
        print(f"  Replicated nodes: {tracker_stats['replicated_nodes']}")

    return results


# =============================================================================
# Main
# =============================================================================


def print_comparison(results: List[Dict]):
    """Print comparison table of results."""
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)

    headers = ["Variant", "F1", "PR-AUC", "P@0.5%", "R@1%", "RA", "Batches"]
    print(
        f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} "
        f"{headers[4]:<10} {headers[5]:<10} {headers[6]:<10}"
    )
    print("-" * 90)

    baseline_f1 = None
    for r in results:
        if r["variant"] == "baseline":
            baseline_f1 = r["f1"]
            break

    for r in results:
        variant = r["variant"]
        f1 = r["f1"]
        pr_auc = r["pr_auc"]
        p_at_05 = r.get("p_at_0.5%", 0)
        r_at_1 = r.get("r_at_1%", 0)
        ra = r["read_amplification"]
        batches = r["batches"]

        f1_str = f"{f1:.4f}"
        if baseline_f1 and f1 > baseline_f1:
            f1_str += " ↑"

        print(
            f"{variant:<20} {f1_str:<10} {pr_auc:.4f}     {p_at_05:.4f}     "
            f"{r_at_1:.4f}     {ra:.2f}x      {batches}"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="FIM-Guided Fraud Detection Benchmark (v2)"
    )

    parser.add_argument("--time", type=float, default=300.0, help="Time limit (s)")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--pages-per-batch", type=int, default=32)
    parser.add_argument("--boundary-budget", type=float, default=0.2)
    parser.add_argument(
        "--boundary-hops",
        type=int,
        default=1,
        help="Hops for boundary expansion (1 or 2). Use 2 for 2-layer GNN.",
    )

    # FIM parameters
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lambda-pagepair", type=float, default=0.5)
    parser.add_argument("--replication-budget", type=float, default=0.005)
    parser.add_argument("--replication-interval", type=int, default=5000)
    parser.add_argument(
        "--importance-root",
        type=str,
        default="fourth_root",
        choices=["raw", "sqrt", "fourth_root"],
    )

    # Paths
    parser.add_argument("--data-dir", type=str, default="..")
    parser.add_argument("--layout", type=str, default="../layout_metis_bfs.npz")

    # W&B
    parser.add_argument("--wandb-project", type=str, default="gnn-fraud-detection")
    parser.add_argument("--no-wandb", action="store_true")

    # Class imbalance
    parser.add_argument(
        "--weighted-loss", action="store_true", help="Use weighted loss for class imbalance"
    )
    parser.add_argument(
        "--pos-weight-scale",
        type=float,
        default=1.0,
        help="Additional scale factor for fraud class weight",
    )
    parser.add_argument(
        "--pos-oversample",
        type=float,
        default=0.0,
        help="Probability of injecting positive-containing page per batch (0.0-1.0)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=2000,
        help="Evaluate every N steps for learning curves (0 to disable)",
    )

    # Ablation
    parser.add_argument("--ablation", action="store_true", help="Run all variants")
    parser.add_argument(
        "--only",
        type=str,
        choices=["neighbor_sampler", "page_batch", "fim_importance", "fim_replication"],
        help="Run only specific variant",
    )

    args = parser.parse_args()

    config = FIMBenchmarkConfig(
        time_limit=args.time,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        pages_per_batch=args.pages_per_batch,
        boundary_budget=args.boundary_budget,
        boundary_hops=args.boundary_hops,
        use_weighted_loss=args.weighted_loss,
        pos_weight_scale=args.pos_weight_scale,
        pos_oversample_prob=args.pos_oversample,
        eval_interval_steps=args.eval_interval,
        beta=args.beta,
        lambda_pagepair=args.lambda_pagepair,
        replication_budget=args.replication_budget,
        replication_interval_steps=args.replication_interval,
        importance_root=args.importance_root,
        data_dir=args.data_dir,
        layout_path=args.layout,
        wandb_project=args.wandb_project,
        use_wandb=WANDB_AVAILABLE and not args.no_wandb,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data, train_mask, val_mask, test_mask, _ = load_dgraphfin(config.data_dir)

    if not os.path.exists(config.layout_path):
        print(f"\nERROR: Layout file not found: {config.layout_path}")
        sys.exit(1)

    page_id, node_order = load_layout(config.layout_path)

    if args.only:
        variants = [args.only]
    elif args.ablation:
        # fim_replication excluded: tested 2025-12-17, no benefit over fim_importance
        # (F1: 0.0757 vs 0.0779, PR-AUC: 0.0340 vs 0.0350)
        variants = ["neighbor_sampler", "page_batch", "fim_importance"]
    else:
        variants = ["neighbor_sampler", "page_batch", "fim_importance"]

    results = []

    for variant in variants:
        if variant == "neighbor_sampler":
            with FIMWandBLogger(config, variant) as logger:
                r = train_neighbor_sampler(
                    data,
                    train_mask,
                    val_mask,
                    test_mask,
                    node_order,
                    config,
                    device,
                    logger,
                )
                results.append(r)

        elif variant == "page_batch":
            with FIMWandBLogger(config, variant) as logger:
                r = train_page_batch(
                    data,
                    train_mask,
                    val_mask,
                    test_mask,
                    page_id,
                    node_order,
                    config,
                    device,
                    logger,
                )
                results.append(r)

        elif variant == "fim_importance":
            with FIMWandBLogger(config, variant) as logger:
                r = train_fim_guided(
                    data,
                    train_mask,
                    val_mask,
                    test_mask,
                    page_id,
                    node_order,
                    config,
                    device,
                    use_replication=False,
                    wandb_logger=logger,
                )
                results.append(r)

        elif variant == "fim_replication":
            with FIMWandBLogger(config, variant) as logger:
                r = train_fim_guided(
                    data,
                    train_mask,
                    val_mask,
                    test_mask,
                    page_id,
                    node_order,
                    config,
                    device,
                    use_replication=True,
                    wandb_logger=logger,
                )
                results.append(r)

    if len(results) > 1:
        print_comparison(results)

    return results


if __name__ == "__main__":
    main()
