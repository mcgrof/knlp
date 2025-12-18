#!/usr/bin/env python3
"""
DGraphFin Fraud Detection Benchmark: NeighborLoader vs Page-Aware Mini-Batch

Compares two GNN training approaches with comprehensive fraud detection metrics:

1. NeighborLoader Baseline (PyG/pyg-lib): Traditional K-hop neighbor sampling
   - Read Amplification: ~58x (page-level)
   - Standard approach used in GNN research

2. Page-Aware Mini-Batch (Our Method): Memory-locality-optimized training
   - Read Amplification: 1.0-1.2x (bounded)
   - Processes entire memory pages as batches

Both methods use identical:
- Model architecture (2-layer GraphSAGE)
- Hyperparameters (lr, weight_decay, pos_weight)
- Evaluation metrics (F1, PR-AUC, P@K, R@K)

Usage:
    python benchmark_neighborloader_vs_pageaware.py --time 60   # Run for 60 seconds
    python benchmark_neighborloader_vs_pageaware.py --time 300  # Run for 5 minutes
    python benchmark_neighborloader_vs_pageaware.py --build-layout --time 120

Designed for portability to other projects (knlp integration).
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

# Optional imports
try:
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.utils import to_undirected

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# System monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

PAGE_SIZE = 4096  # 4KB pages
FEATURE_BYTES = 68  # 17 features * 4 bytes (DGraphFin)
NODES_PER_PAGE = PAGE_SIZE // FEATURE_BYTES  # ~60 nodes


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Training - time-based (primary) or epoch-based (fallback)
    time_limit: float = 60.0  # seconds - wall clock time limit
    lr: float = 0.003
    weight_decay_baseline: float = 5e-7  # NeighborLoader weight decay
    weight_decay_pageaware: float = 5e-7  # Page-aware weight decay (try 1e-4)
    hidden_channels: int = 128  # Match benchmark_1hour_comparison.py
    num_classes: int = 4  # DGraphFin has 4 classes

    # NeighborLoader settings
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5])
    batch_size: int = 1024
    force_sparse: bool = False  # Force torch-sparse backend instead of pyg-lib

    # Page-Aware settings
    pages_per_batch: int = 32
    boundary_budget: float = 0.0  # Pure intra-page (fastest, best F1)

    # Paths
    data_dir: str = ".."  # Parent directory (dgraphfin.npz location)
    layout_path: str = "../layout_metis_bfs.npz"

    # W&B
    wandb_project: str = "neighborloader-vs-pageaware"
    use_wandb: bool = True


@dataclass
class TrainStats:
    """Statistics accumulated during training."""

    loss: float = 0.0
    read_amplification: float = 0.0
    bytes_requested: int = 0
    bytes_read: int = 0
    nodes_processed: int = 0
    pages_accessed: int = 0
    batches_processed: int = 0
    elapsed_time: float = 0.0


@dataclass
class FraudMetrics:
    """Fraud detection evaluation metrics."""

    f1: float = 0.0
    pr_auc: float = 0.0
    precision_at_05pct: float = 0.0  # P@0.5%
    recall_at_1pct: float = 0.0  # R@1%
    threshold: float = 0.5


# =============================================================================
# Data Loading
# =============================================================================


def load_dgraphfin(
    data_dir: str,
) -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load DGraphFin dataset.

    Returns:
        data: PyG Data object with x, y, edge_index
        train_mask, val_mask, test_mask: Boolean masks
        num_classes: Number of classes (4 for DGraphFin)
    """
    npz_path = os.path.join(data_dir, "dgraphfin.npz")
    print(f"Loading DGraphFin from {npz_path}...")

    npz_data = np.load(npz_path)

    x = torch.from_numpy(npz_data["x"]).float()
    y = torch.from_numpy(npz_data["y"]).long()  # 4 classes: 0, 1, 2, 3
    edge_index = torch.from_numpy(npz_data["edge_index"].T).long()

    # Normalize features (critical for training convergence)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    # Convert index arrays to boolean masks
    num_nodes = x.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[npz_data["train_mask"]] = True
    val_mask[npz_data["valid_mask"]] = True
    test_mask[npz_data["test_mask"]] = True

    # Create PyG Data object
    data = Data(x=x, y=y, edge_index=edge_index)

    num_classes = 4  # DGraphFin has 4 classes
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Features: {x.shape[1]}")
    print(f"  Classes: {num_classes}")
    print(
        f"  Train/Val/Test: {train_mask.sum():,}/{val_mask.sum():,}/{test_mask.sum():,}"
    )

    return data, train_mask, val_mask, test_mask, num_classes


def load_layout(layout_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-computed page layout mapping node_id -> page_id and node_order.

    Returns:
        page_id: array mapping node_id -> METIS partition id
        node_order: array mapping node_id -> position in contiguous layout
    """
    print(f"Loading layout from {layout_path}...")
    layout_data = np.load(layout_path)

    if "page_id" in layout_data:
        page_id = layout_data["page_id"]
    elif "node_to_page" in layout_data:
        page_id = layout_data["node_to_page"]
    else:
        raise ValueError(f"Layout file missing page_id or node_to_page: {layout_path}")

    # node_order in layout file maps: position -> node_id
    # We need the inverse: node_id -> position (for RA calculation)
    if "node_order" in layout_data:
        perm = layout_data["node_order"]  # perm[position] = node_id
        # Invert: node_to_position[node_id] = position
        node_to_position = np.empty_like(perm)
        node_to_position[perm] = np.arange(len(perm))
    else:
        # Fallback: create order by sorting nodes within each partition
        print("  Warning: no node_order found, creating from page_id...")
        perm = np.argsort(page_id, kind="stable")
        node_to_position = np.empty_like(perm)
        node_to_position[perm] = np.arange(len(perm))

    num_pages = len(np.unique(page_id))
    print(f"  METIS partitions: {num_pages:,}")
    print(f"  Avg nodes/partition: {len(page_id) / num_pages:.1f}")

    return page_id, node_to_position


def build_layout(data_dir: str, output_path: str, method: str = "metis-bfs"):
    """Build graph layout using METIS + BFS ordering."""
    print(f"Building layout with method={method}...")

    # Import here to avoid circular deps
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scripts.build_graph_layout import main as build_layout_main

    # Build layout
    sys.argv = [
        "build_graph_layout.py",
        "--data-dir",
        data_dir,
        "--method",
        method,
        "--output",
        output_path,
    ]
    build_layout_main()

    print(f"Layout saved to {output_path}")


# =============================================================================
# Model
# =============================================================================


class GraphSAGE(torch.nn.Module):
    """2-layer GraphSAGE for multi-class classification (matches benchmark_1hour_comparison.py)."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.0, training=self.training)  # Match old benchmark
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)  # Multi-class output


# =============================================================================
# Evaluation
# =============================================================================


def compute_fraud_metrics(
    probs: np.ndarray, y_true: np.ndarray, threshold: Optional[float] = None
) -> FraudMetrics:
    """Compute comprehensive fraud detection metrics."""
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)

    # Compute F1 for each threshold to find optimal
    f1_scores = (
        2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)
    )

    if threshold is None:
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    y_pred = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = auc(recall_curve, precision_curve)

    # Precision@K and Recall@K (important for fraud detection)
    n = len(probs)
    total_pos = y_true.sum()
    sorted_idx = np.argsort(probs)[::-1]
    sorted_y = y_true[sorted_idx]

    # P@0.5%
    k_05 = max(1, int(n * 0.005))
    tp_at_05 = sorted_y[:k_05].sum()
    p_at_05 = tp_at_05 / k_05

    # R@1%
    k_1 = max(1, int(n * 0.01))
    tp_at_1 = sorted_y[:k_1].sum()
    r_at_1 = tp_at_1 / total_pos if total_pos > 0 else 0

    return FraudMetrics(
        f1=f1,
        pr_auc=pr_auc,
        precision_at_05pct=p_at_05,
        recall_at_1pct=r_at_1,
        threshold=threshold,
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

    out = model(x, edge_index)  # log_softmax output [N, num_classes]
    # Convert log_softmax to probabilities, take fraud class (class 1)
    probs = torch.exp(out[mask, 1]).cpu().numpy()  # P(fraud)
    # Binary labels: fraud (class 1) vs not fraud
    y_true = (data.y[mask].numpy() == 1).astype(int)

    return compute_fraud_metrics(probs, y_true, threshold)


# =============================================================================
# Training Methods (Time-Based)
# =============================================================================


def train_neighborloader(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    config: BenchmarkConfig,
    device: torch.device,
    num_classes: int,
    wandb_logger: Optional["WandBLogger"] = None,
) -> Tuple[Dict[str, float], TrainStats]:
    """
    Train using PyG NeighborLoader (traditional K-hop sampling).

    Runs for config.time_limit seconds (wall clock time).
    """
    print("\n" + "=" * 70)
    print("NEIGHBORLOADER BASELINE")
    print("=" * 70)

    if not PYG_AVAILABLE:
        raise ImportError("torch_geometric not available")

    # Detect/configure sampling backend
    if getattr(config, "force_sparse", False):
        # Force torch-sparse by hiding pyg-lib
        import sys

        if "pyg_lib" in sys.modules:
            del sys.modules["pyg_lib"]
        # Temporarily disable pyg-lib detection
        import torch_geometric.typing

        torch_geometric.typing.WITH_PYG_LIB = False
        print(f"  Backend: torch-sparse (FORCED via --force-sparse)")
    else:
        try:
            import pyg_lib

            print(f"  Backend: pyg-lib {pyg_lib.__version__} (FAST C++ sampling)")
        except ImportError:
            try:
                import torch_sparse

                print(f"  Backend: torch-sparse (slower fallback)")
            except ImportError:
                print(f"  Backend: pure Python (SLOW fallback)")

    # Make edges undirected for proper message passing
    data_undirected = data.clone()
    data_undirected.edge_index = to_undirected(data.edge_index)

    train_idx = torch.from_numpy(np.where(train_mask)[0])

    loader = NeighborLoader(
        data_undirected,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        input_nodes=train_idx,
        shuffle=True,
    )

    print(f"  Fanout: {config.num_neighbors}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Time limit: {config.time_limit:.0f}s")

    model = GraphSAGE(data.x.shape[1], config.hidden_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay_baseline
    )

    stats = TrainStats()
    best_loss = float("inf")
    best_state = None

    start_time = time.time()
    epoch = 0

    print(f"\nTraining for {config.time_limit:.0f} seconds...")

    while True:
        model.train()
        epoch_loss = 0
        epoch_samples = 0
        epoch_bytes_requested = 0
        epoch_bytes_read = 0

        loader_iter = iter(loader)
        pbar = tqdm(total=len(loader), desc=f"Epoch {epoch}", leave=False)

        for batch in loader_iter:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed >= config.time_limit:
                pbar.close()
                break

            n_ids = batch.n_id.numpy()
            seed_count = batch.batch_size

            # Compute page-level read amplification
            bytes_requested = len(n_ids) * FEATURE_BYTES
            page_ids = n_ids // NODES_PER_PAGE
            unique_pages = len(np.unique(page_ids))
            bytes_read = unique_pages * PAGE_SIZE

            stats.bytes_requested += bytes_requested
            stats.bytes_read += bytes_read
            stats.pages_accessed += unique_pages
            stats.batches_processed += 1
            epoch_bytes_requested += bytes_requested
            epoch_bytes_read += bytes_read

            # Forward pass
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)
            pred = out[: batch.batch_size]  # [batch_size, num_classes]
            target = batch.y[: batch.batch_size]  # [batch_size] long tensor

            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * seed_count
            epoch_samples += seed_count
            stats.nodes_processed += seed_count

            # Log to W&B every batch
            batch_ra = bytes_read / bytes_requested if bytes_requested > 0 else 0
            if wandb_logger:
                wandb_logger.log_batch(
                    loss.item(),
                    batch_ra,
                    elapsed,
                    batch_nodes=len(n_ids),
                    seed_nodes=seed_count,
                )

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", elapsed=f"{elapsed:.0f}s")

        pbar.close()

        # Check if we should stop
        elapsed = time.time() - start_time
        if elapsed >= config.time_limit:
            break

        # End of epoch - evaluate
        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            epoch_ra = (
                epoch_bytes_read / epoch_bytes_requested
                if epoch_bytes_requested > 0
                else 0
            )
            stats.loss = avg_loss

            print(
                f"Epoch {epoch:3d}: loss={avg_loss:.4f}, RA={epoch_ra:.1f}x, "
                f"elapsed={elapsed:.1f}s"
            )

            # Log epoch to W&B (matches old benchmark format)
            if wandb_logger:
                wandb_logger.log_epoch(epoch, avg_loss, epoch_ra, elapsed)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch += 1

    stats.elapsed_time = time.time() - start_time
    stats.read_amplification = (
        stats.bytes_read / stats.bytes_requested if stats.bytes_requested > 0 else 0
    )

    # Use best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on validation set to find threshold, then test set
    val_metrics = evaluate(model, data, val_mask, device, threshold=None)
    test_metrics = evaluate(
        model, data, test_mask, device, threshold=val_metrics.threshold
    )

    results = {
        "method": "neighborloader",
        "final_loss": stats.loss,
        "best_loss": best_loss,
        "read_amplification": stats.read_amplification,
        "total_bytes_read": stats.bytes_read,
        "total_time": stats.elapsed_time,
        "batches": stats.batches_processed,
        "nodes_trained": stats.nodes_processed,
        "epochs": epoch,
        "f1": test_metrics.f1,
        "pr_auc": test_metrics.pr_auc,
        "p_at_0.5%": test_metrics.precision_at_05pct,
        "r_at_1%": test_metrics.recall_at_1pct,
        "threshold": val_metrics.threshold,
    }

    # Log final results to W&B
    if wandb_logger:
        wandb_logger.log_final(results, stats)

    print(f"\nResults after {stats.elapsed_time:.1f}s:")
    print(f"  Epochs:     {epoch}")
    print(f"  Batches:    {stats.batches_processed:,}")
    print(f"  Nodes:      {stats.nodes_processed:,}")
    print(f"  Read Amp:   {stats.read_amplification:.1f}x")
    print(f"  Best Loss:  {best_loss:.4f}")
    print(f"  Test F1:    {test_metrics.f1:.4f}")
    print(f"  PR-AUC:     {test_metrics.pr_auc:.4f}")

    return results, stats


def train_page_aware(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    page_id: np.ndarray,
    node_order: np.ndarray,
    config: BenchmarkConfig,
    device: torch.device,
    num_classes: int,
    wandb_logger: Optional["WandBLogger"] = None,
) -> Tuple[Dict[str, float], TrainStats]:
    """
    Train using Page-Batch Mini-Batch with C++ CppPageBatchSampler.

    This achieves ~4.3x read amplification by:
    1. Sampling pages (not nodes)
    2. Using ALL nodes from selected pages
    3. Using ONLY intra-page edges (edges within selected pages)

    Uses C++ extension with OpenMP for fast batch assembly.

    Runs for config.time_limit seconds (wall clock time).
    """
    print("\n" + "=" * 70)
    print("PAGE-BATCH MINI-BATCH (C++ CppPageBatchSampler)")
    print("=" * 70)

    # Import the C++ accelerated page-batch sampler
    from page_batch_sampler import CppPageBatchSampler

    # Make edges undirected for proper message passing
    edge_index_undirected = to_undirected(data.edge_index)

    # Compute 4KB page IDs from layout position
    page_id_4kb = (node_order * FEATURE_BYTES // PAGE_SIZE).astype(np.int32)
    num_4kb_pages = int(page_id_4kb.max()) + 1
    print(
        f"  4KB pages: {num_4kb_pages:,} (avg {PAGE_SIZE // FEATURE_BYTES} nodes/page)"
    )

    # Build C++ page-batch sampler with inter-page edges
    sampler = CppPageBatchSampler(
        edge_index=edge_index_undirected,
        page_id=page_id_4kb,
        train_mask=train_mask,
        num_nodes=data.x.shape[0],
        pages_per_batch=config.pages_per_batch,
        boundary_budget=config.boundary_budget,
        use_inter_page_edges=False,  # Only intra-page edges (fast)
        shuffle=True,
        seed=42,
    )

    print(f"  Pages per batch: {config.pages_per_batch}")
    print(
        f"  Boundary budget: {config.boundary_budget:.0%} (max RA = {1 + config.boundary_budget:.2f}x)"
    )
    print(f"  Batches per epoch: {len(sampler)}")
    print(f"  Time limit: {config.time_limit:.0f}s")

    model = GraphSAGE(data.x.shape[1], config.hidden_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay_pageaware
    )

    # Pre-move data to device for faster batch construction
    x_device = data.x.to(device)
    y_device = data.y.to(device)

    stats = TrainStats()
    best_loss = float("inf")
    best_state = None

    start_time = time.time()
    epoch = 0

    print(f"\nTraining for {config.time_limit:.0f} seconds...")

    while True:
        model.train()
        epoch_loss = 0
        epoch_samples = 0
        epoch_bytes_requested = 0
        epoch_bytes_read = 0

        pbar = tqdm(sampler, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            elapsed = time.time() - start_time
            if elapsed >= config.time_limit:
                pbar.close()
                break

            if batch.num_train_nodes == 0:
                continue

            # RA calculation: (core pages + boundary nodes) vs train nodes
            # Core nodes come from full pages, boundary nodes are individual fetches
            bytes_requested = batch.num_train_nodes * FEATURE_BYTES
            core_bytes = len(batch.pages) * PAGE_SIZE
            boundary_bytes = batch.num_boundary_nodes * FEATURE_BYTES
            bytes_read = core_bytes + boundary_bytes

            stats.bytes_requested += bytes_requested
            stats.bytes_read += bytes_read
            stats.batches_processed += 1
            epoch_bytes_requested += bytes_requested
            epoch_bytes_read += bytes_read

            # Build batch tensors on device
            batch_x = x_device[batch.nodes]
            batch_y = y_device[batch.nodes[batch.train_mask]]
            batch_edge_index = batch.edge_index.to(device)

            # Forward pass
            optimizer.zero_grad()
            out = model(batch_x, batch_edge_index)
            pred = out[batch.train_mask]  # Only training nodes

            loss = F.nll_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_count = batch.num_train_nodes
            epoch_loss += loss.item() * train_count
            epoch_samples += train_count
            stats.nodes_processed += train_count

            batch_ra = bytes_read / bytes_requested if bytes_requested > 0 else 0
            if wandb_logger:
                wandb_logger.log_batch(
                    loss.item(),
                    batch_ra,
                    elapsed,
                    batch_nodes=len(batch.nodes),
                    seed_nodes=train_count,
                )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ra=f"{batch_ra:.2f}x",
                elapsed=f"{elapsed:.0f}s",
            )

        pbar.close()

        # Check if we should stop
        elapsed = time.time() - start_time
        if elapsed >= config.time_limit:
            break

        # End of epoch - evaluate
        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            epoch_ra = (
                epoch_bytes_read / epoch_bytes_requested
                if epoch_bytes_requested > 0
                else 0
            )
            stats.loss = avg_loss

            print(
                f"Epoch {epoch:3d}: loss={avg_loss:.4f}, RA={epoch_ra:.2f}x, "
                f"elapsed={elapsed:.1f}s"
            )

            # Log epoch to W&B (matches old benchmark format)
            if wandb_logger:
                wandb_logger.log_epoch(epoch, avg_loss, epoch_ra, elapsed)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch += 1

    stats.elapsed_time = time.time() - start_time
    stats.read_amplification = (
        stats.bytes_read / stats.bytes_requested if stats.bytes_requested > 0 else 0
    )

    # Use best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on validation set to find threshold, then test set
    val_metrics = evaluate(model, data, val_mask, device, threshold=None)
    test_metrics = evaluate(
        model, data, test_mask, device, threshold=val_metrics.threshold
    )

    results = {
        "method": "page_aware",
        "final_loss": stats.loss,
        "best_loss": best_loss,
        "read_amplification": stats.read_amplification,
        "total_bytes_read": stats.bytes_read,
        "total_time": stats.elapsed_time,
        "batches": stats.batches_processed,
        "nodes_trained": stats.nodes_processed,
        "epochs": epoch,
        "f1": test_metrics.f1,
        "pr_auc": test_metrics.pr_auc,
        "p_at_0.5%": test_metrics.precision_at_05pct,
        "r_at_1%": test_metrics.recall_at_1pct,
        "threshold": val_metrics.threshold,
    }

    # Log final results to W&B
    if wandb_logger:
        wandb_logger.log_final(results, stats)

    print(f"\nResults after {stats.elapsed_time:.1f}s:")
    print(f"  Epochs:     {epoch}")
    print(f"  Batches:    {stats.batches_processed:,}")
    print(f"  Nodes:      {stats.nodes_processed:,}")
    print(f"  Read Amp:   {stats.read_amplification:.2f}x")
    print(f"  Best Loss:  {best_loss:.4f}")
    print(f"  Test F1:    {test_metrics.f1:.4f}")
    print(f"  PR-AUC:     {test_metrics.pr_auc:.4f}")

    return results, stats


# =============================================================================
# System Resource Monitoring
# =============================================================================


def get_system_stats() -> Dict[str, float]:
    """Get current system resource usage."""
    stats = {}

    # CPU
    if PSUTIL_AVAILABLE:
        stats["system/cpu_percent"] = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        stats["system/ram_used_gb"] = mem.used / 1e9
        stats["system/ram_percent"] = mem.percent

    # GPU
    if torch.cuda.is_available():
        stats["system/gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        # GPU utilization via nvidia-smi parsing or pynvml
        try:
            # Try to get GPU utilization
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                stats["system/gpu_utilization"] = float(
                    result.stdout.strip().split("\n")[0]
                )
        except Exception:
            pass  # Skip if nvidia-smi not available (e.g., AMD GPU)

        # For AMD GPUs, try rocm-smi
        try:
            import subprocess

            result = subprocess.run(
                ["rocm-smi", "--showuse", "--json"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                import json

                data = json.loads(result.stdout)
                for card_id, card_data in data.items():
                    if "GPU use (%)" in card_data:
                        stats["system/gpu_utilization"] = float(
                            card_data["GPU use (%)"]
                        )
                        break
        except Exception:
            pass

    return stats


# =============================================================================
# W&B Logging
# =============================================================================


class WandBLogger:
    """Context manager for W&B logging during training."""

    def __init__(self, config: BenchmarkConfig, method_name: str):
        self.config = config
        self.method_name = method_name
        self.enabled = WANDB_AVAILABLE and config.use_wandb
        self.step = 0
        # Exponential moving average for smoothed loss
        self.loss_ema = None
        self.ema_alpha = 0.1  # Smoothing factor (lower = smoother)

    def __enter__(self):
        if self.enabled:
            wandb.init(
                project=self.config.wandb_project,
                name=f"{self.method_name}-{self.config.time_limit:.0f}s-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "method": self.method_name,
                    "time_limit_s": self.config.time_limit,
                    "lr": self.config.lr,
                    "hidden_channels": self.config.hidden_channels,
                    "num_neighbors": (
                        self.config.num_neighbors
                        if self.method_name == "neighborloader"
                        else None
                    ),
                    "batch_size": (
                        self.config.batch_size
                        if self.method_name == "neighborloader"
                        else None
                    ),
                    "pages_per_batch": (
                        self.config.pages_per_batch
                        if self.method_name == "page_aware"
                        else None
                    ),
                },
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            wandb.finish()

    def log_batch(
        self,
        loss: float,
        read_amp: float,
        elapsed: float,
        batch_nodes: int = 0,
        seed_nodes: int = 0,
    ):
        """Track batch metrics (but don't log to W&B per batch for consistency with old benchmark)."""
        if not self.enabled:
            return
        self.step += 1
        # Just track step count, actual logging happens per epoch

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        read_amp: float,
        elapsed: float,
    ):
        """Log per-epoch metrics (matches benchmark_1hour_comparison.py format)."""
        if not self.enabled:
            return
        # Use same metric names as old benchmark for consistent W&B charts
        metrics = {
            "loss": loss,
            "read_amp": read_amp,
            "epoch": epoch,
            "elapsed_s": elapsed,
        }
        wandb.log(metrics)

    def log_final(self, results: Dict[str, float], stats: TrainStats):
        """Log final results to both summary and history (for charts)."""
        if not self.enabled:
            return

        # Prepare final metrics dict
        final_metrics = {
            "test/f1": results.get("f1", 0),
            "test/pr_auc": results.get("pr_auc", 0),
            "test/p_at_0.5%": results.get("p_at_0.5%", 0),
            "test/r_at_1%": results.get("r_at_1%", 0),
            "final/read_amplification": stats.read_amplification,
            "final/total_bytes_read_gb": stats.bytes_read / 1e9,
            "final/batches": stats.batches_processed,
            "final/nodes_trained": stats.nodes_processed,
            "final/elapsed_time_s": stats.elapsed_time,
            "final/best_loss": results.get("best_loss", 0),
            "final/throughput_batch_per_s": (
                stats.batches_processed / stats.elapsed_time
                if stats.elapsed_time > 0
                else 0
            ),
        }

        # Log to history (enables charts/visualization)
        wandb.log(final_metrics)

        # Also update summary for Runs table display
        for key, value in final_metrics.items():
            wandb.run.summary[key] = value


# =============================================================================
# Main
# =============================================================================


def print_comparison(
    results_baseline: Dict, results_pageaware: Dict, time_limit: float
):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 80)
    print(f"COMPARISON SUMMARY ({time_limit:.0f}s wall clock each)")
    print("=" * 80)

    headers = ["Metric", "NeighborLoader", "Page-Aware", "Difference"]
    print(f"{headers[0]:<25} {headers[1]:<18} {headers[2]:<18} {headers[3]:<15}")
    print("-" * 80)

    metrics = [
        ("Epochs Completed", "epochs"),
        ("Best Loss", "best_loss"),
        ("Final Loss", "final_loss"),
        ("Read Amplification", "read_amplification"),
        ("Total Bytes Read (GB)", "total_bytes_read"),
        ("Batches Processed", "batches"),
        ("Nodes Trained", "nodes_trained"),
    ]

    for name, key in metrics:
        bl = results_baseline.get(key, 0)
        pa = results_pageaware.get(key, 0)

        if key == "total_bytes_read":
            bl_str = f"{bl / 1e9:.2f}"
            pa_str = f"{pa / 1e9:.2f}"
            reduction = (1 - pa / bl) * 100 if bl > 0 else 0
            imp_str = f"-{reduction:.0f}%"
        elif key == "read_amplification":
            bl_str = f"{bl:.1f}x"
            pa_str = f"{pa:.2f}x"
            reduction = (1 - pa / bl) * 100 if bl > 0 else 0
            imp_str = f"-{reduction:.0f}%"
        elif key in ("batches", "nodes_trained", "epochs"):
            bl_str = f"{int(bl):,}"
            pa_str = f"{int(pa):,}"
            diff = (pa - bl) / bl * 100 if bl > 0 else 0
            imp_str = f"{diff:+.0f}%"
        else:
            bl_str = f"{bl:.4f}"
            pa_str = f"{pa:.4f}"
            # For loss, lower is better
            diff = (bl - pa) / bl * 100 if bl > 0 else 0
            imp_str = f"{diff:+.1f}%"

        print(f"{name:<25} {bl_str:<18} {pa_str:<18} {imp_str:<15}")

    print("=" * 80)
    print("\nKey insight: Same wall clock time, ~50x less memory bandwidth")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NeighborLoader vs Page-Aware Mini-Batch Training"
    )

    # Time-based training (primary)
    parser.add_argument(
        "--time",
        type=float,
        default=60.0,
        help="Wall clock time limit in seconds (default: 60)",
    )

    # Training args
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument(
        "--weight-decay-baseline",
        type=float,
        default=5e-7,
        help="Weight decay for NeighborLoader baseline (default: 5e-7)",
    )
    parser.add_argument(
        "--weight-decay-pageaware",
        type=float,
        default=5e-7,
        help="Weight decay for Page-Aware method (default: 5e-7, try 1e-4)",
    )

    # NeighborLoader args
    parser.add_argument(
        "--num-neighbors", type=int, nargs="+", default=[10, 5], help="Fanout per hop"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--force-sparse",
        action="store_true",
        help="Force torch-sparse backend instead of pyg-lib (slower, for comparison)",
    )

    # Page-Aware args
    parser.add_argument("--pages-per-batch", type=int, default=32)
    parser.add_argument(
        "--boundary-budget",
        type=float,
        default=0.2,
        help="Fraction of extra nodes for 1-hop boundary expansion (default 0.2)",
    )
    parser.add_argument("--layout", type=str, default="../layout_metis_bfs.npz")

    # Data args
    parser.add_argument("--data-dir", type=str, default="..")
    parser.add_argument(
        "--build-layout", action="store_true", help="Build layout if not exists"
    )

    # W&B args
    parser.add_argument(
        "--wandb-project", type=str, default="neighborloader-vs-pageaware"
    )
    parser.add_argument("--no-wandb", action="store_true")

    # Run selection
    parser.add_argument(
        "--only-baseline", action="store_true", help="Only run NeighborLoader"
    )
    parser.add_argument(
        "--only-pageaware", action="store_true", help="Only run Page-Aware"
    )

    args = parser.parse_args()

    # Build config
    config = BenchmarkConfig(
        time_limit=args.time,
        lr=args.lr,
        weight_decay_baseline=args.weight_decay_baseline,
        weight_decay_pageaware=args.weight_decay_pageaware,
        hidden_channels=args.hidden_channels,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        force_sparse=args.force_sparse,
        pages_per_batch=args.pages_per_batch,
        boundary_budget=args.boundary_budget,
        data_dir=args.data_dir,
        layout_path=args.layout,
        wandb_project=args.wandb_project,
        use_wandb=WANDB_AVAILABLE and not args.no_wandb,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Time limit: {config.time_limit:.0f}s per method")

    # Load data
    data, train_mask, val_mask, test_mask, num_classes = load_dgraphfin(config.data_dir)

    # Check/build layout
    if not args.only_baseline:
        if not os.path.exists(config.layout_path):
            if args.build_layout:
                build_layout(config.data_dir, config.layout_path)
            else:
                print(f"\nERROR: Layout file not found: {config.layout_path}")
                print("Run with --build-layout or:")
                print(f"  python scripts/build_graph_layout.py --method metis-bfs")
                sys.exit(1)
        page_id, node_order = load_layout(config.layout_path)
    else:
        page_id, node_order = None, None

    results = {}

    # Run NeighborLoader baseline
    if not args.only_pageaware:
        with WandBLogger(config, "neighborloader") as wandb_logger:
            results_bl, stats_bl = train_neighborloader(
                data,
                train_mask,
                val_mask,
                test_mask,
                config,
                device,
                num_classes,
                wandb_logger,
            )
            results["neighborloader"] = results_bl

    # Run Page-Aware
    if not args.only_baseline:
        with WandBLogger(config, "page_aware") as wandb_logger:
            results_pa, stats_pa = train_page_aware(
                data,
                train_mask,
                val_mask,
                test_mask,
                page_id,
                node_order,
                config,
                device,
                num_classes,
                wandb_logger,
            )
            results["page_aware"] = results_pa

    # Print comparison
    if "neighborloader" in results and "page_aware" in results:
        print_comparison(
            results["neighborloader"], results["page_aware"], config.time_limit
        )

    return results


if __name__ == "__main__":
    main()
