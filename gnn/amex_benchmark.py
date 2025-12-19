#!/usr/bin/env python3
"""
AMEX Default Prediction - Streaming Locality Benchmark

This is a SYSTEMS benchmark, not a ML accuracy competition.
Focus: TPS, p99 latency, cache behavior, and feature retrieval overhead.

Key insight: Default-risk signal is learned instantly.
The bottleneck is I/O, data movement, and pipeline overhead.

AMEX is a credit-default dataset, NOT fraud detection.
"""

import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional, Dict, List, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional wandb
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def init_wandb_with_system_metrics(
    project: str, name: str, config: dict, reinit: bool = False
):
    """
    Initialize W&B with all recommended system metrics enabled.

    This enables comprehensive monitoring of:
    - GPU: utilization, memory, temperature, power
    - CPU: utilization per core, frequency
    - Memory: RAM usage, swap
    - Disk: I/O rates, usage
    - Network: bytes sent/received
    """
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        reinit=reinit,
    )
    # Log system info
    if torch.cuda.is_available():
        wandb.config.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return run


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AMEXConfig:
    """Benchmark configuration."""

    data_dir: str = "data/amex"

    # Streaming settings
    window_size: int = 6  # Statements per customer window
    batch_size: int = 512  # Customers per batch

    # Training settings
    hidden_dim: int = 128
    num_epochs: int = 1
    lr: float = 1e-3
    max_time: int = 300  # 5 minutes default

    # Sampler: "random", "page_aware", "fim_importance"
    sampler: str = "random"

    # FIM settings
    fim_ema: float = 0.99
    fim_budget: float = 0.2  # Fraction of customers to prioritize

    # Metrics
    eval_interval: int = 50  # Batches between eval

    # Locality simulation
    page_size: int = 4096  # Bytes per page
    feature_bytes: int = (
        752  # 188 features * 4 bytes (excluding D_63, D_64 categorical)
    )


# =============================================================================
# Data Loading - Streaming with Locality Tracking
# =============================================================================


class AMEXDataset:
    """
    AMEX dataset with streaming and locality instrumentation.

    Treats customer_ID as entity key, S_2 (date) as stream axis.
    Preserves original time ordering for locality benchmarking.
    """

    def __init__(self, config: AMEXConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)

        # Load and preprocess
        print("Loading AMEX dataset...")
        self._load_data()
        self._compute_page_layout()

        print(f"  Customers: {len(self.customer_ids):,}")
        print(f"  Statements: {len(self.data):,}")
        print(f"  Features: {self.num_features}")
        print(f"  Positive rate: {self.labels.mean():.2%}")

    def _load_data(self):
        """Load train data and labels."""
        # Load labels
        labels_df = pd.read_csv(self.data_dir / "train_labels.csv")
        self.customer_to_label = dict(
            zip(labels_df["customer_ID"], labels_df["target"])
        )

        # Load train data - keep time ordering
        print("  Loading train_data.csv...")
        self.data = pd.read_csv(self.data_dir / "train_data.csv", parse_dates=["S_2"])

        # Sort by date to simulate streaming
        self.data = self.data.sort_values("S_2").reset_index(drop=True)

        # Get unique customers (in order of first appearance)
        seen = {}
        for idx, cid in enumerate(self.data["customer_ID"]):
            if cid not in seen:
                seen[cid] = idx
        self.customer_ids = list(seen.keys())
        self.customer_to_idx = {cid: i for i, cid in enumerate(self.customer_ids)}

        # Extract features (drop customer_ID, S_2, and categorical columns)
        # D_63 and D_64 are categorical strings - exclude for simplicity
        exclude_cols = ["customer_ID", "S_2", "D_63", "D_64"]
        feature_cols = [c for c in self.data.columns if c not in exclude_cols]
        self.num_features = len(feature_cols)

        # Convert to numpy for speed
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0)  # Handle NaN

        self.customer_col = self.data["customer_ID"].values
        self.date_col = self.data["S_2"].values

        # Create labels array (one per customer)
        self.labels = np.array(
            [self.customer_to_label.get(cid, 0) for cid in self.customer_ids],
            dtype=np.float32,
        )

        # Build customer -> row indices mapping
        self.customer_rows = {}
        for idx, cid in enumerate(self.customer_col):
            if cid not in self.customer_rows:
                self.customer_rows[cid] = []
            self.customer_rows[cid].append(idx)

    def _compute_page_layout(self):
        """
        Compute page layout for locality metrics.

        Simulates storage where rows are stored contiguously.
        Each page holds PAGE_SIZE / FEATURE_BYTES rows.
        """
        rows_per_page = self.config.page_size // self.config.feature_bytes
        self.rows_per_page = max(1, rows_per_page)  # At least 1

        # Assign each row to a page
        self.row_to_page = np.arange(len(self.data)) // self.rows_per_page
        self.num_pages = int(self.row_to_page.max()) + 1

        print(f"  Rows per page: {self.rows_per_page}")
        print(f"  Total pages: {self.num_pages:,}")

    def get_customer_window(self, customer_id: str, window_size: int) -> np.ndarray:
        """Get last N statements for a customer."""
        rows = self.customer_rows.get(customer_id, [])
        if not rows:
            return np.zeros((window_size, self.num_features), dtype=np.float32)

        # Take last window_size rows
        rows = rows[-window_size:]

        # Pad if needed
        features = self.features[rows]
        if len(features) < window_size:
            pad = np.zeros(
                (window_size - len(features), self.num_features), dtype=np.float32
            )
            features = np.vstack([pad, features])

        return features

    def get_pages_touched(self, customer_ids: List[str]) -> set:
        """Get set of pages touched when loading these customers."""
        pages = set()
        for cid in customer_ids:
            rows = self.customer_rows.get(cid, [])
            for r in rows[-self.config.window_size :]:
                pages.add(self.row_to_page[r])
        return pages


# =============================================================================
# Samplers
# =============================================================================


class BaseSampler:
    """Base class for customer samplers."""

    def __init__(self, dataset: AMEXDataset, config: AMEXConfig):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size

    def __iter__(self) -> Iterator[List[str]]:
        raise NotImplementedError

    def update_importance(self, customer_ids: List[str], gradients: torch.Tensor):
        """Update importance scores (only used by FIM sampler)."""
        pass


class RandomSampler(BaseSampler):
    """
    Random customer sampling.
    Worst-case I/O baseline - no locality guarantees.
    """

    def __iter__(self):
        customers = self.dataset.customer_ids.copy()
        random.shuffle(customers)

        for i in range(0, len(customers), self.batch_size):
            yield customers[i : i + self.batch_size]


class PageAwareSampler(BaseSampler):
    """
    Page-aware batching - maximize cache reuse.

    Batches consecutive customers based on storage order.
    Equivalent to page-aware batching from DGraphFin.
    """

    def __iter__(self):
        # Customers ordered by their first row index (storage order)
        customers = sorted(
            self.dataset.customer_ids, key=lambda c: self.dataset.customer_rows[c][0]
        )

        for i in range(0, len(customers), self.batch_size):
            yield customers[i : i + self.batch_size]


class FIMImportanceSampler(BaseSampler):
    """
    FIM-guided importance sampling.

    Tracks gradient magnitude per customer and prioritizes
    high-importance customers for revisiting.

    Note: NOT expected to improve convergence speed.
    Included for stability evaluation vs random baseline.
    """

    def __init__(self, dataset: AMEXDataset, config: AMEXConfig):
        super().__init__(dataset, config)

        # Initialize importance scores
        self.importance = np.ones(len(dataset.customer_ids), dtype=np.float32)
        self.ema = config.fim_ema
        self.budget = config.fim_budget

    def __iter__(self):
        n_customers = len(self.dataset.customer_ids)
        n_priority = int(n_customers * self.budget)

        # Split into priority (high FIM) and regular
        priority_idx = np.argsort(self.importance)[-n_priority:]
        regular_idx = np.argsort(self.importance)[:-n_priority]

        # Shuffle within groups
        np.random.shuffle(priority_idx)
        np.random.shuffle(regular_idx)

        # Interleave: each batch has some priority + some regular
        priority_per_batch = max(1, self.batch_size // 5)
        regular_per_batch = self.batch_size - priority_per_batch

        pi, ri = 0, 0
        while pi < len(priority_idx) or ri < len(regular_idx):
            batch_idx = []

            # Add priority customers
            for _ in range(priority_per_batch):
                if pi < len(priority_idx):
                    batch_idx.append(priority_idx[pi])
                    pi += 1

            # Add regular customers
            for _ in range(regular_per_batch):
                if ri < len(regular_idx):
                    batch_idx.append(regular_idx[ri])
                    ri += 1

            if batch_idx:
                yield [self.dataset.customer_ids[i] for i in batch_idx]

    def update_importance(self, customer_ids: List[str], gradients: torch.Tensor):
        """Update FIM scores using gradient magnitude."""
        # Handle 1D (batch,) or 2D (batch, features) gradients
        if gradients.dim() == 1:
            grad_norms = gradients.abs().cpu().numpy()
        else:
            grad_norms = gradients.norm(dim=-1).cpu().numpy()

        for cid, gn in zip(customer_ids, grad_norms):
            idx = self.dataset.customer_to_idx[cid]
            self.importance[idx] = self.ema * self.importance[idx] + (1 - self.ema) * gn


def get_sampler(name: str, dataset: AMEXDataset, config: AMEXConfig) -> BaseSampler:
    """Factory for samplers."""
    samplers = {
        "random": RandomSampler,
        "page_aware": PageAwareSampler,
        "fim_importance": FIMImportanceSampler,
    }
    return samplers[name](dataset, config)


# =============================================================================
# Model - Simple MLP (not the focus)
# =============================================================================


class DefaultPredictor(nn.Module):
    """
    Simple MLP for default prediction.

    Architecture is intentionally simple - this is a systems benchmark,
    not a model accuracy competition.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Aggregate window with attention-like pooling
        self.window_proj = nn.Linear(input_dim, hidden_dim)
        self.window_attn = nn.Linear(hidden_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window, features)
        returns: (batch,) logits
        """
        # Project features
        h = F.relu(self.window_proj(x))  # (batch, window, hidden)

        # Attention pooling over window
        attn = F.softmax(self.window_attn(h), dim=1)  # (batch, window, 1)
        pooled = (h * attn).sum(dim=1)  # (batch, hidden)

        # Classify
        return self.classifier(pooled).squeeze(-1)


# =============================================================================
# Metrics - Systems metrics are the focus
# =============================================================================


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""

    batch_idx: int
    batch_size: int

    # Timing
    load_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float

    # Locality
    pages_touched: int
    rows_loaded: int
    ra_fetch: float  # Read amplification

    # Learning (minimal)
    loss: float

    # Throughput
    samples_per_sec: float
    bytes_read: int


@dataclass
class EvalMetrics:
    """Evaluation metrics."""

    step: int
    runtime_sec: float

    # Learning metrics (minimal)
    f1: float
    pr_auc: float
    precision_at_half_pct: float
    recall_at_1_pct: float

    # Quartile metrics (prediction score distribution)
    q1: float
    q2: float
    q3: float
    q4: float

    # Aggregate systems metrics
    avg_ra_fetch: float
    avg_samples_per_sec: float
    total_pages_touched: int


class MetricsTracker:
    """Track and log metrics."""

    F1_THRESHOLD = 0.07  # Time-to-F1 target

    def __init__(self, config: AMEXConfig, use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb and HAS_WANDB
        self.batch_metrics: List[BatchMetrics] = []
        self.eval_metrics: List[EvalMetrics] = []
        self.start_time = time.time()
        self.all_pages = set()

        # Time-to-F1 tracking
        self.time_to_f1_threshold: Optional[float] = None
        self.batches_to_f1_threshold: Optional[int] = None

        # Cache hit tracking (pages seen before)
        self.pages_seen_ever: set = set()
        self.cumulative_cache_hits = 0
        self.cumulative_page_accesses = 0

    def update_cache_stats(self, pages: set):
        """Track cache hit rate based on pages seen before."""
        hits = len(pages & self.pages_seen_ever)
        self.cumulative_cache_hits += hits
        self.cumulative_page_accesses += len(pages)
        self.pages_seen_ever.update(pages)

    def get_cache_hit_rate(self) -> float:
        """Return cumulative cache hit rate."""
        if self.cumulative_page_accesses == 0:
            return 0.0
        return self.cumulative_cache_hits / self.cumulative_page_accesses

    def log_batch(self, m: BatchMetrics):
        self.batch_metrics.append(m)

        if self.use_wandb:
            log_dict = {
                "batch/load_time_ms": m.load_time_ms,
                "batch/forward_time_ms": m.forward_time_ms,
                "batch/backward_time_ms": m.backward_time_ms,
                "batch/total_time_ms": m.total_time_ms,
                "batch/pages_touched": m.pages_touched,
                "batch/ra_fetch": m.ra_fetch,
                "batch/loss": m.loss,
                "batch/samples_per_sec": m.samples_per_sec,
                "batch/bytes_read": m.bytes_read,
                "batch/cache_hit_rate": self.get_cache_hit_rate(),
            }
            # Add GPU metrics if available
            if torch.cuda.is_available():
                log_dict["gpu/memory_allocated_gb"] = (
                    torch.cuda.memory_allocated() / 1e9
                )
                log_dict["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                log_dict["gpu/max_memory_allocated_gb"] = (
                    torch.cuda.max_memory_allocated() / 1e9
                )
            wandb.log(log_dict, step=m.batch_idx)

    def log_eval(self, m: EvalMetrics):
        self.eval_metrics.append(m)

        # Track time-to-F1 threshold
        if self.time_to_f1_threshold is None and m.f1 >= self.F1_THRESHOLD:
            self.time_to_f1_threshold = m.runtime_sec
            self.batches_to_f1_threshold = m.step
            print(
                f"  *** F1 threshold {self.F1_THRESHOLD} reached at {m.runtime_sec:.1f}s (batch {m.step})"
            )

        if self.use_wandb:
            log_dict = {
                "eval/f1": m.f1,
                "eval/pr_auc": m.pr_auc,
                "eval/p_at_0.5%": m.precision_at_half_pct,
                "eval/r_at_1%": m.recall_at_1_pct,
                "eval/q1": m.q1,
                "eval/q2": m.q2,
                "eval/q3": m.q3,
                "eval/q4": m.q4,
                "eval/avg_ra_fetch": m.avg_ra_fetch,
                "eval/avg_samples_per_sec": m.avg_samples_per_sec,
                "eval/total_pages": m.total_pages_touched,
                "eval/runtime_sec": m.runtime_sec,
                "eval/cache_hit_rate": self.get_cache_hit_rate(),
            }
            # Log time-to-F1 if reached
            if self.time_to_f1_threshold is not None:
                log_dict["eval/time_to_f1_threshold"] = self.time_to_f1_threshold
                log_dict["eval/batches_to_f1_threshold"] = self.batches_to_f1_threshold
            wandb.log(log_dict, step=m.step)

    def get_summary(self) -> Dict:
        """Get final summary metrics."""
        if not self.batch_metrics:
            return {}

        return {
            "total_batches": len(self.batch_metrics),
            "total_runtime_sec": time.time() - self.start_time,
            "avg_ra_fetch": np.mean([m.ra_fetch for m in self.batch_metrics]),
            "avg_samples_per_sec": np.mean(
                [m.samples_per_sec for m in self.batch_metrics]
            ),
            "total_pages_touched": len(self.all_pages),
            "final_f1": self.eval_metrics[-1].f1 if self.eval_metrics else 0,
            "final_pr_auc": self.eval_metrics[-1].pr_auc if self.eval_metrics else 0,
            "cache_hit_rate": self.get_cache_hit_rate(),
            "time_to_f1_threshold": self.time_to_f1_threshold,
            "batches_to_f1_threshold": self.batches_to_f1_threshold,
            "total_bytes_read": sum(m.bytes_read for m in self.batch_metrics),
        }


# =============================================================================
# Training Loop
# =============================================================================


def evaluate(
    model: nn.Module,
    dataset: AMEXDataset,
    config: AMEXConfig,
    device: torch.device,
    n_samples: int = 10000,
) -> Dict:
    """Evaluate model on validation set."""
    model.eval()

    # Sample customers for eval
    eval_customers = random.sample(
        dataset.customer_ids, min(n_samples, len(dataset.customer_ids))
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(eval_customers), config.batch_size):
            batch_customers = eval_customers[i : i + config.batch_size]

            # Load windows
            windows = np.stack(
                [
                    dataset.get_customer_window(cid, config.window_size)
                    for cid in batch_customers
                ]
            )
            labels = np.array(
                [dataset.customer_to_label.get(cid, 0) for cid in batch_customers]
            )

            # Forward
            x = torch.from_numpy(windows).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    from sklearn.metrics import f1_score, precision_recall_curve, auc

    # F1 at 0.5 threshold
    pred_labels = (all_preds >= 0.5).astype(int)
    f1 = f1_score(all_labels, pred_labels, zero_division=0)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)

    # Precision at 0.5% and Recall at 1%
    sorted_idx = np.argsort(all_preds)[::-1]
    n_05_pct = max(1, int(len(all_preds) * 0.005))
    n_1_pct = max(1, int(len(all_preds) * 0.01))

    p_at_05 = all_labels[sorted_idx[:n_05_pct]].mean()
    r_at_1 = all_labels[sorted_idx[:n_1_pct]].sum() / max(1, all_labels.sum())

    # Quartile metrics (prediction score distribution)
    q1, q2, q3, q4 = np.percentile(all_preds, [25, 50, 75, 100])

    model.train()

    return {
        "f1": f1,
        "pr_auc": pr_auc,
        "p_at_0.5%": p_at_05,
        "r_at_1%": r_at_1,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
    }


def train(config: AMEXConfig, use_wandb: bool = True):
    """Run the benchmark."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    dataset = AMEXDataset(config)

    # Create sampler
    sampler = get_sampler(config.sampler, dataset, config)
    print(f"Sampler: {config.sampler}")

    # Create model
    model = DefaultPredictor(
        input_dim=dataset.num_features, hidden_dim=config.hidden_dim
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Metrics
    tracker = MetricsTracker(config, use_wandb=use_wandb)

    # Training loop
    print(f"\nTraining for {config.max_time}s...")
    start_time = time.time()
    batch_idx = 0
    epoch = 0

    while time.time() - start_time < config.max_time:
        epoch += 1

        for customer_batch in sampler:
            if time.time() - start_time >= config.max_time:
                break

            batch_idx += 1
            batch_start = time.time()

            # === Load batch ===
            load_start = time.time()
            windows = np.stack(
                [
                    dataset.get_customer_window(cid, config.window_size)
                    for cid in customer_batch
                ]
            )
            labels = np.array(
                [dataset.customer_to_label.get(cid, 0) for cid in customer_batch],
                dtype=np.float32,
            )

            # Track pages and cache hits
            pages = dataset.get_pages_touched(customer_batch)
            tracker.update_cache_stats(pages)
            tracker.all_pages.update(pages)

            x = torch.from_numpy(windows).to(device)
            y = torch.from_numpy(labels).to(device)
            load_time = (time.time() - load_start) * 1000

            # === Forward ===
            forward_start = time.time()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            forward_time = (time.time() - forward_start) * 1000

            # === Backward ===
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = (time.time() - backward_start) * 1000

            # === Update FIM (if applicable) ===
            if hasattr(sampler, "update_importance"):
                with torch.no_grad():
                    # Use output gradient as importance proxy
                    sampler.update_importance(customer_batch, logits.detach())

            # === Compute metrics ===
            total_time = (time.time() - batch_start) * 1000
            rows_loaded = len(customer_batch) * config.window_size
            bytes_read = len(pages) * config.page_size
            bytes_needed = rows_loaded * config.feature_bytes
            ra_fetch = bytes_read / max(1, bytes_needed)

            metrics = BatchMetrics(
                batch_idx=batch_idx,
                batch_size=len(customer_batch),
                load_time_ms=load_time,
                forward_time_ms=forward_time,
                backward_time_ms=backward_time,
                total_time_ms=total_time,
                pages_touched=len(pages),
                rows_loaded=rows_loaded,
                ra_fetch=ra_fetch,
                loss=loss.item(),
                samples_per_sec=len(customer_batch) / (total_time / 1000),
                bytes_read=bytes_read,
            )
            tracker.log_batch(metrics)

            # === Evaluate ===
            if batch_idx % config.eval_interval == 0:
                eval_results = evaluate(model, dataset, config, device)

                recent_batches = tracker.batch_metrics[-config.eval_interval :]
                eval_m = EvalMetrics(
                    step=batch_idx,
                    runtime_sec=time.time() - start_time,
                    f1=eval_results["f1"],
                    pr_auc=eval_results["pr_auc"],
                    precision_at_half_pct=eval_results["p_at_0.5%"],
                    recall_at_1_pct=eval_results["r_at_1%"],
                    q1=eval_results["q1"],
                    q2=eval_results["q2"],
                    q3=eval_results["q3"],
                    q4=eval_results["q4"],
                    avg_ra_fetch=np.mean([m.ra_fetch for m in recent_batches]),
                    avg_samples_per_sec=np.mean(
                        [m.samples_per_sec for m in recent_batches]
                    ),
                    total_pages_touched=len(tracker.all_pages),
                )
                tracker.log_eval(eval_m)

                print(
                    f"  [{batch_idx:5d}] F1={eval_m.f1:.4f} PR-AUC={eval_m.pr_auc:.4f} "
                    f"RA={eval_m.avg_ra_fetch:.2f}x Samples/s={eval_m.avg_samples_per_sec:.0f}"
                )

    # Final summary
    summary = tracker.get_summary()
    print(f"\n=== Final Summary ===")
    print(f"Batches: {summary['total_batches']}")
    print(f"Runtime: {summary['total_runtime_sec']:.1f}s")
    print(f"Avg RA_fetch: {summary['avg_ra_fetch']:.2f}x")
    print(f"Avg Samples/s: {summary['avg_samples_per_sec']:.0f}")
    print(f"Total pages: {summary['total_pages_touched']:,}")
    print(f"Cache hit rate: {summary['cache_hit_rate']:.2%}")
    print(f"Total bytes read: {summary['total_bytes_read'] / 1e9:.2f} GB")
    if summary["time_to_f1_threshold"] is not None:
        print(
            f"Time to F1>={MetricsTracker.F1_THRESHOLD}: {summary['time_to_f1_threshold']:.1f}s (batch {summary['batches_to_f1_threshold']})"
        )
    else:
        print(f"Time to F1>={MetricsTracker.F1_THRESHOLD}: NOT REACHED")
    print(f"Final F1: {summary['final_f1']:.4f}")
    print(f"Final PR-AUC: {summary['final_pr_auc']:.4f}")

    if use_wandb and HAS_WANDB:
        wandb.log({"final/" + k: v for k, v in summary.items()})

    return summary


# =============================================================================
# Streaming Inference Benchmark
# =============================================================================


@dataclass
class StreamingInferenceResult:
    """Results from streaming inference benchmark."""

    target_tps: int
    achieved_tps: float
    duration_sec: float
    total_events: int

    # Latency percentiles (ms)
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    mean_latency: float

    # Systems metrics
    cache_hit_rate: float
    bytes_per_event: float

    # Cross-entity retrieval (k=0 means disabled)
    k_related: int = 0
    related_cache_hit_rate: float = 0.0
    total_entities_fetched: int = 0


def infer_stream(
    config: AMEXConfig,
    target_tps: int = 10000,
    duration: int = 60,
    use_wandb: bool = True,
    use_cpp_sampler: bool = True,
    k_related: int = 0,
) -> StreamingInferenceResult:
    """
    Streaming inference benchmark with TPS load generator.

    Measures end-to-end latency distribution under configurable load.
    Reports p50/p95/p99 latency and achieved throughput.

    Args:
        k_related: Number of related entities to fetch per event (0=disabled).
                   Used for cross-entity retrieval locality stress testing.
                   Higher k increases data locality pressure.
    """
    import time

    # Try to use C++ sampler for lower latency
    cpp_available = False
    related_lookup = None
    if use_cpp_sampler:
        try:
            import sys

            sys.path.insert(0, "gnn/cpp_extension")
            import amex_sampler_cpp as cpp

            cpp_available = True
            print("Using C++ samplers for lower latency")
        except ImportError:
            print("C++ samplers not available, using Python")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    dataset = AMEXDataset(config)

    # Create model (inference only)
    model = DefaultPredictor(
        input_dim=dataset.num_features, hidden_dim=config.hidden_dim
    ).to(device)
    model.eval()

    # Setup related entity lookup for cross-entity retrieval
    num_entities = len(dataset.customer_ids)
    if k_related > 0:
        if cpp_available:
            related_lookup = cpp.RelatedEntityLookup(num_entities, k_related, seed=42)
            print(f"Cross-entity retrieval: k={k_related} related entities per event")
        else:
            # Python fallback for related entity lookup
            print(f"Cross-entity retrieval (Python): k={k_related}")
            rng_related = random.Random(42)
            related_mapping = {}
            for i in range(num_entities):
                candidates = list(range(num_entities))
                candidates.remove(i)
                related_mapping[i] = rng_related.sample(
                    candidates, min(k_related, len(candidates))
                )

    # Setup latency tracking
    if cpp_available:
        latency_tracker = cpp.LatencyTracker(duration * target_tps)
    else:
        latencies = []

    # Setup entity indices for streaming
    entity_indices = list(range(num_entities))
    random.shuffle(entity_indices)
    entity_ptr = 0

    # Cache tracking (primary entity)
    pages_seen = set()
    total_pages_accessed = 0
    cache_hits = 0
    total_bytes = 0

    # Cache tracking (related entities)
    related_entities_seen = set()
    total_related_fetched = 0
    related_cache_hits = 0

    # Event timing
    event_interval = 1.0 / target_tps  # seconds between events
    events_processed = 0

    print(f"\nStreaming inference: target {target_tps} TPS for {duration}s...")
    print(f"Expected events: {target_tps * duration:,}")

    start_time = time.time()
    next_event_time = start_time

    with torch.no_grad():
        while time.time() - start_time < duration:
            # Rate limiting - wait for next event slot
            current_time = time.time()
            if current_time < next_event_time:
                # Busy wait for accuracy (sleep has too much jitter)
                while time.time() < next_event_time:
                    pass

            event_start = time.time()

            # Get next entity (circular)
            entity_idx = entity_indices[entity_ptr]
            entity_ptr = (entity_ptr + 1) % len(entity_indices)
            customer_id = dataset.customer_ids[entity_idx]

            # Get features for this entity
            window = dataset.get_customer_window(customer_id, config.window_size)

            # Track cache behavior (primary entity)
            rows = dataset.customer_rows.get(customer_id, [])
            for r in rows[-config.window_size :]:
                page = dataset.row_to_page[r]
                total_pages_accessed += 1
                if page in pages_seen:
                    cache_hits += 1
                else:
                    pages_seen.add(page)
            total_bytes += config.window_size * config.feature_bytes

            # Cross-entity retrieval (if enabled)
            if k_related > 0:
                if cpp_available and related_lookup is not None:
                    related_indices = related_lookup.get_related(entity_idx)
                    related_list = related_indices.tolist()
                else:
                    related_list = related_mapping.get(entity_idx, [])

                # Fetch related entity windows and track cache
                for rel_idx in related_list:
                    total_related_fetched += 1
                    if rel_idx in related_entities_seen:
                        related_cache_hits += 1
                    else:
                        related_entities_seen.add(rel_idx)

                    # Actually fetch the related entity's window (simulates data access)
                    rel_customer_id = dataset.customer_ids[rel_idx]
                    rel_window = dataset.get_customer_window(
                        rel_customer_id, config.window_size
                    )

                    # Track page accesses for related entities
                    rel_rows = dataset.customer_rows.get(rel_customer_id, [])
                    for r in rel_rows[-config.window_size :]:
                        page = dataset.row_to_page[r]
                        total_pages_accessed += 1
                        if page in pages_seen:
                            cache_hits += 1
                        else:
                            pages_seen.add(page)

                    total_bytes += config.window_size * config.feature_bytes

            # Run inference
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            logits = model(x)
            _ = torch.sigmoid(logits)

            # Record latency
            event_end = time.time()
            latency_ms = (event_end - event_start) * 1000

            if cpp_available:
                latency_tracker.record(latency_ms)
            else:
                latencies.append(latency_ms)

            events_processed += 1
            next_event_time += event_interval

            # Progress update
            if events_processed % 10000 == 0:
                elapsed = time.time() - start_time
                actual_tps = events_processed / elapsed
                print(f"  [{events_processed:,}] TPS={actual_tps:.0f}")

    end_time = time.time()
    actual_duration = end_time - start_time
    achieved_tps = events_processed / actual_duration

    # Compute latency percentiles
    if cpp_available:
        pcts = latency_tracker.get_percentiles()
        p50 = pcts[0].item()
        p95 = pcts[1].item()
        p99 = pcts[2].item()
        max_lat = pcts[3].item()
        mean_lat = latency_tracker.mean()
    else:
        latencies.sort()
        n = len(latencies)
        p50 = latencies[n * 50 // 100] if n > 0 else 0
        p95 = latencies[n * 95 // 100] if n > 0 else 0
        p99 = latencies[n * 99 // 100] if n > 0 else 0
        max_lat = latencies[-1] if n > 0 else 0
        mean_lat = sum(latencies) / n if n > 0 else 0

    cache_hit_rate = (
        cache_hits / total_pages_accessed if total_pages_accessed > 0 else 0
    )
    bytes_per_event = total_bytes / events_processed if events_processed > 0 else 0

    # Related entity cache metrics
    related_hit_rate = (
        related_cache_hits / total_related_fetched if total_related_fetched > 0 else 0
    )

    result = StreamingInferenceResult(
        target_tps=target_tps,
        achieved_tps=achieved_tps,
        duration_sec=actual_duration,
        total_events=events_processed,
        p50_latency=p50,
        p95_latency=p95,
        p99_latency=p99,
        max_latency=max_lat,
        mean_latency=mean_lat,
        cache_hit_rate=cache_hit_rate,
        bytes_per_event=bytes_per_event,
        k_related=k_related,
        related_cache_hit_rate=related_hit_rate,
        total_entities_fetched=total_related_fetched,
    )

    # Print results
    print(f"\n=== Streaming Inference Results ===")
    print(f"Target TPS: {target_tps:,}")
    print(f"Achieved TPS: {achieved_tps:,.0f}")
    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total events: {events_processed:,}")
    print(f"\nLatency (ms):")
    print(f"  p50:  {p50:.3f}")
    print(f"  p95:  {p95:.3f}")
    print(f"  p99:  {p99:.3f}")
    print(f"  max:  {max_lat:.3f}")
    print(f"  mean: {mean_lat:.3f}")
    print(f"\nSystems:")
    print(f"  Cache hit rate: {cache_hit_rate:.2%}")
    print(f"  Bytes/event: {bytes_per_event:.0f}")
    if k_related > 0:
        print(f"\nCross-Entity Retrieval (k={k_related}):")
        print(f"  Related entities fetched: {total_related_fetched:,}")
        print(f"  Related cache hit rate: {related_hit_rate:.2%}")

    if use_wandb and HAS_WANDB:
        log_dict = {
            "infer/target_tps": target_tps,
            "infer/achieved_tps": achieved_tps,
            "infer/p50_latency_ms": p50,
            "infer/p95_latency_ms": p95,
            "infer/p99_latency_ms": p99,
            "infer/max_latency_ms": max_lat,
            "infer/mean_latency_ms": mean_lat,
            "infer/cache_hit_rate": cache_hit_rate,
            "infer/bytes_per_event": bytes_per_event,
            "infer/k_related": k_related,
        }
        if k_related > 0:
            log_dict["infer/related_cache_hit_rate"] = related_hit_rate
            log_dict["infer/total_related_fetched"] = total_related_fetched
        wandb.log(log_dict)

    return result


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="AMEX Streaming Locality Benchmark")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="train",
        help="Benchmark mode: train (throughput) or infer (latency)",
    )

    # Sampler settings
    parser.add_argument(
        "--sampler",
        choices=["random", "page_aware", "fim_importance"],
        default="random",
        help="Sampler type",
    )

    # Time/duration settings
    parser.add_argument("--time", type=int, default=300, help="Max time (seconds)")

    # Batch and window settings
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Customers per batch"
    )
    parser.add_argument(
        "--window-size", type=int, default=6, help="Statements per customer"
    )

    # Inference-specific settings
    parser.add_argument(
        "--target-tps", type=int, default=10000, help="Target TPS for infer mode"
    )
    parser.add_argument(
        "--tps-sweep",
        type=str,
        default=None,
        help="Comma-separated TPS values to sweep (e.g., '1000,5000,10000,50000')",
    )

    # Cross-entity retrieval settings
    parser.add_argument(
        "--k-related",
        type=int,
        default=0,
        help="Number of related entities to fetch per event (0=disabled, try 8,16,32)",
    )
    parser.add_argument(
        "--k-sweep",
        type=str,
        default=None,
        help="Comma-separated k values to sweep (e.g., '0,8,16,32')",
    )

    # Ablation and output settings
    parser.add_argument("--ablation", action="store_true", help="Run all samplers")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--wandb-project", default="amex-locality-benchmark")

    args = parser.parse_args()

    config = AMEXConfig(
        sampler=args.sampler,
        max_time=args.time,
        batch_size=args.batch_size,
        window_size=args.window_size,
    )

    use_wandb = not args.no_wandb and HAS_WANDB

    # Handle inference mode
    if args.mode == "infer":
        if args.k_sweep:
            # Cross-entity retrieval sweep mode
            k_values = [int(x.strip()) for x in args.k_sweep.split(",")]
            results = []

            for k in k_values:
                print(f"\n{'='*60}")
                print(f"Cross-entity retrieval: k={k} at {args.target_tps:,} TPS")
                print(f"{'='*60}")

                if use_wandb:
                    init_wandb_with_system_metrics(
                        project=args.wandb_project,
                        name=f"infer-k{k}-{args.target_tps}tps-{args.time}s",
                        config=vars(config),
                        reinit=True,
                    )

                result = infer_stream(
                    config,
                    target_tps=args.target_tps,
                    duration=args.time,
                    use_wandb=use_wandb,
                    k_related=k,
                )
                results.append(result)

                if use_wandb:
                    wandb.finish()

            # Print comparison
            print(f"\n{'='*60}")
            print("CROSS-ENTITY RETRIEVAL SWEEP RESULTS")
            print(f"{'='*60}")
            print(
                f"{'k':>4} {'TPS':>12} {'p50':>8} {'p95':>8} {'p99':>8} "
                f"{'Cache':>8} {'Rel Cache':>10} {'Bytes/evt':>10}"
            )
            print("-" * 80)
            for r in results:
                print(
                    f"{r.k_related:>4} {r.achieved_tps:>12,.0f} "
                    f"{r.p50_latency:>8.3f} {r.p95_latency:>8.3f} {r.p99_latency:>8.3f} "
                    f"{r.cache_hit_rate:>8.2%} {r.related_cache_hit_rate:>10.2%} "
                    f"{r.bytes_per_event:>10.0f}"
                )

        elif args.tps_sweep:
            # TPS sweep mode
            tps_values = [int(x.strip()) for x in args.tps_sweep.split(",")]
            results = []

            for tps in tps_values:
                print(f"\n{'='*60}")
                print(f"Streaming inference: target {tps:,} TPS")
                print(f"{'='*60}")

                if use_wandb:
                    init_wandb_with_system_metrics(
                        project=args.wandb_project,
                        name=f"infer-{tps}tps-{args.time}s",
                        config=vars(config),
                        reinit=True,
                    )

                result = infer_stream(
                    config,
                    target_tps=tps,
                    duration=args.time,
                    use_wandb=use_wandb,
                    k_related=args.k_related,
                )
                results.append(result)

                if use_wandb:
                    wandb.finish()

            # Print comparison
            print(f"\n{'='*60}")
            print("TPS SWEEP RESULTS")
            print(f"{'='*60}")
            print(
                f"{'Target TPS':>12} {'Achieved':>12} {'p50':>8} {'p95':>8} {'p99':>8} {'Cache Hit':>10}"
            )
            print("-" * 70)
            for r in results:
                print(
                    f"{r.target_tps:>12,} {r.achieved_tps:>12,.0f} "
                    f"{r.p50_latency:>8.3f} {r.p95_latency:>8.3f} {r.p99_latency:>8.3f} "
                    f"{r.cache_hit_rate:>10.2%}"
                )
        else:
            # Single TPS run
            if use_wandb:
                k_suffix = f"-k{args.k_related}" if args.k_related > 0 else ""
                init_wandb_with_system_metrics(
                    project=args.wandb_project,
                    name=f"infer-{args.target_tps}tps{k_suffix}-{args.time}s",
                    config=vars(config),
                )

            infer_stream(
                config,
                target_tps=args.target_tps,
                duration=args.time,
                use_wandb=use_wandb,
                k_related=args.k_related,
            )

            if use_wandb:
                wandb.finish()

    elif args.ablation:
        # Run all samplers (train mode)
        samplers = ["random", "page_aware", "fim_importance"]
        results = {}

        for sampler in samplers:
            print(f"\n{'='*60}")
            print(f"Running {sampler} sampler")
            print(f"{'='*60}")

            config.sampler = sampler

            if use_wandb:
                init_wandb_with_system_metrics(
                    project=args.wandb_project,
                    name=f"{sampler}-{args.time}s",
                    config=vars(config),
                    reinit=True,
                )

            results[sampler] = train(config, use_wandb=use_wandb)

            if use_wandb:
                wandb.finish()

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(
            f"{'Sampler':<15} {'RA_fetch':>10} {'Samples/s':>12} {'F1':>8} {'PR-AUC':>8}"
        )
        print("-" * 60)
        for sampler, r in results.items():
            print(
                f"{sampler:<15} {r['avg_ra_fetch']:>10.2f}x {r['avg_samples_per_sec']:>12.0f} "
                f"{r['final_f1']:>8.4f} {r['final_pr_auc']:>8.4f}"
            )

    else:
        # Single sampler run (train mode)
        if use_wandb:
            init_wandb_with_system_metrics(
                project=args.wandb_project,
                name=f"{args.sampler}-{args.time}s",
                config=vars(config),
            )

        train(config, use_wandb=use_wandb)

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
