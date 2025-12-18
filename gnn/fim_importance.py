#!/usr/bin/env python3
"""
FIM-Guided Fraud Detection: Node Importance via Backward Hooks

Uses backward hooks to capture node-level grad² during normal training,
avoiding expensive separate autograd.grad() calls. This gives ~zero overhead
importance tracking that piggybacks on the existing backward pass.

Key changes from v1:
- No autograd.grad() - uses register_hook() on hidden states
- Sparse updates - only updates nodes in each batch
- Class-conditional weighting - scales updates by fraud mask
- Step-based replication - updates on step schedule, not epoch boundaries

Usage:
    tracker = NodeImportanceTracker(num_nodes, page_id)

    # In training loop:
    hidden = model.get_hidden()  # [N_batch, D]
    tracker.register_hook(hidden, batch_nodes, batch_y)

    loss.backward()  # Hook fires automatically

    # Periodically update replication set
    if step % replication_interval == 0:
        tracker.update_replication_set()
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


@dataclass
class FIMConfig:
    """Configuration for FIM-guided importance tracking."""

    # Update frequency (now cheap, can be frequent)
    importance_interval: int = 1  # Update every batch (it's free now)

    # EMA parameters
    beta: float = 0.01  # EMA smoothing for node importance

    # Pagepair boost for cross-page sampling
    lambda_pagepair: float = 0.5

    # Gradient clipping (cap at p99 to avoid exploding importance)
    grad_clip_percentile: float = 99.0

    # Replication settings
    replication_budget: float = 0.005  # 0.5% of nodes
    replication_interval_steps: int = 5000  # Update replication every N steps
    replication_interval_seconds: float = 60.0  # Or every N seconds

    # Importance computation
    importance_root: str = "fourth_root"  # "sqrt", "fourth_root", or "raw"

    # Class-conditional weighting
    fraud_weight: float = 1.0  # Weight for fraud positives
    negative_weight: float = 0.1  # Weight for negatives
    hard_negative_threshold: float = 0.7  # P(fraud) > this for hard negatives
    hard_negative_weight: float = 0.5  # Weight for hard negatives


class NodeImportanceTable:
    """
    Per-node importance table with sparse EMA updates.

    Only updates nodes that appear in each batch, keeping overhead
    proportional to batch size, not graph size.
    """

    def __init__(self, num_nodes: int, beta: float = 0.01):
        self.num_nodes = num_nodes
        self.beta = beta
        self.importance = np.zeros(num_nodes, dtype=np.float32)
        self.update_count = np.zeros(num_nodes, dtype=np.int32)
        self.cross_page_incidents = np.zeros(num_nodes, dtype=np.int32)
        self._nodes_seen = 0  # Track coverage incrementally

    def update_sparse(
        self,
        node_ids: np.ndarray,
        scores: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Sparse EMA update for batch nodes only (vectorized).

        Args:
            node_ids: Global node IDs in this batch
            scores: Importance scores (grad² based)
            weights: Optional per-node weights (e.g., fraud mask)
        """
        if weights is not None:
            scores = scores * weights

        # Vectorized EMA update
        # First-time nodes: importance = score
        # Seen nodes: importance = (1-beta)*old + beta*new
        first_time = self.update_count[node_ids] == 0
        seen = ~first_time

        # Track new nodes for coverage
        new_node_count = first_time.sum()
        self._nodes_seen += new_node_count

        if first_time.any():
            self.importance[node_ids[first_time]] = scores[first_time]
        if seen.any():
            self.importance[node_ids[seen]] = (
                (1 - self.beta) * self.importance[node_ids[seen]]
                + self.beta * scores[seen]
            )
        self.update_count[node_ids] += 1

    def update_cross_page_incidents(self, node_ids: np.ndarray):
        """Track how many times each node appears in cross-page edges."""
        np.add.at(self.cross_page_incidents, node_ids, 1)

    def get(self, node_ids: np.ndarray) -> np.ndarray:
        """Get importance scores for given nodes."""
        return self.importance[node_ids]

    def get_replication_scores(self, cross_page_degree: np.ndarray) -> np.ndarray:
        """
        Compute replication scores: importance * cross_page_degree.

        Args:
            cross_page_degree: Array of cross-page degree per node

        Returns:
            Replication scores for all nodes
        """
        return self.importance * cross_page_degree

    def get_replication_scores_from_incidents(self) -> np.ndarray:
        """
        Compute replication scores using tracked cross-page incidents.

        Returns:
            Replication scores: importance * cross_page_incidents_seen
        """
        return self.importance * self.cross_page_incidents

    def stats(self) -> Dict[str, float]:
        """Return statistics about importance distribution (fast path)."""
        # Use cached coverage for speed (O(1) instead of O(num_nodes))
        coverage = self._nodes_seen / self.num_nodes if self.num_nodes > 0 else 0
        return {
            "coverage": float(coverage),
            "nodes_seen": int(self._nodes_seen),
        }

    def full_stats(self) -> Dict[str, float]:
        """Return full statistics (expensive - scans all nodes)."""
        active_mask = self.update_count > 0
        active = self.importance[active_mask]
        if len(active) == 0:
            return {"mean": 0, "std": 0, "max": 0, "min": 0, "coverage": 0}
        return {
            "mean": float(active.mean()),
            "std": float(active.std()),
            "max": float(active.max()),
            "min": float(active.min()),
            "coverage": float(active_mask.sum() / self.num_nodes),
            "total_updates": int(self.update_count.sum()),
        }


class PagePairImportanceTable:
    """
    Cross-page edge importance table (sparse hashmap).

    Tracks importance of page-pair connections for guiding cross-page
    neighbor selection.
    """

    def __init__(self, max_entries: int = 1_000_000):
        self.max_entries = max_entries
        self.importance: Dict[Tuple[int, int], float] = defaultdict(float)
        self.access_order: List[Tuple[int, int]] = []

    def update_batch(
        self,
        page_src: np.ndarray,
        page_dst: np.ndarray,
        scores: np.ndarray,
    ):
        """Batch update for cross-page edges (vectorized aggregation)."""
        cross_mask = page_src != page_dst
        if not cross_mask.any():
            return

        # Get cross-page subset
        ps_cross = page_src[cross_mask]
        pd_cross = page_dst[cross_mask]
        scores_cross = scores[cross_mask]

        # Aggregate scores by unique page pairs using pandas-style groupby
        # Create combined key for fast aggregation
        keys = ps_cross.astype(np.int64) * 1_000_000 + pd_cross.astype(np.int64)
        unique_keys, inverse = np.unique(keys, return_inverse=True)

        # Sum scores per unique key
        aggregated = np.bincount(inverse, weights=scores_cross)

        # Update importance dict (still needs loop but much smaller)
        for idx, key in enumerate(unique_keys):
            ps = int(key // 1_000_000)
            pd = int(key % 1_000_000)
            self.importance[(ps, pd)] += aggregated[idx]  # defaultdict handles missing keys

        # Simple eviction: clear oldest half if over capacity (no LRU tracking)
        if len(self.importance) > self.max_entries:
            items = list(self.importance.items())
            items.sort(key=lambda x: x[1])  # Sort by importance
            # Keep top half as defaultdict
            self.importance = defaultdict(float, items[len(items) // 2 :])

    def get_normalized(self, page_src: int, page_dst: int) -> float:
        """Get normalized importance (0-1 range)."""
        raw = self.importance.get((page_src, page_dst), 0.0)
        max_imp = max(self.importance.values()) if self.importance else 1.0
        return raw / max_imp if max_imp > 0 else 0.0

    def stats(self) -> Dict[str, float]:
        """Return statistics about page-pair importance."""
        if not self.importance:
            return {"num_pairs": 0, "mean": 0, "max": 0}
        values = list(self.importance.values())
        return {
            "num_pairs": len(self.importance),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }


class NodeImportanceTracker:
    """
    Tracks node importance using backward hooks on hidden states.

    Optimized for throughput:
    - Persistent hook (register once, not per batch)
    - GPU-side grad² accumulation with batched CPU drain
    - Minimal per-batch Python overhead
    """

    def __init__(
        self,
        num_nodes: int,
        page_id: np.ndarray,
        config: Optional[FIMConfig] = None,
        device: str = "cuda",
        drain_interval: int = 100,  # Drain GPU buffer every N batches
    ):
        self.num_nodes = num_nodes
        self.page_id = page_id
        self.config = config or FIMConfig()
        self.device = device
        self.drain_interval = drain_interval

        # Importance tables (CPU)
        self.node_table = NodeImportanceTable(num_nodes, self.config.beta)
        self.pagepair_table = PagePairImportanceTable()

        # Replication state
        self.replicated_nodes: Set[int] = set()
        self.last_replication_step = 0
        self.last_replication_time = time.time()

        # Persistent hook state - set once, reused
        self._current_batch_nodes: Optional[torch.Tensor] = None  # GPU tensor
        self._current_batch_weight: float = 1.0  # Scalar weight
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._hook_target: Optional[torch.Tensor] = None

        # GPU accumulation buffer for batched drain
        self._gpu_node_buffer: List[torch.Tensor] = []
        self._gpu_grad_buffer: List[torch.Tensor] = []
        self._pending_batches = 0

        # Stats
        self.total_updates = 0
        self.batches_with_fraud = 0
        self._grad_clip_value: Optional[float] = None

    def set_batch(self, batch_nodes: torch.Tensor, batch_weight: float = 1.0):
        """
        Set current batch info for persistent hook (minimal overhead).

        Args:
            batch_nodes: Node IDs as torch tensor (keep on GPU)
            batch_weight: Scalar weight for this batch
        """
        self._current_batch_nodes = batch_nodes
        self._current_batch_weight = batch_weight

    def _backward_hook(self, grad: torch.Tensor):
        """
        Persistent hook - accumulates grad² on GPU, drains periodically.
        """
        if self._current_batch_nodes is None:
            return

        # Compute grad² on GPU (no CPU sync)
        grad_sq = (grad.detach() ** 2).sum(dim=-1)  # stays on GPU

        # Scale by batch weight
        if self._current_batch_weight != 1.0:
            grad_sq = grad_sq * self._current_batch_weight

        # Accumulate in GPU buffer
        self._gpu_node_buffer.append(self._current_batch_nodes)
        self._gpu_grad_buffer.append(grad_sq)
        self._pending_batches += 1
        self.total_updates += 1

        # Drain to CPU periodically
        if self._pending_batches >= self.drain_interval:
            self._drain_to_cpu()

    def _drain_to_cpu(self):
        """Drain GPU buffers to CPU importance table (batched transfer)."""
        if not self._gpu_node_buffer:
            return

        # Single batched transfer to CPU
        all_nodes = torch.cat(self._gpu_node_buffer).cpu().numpy()
        all_grads = torch.cat(self._gpu_grad_buffer).cpu().numpy()

        # Apply transform
        if self.config.importance_root == "sqrt":
            all_grads = np.sqrt(all_grads + 1e-10)
        elif self.config.importance_root == "fourth_root":
            all_grads = np.power(all_grads + 1e-10, 0.25)

        # Clip
        if self._grad_clip_value is not None:
            all_grads = np.clip(all_grads, 0, self._grad_clip_value)
        else:
            # Initialize clip value
            self._grad_clip_value = np.percentile(all_grads, 99) if len(all_grads) > 100 else None

        # Update importance table
        self.node_table.update_sparse(all_nodes, all_grads)

        # Clear buffers
        self._gpu_node_buffer.clear()
        self._gpu_grad_buffer.clear()
        self._pending_batches = 0

    def register_hook(
        self,
        hidden_states: torch.Tensor,
        batch_nodes,
        batch_y=None,
        batch_edge_index=None,
        batch_probs=None,
    ):
        """
        Register hook on this batch's hidden tensor.
        Hook must be registered each batch since hidden tensor changes.
        """
        # Convert nodes to tensor if needed (keep on GPU)
        if isinstance(batch_nodes, np.ndarray):
            batch_nodes = torch.from_numpy(batch_nodes).long().to(hidden_states.device)
        elif isinstance(batch_nodes, torch.Tensor) and batch_nodes.device != hidden_states.device:
            batch_nodes = batch_nodes.to(hidden_states.device)

        self.set_batch(batch_nodes, 1.0)

        # Must register on EACH batch's hidden tensor (tensor changes each forward)
        # Remove old hook if exists
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._hook_handle = hidden_states.register_hook(self._backward_hook)

    def clear_hook(self):
        """No-op - hook is persistent now."""
        pass

    def finish(self):
        """Call at end of training to drain remaining data and remove hook."""
        self._drain_to_cpu()
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def should_update_replication(self, current_step: int) -> bool:
        """Check if replication set should be updated."""
        steps_since = current_step - self.last_replication_step
        time_since = time.time() - self.last_replication_time

        return (
            steps_since >= self.config.replication_interval_steps
            or time_since >= self.config.replication_interval_seconds
        )

    def update_replication_set(self, current_step: int) -> Set[int]:
        """
        Update the set of replicated nodes based on current importance.

        Uses: rep_score[u] = node_imp[u] * cross_page_incidents_seen[u]

        Returns:
            Set of node IDs to replicate
        """
        # Compute replication scores
        rep_scores = self.node_table.get_replication_scores_from_incidents()

        # Select top budget% nodes
        k = int(self.num_nodes * self.config.replication_budget)
        if k == 0:
            self.replicated_nodes = set()
        else:
            top_idx = np.argpartition(rep_scores, -k)[-k:]
            self.replicated_nodes = set(top_idx.tolist())

        self.last_replication_step = current_step
        self.last_replication_time = time.time()

        return self.replicated_nodes

    def get_replicated_nodes(self) -> Set[int]:
        """Get current set of replicated nodes."""
        return self.replicated_nodes

    def stats(self) -> Dict[str, float]:
        """Get tracker statistics (fast path)."""
        node_stats = self.node_table.stats()  # Fast O(1) stats

        return {
            "total_updates": self.total_updates,
            "batches_with_fraud": self.batches_with_fraud,
            "node_coverage": node_stats["coverage"],
            "replicated_nodes": len(self.replicated_nodes),
        }

    def full_stats(self) -> Dict[str, float]:
        """Get full tracker statistics (expensive)."""
        node_stats = self.node_table.full_stats()
        pagepair_stats = self.pagepair_table.stats()

        return {
            "total_updates": self.total_updates,
            "batches_with_fraud": self.batches_with_fraud,
            "node_coverage": node_stats["coverage"],
            "node_importance_mean": node_stats["mean"],
            "node_importance_max": node_stats["max"],
            "pagepair_count": pagepair_stats["num_pairs"],
            "replicated_nodes": len(self.replicated_nodes),
        }


def create_importance_tracker(
    num_nodes: int,
    page_id: np.ndarray,
    config: Optional[FIMConfig] = None,
) -> NodeImportanceTracker:
    """
    Create a node importance tracker.

    Args:
        num_nodes: Total number of nodes
        page_id: Array mapping node_id -> page_id
        config: Optional FIMConfig

    Returns:
        NodeImportanceTracker instance
    """
    return NodeImportanceTracker(num_nodes, page_id, config)
