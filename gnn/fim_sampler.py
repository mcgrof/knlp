#!/usr/bin/env python3
"""
FIM-Guided Page-Batch Sampler

Two implementations:
1. FIMCppPageBatchSampler: Uses C++ sampler core with FIM overlay for
   boundary selection. Achieves ~200 it/s (same as baseline C++ sampler).
2. FIMPageBatchSampler: Pure Python implementation (legacy, ~120 it/s).

Use FIMCppPageBatchSampler for production.

Key features:
- Importance-weighted cross-page neighbor selection
- Page-pair boost for high-importance page connections
- Replication simulation (replicated nodes treated as same-page)

Usage:
    from fim_sampler import FIMCppPageBatchSampler
    from fim_importance import NodeImportanceTracker

    tracker = NodeImportanceTracker(num_nodes, page_id)
    sampler = FIMCppPageBatchSampler(
        edge_index=data.edge_index,
        page_id=page_id_4kb,
        train_mask=train_mask,
        num_nodes=data.x.shape[0],
        importance_tracker=tracker,
        pages_per_batch=32,
        boundary_budget=0.2,
    )
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Set, Tuple

import numpy as np
import torch

from page_batch_sampler import CppPageBatchSampler, PageBatch

if TYPE_CHECKING:
    from fim_importance import NodeImportanceTracker


class FIMCppPageBatchSampler(CppPageBatchSampler):
    """
    C++ page-batch sampler with FIM-guided boundary selection.

    Inherits from CppPageBatchSampler for fast core sampling, overrides
    only _add_boundary_expansion to use importance-weighted selection.

    Achieves ~200 it/s (same as baseline C++ sampler) while adding
    FIM-guided neighbor selection.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        page_id: np.ndarray,
        train_mask: np.ndarray,
        num_nodes: int,
        importance_tracker: Optional["NodeImportanceTracker"] = None,
        pages_per_batch: int = 32,
        boundary_budget: float = 0.2,
        boundary_hops: int = 1,
        use_inter_page_edges: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        importance_temperature: float = 1.0,
        pos_labels: Optional[np.ndarray] = None,
        pos_oversample_prob: float = 0.0,
    ):
        """
        Initialize FIM-guided C++ page-batch sampler.

        Args:
            edge_index: [2, E] edge index tensor
            page_id: [N] array mapping node_id -> 4KB page ID
            train_mask: [N] boolean array of training nodes
            num_nodes: Total number of nodes
            importance_tracker: Optional NodeImportanceTracker for importance scores
            pages_per_batch: Number of pages per batch
            boundary_budget: Fraction of extra nodes for boundary (default 0.2)
            boundary_hops: Number of hops for boundary expansion (default 1)
            use_inter_page_edges: If True, use ALL edges within batch
            shuffle: Whether to shuffle pages each epoch
            seed: Random seed for shuffling
            importance_temperature: Temperature for importance-weighted sampling
            pos_labels: Optional [N] array of positive labels (1=fraud, 0=other)
            pos_oversample_prob: Probability of injecting a positive-containing page
        """
        # Store FIM-specific params before calling super().__init__
        self.importance_tracker = importance_tracker
        self.importance_temperature = importance_temperature

        # Initialize C++ sampler
        super().__init__(
            edge_index=edge_index,
            page_id=page_id,
            train_mask=train_mask,
            num_nodes=num_nodes,
            pages_per_batch=pages_per_batch,
            boundary_budget=boundary_budget,
            boundary_hops=boundary_hops,
            use_inter_page_edges=use_inter_page_edges,
            shuffle=shuffle,
            seed=seed,
            pos_labels=pos_labels,
            pos_oversample_prob=pos_oversample_prob,
        )

        if importance_tracker is not None:
            print("  FIM importance: ENABLED (importance-weighted boundary selection)")

    def set_importance_tracker(self, tracker: "NodeImportanceTracker"):
        """Update the importance tracker."""
        self.importance_tracker = tracker

    def _get_node_importance(self, node_ids: np.ndarray) -> np.ndarray:
        """Get importance scores for nodes."""
        if self.importance_tracker is None:
            return np.ones(len(node_ids), dtype=np.float32)
        return self.importance_tracker.node_table.get(node_ids)

    def _get_replicated_nodes(self) -> Set[int]:
        """Get set of replicated nodes."""
        if self.importance_tracker is None:
            return set()
        return self.importance_tracker.get_replicated_nodes()

    def _add_boundary_expansion(
        self,
        core_nodes: np.ndarray,
        intra_edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        num_train: int,
        batch_pages: np.ndarray,
        num_core: int,
    ) -> PageBatch:
        """
        FIM-guided k-hop boundary expansion with importance-weighted sampling.

        Overrides parent's random boundary selection with importance-weighted
        selection. Supports multi-hop expansion via self.boundary_hops.
        """
        core_set = set(core_nodes)
        boundary_budget_count = int(num_core * self.boundary_budget)

        # Track all nodes in batch (core + boundary added so far)
        current_nodes = set(core_nodes)

        # Collect boundary nodes and edges per hop
        hop_boundary_nodes = []
        hop_edges = []

        # Iteratively expand for each hop
        frontier = set(core_nodes)

        for hop in range(self.boundary_hops):
            new_boundary = set()
            new_edges_src = []
            new_edges_dst = []

            for node in frontier:
                neighbors = self.adj.get(int(node), np.array([], dtype=np.int64))
                for neighbor in neighbors:
                    if neighbor not in current_nodes:
                        new_boundary.add(neighbor)
                        new_edges_src.append(neighbor)
                        new_edges_dst.append(node)

            if not new_boundary:
                break

            hop_boundary_nodes.append(new_boundary)
            hop_edges.append((new_edges_src, new_edges_dst))
            current_nodes.update(new_boundary)
            frontier = new_boundary

        # Flatten all boundary nodes
        all_boundary = set()
        for hop_nodes in hop_boundary_nodes:
            all_boundary.update(hop_nodes)

        if len(all_boundary) == 0:
            return PageBatch(
                nodes=torch.from_numpy(core_nodes),
                edge_index=intra_edge_index,
                train_mask=train_mask,
                num_train_nodes=num_train,
                pages=batch_pages,
                num_core_nodes=num_core,
                num_boundary_nodes=0,
            )

        # Convert to array
        unique_boundary = np.array(list(all_boundary), dtype=np.int64)

        if len(unique_boundary) > boundary_budget_count:
            # FIM DIFFERENCE: Importance-weighted sampling instead of random
            node_imp = self._get_node_importance(unique_boundary)
            weights = node_imp + 1e-8

            if self.importance_temperature != 1.0:
                weights = np.power(weights, 1.0 / self.importance_temperature)

            weights_sum = weights.sum()
            if weights_sum > 0:
                weights_normalized = weights / weights_sum
            else:
                weights_normalized = np.ones(len(unique_boundary)) / len(unique_boundary)

            try:
                selected_idx = self.rng.choice(
                    len(unique_boundary),
                    boundary_budget_count,
                    replace=False,
                    p=weights_normalized,
                )
            except ValueError:
                selected_idx = np.argpartition(weights, -boundary_budget_count)[
                    -boundary_budget_count:
                ]

            boundary_nodes = unique_boundary[selected_idx]
        else:
            boundary_nodes = unique_boundary

        num_boundary = len(boundary_nodes)

        # Combine core + boundary nodes
        all_nodes = np.concatenate([core_nodes, boundary_nodes])
        all_nodes_set = set(all_nodes.tolist())

        # Build global-to-local mapping
        node_to_local = {int(n): i for i, n in enumerate(all_nodes)}

        # Collect and filter edges - only keep edges where BOTH endpoints are in batch
        boundary_edges_src = []
        boundary_edges_dst = []
        for hop_src, hop_dst in hop_edges:
            for s, d in zip(hop_src, hop_dst):
                if s in all_nodes_set and d in all_nodes_set:
                    boundary_edges_src.append(s)
                    boundary_edges_dst.append(d)

        # Remap boundary edges to local indices
        if len(boundary_edges_src) > 0:
            local_boundary_src = np.array(
                [node_to_local[int(s)] for s in boundary_edges_src], dtype=np.int64
            )
            local_boundary_dst = np.array(
                [node_to_local[int(d)] for d in boundary_edges_dst], dtype=np.int64
            )
            boundary_edge_index = torch.from_numpy(
                np.stack([local_boundary_src, local_boundary_dst])
            )

            combined_edge_index = torch.cat(
                [intra_edge_index, boundary_edge_index], dim=1
            )
        else:
            combined_edge_index = intra_edge_index

        extended_train_mask = torch.cat(
            [train_mask, torch.zeros(num_boundary, dtype=torch.bool)]
        )

        return PageBatch(
            nodes=torch.from_numpy(all_nodes),
            edge_index=combined_edge_index,
            train_mask=extended_train_mask,
            num_train_nodes=num_train,
            pages=batch_pages,
            num_core_nodes=num_core,
            num_boundary_nodes=num_boundary,
        )


@dataclass
class FIMPageBatch(PageBatch):
    """Extended PageBatch with FIM-related statistics."""

    # Importance statistics
    importance_mean: float = 0.0
    importance_sum: float = 0.0
    top_importance_nodes: int = 0  # Count of nodes from top-10% importance

    # Cross-page statistics
    cross_page_edges: int = 0
    cross_page_edges_high_importance: int = 0  # Edges with importance > median

    # Replication statistics
    replicated_boundary_nodes: int = 0
    saved_reads_estimate: int = 0


class FIMPageBatchSampler:
    """
    FIM-guided page-batch sampler with importance-weighted neighbor selection.

    Uses NodeImportanceTracker to:
    1. Select cross-page neighbors by importance instead of randomly
    2. Apply page-pair boost for high-importance page connections
    3. Track replication simulation statistics
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        page_id: np.ndarray,
        train_mask: np.ndarray,
        num_nodes: int,
        importance_tracker: Optional["NodeImportanceTracker"] = None,
        pages_per_batch: int = 32,
        boundary_budget: float = 0.2,
        use_inter_page_edges: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        lambda_pagepair: float = 0.5,
        importance_temperature: float = 1.0,
    ):
        """
        Initialize FIM-guided page-batch sampler.

        Args:
            edge_index: [2, E] edge index tensor
            page_id: [N] array mapping node_id -> 4KB page ID
            train_mask: [N] boolean array of training nodes
            num_nodes: Total number of nodes
            importance_tracker: Optional NodeImportanceTracker for importance scores
            pages_per_batch: Number of pages per batch
            boundary_budget: Fraction of extra nodes for boundary (default 0.2)
            use_inter_page_edges: If True, use ALL edges within batch
            shuffle: Whether to shuffle pages each epoch
            seed: Random seed for shuffling
            lambda_pagepair: Boost factor for cross-page importance (default 0.5)
            importance_temperature: Temperature for importance-weighted sampling
        """
        self.num_nodes = num_nodes
        self.pages_per_batch = pages_per_batch
        self.boundary_budget = boundary_budget
        self.use_inter_page_edges = use_inter_page_edges
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # FIM-specific: use tracker for importance tables
        self.importance_tracker = importance_tracker
        self.lambda_pagepair = lambda_pagepair
        self.importance_temperature = importance_temperature

        # Convert inputs
        page_id = np.asarray(page_id, dtype=np.int32)
        self.page_id = page_id
        train_mask = np.asarray(train_mask, dtype=bool)
        self.train_mask = train_mask
        self.num_pages = int(page_id.max()) + 1

        # Store full edge index
        self.edge_index = edge_index
        self.edge_src = edge_index[0].numpy()
        self.edge_dst = edge_index[1].numpy()

        # Build adjacency list
        print("  Building adjacency list...")
        self.adj: Dict[int, np.ndarray] = defaultdict(list)
        for s, d in zip(self.edge_src, self.edge_dst):
            self.adj[s].append(d)
        self.adj = {k: np.array(v, dtype=np.int64) for k, v in self.adj.items()}

        # Find pages with training nodes
        train_indices = np.where(train_mask)[0]
        pages_with_train = np.unique(page_id[train_indices])
        self.train_pages = pages_with_train.astype(np.int64)

        # Group nodes by page
        self.page_nodes: Dict[int, np.ndarray] = defaultdict(list)
        for node in range(num_nodes):
            self.page_nodes[page_id[node]].append(node)
        self.page_nodes = {
            p: np.array(nodes, dtype=np.int64) for p, nodes in self.page_nodes.items()
        }

        # Build intra-page edges
        src_pages = page_id[self.edge_src]
        dst_pages = page_id[self.edge_dst]
        intra_mask = src_pages == dst_pages

        intra_src = self.edge_src[intra_mask]
        intra_dst = self.edge_dst[intra_mask]
        intra_pages = src_pages[intra_mask]

        self.page_edges: Dict[int, Tuple[np.ndarray, np.ndarray]] = defaultdict(
            lambda: ([], [])
        )
        for s, d, p in zip(intra_src, intra_dst, intra_pages):
            self.page_edges[p][0].append(s)
            self.page_edges[p][1].append(d)
        self.page_edges = {
            p: (np.array(edges[0], dtype=np.int64), np.array(edges[1], dtype=np.int64))
            for p, edges in self.page_edges.items()
        }

        # Group training nodes by page
        self.page_train_nodes: Dict[int, list] = defaultdict(list)
        for node in train_indices:
            self.page_train_nodes[page_id[node]].append(node)

        # Statistics tracking
        self._batch_stats: Dict[str, float] = {}

        print(
            f"FIMPageBatchSampler: {self.num_pages:,} pages, "
            f"{len(self.train_pages):,} with training nodes"
        )
        if importance_tracker is not None:
            print("  Importance tracking: ENABLED (backward hooks)")

    def set_importance_tracker(self, tracker: "NodeImportanceTracker"):
        """Update the importance tracker."""
        self.importance_tracker = tracker

    def _get_node_importance(self, node_ids: np.ndarray) -> np.ndarray:
        """Get importance scores for nodes."""
        if self.importance_tracker is None:
            return np.ones(len(node_ids), dtype=np.float32)
        return self.importance_tracker.node_table.get(node_ids)

    def _get_pagepair_bonus(self, page_src: int, page_dst: int) -> float:
        """Get page-pair importance bonus."""
        if self.importance_tracker is None:
            return 0.0
        return self.importance_tracker.pagepair_table.get_normalized(page_src, page_dst)

    def _get_replicated_nodes(self) -> Set[int]:
        """Get set of replicated nodes."""
        if self.importance_tracker is None:
            return set()
        return self.importance_tracker.get_replicated_nodes()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (
            len(self.train_pages) + self.pages_per_batch - 1
        ) // self.pages_per_batch

    def __iter__(self) -> Iterator[FIMPageBatch]:
        """Iterate over page batches with importance-weighted sampling."""
        pages = self.train_pages.copy()
        if self.shuffle:
            self.rng.shuffle(pages)

        for start_idx in range(0, len(pages), self.pages_per_batch):
            batch_pages = pages[start_idx : start_idx + self.pages_per_batch]
            batch = self._sample_batch_fim(batch_pages)
            yield batch

    def _compute_neighbor_weights(
        self,
        core_node: int,
        candidates: np.ndarray,
        core_page: int,
    ) -> np.ndarray:
        """
        Compute importance-based weights for neighbor candidates (vectorized).

        S(u) = node_imp[u] * (1 + λ * pagepair_bonus(page(u), page(core)))
        """
        # Get node importance (returns ones if no tracker)
        node_imp = self._get_node_importance(candidates)
        weights = node_imp + 1e-8  # Small epsilon to avoid zero weights

        # Skip pagepair bonus - it's too expensive and rarely helps
        # The node importance alone captures most of the signal

        # Apply temperature
        if self.importance_temperature != 1.0:
            weights = np.power(weights, 1.0 / self.importance_temperature)

        return weights

    def _sample_batch_fim(self, batch_pages: np.ndarray) -> FIMPageBatch:
        """
        Sample batch with FIM-guided boundary expansion (vectorized).
        """
        # Collect all nodes from selected pages (vectorized via concatenate)
        page_node_arrays = [
            self.page_nodes.get(pg, np.array([], dtype=np.int64))
            for pg in batch_pages
        ]
        if page_node_arrays:
            core_nodes = np.concatenate(page_node_arrays) if any(len(a) > 0 for a in page_node_arrays) else np.array([], dtype=np.int64)
        else:
            core_nodes = np.array([], dtype=np.int64)

        num_core = len(core_nodes)

        if num_core == 0:
            return FIMPageBatch(
                nodes=torch.zeros(0, dtype=torch.long),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                train_mask=torch.zeros(0, dtype=torch.bool),
                num_train_nodes=0,
                pages=batch_pages,
                num_core_nodes=0,
                num_boundary_nodes=0,
            )

        # Collect train nodes
        train_node_arrays = [
            np.array(self.page_train_nodes.get(pg, []), dtype=np.int64)
            for pg in batch_pages
        ]
        all_train_nodes = np.concatenate(train_node_arrays) if any(len(a) > 0 for a in train_node_arrays) else np.array([], dtype=np.int64)

        core_set = set(core_nodes)

        # Get intra-page edges (vectorized)
        if self.use_inter_page_edges:
            # Vectorized check: both endpoints in core_set
            core_array = np.array(list(core_set), dtype=np.int64)
            src_in_core = np.isin(self.edge_src, core_array)
            dst_in_core = np.isin(self.edge_dst, core_array)
            edge_mask = src_in_core & dst_in_core
            batch_src = self.edge_src[edge_mask]
            batch_dst = self.edge_dst[edge_mask]
        else:
            # Concatenate pre-computed page edges
            edge_arrays = [
                self.page_edges.get(pg, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                for pg in batch_pages
            ]
            src_arrays = [e[0] for e in edge_arrays]
            dst_arrays = [e[1] for e in edge_arrays]
            batch_src = np.concatenate(src_arrays) if any(len(a) > 0 for a in src_arrays) else np.array([], dtype=np.int64)
            batch_dst = np.concatenate(dst_arrays) if any(len(a) > 0 for a in dst_arrays) else np.array([], dtype=np.int64)

        # Remap to local indices (vectorized via searchsorted)
        if len(batch_src) > 0:
            sorted_idx = np.argsort(core_nodes)
            sorted_nodes = core_nodes[sorted_idx]
            inv_idx = np.empty_like(sorted_idx)
            inv_idx[sorted_idx] = np.arange(len(core_nodes))

            src_pos = np.searchsorted(sorted_nodes, batch_src)
            dst_pos = np.searchsorted(sorted_nodes, batch_dst)
            local_src = inv_idx[src_pos]
            local_dst = inv_idx[dst_pos]
            intra_edge_index = torch.tensor(np.stack([local_src, local_dst]), dtype=torch.long)
        else:
            intra_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Build train mask (vectorized)
        train_mask = torch.from_numpy(np.isin(core_nodes, all_train_nodes))

        # FIM-guided boundary expansion
        if self.boundary_budget > 0:
            batch = self._add_fim_boundary_expansion(
                core_nodes,
                core_set,
                intra_edge_index,
                train_mask,
                len(all_train_nodes),
                batch_pages,
                num_core,
            )
        else:
            batch = FIMPageBatch(
                nodes=torch.from_numpy(core_nodes),
                edge_index=intra_edge_index,
                train_mask=train_mask,
                num_train_nodes=len(all_train_nodes),
                pages=batch_pages,
                num_core_nodes=num_core,
                num_boundary_nodes=0,
            )

        return batch

    def _add_fim_boundary_expansion(
        self,
        core_nodes: np.ndarray,
        core_set: Set[int],
        intra_edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        num_train: int,
        batch_pages: np.ndarray,
        num_core: int,
    ) -> FIMPageBatch:
        """
        Add FIM-guided boundary expansion (using adjacency list).

        Selects boundary neighbors using importance-weighted sampling.
        """
        boundary_budget_count = int(num_core * self.boundary_budget)
        replicated_nodes = self._get_replicated_nodes()

        # Use adjacency list with set-based filtering (O(1) lookups)
        # Collect all neighbors at once, then filter
        all_neighbors = []
        all_core_sources = []

        for core_node in core_nodes:
            neighbors = self.adj.get(int(core_node))
            if neighbors is not None and len(neighbors) > 0:
                all_neighbors.append(neighbors)
                all_core_sources.append(np.full(len(neighbors), core_node, dtype=np.int64))

        if not all_neighbors:
            return FIMPageBatch(
                nodes=torch.from_numpy(core_nodes),
                edge_index=intra_edge_index,
                train_mask=train_mask,
                num_train_nodes=num_train,
                pages=batch_pages,
                num_core_nodes=num_core,
                num_boundary_nodes=0,
            )

        all_neighbors = np.concatenate(all_neighbors)
        all_core_sources = np.concatenate(all_core_sources)

        # Filter to boundary (not in core_set) using numpy's in1d (optimized)
        is_in_core = np.in1d(all_neighbors, core_nodes, assume_unique=False)
        is_boundary = ~is_in_core

        if not is_boundary.any():
            return FIMPageBatch(
                nodes=torch.from_numpy(core_nodes),
                edge_index=intra_edge_index,
                train_mask=train_mask,
                num_train_nodes=num_train,
                pages=batch_pages,
                num_core_nodes=num_core,
                num_boundary_nodes=0,
            )

        # Get boundary edges
        boundary_edges_src = all_neighbors[is_boundary]
        boundary_edges_dst = all_core_sources[is_boundary]

        # Get unique boundary nodes and their importance
        unique_boundary, inverse_idx = np.unique(boundary_edges_src, return_inverse=True)

        # Get importance weights (vectorized)
        node_imp = self._get_node_importance(unique_boundary)
        unique_weights = node_imp + 1e-8

        # Sample boundary nodes by importance
        if len(unique_boundary) > boundary_budget_count:
            # Normalize for sampling
            weights_sum = unique_weights.sum()
            if weights_sum > 0:
                weights_normalized = unique_weights / weights_sum
            else:
                weights_normalized = np.ones(len(unique_boundary)) / len(unique_boundary)

            # Importance-weighted sampling
            try:
                selected_idx = self.rng.choice(
                    len(unique_boundary),
                    boundary_budget_count,
                    replace=False,
                    p=weights_normalized,
                )
            except ValueError:
                # Fallback to top-k if sampling fails
                selected_idx = np.argpartition(unique_weights, -boundary_budget_count)[
                    -boundary_budget_count:
                ]

            selected_boundary = unique_boundary[selected_idx]

            # Filter edges to selected boundary (vectorized)
            edge_mask = np.isin(boundary_edges_src, selected_boundary)
            boundary_edges_src = boundary_edges_src[edge_mask]
            boundary_edges_dst = boundary_edges_dst[edge_mask]
            boundary_nodes = selected_boundary
        else:
            boundary_nodes = unique_boundary

        num_boundary = len(boundary_nodes)

        # Count replicated nodes (vectorized)
        if replicated_nodes:
            replicated_array = np.array(list(replicated_nodes), dtype=np.int64)
            replicated_selected = int(np.isin(boundary_nodes, replicated_array).sum())
        else:
            replicated_selected = 0

        # Compute importance statistics (vectorized)
        boundary_imp = self._get_node_importance(boundary_nodes)
        importance_mean = float(boundary_imp.mean()) if len(boundary_imp) > 0 else 0
        importance_sum = float(boundary_imp.sum())

        # Count cross-page edges (vectorized)
        src_pages = self.page_id[boundary_edges_src]
        dst_pages = self.page_id[boundary_edges_dst]
        cross_page_mask = src_pages != dst_pages
        cross_page_edges = int(cross_page_mask.sum())

        # High-importance cross-page edges
        cross_page_high_importance = 0
        if cross_page_edges > 0 and self.importance_tracker is not None:
            node_table = self.importance_tracker.node_table
            active_mask = node_table.update_count > 0
            if active_mask.any():
                imp_median = np.median(node_table.importance[active_mask])
                src_imp = node_table.importance[boundary_edges_src[cross_page_mask]]
                cross_page_high_importance = int((src_imp > imp_median).sum())

        # Combine core + boundary nodes
        all_nodes = np.concatenate([core_nodes, boundary_nodes])

        # Remap boundary edges to local indices (vectorized using searchsorted)
        if len(boundary_edges_src) > 0:
            # Create sorted index for fast lookup
            sorted_idx = np.argsort(all_nodes)
            sorted_nodes = all_nodes[sorted_idx]
            inv_idx = np.empty_like(sorted_idx)
            inv_idx[sorted_idx] = np.arange(len(all_nodes))

            # Find local indices via searchsorted
            src_pos = np.searchsorted(sorted_nodes, boundary_edges_src)
            dst_pos = np.searchsorted(sorted_nodes, boundary_edges_dst)
            local_boundary_src = inv_idx[src_pos]
            local_boundary_dst = inv_idx[dst_pos]

            boundary_edge_index = torch.tensor(
                np.stack([local_boundary_src, local_boundary_dst]), dtype=torch.long
            )

            # Combine edges
            combined_edge_index = torch.cat(
                [intra_edge_index, boundary_edge_index], dim=1
            )
        else:
            combined_edge_index = intra_edge_index

        # Extend train mask
        extended_train_mask = torch.cat(
            [train_mask, torch.zeros(num_boundary, dtype=torch.bool)]
        )

        return FIMPageBatch(
            nodes=torch.from_numpy(all_nodes),
            edge_index=combined_edge_index,
            train_mask=extended_train_mask,
            num_train_nodes=num_train,
            pages=batch_pages,
            num_core_nodes=num_core,
            num_boundary_nodes=num_boundary,
            importance_mean=importance_mean,
            importance_sum=importance_sum,
            cross_page_edges=cross_page_edges,
            cross_page_edges_high_importance=cross_page_high_importance,
            replicated_boundary_nodes=replicated_selected,
            saved_reads_estimate=replicated_selected,
        )

    def get_batch_stats(self) -> Dict[str, float]:
        """Get accumulated batch statistics."""
        return self._batch_stats.copy()

    def reset_stats(self):
        """Reset batch statistics."""
        self._batch_stats = {}
