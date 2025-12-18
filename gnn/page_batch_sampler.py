#!/usr/bin/env python3
"""
C++ Page-Batch Sampler Wrapper with Boundary Expansion

High-performance page-batch sampling using C++ extension with OpenMP.
Processes entire memory pages as batches with bounded boundary expansion.

Key features:
- Page-batch core: all nodes from selected pages
- Boundary expansion: 1-hop neighbors with budget limit (default 20%)
- Bounded RA: max RA = 1 + boundary_budget (e.g., 1.2x with 0.2 budget)

Usage:
    from page_batch_sampler_cpp import CppPageBatchSampler

    sampler = CppPageBatchSampler(
        edge_index=data.edge_index,
        page_id=page_id_4kb,
        train_mask=train_mask,
        num_nodes=data.x.shape[0],
        pages_per_batch=32,
        boundary_budget=0.2,  # 20% extra nodes from boundary
    )

    for batch in sampler:
        # batch.nodes: global node IDs (core + boundary)
        # batch.edge_index: local edge indices [2, E]
        # batch.train_mask: boolean mask for training nodes
        # batch.num_train_nodes: count for loss weighting
        ...
"""

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

# Try to import the C++ extension
import sys

# Add cpp_extension directory to path
_cpp_ext_dir = os.path.join(os.path.dirname(__file__), "cpp_extension")
if _cpp_ext_dir not in sys.path:
    sys.path.insert(0, _cpp_ext_dir)

try:
    import cpp_page_batch_sampler

    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    raise ImportError(
        f"cpp_page_batch_sampler not available: {e}\n"
        f"Build with: cd {_cpp_ext_dir} && python setup.py build_ext --inplace"
    )


@dataclass
class PageBatch:
    """A batch of nodes and edges from selected pages."""

    nodes: torch.Tensor  # [N] global node IDs (core + boundary)
    edge_index: torch.Tensor  # [2, E] local edge indices
    train_mask: torch.Tensor  # [N] boolean mask for training nodes
    num_train_nodes: int  # count of training nodes
    pages: np.ndarray  # page IDs in this batch
    num_core_nodes: int = 0  # nodes from selected pages (before boundary)
    num_boundary_nodes: int = 0  # nodes added via boundary expansion


class CppPageBatchSampler:
    """
    C++ accelerated page-batch sampler with boundary expansion.

    Achieves bounded read amplification by:
    1. Pre-building CSR structures for pages (one-time)
    2. Sampling pages (not nodes)
    3. Using ALL nodes from selected pages (core)
    4. Adding 1-hop boundary neighbors with budget limit
    5. Including edges: intra-page + core-to-boundary

    With boundary_budget=0.2, max RA = 1.2x (bounded by construction).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        page_id: np.ndarray,
        train_mask: np.ndarray,
        num_nodes: int,
        pages_per_batch: int = 32,
        boundary_budget: float = 0.2,
        boundary_hops: int = 1,
        use_inter_page_edges: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        pos_labels: Optional[np.ndarray] = None,
        pos_oversample_prob: float = 0.0,
    ):
        """
        Initialize the page-batch sampler with boundary expansion.

        Args:
            edge_index: [2, E] edge index tensor
            page_id: [N] array mapping node_id -> 4KB page ID
            train_mask: [N] boolean array of training nodes
            num_nodes: Total number of nodes
            pages_per_batch: Number of pages per batch
            boundary_budget: Fraction of extra nodes for boundary (default 0.2)
            boundary_hops: Number of hops for boundary expansion (default 1).
                For 2-layer GNN, use boundary_hops=2 for proper neighborhood.
                WARNING: 2-hop significantly increases batch size and RA.
            use_inter_page_edges: If True, use ALL edges within batch (not just
                intra-page). This dramatically improves connectivity without
                increasing RA since we've already loaded all the nodes.
            shuffle: Whether to shuffle pages each epoch
            seed: Random seed for shuffling
            pos_labels: Optional [N] array of positive labels (1=fraud, 0=other)
            pos_oversample_prob: Probability of injecting a positive-containing
                page into each batch (default 0.0 = no oversampling)
        """
        self.num_nodes = num_nodes
        self.pages_per_batch = pages_per_batch
        self.boundary_budget = boundary_budget
        self.boundary_hops = boundary_hops
        self.use_inter_page_edges = use_inter_page_edges
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.pos_oversample_prob = pos_oversample_prob

        # Convert inputs
        page_id = np.asarray(page_id, dtype=np.int32)
        self.page_id = page_id
        train_mask = np.asarray(train_mask, dtype=bool)
        self.train_mask = train_mask
        self.num_pages = int(page_id.max()) + 1

        # Store full edge index for inter-page edges and boundary expansion
        self.edge_index = edge_index
        self.edge_src = edge_index[0].numpy()
        self.edge_dst = edge_index[1].numpy()

        # Build adjacency list for boundary expansion
        if boundary_budget > 0:
            print("  Building adjacency list for boundary expansion...")
            self.adj = defaultdict(list)
            for s, d in zip(self.edge_src, self.edge_dst):
                self.adj[s].append(d)
            self.adj = {k: np.array(v, dtype=np.int64) for k, v in self.adj.items()}
        else:
            self.adj = {}

        # Find pages with training nodes
        train_indices = np.where(train_mask)[0]
        pages_with_train = np.unique(page_id[train_indices])
        self.train_pages = pages_with_train.astype(np.int64)

        # Find pages with positive training nodes (for oversampling)
        self.pos_train_pages = None
        if pos_labels is not None and pos_oversample_prob > 0:
            pos_labels = np.asarray(pos_labels)
            # Positive training nodes
            pos_train_mask = train_mask & (pos_labels == 1)
            pos_train_indices = np.where(pos_train_mask)[0]
            if len(pos_train_indices) > 0:
                pos_pages = np.unique(page_id[pos_train_indices])
                self.pos_train_pages = pos_pages.astype(np.int64)

        print(
            f"CppPageBatchSampler: {self.num_pages:,} pages, "
            f"{len(self.train_pages):,} with training nodes"
        )
        if self.pos_train_pages is not None:
            print(
                f"  Positive oversampling: {pos_oversample_prob:.0%} prob, "
                f"{len(self.pos_train_pages):,} pages with positives"
            )
        if use_inter_page_edges:
            print(f"  Inter-page edges: ENABLED (use all edges within batch)")
        else:
            print(f"  Inter-page edges: DISABLED (only intra-page edges)")
        if boundary_budget > 0:
            print(
                f"  Boundary budget: {boundary_budget:.0%} "
                f"(adds 1-hop neighbors, edges: boundary→core)"
            )

        if CPP_AVAILABLE:
            self._init_cpp(edge_index, page_id, train_mask)
        else:
            self._init_python(edge_index, page_id, train_mask)

    def _init_cpp(
        self,
        edge_index: torch.Tensor,
        page_id: np.ndarray,
        train_mask: np.ndarray,
    ):
        """Initialize using C++ extension."""
        print("  Building C++ page structures...")
        start = time.time()

        # Prepare tensors
        page_id_tensor = torch.from_numpy(page_id).int()
        edge_src = edge_index[0].long()
        edge_dst = edge_index[1].long()
        train_mask_tensor = torch.from_numpy(train_mask).bool()

        # Build page structures
        (
            self.page_node_offsets,
            self.page_node_indices,
            self.page_edge_offsets,
            self.page_edge_src,
            self.page_edge_dst,
            self.page_train_offsets,
            self.page_train_indices,
        ) = cpp_page_batch_sampler.build_page_structures(
            page_id_tensor,
            edge_src,
            edge_dst,
            train_mask_tensor,
            self.num_pages,
        )

        elapsed = time.time() - start
        total_intra = self.page_edge_src.shape[0]
        total_edges = edge_index.shape[1]
        print(f"  Built in {elapsed:.2f}s")
        print(
            f"  Intra-page edges: {total_intra:,} ({100*total_intra/total_edges:.1f}%)"
        )

        self.use_cpp = True

    def _init_python(
        self,
        edge_index: torch.Tensor,
        page_id: np.ndarray,
        train_mask: np.ndarray,
    ):
        """Initialize using Python fallback."""
        from collections import defaultdict

        print("  Building Python page structures (fallback)...")
        start = time.time()

        # Group nodes by page
        self.page_nodes = defaultdict(list)
        for node in range(self.num_nodes):
            self.page_nodes[page_id[node]].append(node)
        self.page_nodes = {p: np.array(nodes) for p, nodes in self.page_nodes.items()}

        # Build intra-page edges
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        src_pages = page_id[src]
        dst_pages = page_id[dst]
        intra_mask = src_pages == dst_pages

        intra_src = src[intra_mask]
        intra_dst = dst[intra_mask]
        intra_pages = src_pages[intra_mask]

        self.page_edges = defaultdict(lambda: ([], []))
        for s, d, p in zip(intra_src, intra_dst, intra_pages):
            self.page_edges[p][0].append(s)
            self.page_edges[p][1].append(d)
        self.page_edges = {
            p: (np.array(edges[0]), np.array(edges[1]))
            for p, edges in self.page_edges.items()
        }

        # Group training nodes by page
        train_indices = np.where(train_mask)[0]
        self.page_train_nodes = defaultdict(list)
        for node in train_indices:
            self.page_train_nodes[page_id[node]].append(node)

        elapsed = time.time() - start
        total_intra = sum(len(e[0]) for e in self.page_edges.values())
        total_edges = edge_index.shape[1]
        print(f"  Built in {elapsed:.2f}s (Python fallback)")
        print(
            f"  Intra-page edges: {total_intra:,} ({100*total_intra/total_edges:.1f}%)"
        )

        self.use_cpp = False

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (
            len(self.train_pages) + self.pages_per_batch - 1
        ) // self.pages_per_batch

    def __iter__(self) -> Iterator[PageBatch]:
        """Iterate over page batches."""
        pages = self.train_pages.copy()
        if self.shuffle:
            self.rng.shuffle(pages)

        for start_idx in range(0, len(pages), self.pages_per_batch):
            batch_pages = pages[start_idx : start_idx + self.pages_per_batch]

            # With probability pos_oversample_prob, inject a positive page
            if (
                self.pos_train_pages is not None
                and self.pos_oversample_prob > 0
                and self.rng.random() < self.pos_oversample_prob
            ):
                # Pick a random positive page not already in batch
                batch_pages_set = set(batch_pages.tolist())
                available_pos = [
                    p for p in self.pos_train_pages if p not in batch_pages_set
                ]
                if available_pos:
                    # Replace last page with a positive-containing page
                    pos_page = self.rng.choice(available_pos)
                    batch_pages = np.append(batch_pages[:-1], pos_page)

            if self.use_cpp:
                batch = self._sample_batch_cpp(batch_pages)
            else:
                batch = self._sample_batch_python(batch_pages)

            yield batch

    def _sample_batch_cpp(self, batch_pages: np.ndarray) -> PageBatch:
        """Sample batch using C++ extension with boundary expansion."""
        batch_pages_tensor = torch.from_numpy(batch_pages).long()

        # Get core batch (intra-page only)
        core_nodes, intra_edge_index, train_mask, num_train = (
            cpp_page_batch_sampler.sample_batch(
                batch_pages_tensor,
                self.page_node_offsets,
                self.page_node_indices,
                self.page_edge_offsets,
                self.page_edge_src,
                self.page_edge_dst,
                self.page_train_offsets,
                self.page_train_indices,
                self.num_nodes,
            )
        )

        num_core = len(core_nodes)

        # Add boundary expansion if budget > 0
        if self.boundary_budget > 0:
            batch = self._add_boundary_expansion(
                core_nodes.numpy(),
                intra_edge_index,
                train_mask,
                num_train,
                batch_pages,
                num_core,
            )
        else:
            batch = PageBatch(
                nodes=core_nodes,
                edge_index=intra_edge_index,
                train_mask=train_mask,
                num_train_nodes=num_train,
                pages=batch_pages,
                num_core_nodes=num_core,
                num_boundary_nodes=0,
            )

        return batch

    def _sample_batch_python(self, batch_pages: np.ndarray) -> PageBatch:
        """Sample batch using Python fallback with inter-page edges and boundary expansion."""
        # Collect all nodes from selected pages
        all_nodes = []
        all_train_nodes = []

        for pg in batch_pages:
            nodes = self.page_nodes.get(pg, np.array([], dtype=np.int64))
            all_nodes.extend(nodes)
            all_train_nodes.extend(self.page_train_nodes.get(pg, []))

        core_nodes = np.array(all_nodes, dtype=np.int64)
        num_core = len(core_nodes)

        if num_core == 0:
            return PageBatch(
                nodes=torch.zeros(0, dtype=torch.long),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                train_mask=torch.zeros(0, dtype=torch.bool),
                num_train_nodes=0,
                pages=batch_pages,
                num_core_nodes=0,
                num_boundary_nodes=0,
            )

        # Build node set for fast lookup
        core_set = set(core_nodes)
        node_to_local = {n: i for i, n in enumerate(core_nodes)}

        # Get edges based on inter_page setting
        if self.use_inter_page_edges:
            # Use ALL edges where both endpoints are in the batch
            # This includes both intra-page AND inter-page edges
            edge_mask = np.array(
                [
                    (s in core_set) and (d in core_set)
                    for s, d in zip(self.edge_src, self.edge_dst)
                ]
            )
            batch_src = self.edge_src[edge_mask]
            batch_dst = self.edge_dst[edge_mask]
        else:
            # Use only intra-page edges (old behavior)
            edge_src = []
            edge_dst = []
            for pg in batch_pages:
                if pg in self.page_edges:
                    s, d = self.page_edges[pg]
                    edge_src.extend(s)
                    edge_dst.extend(d)
            batch_src = (
                np.array(edge_src, dtype=np.int64)
                if edge_src
                else np.array([], dtype=np.int64)
            )
            batch_dst = (
                np.array(edge_dst, dtype=np.int64)
                if edge_dst
                else np.array([], dtype=np.int64)
            )

        # Remap to local indices
        if len(batch_src) > 0:
            local_src = np.array([node_to_local[s] for s in batch_src])
            local_dst = np.array([node_to_local[d] for d in batch_dst])
            batch_edge_index = torch.tensor([local_src, local_dst], dtype=torch.long)
        else:
            batch_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Build train mask
        train_set = set(all_train_nodes)
        train_mask = torch.tensor(
            [n in train_set for n in core_nodes], dtype=torch.bool
        )

        # Add boundary expansion if budget > 0
        if self.boundary_budget > 0:
            batch = self._add_boundary_expansion(
                core_nodes,
                batch_edge_index,
                train_mask,
                len(all_train_nodes),
                batch_pages,
                num_core,
            )
        else:
            batch = PageBatch(
                nodes=torch.from_numpy(core_nodes),
                edge_index=batch_edge_index,
                train_mask=train_mask,
                num_train_nodes=len(all_train_nodes),
                pages=batch_pages,
                num_core_nodes=num_core,
                num_boundary_nodes=0,
            )

        return batch

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
        Add k-hop boundary neighbors with budget limit.

        For each core node, iteratively find neighbors outside current set.
        Limit total boundary nodes to boundary_budget * num_core.
        Add edges from boundary nodes to core/boundary nodes.
        """
        core_set = set(core_nodes)
        boundary_budget_count = int(num_core * self.boundary_budget)

        # Track all nodes in batch (core + boundary added so far)
        current_nodes = set(core_nodes)

        # Collect boundary nodes and edges per hop (needed for proper filtering)
        hop_boundary_nodes = []  # List of sets, one per hop
        hop_edges = []  # List of (src_list, dst_list) tuples per hop

        # Iteratively expand for each hop
        frontier = set(core_nodes)  # Nodes to expand from

        for hop in range(self.boundary_hops):
            new_boundary = set()
            new_edges_src = []
            new_edges_dst = []

            for node in frontier:
                neighbors = self.adj.get(int(node), np.array([], dtype=np.int64))
                for neighbor in neighbors:
                    if neighbor not in current_nodes:
                        new_boundary.add(neighbor)
                        # Edge: neighbor → node (neighbor sends to node)
                        new_edges_src.append(neighbor)
                        new_edges_dst.append(node)

            if not new_boundary:
                break

            # Store this hop's data
            hop_boundary_nodes.append(new_boundary)
            hop_edges.append((new_edges_src, new_edges_dst))

            # Update tracking
            current_nodes.update(new_boundary)
            frontier = new_boundary

        # Flatten all boundary nodes
        all_boundary = set()
        for hop_nodes in hop_boundary_nodes:
            all_boundary.update(hop_nodes)

        if len(all_boundary) == 0:
            # No boundary neighbors, return core-only batch
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

        # Sample boundary nodes up to budget
        if len(unique_boundary) > boundary_budget_count:
            selected_idx = self.rng.choice(
                len(unique_boundary), boundary_budget_count, replace=False
            )
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

            # Combine intra-page edges + boundary edges
            combined_edge_index = torch.cat(
                [intra_edge_index, boundary_edge_index], dim=1
            )
        else:
            combined_edge_index = intra_edge_index

        # Extend train mask (boundary nodes are not training nodes)
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


def build_cpp_extension():
    """Build the C++ extension."""
    from torch.utils.cpp_extension import load

    cpp_dir = os.path.join(os.path.dirname(__file__), "cpp_extension")
    cpp_file = os.path.join(cpp_dir, "page_batch_sampler.cpp")

    if not os.path.exists(cpp_file):
        raise FileNotFoundError(f"C++ source not found: {cpp_file}")

    print(f"Building C++ page-batch sampler from {cpp_file}...")

    module = load(
        name="cpp_page_batch_sampler",
        sources=[cpp_file],
        extra_cflags=["-O3", "-fopenmp", "-march=native"],
        extra_ldflags=["-fopenmp"],
        verbose=True,
    )

    return module


if __name__ == "__main__":
    # Build and test the extension
    print("Building C++ page-batch sampler extension...")
    module = build_cpp_extension()
    print("Build successful!")

    # Quick test
    print("\nRunning quick test...")
    num_nodes = 1000
    num_edges = 5000
    num_pages = 20

    page_id = torch.randint(0, num_pages, (num_nodes,), dtype=torch.int32)
    edge_src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int64)
    edge_dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int64)
    train_mask = torch.rand(num_nodes) < 0.3

    print("Building page structures...")
    result = module.build_page_structures(
        page_id, edge_src, edge_dst, train_mask, num_pages
    )
    print(f"  page_node_offsets: {result[0].shape}")
    print(f"  page_node_indices: {result[1].shape}")
    print(f"  page_edge_offsets: {result[2].shape}")
    print(f"  page_edge_src: {result[3].shape}")
    print(f"  page_edge_dst: {result[4].shape}")

    print("\nSampling batch...")
    batch_pages = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    batch_result = module.sample_batch(
        batch_pages,
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
        result[6],
        num_nodes,
    )
    print(f"  batch_nodes: {batch_result[0].shape}")
    print(f"  batch_edge_index: {batch_result[1].shape}")
    print(f"  batch_train_mask: {batch_result[2].shape}")
    print(f"  num_train_nodes: {batch_result[3]}")

    print("\nTest passed!")
