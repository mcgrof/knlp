#!/usr/bin/env python3
"""
Synthetic test for Reciprocal Attention in GNNs.

Creates a toy graph with:
- Two clean clusters (easy nodes)
- Bridge nodes connecting both clusters (hard/uncertain nodes)

Tests whether RA helps bridge nodes more than cluster nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
import numpy as np
from collections import defaultdict


def create_synthetic_graph(
    cluster_size: int = 500,
    num_bridges: int = 50,
    feature_dim: int = 16,
    intra_cluster_edges: int = 5,  # edges per node within cluster
    bridge_edges_per_side: int = 10,  # edges from each bridge to each cluster
    seed: int = 42,
):
    """
    Create synthetic graph with two clusters and bridge nodes.

    Cluster A: nodes 0 to cluster_size-1, label=0
    Cluster B: nodes cluster_size to 2*cluster_size-1, label=1
    Bridges: nodes 2*cluster_size to 2*cluster_size+num_bridges-1, label=mixed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_total = 2 * cluster_size + num_bridges

    # Node features: cluster A has feature[0] high, cluster B has feature[1] high
    # Bridges have mixed features
    x = torch.randn(n_total, feature_dim) * 0.1

    # Cluster A: boost feature 0
    x[:cluster_size, 0] += 1.0
    # Cluster B: boost feature 1
    x[cluster_size : 2 * cluster_size, 1] += 1.0
    # Bridges: mix of both (ambiguous)
    x[2 * cluster_size :, 0] += 0.5
    x[2 * cluster_size :, 1] += 0.5

    # Labels
    y = torch.zeros(n_total, dtype=torch.long)
    y[cluster_size : 2 * cluster_size] = 1
    # Bridge labels: 50% each class (the hard cases)
    bridge_labels = torch.randint(0, 2, (num_bridges,))
    y[2 * cluster_size :] = bridge_labels

    # Build edges
    edge_list = []

    # Intra-cluster edges (cluster A)
    for i in range(cluster_size):
        neighbors = np.random.choice(cluster_size, intra_cluster_edges, replace=False)
        for j in neighbors:
            if i != j:
                edge_list.append([i, j])
                edge_list.append([j, i])

    # Intra-cluster edges (cluster B)
    for i in range(cluster_size, 2 * cluster_size):
        neighbors = np.random.choice(
            range(cluster_size, 2 * cluster_size), intra_cluster_edges, replace=False
        )
        for j in neighbors:
            if i != j:
                edge_list.append([i, j])
                edge_list.append([j, i])

    # Bridge edges (connect to both clusters)
    for i in range(2 * cluster_size, n_total):
        # Connect to cluster A
        neighbors_a = np.random.choice(
            cluster_size, bridge_edges_per_side, replace=False
        )
        for j in neighbors_a:
            edge_list.append([i, j])
            edge_list.append([j, i])

        # Connect to cluster B
        neighbors_b = np.random.choice(
            range(cluster_size, 2 * cluster_size), bridge_edges_per_side, replace=False
        )
        for j in neighbors_b:
            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    # Train/val/test split
    # Use 60% cluster nodes for train, 20% val, 20% test
    # All bridges go to test (we want to evaluate on them)
    cluster_indices = torch.arange(2 * cluster_size)
    perm = cluster_indices[torch.randperm(2 * cluster_size)]

    n_train = int(0.6 * 2 * cluster_size)
    n_val = int(0.2 * 2 * cluster_size)

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True
    test_mask[2 * cluster_size :] = True  # All bridges in test

    # Track which nodes are bridges
    is_bridge = torch.zeros(n_total, dtype=torch.bool)
    is_bridge[2 * cluster_size :] = True

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.is_bridge = is_bridge
    data.num_classes = 2

    return data


class BaselineGraphSAGE(nn.Module):
    """Standard GraphSAGE with mean aggregation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class RAConv(nn.Module):
    """
    Reciprocal Attention convolution layer (vectorized).

    Uses scatter operations for efficiency.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_reciprocal: bool = True,
        use_variance: bool = True,
    ):
        super().__init__()
        self.use_reciprocal = use_reciprocal
        self.use_variance = use_variance

        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

        if use_variance:
            self.var_proj = nn.Linear(in_dim, out_dim)

        self.scale = in_dim**-0.5

    def forward(self, x, edge_index):
        from torch_scatter import scatter_softmax, scatter_mean, scatter_add

        src, dst = edge_index  # src -> dst edges

        # Get neighbor features
        x_src = x[src]  # [E, in_dim]
        x_dst = x[dst]  # [E, in_dim]

        if self.use_reciprocal:
            # Reciprocal attention: neighbors query center (K @ Q.T)
            # attn[e] = x_src[e] @ x_dst[e] (neighbor queries center)
            attn_logits = (x_src * x_dst).sum(dim=-1) * self.scale  # [E]
            attn = scatter_softmax(attn_logits, dst, dim=0)  # [E]

            # Weighted aggregation
            weighted = x_src * attn.unsqueeze(-1)  # [E, in_dim]
            msg = scatter_add(weighted, dst, dim=0, dim_size=x.size(0))  # [N, in_dim]
        else:
            # Mean aggregation
            msg = scatter_mean(x_src, dst, dim=0, dim_size=x.size(0))  # [N, in_dim]

        out = self.lin_self(x) + self.lin_neigh(msg)

        # Variance residual
        if self.use_variance:
            # Compute variance per node
            msg_mean = scatter_mean(x_src, dst, dim=0, dim_size=x.size(0))
            diff_sq = (x_src - msg_mean[dst]) ** 2
            var = scatter_mean(diff_sq, dst, dim=0, dim_size=x.size(0))
            out = out + self.var_proj(var)

        return out


class RAGraphSAGE(nn.Module):
    """GraphSAGE with Reciprocal Attention aggregation (vectorized)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_reciprocal: bool = True,
        use_variance: bool = True,
    ):
        super().__init__()
        self.conv1 = RAConv(in_dim, hidden_dim, use_reciprocal, use_variance)
        self.conv2 = RAConv(hidden_dim, out_dim, use_reciprocal, use_variance)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    results = {}

    # Overall accuracy
    for split, mask in [
        ("train", data.train_mask),
        ("val", data.val_mask),
        ("test", data.test_mask),
    ]:
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        results[f"{split}_acc"] = correct / total if total > 0 else 0

    # Bridge vs cluster accuracy (test set only)
    test_mask = data.test_mask
    bridge_mask = test_mask & data.is_bridge
    cluster_mask = test_mask & ~data.is_bridge

    if bridge_mask.sum() > 0:
        correct = (pred[bridge_mask] == data.y[bridge_mask]).sum().item()
        results["bridge_acc"] = correct / bridge_mask.sum().item()
    else:
        results["bridge_acc"] = 0

    if cluster_mask.sum() > 0:
        correct = (pred[cluster_mask] == data.y[cluster_mask]).sum().item()
        results["cluster_acc"] = correct / cluster_mask.sum().item()
    else:
        results["cluster_acc"] = 0

    return results


def run_experiment(
    model_class,
    model_kwargs,
    data,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
):
    torch.manual_seed(seed)

    model = model_class(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0
    best_results = None

    for epoch in range(epochs):
        loss = train_epoch(model, data, optimizer)
        results = evaluate(model, data)

        if results["val_acc"] > best_val_acc:
            best_val_acc = results["val_acc"]
            best_results = results.copy()
            best_results["epoch"] = epoch

    return best_results


def main():
    print("=" * 60)
    print("Reciprocal Attention GNN: Synthetic Bridge Node Test")
    print("=" * 60)
    print()

    # Create synthetic graph
    print("Creating synthetic graph...")
    data = create_synthetic_graph(
        cluster_size=500,
        num_bridges=100,
        feature_dim=16,
        seed=42,
    )

    print(f"  Nodes: {data.x.size(0)}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Cluster nodes: {(~data.is_bridge).sum().item()}")
    print(f"  Bridge nodes: {data.is_bridge.sum().item()}")
    print(f"  Train: {data.train_mask.sum().item()}")
    print(f"  Val: {data.val_mask.sum().item()}")
    print(f"  Test: {data.test_mask.sum().item()} (includes all bridges)")
    print()

    # Model configs to test
    configs = [
        (
            "Baseline (mean pool)",
            BaselineGraphSAGE,
            {"in_dim": 16, "hidden_dim": 32, "out_dim": 2},
        ),
        (
            "RA only",
            RAGraphSAGE,
            {
                "in_dim": 16,
                "hidden_dim": 32,
                "out_dim": 2,
                "use_reciprocal": True,
                "use_variance": False,
            },
        ),
        (
            "Variance only",
            RAGraphSAGE,
            {
                "in_dim": 16,
                "hidden_dim": 32,
                "out_dim": 2,
                "use_reciprocal": False,
                "use_variance": True,
            },
        ),
        (
            "RA + Variance",
            RAGraphSAGE,
            {
                "in_dim": 16,
                "hidden_dim": 32,
                "out_dim": 2,
                "use_reciprocal": True,
                "use_variance": True,
            },
        ),
    ]

    # Run experiments
    print("Running experiments (5 seeds each)...")
    print()

    all_results = {}

    for name, model_class, model_kwargs in configs:
        print(f"  {name}...", end=" ", flush=True)

        seed_results = []
        for seed in range(5):
            results = run_experiment(
                model_class, model_kwargs, data, epochs=200, lr=0.01, seed=seed
            )
            seed_results.append(results)

        # Aggregate across seeds
        agg = {}
        for key in seed_results[0].keys():
            if key != "epoch":
                vals = [r[key] for r in seed_results]
                agg[f"{key}_mean"] = np.mean(vals)
                agg[f"{key}_std"] = np.std(vals)

        all_results[name] = agg
        print("done")

    # Print results table
    print()
    print("=" * 60)
    print("Results (mean ± std over 5 seeds)")
    print("=" * 60)
    print()
    print(f"{'Model':<20} {'Test Acc':<15} {'Cluster Acc':<15} {'Bridge Acc':<15}")
    print("-" * 65)

    for name in all_results:
        r = all_results[name]
        test = f"{r['test_acc_mean']*100:.1f} ± {r['test_acc_std']*100:.1f}"
        cluster = f"{r['cluster_acc_mean']*100:.1f} ± {r['cluster_acc_std']*100:.1f}"
        bridge = f"{r['bridge_acc_mean']*100:.1f} ± {r['bridge_acc_std']*100:.1f}"
        print(f"{name:<20} {test:<15} {cluster:<15} {bridge:<15}")

    print()
    print("=" * 60)
    print("Analysis")
    print("=" * 60)
    print()

    baseline_bridge = all_results["Baseline (mean pool)"]["bridge_acc_mean"]
    ra_var_bridge = all_results["RA + Variance"]["bridge_acc_mean"]

    bridge_delta = (ra_var_bridge - baseline_bridge) * 100

    baseline_cluster = all_results["Baseline (mean pool)"]["cluster_acc_mean"]
    ra_var_cluster = all_results["RA + Variance"]["cluster_acc_mean"]

    cluster_delta = (ra_var_cluster - baseline_cluster) * 100

    print(f"Bridge node improvement (RA+Var vs Baseline): {bridge_delta:+.1f}%")
    print(f"Cluster node change (RA+Var vs Baseline): {cluster_delta:+.1f}%")
    print()

    if bridge_delta > 2.0 and cluster_delta > -2.0:
        print("✓ Hypothesis SUPPORTED: RA helps bridge nodes without hurting clusters")
        print("  → Proceed to DGraphFin evaluation")
    elif bridge_delta > 2.0:
        print("~ Partial support: RA helps bridges but may hurt clusters")
        print("  → Consider FIM-guided selective application")
    else:
        print("✗ Hypothesis NOT supported: RA doesn't help bridge nodes")
        print("  → Revisit the approach before scaling to real data")


if __name__ == "__main__":
    main()
