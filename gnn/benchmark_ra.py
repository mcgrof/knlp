#!/usr/bin/env python3
"""
Reciprocal Attention GNN Benchmark on DGraphFin.

Compares:
1. Baseline GraphSAGE (mean pooling)
2. RA-GraphSAGE (reciprocal attention + variance residual, all nodes)
3. FIM-guided RA (reciprocal attention only on low-FIM nodes)

Usage:
    python gnn/benchmark_ra.py --time 300
    python gnn/benchmark_ra.py --time 3600 --ablation
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_add, scatter_mean, scatter_softmax
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RABenchmarkConfig:
    """Configuration for RA benchmark."""

    time_limit: float = 300.0
    lr: float = 0.003
    weight_decay: float = 5e-7
    hidden_channels: int = 128
    num_classes: int = 2
    batch_size: int = 1024
    num_neighbors: list = None  # [10, 5]
    fim_update_interval: int = 100  # Steps between FIM updates
    fim_percentile: float = 33.0  # Bottom X% nodes get RA treatment

    def __post_init__(self):
        if self.num_neighbors is None:
            self.num_neighbors = [10, 5]


# =============================================================================
# Data Loading
# =============================================================================


def load_dgraphfin(
    data_dir: str = ".",
) -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    """Load DGraphFin dataset."""
    npz_path = os.path.join(data_dir, "dgraphfin.npz")
    print(f"Loading DGraphFin from {npz_path}...")

    npz_data = np.load(npz_path)

    x = torch.from_numpy(npz_data["x"]).float()
    y = torch.from_numpy(npz_data["y"]).long()
    edge_index = torch.from_numpy(npz_data["edge_index"].T).long()

    # Normalize features
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
    print(f"  Features: {x.shape[1]}")
    print(
        f"  Train/Val/Test: {train_mask.sum():,}/{val_mask.sum():,}/{test_mask.sum():,}"
    )

    # Class distribution
    y_np = y.numpy()
    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        n_pos = (y_np[mask] == 1).sum()
        n_total = mask.sum()
        print(f"  {name} fraud rate: {100 * n_pos / n_total:.2f}%")

    return data, train_mask, val_mask, test_mask


# =============================================================================
# Models
# =============================================================================


class RAConv(nn.Module):
    """
    Reciprocal Attention convolution layer.

    Instead of Q@K.T (node queries neighbors), we do K@Q.T (neighbors query node).
    Optionally adds variance residual to preserve disagreement signal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_reciprocal: bool = True,
        use_variance: bool = True,
    ):
        super().__init__()
        self.use_reciprocal = use_reciprocal
        self.use_variance = use_variance

        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

        if use_variance:
            self.var_proj = nn.Linear(in_channels, out_channels)

        self.scale = in_channels**-0.5

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ra_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            ra_mask: Optional boolean mask [N] - True = use RA, False = use mean

        Returns:
            Updated features [N, out_channels]
        """
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if self.use_reciprocal:
            if ra_mask is not None:
                # FIM-guided: selective RA
                msg = self._selective_aggregate(x, x_src, x_dst, src, dst, ra_mask)
            else:
                # RA for all nodes
                msg = self._ra_aggregate(x_src, x_dst, dst, x.size(0))
        else:
            # Mean aggregation (baseline)
            msg = scatter_mean(x_src, dst, dim=0, dim_size=x.size(0))

        out = self.lin_self(x) + self.lin_neigh(msg)

        # Variance residual
        if self.use_variance:
            msg_mean = scatter_mean(x_src, dst, dim=0, dim_size=x.size(0))
            diff_sq = (x_src - msg_mean[dst]) ** 2
            var = scatter_mean(diff_sq, dst, dim=0, dim_size=x.size(0))
            out = out + self.var_proj(var)

        return out

    def _ra_aggregate(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Reciprocal attention aggregation for all nodes."""
        # attn[e] = x_src[e] @ x_dst[e] (neighbor queries center)
        attn_logits = (x_src * x_dst).sum(dim=-1) * self.scale
        attn = scatter_softmax(attn_logits, dst, dim=0)
        weighted = x_src * attn.unsqueeze(-1)
        return scatter_add(weighted, dst, dim=0, dim_size=num_nodes)

    def _selective_aggregate(
        self,
        x: torch.Tensor,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        ra_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Selective aggregation: RA for masked nodes, mean for others."""
        num_nodes = x.size(0)

        # Compute both aggregations
        msg_mean = scatter_mean(x_src, dst, dim=0, dim_size=num_nodes)
        msg_ra = self._ra_aggregate(x_src, x_dst, dst, num_nodes)

        # Select based on mask
        ra_mask_expanded = ra_mask.unsqueeze(-1)
        return torch.where(ra_mask_expanded, msg_ra, msg_mean)


class BaselineGraphSAGE(nn.Module):
    """Standard GraphSAGE with mean aggregation."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


class RAGraphSAGE(nn.Module):
    """GraphSAGE with Reciprocal Attention aggregation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        use_reciprocal: bool = True,
        use_variance: bool = True,
    ):
        super().__init__()
        self.conv1 = RAConv(in_channels, hidden_channels, use_reciprocal, use_variance)
        self.conv2 = RAConv(hidden_channels, out_channels, use_reciprocal, use_variance)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ra_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, ra_mask)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, ra_mask)
        return F.log_softmax(x, dim=-1)


class FIMGuidedRAGraphSAGE(nn.Module):
    """
    GraphSAGE with FIM-guided Reciprocal Attention.

    Computes FIM trace per node and applies RA only to low-FIM (uncertain) nodes.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        fim_percentile: float = 33.0,
    ):
        super().__init__()
        self.conv1 = RAConv(
            in_channels, hidden_channels, use_reciprocal=True, use_variance=True
        )
        self.conv2 = RAConv(
            hidden_channels, out_channels, use_reciprocal=True, use_variance=True
        )
        self.fim_percentile = fim_percentile

        # FIM tracking
        self.register_buffer("fim_trace", None)
        self.register_buffer("ra_mask", None)
        self._grad_accumulator = {}

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get RA mask for current batch
        ra_mask = None
        if self.ra_mask is not None and node_ids is not None:
            ra_mask = self.ra_mask[node_ids]

        x = self.conv1(x, edge_index, ra_mask)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, ra_mask)
        return F.log_softmax(x, dim=-1)

    def update_fim(self, num_nodes: int):
        """Update FIM trace and RA mask from accumulated gradients."""
        if not self._grad_accumulator:
            return

        # Aggregate FIM trace
        fim = torch.zeros(num_nodes, device=next(self.parameters()).device)
        counts = torch.zeros(num_nodes, device=fim.device)

        for node_id, grad_sq in self._grad_accumulator.items():
            fim[node_id] += grad_sq
            counts[node_id] += 1

        # Normalize by count
        mask = counts > 0
        fim[mask] /= counts[mask]

        self.fim_trace = fim

        # Compute RA mask: bottom fim_percentile% get RA
        threshold = torch.quantile(fim[mask], self.fim_percentile / 100.0)
        self.ra_mask = fim <= threshold

        # Clear accumulator
        self._grad_accumulator.clear()

        # Stats
        n_ra = self.ra_mask.sum().item()
        print(
            f"  FIM update: {n_ra:,} nodes ({100 * n_ra / num_nodes:.1f}%) get RA treatment"
        )

    def accumulate_fim(self, node_ids: torch.Tensor, hidden: torch.Tensor):
        """Accumulate FIM trace from gradients."""
        if hidden.grad is not None:
            grad_sq = (hidden.grad**2).sum(dim=-1)
            for i, nid in enumerate(node_ids.tolist()):
                if nid not in self._grad_accumulator:
                    self._grad_accumulator[nid] = 0.0
                self._grad_accumulator[nid] += grad_sq[i].item()


# =============================================================================
# Training & Evaluation
# =============================================================================


def compute_metrics(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute fraud detection metrics."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    y_pred = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = auc(recall, precision)

    return {"f1": f1, "pr_auc": pr_auc, "threshold": threshold}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Evaluate model on masked nodes."""
    model.eval()

    # Full graph evaluation
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    if hasattr(model, "fim_trace"):
        # FIM-guided model needs node_ids
        out = model(x, edge_index, torch.arange(x.size(0), device=device))
    else:
        out = model(x, edge_index)

    probs = out.exp()[:, 1].cpu().numpy()
    y_true = data.y.cpu().numpy()

    return compute_metrics(probs[mask], y_true[mask])


def train_baseline(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    config: RABenchmarkConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train baseline GraphSAGE."""
    print("\n" + "=" * 60)
    print("Training: Baseline GraphSAGE")
    print("=" * 60)

    model = BaselineGraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=config.hidden_channels,
        out_channels=config.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Create loader
    train_idx = torch.from_numpy(np.where(train_mask)[0])
    loader = NeighborLoader(
        data,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        input_nodes=train_idx,
        shuffle=True,
    )

    start_time = time.time()
    best_val = {"f1": 0, "pr_auc": 0}
    step = 0

    pbar = tqdm(total=int(config.time_limit), desc="Training", unit="s")

    while time.time() - start_time < config.time_limit:
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)
            # Only supervise training nodes in batch
            train_mask_batch = (
                batch.train_mask if hasattr(batch, "train_mask") else slice(None)
            )
            loss = F.nll_loss(out[: batch.batch_size], batch.y[: batch.batch_size])

            loss.backward()
            optimizer.step()
            step += 1

            if time.time() - start_time >= config.time_limit:
                break

        # Evaluate
        val_metrics = evaluate(model, data, val_mask, device)
        if val_metrics["f1"] > best_val["f1"]:
            best_val = val_metrics

        elapsed = time.time() - start_time
        pbar.n = min(int(elapsed), int(config.time_limit))
        pbar.set_postfix({"F1": f"{val_metrics['f1']:.4f}", "step": step})
        pbar.refresh()

    pbar.close()
    print(f"  Best val F1: {best_val['f1']:.4f}, PR-AUC: {best_val['pr_auc']:.4f}")

    return model, best_val


def train_ra(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    config: RABenchmarkConfig,
    device: torch.device,
    use_reciprocal: bool = True,
    use_variance: bool = True,
    name: str = "RA",
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train RA-GraphSAGE (all nodes get RA)."""
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    model = RAGraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=config.hidden_channels,
        out_channels=config.num_classes,
        use_reciprocal=use_reciprocal,
        use_variance=use_variance,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_idx = torch.from_numpy(np.where(train_mask)[0])
    loader = NeighborLoader(
        data,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        input_nodes=train_idx,
        shuffle=True,
    )

    start_time = time.time()
    best_val = {"f1": 0, "pr_auc": 0}
    step = 0

    pbar = tqdm(total=int(config.time_limit), desc="Training", unit="s")

    while time.time() - start_time < config.time_limit:
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[: batch.batch_size], batch.y[: batch.batch_size])

            loss.backward()
            optimizer.step()
            step += 1

            if time.time() - start_time >= config.time_limit:
                break

        val_metrics = evaluate(model, data, val_mask, device)
        if val_metrics["f1"] > best_val["f1"]:
            best_val = val_metrics

        elapsed = time.time() - start_time
        pbar.n = min(int(elapsed), int(config.time_limit))
        pbar.set_postfix({"F1": f"{val_metrics['f1']:.4f}", "step": step})
        pbar.refresh()

    pbar.close()
    print(f"  Best val F1: {best_val['f1']:.4f}, PR-AUC: {best_val['pr_auc']:.4f}")

    return model, best_val


def train_fim_guided(
    data: Data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    config: RABenchmarkConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train FIM-guided RA-GraphSAGE."""
    print("\n" + "=" * 60)
    print(f"Training: FIM-Guided RA (bottom {config.fim_percentile}% nodes)")
    print("=" * 60)

    model = FIMGuidedRAGraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=config.hidden_channels,
        out_channels=config.num_classes,
        fim_percentile=config.fim_percentile,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_idx = torch.from_numpy(np.where(train_mask)[0])
    loader = NeighborLoader(
        data,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        input_nodes=train_idx,
        shuffle=True,
    )

    start_time = time.time()
    best_val = {"f1": 0, "pr_auc": 0}
    step = 0

    pbar = tqdm(total=int(config.time_limit), desc="Training", unit="s")

    while time.time() - start_time < config.time_limit:
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward with node tracking
            node_ids = batch.n_id.to(device)
            out = model(batch.x, batch.edge_index, node_ids)
            loss = F.nll_loss(out[: batch.batch_size], batch.y[: batch.batch_size])

            loss.backward()
            optimizer.step()
            step += 1

            # Update FIM periodically
            if step % config.fim_update_interval == 0:
                model.update_fim(data.x.size(0))

            if time.time() - start_time >= config.time_limit:
                break

        val_metrics = evaluate(model, data, val_mask, device)
        if val_metrics["f1"] > best_val["f1"]:
            best_val = val_metrics

        elapsed = time.time() - start_time
        pbar.n = min(int(elapsed), int(config.time_limit))
        pbar.set_postfix({"F1": f"{val_metrics['f1']:.4f}", "step": step})
        pbar.refresh()

    pbar.close()
    print(f"  Best val F1: {best_val['f1']:.4f}, PR-AUC: {best_val['pr_auc']:.4f}")

    return model, best_val


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="RA-GNN Benchmark on DGraphFin")
    parser.add_argument(
        "--time", type=float, default=300, help="Training time per model (seconds)"
    )
    parser.add_argument("--ablation", action="store_true", help="Run full ablation")
    parser.add_argument(
        "--fim-percentile", type=float, default=33.0, help="FIM percentile for RA"
    )
    parser.add_argument(
        "--data-dir", type=str, default=".", help="Directory containing dgraphfin.npz"
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Config
    config = RABenchmarkConfig(
        time_limit=args.time,
        fim_percentile=args.fim_percentile,
    )

    # Load data
    data, train_mask, val_mask, test_mask = load_dgraphfin(args.data_dir)
    data = data.to(device)

    # Define experiments
    if args.ablation:
        experiments = [
            (
                "Baseline",
                lambda: train_baseline(data, train_mask, val_mask, config, device),
            ),
            (
                "RA only",
                lambda: train_ra(
                    data,
                    train_mask,
                    val_mask,
                    config,
                    device,
                    use_reciprocal=True,
                    use_variance=False,
                    name="RA only",
                ),
            ),
            (
                "Var only",
                lambda: train_ra(
                    data,
                    train_mask,
                    val_mask,
                    config,
                    device,
                    use_reciprocal=False,
                    use_variance=True,
                    name="Var only",
                ),
            ),
            (
                "RA + Var",
                lambda: train_ra(
                    data,
                    train_mask,
                    val_mask,
                    config,
                    device,
                    use_reciprocal=True,
                    use_variance=True,
                    name="RA + Var",
                ),
            ),
            (
                "FIM-guided",
                lambda: train_fim_guided(data, train_mask, val_mask, config, device),
            ),
        ]
    else:
        experiments = [
            (
                "Baseline",
                lambda: train_baseline(data, train_mask, val_mask, config, device),
            ),
            (
                "RA + Var",
                lambda: train_ra(
                    data,
                    train_mask,
                    val_mask,
                    config,
                    device,
                    use_reciprocal=True,
                    use_variance=True,
                    name="RA + Var",
                ),
            ),
            (
                "FIM-guided",
                lambda: train_fim_guided(data, train_mask, val_mask, config, device),
            ),
        ]

    # Run experiments
    results = {}
    for name, train_fn in experiments:
        model, val_metrics = train_fn()

        # Test evaluation
        test_metrics = evaluate(model, data, test_mask, device)
        results[name] = {
            "val_f1": val_metrics["f1"],
            "val_pr_auc": val_metrics["pr_auc"],
            "test_f1": test_metrics["f1"],
            "test_pr_auc": test_metrics["pr_auc"],
        }
        print(
            f"  Test F1: {test_metrics['f1']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}"
        )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Val F1':<10} {'Test F1':<10} {'Test PR-AUC':<12}")
    print("-" * 50)
    for name, r in results.items():
        print(
            f"{name:<15} {r['val_f1']:<10.4f} {r['test_f1']:<10.4f} {r['test_pr_auc']:<12.4f}"
        )

    # Delta analysis
    if "Baseline" in results and "RA + Var" in results:
        baseline_f1 = results["Baseline"]["test_f1"]
        ra_f1 = results["RA + Var"]["test_f1"]
        delta = (ra_f1 - baseline_f1) * 100
        print(f"\nRA + Var vs Baseline: {delta:+.2f}% F1")

    if "Baseline" in results and "FIM-guided" in results:
        baseline_f1 = results["Baseline"]["test_f1"]
        fim_f1 = results["FIM-guided"]["test_f1"]
        delta = (fim_f1 - baseline_f1) * 100
        print(f"FIM-guided vs Baseline: {delta:+.2f}% F1")


if __name__ == "__main__":
    main()
