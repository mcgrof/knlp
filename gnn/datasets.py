#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
GNN Dataset Loaders with HuggingFace/Download Support

Supports:
- DGraphFin: Financial fraud detection (3.7M nodes)
- YelpChi: Review spam detection (45K nodes)
- Amazon: Product fraud detection (11K nodes)
- Elliptic: Bitcoin fraud detection (203K nodes)
- ogbn-products: Product classification (2.4M nodes)
- ogbn-proteins: Protein function prediction (132K nodes)
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def download_prompt(dataset_name: str, instructions: str):
    """Print download instructions and exit."""
    print(f"\n{'='*60}")
    print(f"Dataset not found: {dataset_name}")
    print(f"{'='*60}")
    print(instructions)
    print(f"{'='*60}\n")
    sys.exit(1)


def load_dgraphfin(root: str = ".") -> Data:
    """Load DGraphFin financial fraud detection dataset.

    Args:
        root: Directory containing dgraphfin.npz

    Returns:
        PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask
    """
    npz_path = os.path.join(root, "dgraphfin.npz")

    if not os.path.exists(npz_path):
        download_prompt("DGraphFin", f"""
DGraphFin is a financial fraud detection graph dataset with 3.7M nodes.

To download via HuggingFace:
    pip install huggingface_hub
    python -c "from huggingface_hub import hf_hub_download; \\
        hf_hub_download('YinzhenWan/DGraphFin', 'dgraphfin.npz', \\
        local_dir='{root}')"

Or manually download from:
    https://huggingface.co/datasets/YinzhenWan/DGraphFin

Expected file: {npz_path}
""")

    print(f"Loading DGraphFin from {npz_path}...")
    npz_data = np.load(npz_path)

    x = torch.from_numpy(npz_data["x"]).float()
    y = torch.from_numpy(npz_data["y"]).long()
    edge_index = torch.from_numpy(npz_data["edge_index"].T).long()

    # Normalize features
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    # Convert index arrays to boolean masks
    num_nodes = x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[npz_data["train_mask"]] = True
    val_mask[npz_data["valid_mask"]] = True
    test_mask[npz_data["test_mask"]] = True

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_classes = 4

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Features: {x.shape[1]}")
    print(f"  Classes: 4")
    print(f"  Train/Val/Test: {train_mask.sum():,}/{val_mask.sum():,}/{test_mask.sum():,}")

    return data


def load_yelpchi(root: str = ".", edge_type: str = "net_rtr") -> Data:
    """Load YelpChi review spam detection dataset.

    Args:
        root: Directory containing fraud/YelpChi.mat
        edge_type: Edge type to use (net_rtr recommended, net_rur, net_rsr, homo)

    Returns:
        PyG Data object
    """
    import scipy.io

    mat_path = os.path.join(root, "fraud", "YelpChi.mat")

    if not os.path.exists(mat_path):
        download_prompt("YelpChi", f"""
YelpChi is a review spam detection dataset with 45K review nodes.

To download:
    mkdir -p {root}/fraud
    wget https://data.dgl.ai/dataset/FraudYelp.zip -O /tmp/FraudYelp.zip
    unzip /tmp/FraudYelp.zip -d {root}/fraud/

Expected file: {mat_path}
""")

    print(f"Loading YelpChi from {mat_path} (edge_type={edge_type})...")
    mat_data = scipy.io.loadmat(mat_path)

    # Features are sparse - convert to dense
    x = torch.from_numpy(mat_data["features"].toarray()).float()
    y = torch.from_numpy(mat_data["label"].flatten()).long()

    # Get edge_index from specified edge type
    adj_matrix = mat_data[edge_type]
    coo = adj_matrix.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

    num_nodes = x.size(0)

    # Create train/val/test masks (60/20/20 split)
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_classes = 2

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Features: {x.shape[1]}")
    print(f"  Classes: 2 (spam/legitimate)")

    return data


def load_amazon(root: str = ".", edge_type: str = "net_upu") -> Data:
    """Load Amazon product fraud detection dataset.

    Args:
        root: Directory containing fraud/Amazon.mat
        edge_type: Edge type to use (net_upu, net_usu, net_uvu)

    Returns:
        PyG Data object
    """
    import scipy.io

    mat_path = os.path.join(root, "fraud", "Amazon.mat")

    if not os.path.exists(mat_path):
        download_prompt("Amazon", f"""
Amazon is a product fraud detection dataset with 11K user nodes.

To download:
    mkdir -p {root}/fraud
    wget https://data.dgl.ai/dataset/FraudAmazon.zip -O /tmp/FraudAmazon.zip
    unzip /tmp/FraudAmazon.zip -d {root}/fraud/

Expected file: {mat_path}
""")

    print(f"Loading Amazon from {mat_path} (edge_type={edge_type})...")
    mat_data = scipy.io.loadmat(mat_path)

    x = torch.from_numpy(mat_data["features"].toarray()).float()
    y = torch.from_numpy(mat_data["label"].flatten()).long()

    adj_matrix = mat_data[edge_type]
    coo = adj_matrix.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

    num_nodes = x.size(0)

    # Create masks
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_classes = 2

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Features: {x.shape[1]}")
    print(f"  Classes: 2 (fraud/legitimate)")

    return data


def load_elliptic(root: str = ".") -> Data:
    """Load Elliptic Bitcoin fraud detection dataset.

    Args:
        root: Directory containing elliptic_bitcoin_dataset/

    Returns:
        PyG Data object
    """
    data_dir = os.path.join(root, "elliptic_bitcoin_dataset")

    if not os.path.exists(data_dir):
        download_prompt("Elliptic", f"""
Elliptic is a Bitcoin transaction fraud detection dataset with 203K nodes.

To download from Kaggle:
    pip install kaggle
    kaggle datasets download -d ellipticco/elliptic-data-set
    unzip elliptic-data-set.zip -d {root}/

Or manually download from:
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Expected directory: {data_dir}
""")

    print(f"Loading Elliptic from {data_dir}...")

    # Load features
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    features_df = np.loadtxt(features_path, delimiter=',', skiprows=1)
    node_ids = features_df[:, 0].astype(int)
    x = torch.from_numpy(features_df[:, 1:]).float()

    # Create node_id to index mapping
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # Load edges
    edges_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    edges_df = np.loadtxt(edges_path, delimiter=',', skiprows=1, dtype=int)

    # Filter edges to only include nodes in features
    valid_edges = []
    for src, dst in edges_df:
        if src in node_to_idx and dst in node_to_idx:
            valid_edges.append([node_to_idx[src], node_to_idx[dst]])
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t()

    # Load labels (1=illicit, 2=licit, unknown otherwise)
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    with open(classes_path, 'r') as f:
        next(f)  # skip header
        labels = {}
        for line in f:
            parts = line.strip().split(',')
            node_id = int(parts[0])
            label = parts[1]
            if label == '1':
                labels[node_id] = 1  # illicit
            elif label == '2':
                labels[node_id] = 0  # licit
            # unknown stays as -1

    y = torch.full((len(node_ids),), -1, dtype=torch.long)
    for nid, idx in node_to_idx.items():
        if nid in labels:
            y[idx] = labels[nid]

    # Create masks based on labeled nodes
    labeled_mask = y >= 0
    labeled_indices = torch.where(labeled_mask)[0]
    perm = labeled_indices[torch.randperm(len(labeled_indices))]

    train_size = int(0.6 * len(perm))
    val_size = int(0.2 * len(perm))

    num_nodes = x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True

    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_classes = 2

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Features: {x.shape[1]}")
    print(f"  Labeled: {labeled_mask.sum():,}")
    print(f"  Classes: 2 (illicit/licit)")

    return data


def load_ogbn_products(root: str = ".") -> Data:
    """Load ogbn-products node classification dataset.

    Args:
        root: Directory for OGB data

    Returns:
        PyG Data object
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        download_prompt("ogbn-products", """
OGB (Open Graph Benchmark) package required.

To install:
    pip install ogb

Then run again - the dataset will download automatically (~1.5GB).
""")

    print("Loading ogbn-products...")
    dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    num_nodes = data.x.size(0)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.y = data.y.squeeze()
    data.num_classes = dataset.num_classes

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Classes: {dataset.num_classes}")

    return data


def load_ogbn_proteins(root: str = ".") -> Data:
    """Load ogbn-proteins node classification dataset.

    Args:
        root: Directory for OGB data

    Returns:
        PyG Data object
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        download_prompt("ogbn-proteins", """
OGB (Open Graph Benchmark) package required.

To install:
    pip install ogb

Then run again - the dataset will download automatically.
""")

    print("Loading ogbn-proteins...")
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root=root)
    data = dataset[0]

    # ogbn-proteins uses edge features as node features
    # Aggregate edge features to node features
    from torch_geometric.utils import scatter
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, dim=0, reduce='mean')

    split_idx = dataset.get_idx_split()
    num_nodes = data.x.size(0)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_classes = 112  # multi-label

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Classes: 112 (multi-label)")

    return data


# Dataset registry
DATASETS = {
    'dgraphfin': load_dgraphfin,
    'yelpchi': load_yelpchi,
    'amazon': load_amazon,
    'elliptic': load_elliptic,
    'ogbn-products': load_ogbn_products,
    'ogbn-proteins': load_ogbn_proteins,
}


def load_dataset(name: str, root: str = ".", **kwargs) -> Data:
    """Load a dataset by name.

    Args:
        name: Dataset name (dgraphfin, yelpchi, amazon, elliptic, ogbn-products, ogbn-proteins)
        root: Data directory
        **kwargs: Additional arguments for specific loaders

    Returns:
        PyG Data object
    """
    name = name.lower()
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    return DATASETS[name](root, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--dataset", default="dgraphfin", choices=list(DATASETS.keys()))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    data = load_dataset(args.dataset, args.root)
    print(f"\nLoaded {args.dataset}: {data}")
