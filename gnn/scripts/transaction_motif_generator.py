#!/usr/bin/env python3
"""
Typed Transaction-Motif Generator

Generates graphs that mimic fraud transaction networks with:
- Typed nodes: accounts, merchants, devices
- Star/forest-like local structure (accounts connect to merchants/devices)
- Community structure (groups of accounts sharing merchants)
- High leaf fraction (merchants/devices often have few connections)
- Bounded degree variance, no isolates

This aims to reproduce DGraphFin's structural properties:
- 44% leaves
- 0% isolates
- Low METIS cut ratio
- High intra-page edges after BFS layout
"""

import argparse
import numpy as np
import json
import os
from collections import defaultdict


def generate_transaction_graph(
    n_accounts=25000,
    n_merchants=15000,
    n_devices=10000,
    n_communities=100,
    avg_merchants_per_account=2,
    avg_devices_per_account=1,
    inter_community_rate=0.01,
    seed=42,
):
    """
    Generate a transaction graph with typed nodes.

    Structure:
    - Each account belongs to one community
    - Merchants are local to communities (shared by accounts in same community)
    - Devices are personal (usually connected to one account)
    - Some cross-community edges (account uses merchant from other community)
    """
    np.random.seed(seed)

    n_total = n_accounts + n_merchants + n_devices
    print(f"Generating transaction graph: {n_total} nodes")
    print(f"  Accounts: {n_accounts}")
    print(f"  Merchants: {n_merchants}")
    print(f"  Devices: {n_devices}")
    print(f"  Communities: {n_communities}")

    # Assign accounts to communities
    accounts_per_community = n_accounts // n_communities
    account_community = np.repeat(np.arange(n_communities), accounts_per_community)
    if len(account_community) < n_accounts:
        account_community = np.concatenate(
            [
                account_community,
                np.zeros(n_accounts - len(account_community), dtype=int),
            ]
        )
    np.random.shuffle(account_community)

    # Assign merchants to communities (some merchants span communities)
    merchants_per_community = n_merchants // n_communities
    merchant_community = np.repeat(np.arange(n_communities), merchants_per_community)
    if len(merchant_community) < n_merchants:
        merchant_community = np.concatenate(
            [
                merchant_community,
                np.random.randint(
                    0, n_communities, n_merchants - len(merchant_community)
                ),
            ]
        )
    np.random.shuffle(merchant_community)

    # Build community -> merchants mapping (vectorized)
    print("Building community-merchant mapping...")
    community_merchants = defaultdict(list)
    for m, c in enumerate(merchant_community):
        community_merchants[c].append(m + n_accounts)
    # Convert to numpy arrays for faster sampling
    community_merchants_arr = {
        c: np.array(merchants) for c, merchants in community_merchants.items()
    }

    # Node ID mapping:
    # 0 to n_accounts-1: accounts
    # n_accounts to n_accounts+n_merchants-1: merchants
    # n_accounts+n_merchants to n_total-1: devices
    device_start = n_accounts + n_merchants

    print("Generating account-merchant edges (vectorized)...")
    # Vectorized: generate number of connections per account
    n_connections_per_account = np.maximum(
        1, np.random.poisson(avg_merchants_per_account, n_accounts)
    )
    total_merchant_edges = n_connections_per_account.sum()

    # Pre-allocate edge arrays
    merchant_edges_src = np.zeros(total_merchant_edges, dtype=np.int32)
    merchant_edges_dst = np.zeros(total_merchant_edges, dtype=np.int32)

    # Generate edges in chunks for memory efficiency
    edge_idx = 0
    chunk_size = 100000
    for chunk_start in range(0, n_accounts, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_accounts)
        for account in range(chunk_start, chunk_end):
            community = account_community[account]
            n_conn = n_connections_per_account[account]

            for _ in range(n_conn):
                if np.random.random() < inter_community_rate:
                    other_community = np.random.randint(n_communities)
                    merchants = community_merchants_arr.get(other_community)
                else:
                    merchants = community_merchants_arr.get(community)

                if merchants is not None and len(merchants) > 0:
                    merchant = np.random.choice(merchants)
                    merchant_edges_src[edge_idx] = account
                    merchant_edges_dst[edge_idx] = merchant
                    edge_idx += 1

        if chunk_start % 1000000 == 0 and chunk_start > 0:
            print(f"  Processed {chunk_start:,} accounts...")

    merchant_edges_src = merchant_edges_src[:edge_idx]
    merchant_edges_dst = merchant_edges_dst[:edge_idx]
    print(f"  Generated {edge_idx:,} account-merchant edges")

    print("Generating account-device edges (vectorized)...")
    # Vectorized device assignment
    n_devices_per_account = np.maximum(
        0, np.random.poisson(avg_devices_per_account, n_accounts)
    )
    total_device_edges = min(n_devices_per_account.sum(), n_devices)

    # Assign devices to accounts
    device_edges_src = np.zeros(n_devices, dtype=np.int32)
    device_edges_dst = np.arange(device_start, device_start + n_devices, dtype=np.int32)

    device_idx = 0
    for account in range(n_accounts):
        n_dev = n_devices_per_account[account]
        for _ in range(n_dev):
            if device_idx < n_devices:
                device_edges_src[device_idx] = account
                device_idx += 1

    # Remaining devices get random accounts
    if device_idx < n_devices:
        device_edges_src[device_idx:] = np.random.randint(
            0, n_accounts, n_devices - device_idx
        )

    print(f"  Generated {n_devices:,} account-device edges")

    # Combine edges
    edges_src = np.concatenate([merchant_edges_src, device_edges_src])
    edges_dst = np.concatenate([merchant_edges_dst, device_edges_dst])

    print("Ensuring no merchant isolates...")
    # Check merchant connectivity (vectorized)
    merchant_ids = np.arange(n_accounts, device_start)
    merchant_in_edges = np.isin(merchant_ids, edges_dst)
    unconnected_merchants = merchant_ids[~merchant_in_edges]

    if len(unconnected_merchants) > 0:
        print(f"  Connecting {len(unconnected_merchants):,} isolated merchants...")
        # Connect each unconnected merchant to a random account in its community
        extra_src = []
        extra_dst = []
        for m in unconnected_merchants:
            comm = merchant_community[m - n_accounts]
            comm_accounts = np.where(account_community == comm)[0]
            if len(comm_accounts) > 0:
                account = np.random.choice(comm_accounts)
                extra_src.append(account)
                extra_dst.append(m)
        if extra_src:
            edges_src = np.concatenate([edges_src, np.array(extra_src, dtype=np.int32)])
            edges_dst = np.concatenate([edges_dst, np.array(extra_dst, dtype=np.int32)])

    # Remove duplicates
    print("Removing duplicate edges...")
    edges_combined = np.stack(
        [np.minimum(edges_src, edges_dst), np.maximum(edges_src, edges_dst)], axis=1
    )
    edges_unique = np.unique(edges_combined, axis=0)

    print(f"Generated {len(edges_unique):,} unique edges")

    # edges_unique is already numpy array
    edges_array = edges_unique

    # Compute stats (vectorized)
    degrees = np.zeros(n_total, dtype=int)
    np.add.at(degrees, edges_array[:, 0], 1)
    np.add.at(degrees, edges_array[:, 1], 1)

    isolates = np.sum(degrees == 0)
    leaves = np.sum(degrees == 1)

    print(f"Stats:")
    print(f"  Avg degree: {degrees.mean():.2f}")
    print(f"  Isolates: {isolates} ({isolates/n_total:.1%})")
    print(f"  Leaves: {leaves} ({leaves/n_total:.1%})")

    # Node types
    node_types = np.zeros(n_total, dtype=int)
    node_types[n_accounts : n_accounts + n_merchants] = 1  # merchants
    node_types[n_accounts + n_merchants :] = 2  # devices

    return edges_array, n_total, node_types, account_community, merchant_community


def generate_features_and_labels(
    n_total, node_types, account_community, n_communities, fraud_rate=0.02, feat_dim=16
):
    """
    Generate synthetic features and fraud labels (vectorized for speed).

    Features are community-aware: nodes in the same community have similar features.
    Labels mark ~2% of accounts as fraudulent, with fraud concentrated in certain
    communities (simulating fraud rings).
    """
    np.random.seed(43)
    print("Generating features (vectorized)...")

    # Generate community embeddings
    community_embeddings = np.random.randn(n_communities, feat_dim).astype(np.float32)

    # Type embeddings
    type_embeddings = np.random.randn(3, feat_dim).astype(np.float32) * 0.5

    # Vectorized feature generation
    features = np.zeros((n_total, feat_dim), dtype=np.float32)

    # Accounts: community embedding + type embedding
    account_mask = node_types == 0
    n_accounts = account_mask.sum()
    # account_community is indexed 0 to n_accounts-1, so use direct indexing
    account_indices = np.where(account_mask)[0]
    features[account_indices] = (
        community_embeddings[account_community[: len(account_indices)]]
        + type_embeddings[0]
    )

    # Merchants: averaged community features + type embedding
    merchant_mask = node_types == 1
    merchant_mean = community_embeddings.mean(axis=0)
    features[merchant_mask] = merchant_mean + type_embeddings[1]

    # Devices: just type embedding
    device_mask = node_types == 2
    features[device_mask] = type_embeddings[2]

    # Add noise (vectorized)
    features += np.random.randn(n_total, feat_dim).astype(np.float32) * 0.3

    print("Generating labels...")
    # Generate labels (only accounts can be fraudulent)
    labels = np.zeros(n_total, dtype=np.int64)

    # Select ~10% of communities as "fraud rings"
    n_fraud_communities = max(1, n_communities // 10)
    fraud_communities = set(
        np.random.choice(n_communities, n_fraud_communities, replace=False)
    )

    # Vectorized label generation
    # account_community is indexed 0 to n_accounts-1
    is_fraud_community = np.array(
        [account_community[i] in fraud_communities for i in range(n_accounts)]
    )
    rand_vals = np.random.random(n_accounts)

    # 15% fraud rate in fraud communities, 0.5% elsewhere
    fraud_mask = (is_fraud_community & (rand_vals < 0.15)) | (
        ~is_fraud_community & (rand_vals < 0.005)
    )
    # Accounts are nodes 0 to n_accounts-1
    labels[np.where(fraud_mask)[0]] = 1

    fraud_count = np.sum(labels == 1)
    print(
        f"Generated {fraud_count} fraud labels ({fraud_count/n_accounts:.1%} of accounts)"
    )

    return features, labels


def main():
    parser = argparse.ArgumentParser(
        description="Transaction-motif graph generator for fraud detection benchmarks"
    )
    parser.add_argument("--n-accounts", type=int, default=25000)
    parser.add_argument("--n-merchants", type=int, default=15000)
    parser.add_argument("--n-devices", type=int, default=10000)
    parser.add_argument("--n-communities", type=int, default=100)
    parser.add_argument("--avg-merchants-per-account", type=float, default=2.0)
    parser.add_argument("--avg-devices-per-account", type=float, default=1.0)
    parser.add_argument("--inter-community-rate", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="/tmp/transaction_graph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-features", action="store_true", help="Generate features and labels"
    )
    parser.add_argument("--feat-dim", type=int, default=16, help="Feature dimension")
    args = parser.parse_args()

    edges, n_total, node_types, account_comm, merchant_comm = (
        generate_transaction_graph(
            n_accounts=args.n_accounts,
            n_merchants=args.n_merchants,
            n_devices=args.n_devices,
            n_communities=args.n_communities,
            avg_merchants_per_account=args.avg_merchants_per_account,
            avg_devices_per_account=args.avg_devices_per_account,
            inter_community_rate=args.inter_community_rate,
            seed=args.seed,
        )
    )

    # Compute final stats
    degrees = np.zeros(n_total, dtype=int)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    # Save
    os.makedirs(args.output, exist_ok=True)
    np.save(f"{args.output}/edges.npy", edges)
    np.save(f"{args.output}/node_types.npy", node_types)

    # Generate features and labels if requested
    if args.with_features:
        features, labels = generate_features_and_labels(
            n_total,
            node_types,
            account_comm,
            args.n_communities,
            feat_dim=args.feat_dim,
        )
        np.save(f"{args.output}/features.npy", features)
        np.save(f"{args.output}/labels.npy", labels)

        # Generate train/val/test masks (70/15/15 split on accounts only)
        n_accounts = args.n_accounts
        perm = np.random.permutation(n_accounts)
        train_end = int(0.7 * n_accounts)
        val_end = int(0.85 * n_accounts)

        train_mask = np.zeros(n_total, dtype=bool)
        val_mask = np.zeros(n_total, dtype=bool)
        test_mask = np.zeros(n_total, dtype=bool)

        train_mask[perm[:train_end]] = True
        val_mask[perm[train_end:val_end]] = True
        test_mask[perm[val_end:n_accounts]] = True

        np.save(f"{args.output}/train_mask.npy", train_mask)
        np.save(f"{args.output}/val_mask.npy", val_mask)
        np.save(f"{args.output}/test_mask.npy", test_mask)

    meta = {
        "num_nodes": n_total,
        "num_edges": len(edges),
        "avg_degree": float(degrees.mean()),
        "isolates": int(np.sum(degrees == 0)),
        "leaf_fraction": float(np.sum(degrees == 1) / n_total),
        "generator": "Transaction-Motif",
        "params": {
            "n_accounts": args.n_accounts,
            "n_merchants": args.n_merchants,
            "n_devices": args.n_devices,
            "n_communities": args.n_communities,
            "avg_merchants_per_account": args.avg_merchants_per_account,
            "avg_devices_per_account": args.avg_devices_per_account,
            "inter_community_rate": args.inter_community_rate,
        },
    }
    if args.with_features:
        meta["feat_dim"] = args.feat_dim
        meta["num_classes"] = 2

    with open(f"{args.output}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
