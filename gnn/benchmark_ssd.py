#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Force-SSD read-amplification comparison across page layouts.

This is the engine behind ``make defconfig-gnn-dgraphfin-force-ssd``. It
serves DGraphFin features from an on-disk store (ssd_feature_store) and
reads them back with real I/O so external tooling (eBPF
biosnoop/biolatency, blktrace, iostat) can verify the read intent per
I/O instead of trusting the in-RAM page-touch counter.

For each layout (natural, random, bfs, metis) it runs two access
patterns and measures real device read amplification:

  - neighbor: random training seeds expanded to their sampled
    neighborhood, the locality-sensitive pattern. RA_fetch here drops
    sharply from natural to metis -- this is the value of the layout.

  - page: page-aligned sweep over training pages, the page-aware
    pattern. RA_fetch is ~1x for every layout, but RA_signal stays near
    the ~4.3x floor (only a fraction of nodes per page are supervised)
    no matter how good the layout is -- this is the gap the layout
    cannot close, which is exactly what we want a reader with their own
    eBPF tooling to confirm.

The smoke path (--smoke) runs the whole comparison on a synthetic
community graph with numpy only (no torch), so the I/O and accounting
can be validated anywhere:

    python gnn/benchmark_ssd.py --smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from ssd_feature_store import (  # noqa: E402
    PAGE_SIZE,
    SSDFeatureReader,
    drop_caches,
    write_feature_file,
)


# --------------------------------------------------------------------------
# Graph helpers (numpy-only)
# --------------------------------------------------------------------------
def build_csr(edge_index, num_nodes):
    """Undirected CSR adjacency from a [2, E] edge_index."""
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    order = np.argsort(src, kind="stable")
    src_sorted = src[order]
    indices = dst[order].astype(np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    counts = np.bincount(src_sorted, minlength=num_nodes)
    indptr[1:] = np.cumsum(counts)
    return indptr, indices


def sample_neighbors(seeds, indptr, indices, fanouts, rng):
    """Expand seeds to a node set via per-hop neighbor sampling."""
    nodes = set(int(s) for s in seeds)
    current = np.asarray(seeds, dtype=np.int64)
    for k in fanouts:
        nxt = []
        for u in current:
            a, b = int(indptr[u]), int(indptr[u + 1])
            deg = b - a
            if deg == 0:
                continue
            if deg <= k:
                nb = indices[a:b]
            else:
                sel = rng.choice(deg, size=k, replace=False)
                nb = indices[a + sel]
            nxt.append(nb)
            nodes.update(int(v) for v in nb)
        if not nxt:
            break
        current = np.unique(np.concatenate(nxt))
    return np.fromiter(nodes, dtype=np.int64, count=len(nodes))


def page_node_groups(page_id):
    """Map page index -> array of node ids on that page."""
    order = np.argsort(page_id, kind="stable")
    sp = page_id[order]
    if len(sp) == 0:
        return {}
    change = np.where(np.diff(sp) != 0)[0] + 1
    groups = np.split(order, change)
    pages = sp[np.concatenate([[0], change])]
    return {int(p): g.astype(np.int64) for p, g in zip(pages, groups)}


# --------------------------------------------------------------------------
# Layout construction
# --------------------------------------------------------------------------
def build_layout(method, edge_index, num_nodes, nodes_per_page, seed=0):
    """Return (node_order, page_id) as numpy arrays for ``method``.

    natural/random are computed here in numpy. bfs/metis defer to the
    torch-based builders in scripts/build_graph_layout.py.
    """
    if method == "natural":
        node_order = np.arange(num_nodes, dtype=np.int64)
        page_id = (np.arange(num_nodes, dtype=np.int64) // nodes_per_page).astype(
            np.int64
        )
        return node_order, page_id
    if method == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(num_nodes).astype(np.int64)
        page_id = np.zeros(num_nodes, dtype=np.int64)
        page_id[order] = np.arange(num_nodes, dtype=np.int64) // nodes_per_page
        return order, page_id

    # bfs / metis need the torch builders.
    import torch
    import build_graph_layout as bgl

    ei = torch.from_numpy(np.asarray(edge_index)).long()
    if method == "bfs":
        node_order, page_id, _ = bgl.build_bfs_layout(ei, num_nodes, nodes_per_page)
    elif method == "metis":
        node_order, page_id, _ = bgl.build_metis_layout(ei, num_nodes, nodes_per_page)
    else:
        raise ValueError(f"unknown layout method: {method}")
    return np.asarray(node_order).astype(np.int64), np.asarray(page_id).astype(np.int64)


# --------------------------------------------------------------------------
# Access patterns
# --------------------------------------------------------------------------
def neighbor_batches(train_idx, indptr, indices, batch_size, fanouts, seed):
    """Yield (nodes, num_seed) for one shuffled epoch over training seeds."""
    rng = np.random.default_rng(seed)
    seeds_all = np.asarray(train_idx, dtype=np.int64).copy()
    rng.shuffle(seeds_all)
    for i in range(0, len(seeds_all), batch_size):
        seeds = seeds_all[i : i + batch_size]
        nodes = sample_neighbors(seeds, indptr, indices, fanouts, rng)
        yield nodes, len(seeds)


def page_batches(groups, page_id, train_mask, pages_per_batch, seed):
    """Yield (nodes, num_train) for page-aligned batches over train pages."""
    train_idx = np.where(train_mask)[0]
    pages_with_train = np.unique(page_id[train_idx])
    rng = np.random.default_rng(seed)
    rng.shuffle(pages_with_train)
    for i in range(0, len(pages_with_train), pages_per_batch):
        chunk = pages_with_train[i : i + pages_per_batch]
        nodes = np.concatenate([groups[int(p)] for p in chunk])
        num_train = int(train_mask[nodes].sum())
        yield nodes, num_train


def run_access(reader, batch_iter, time_limit):
    """Drive real reads for one access pattern; return measured stats."""
    reader.reset_stats()
    signal_bytes = 0
    t0 = time.time()
    for nodes, num_signal in batch_iter:
        reader.read_nodes(nodes)
        signal_bytes += int(num_signal) * reader.FB
        if time.time() - t0 > time_limit:
            break
    dt = time.time() - t0
    st = reader.stats()
    st["seconds"] = dt
    st["signal_bytes"] = signal_bytes
    st["ra_signal"] = (reader.bytes_read / signal_bytes) if signal_bytes else 0.0
    st["throughput_mb_s"] = (st["bytes_read"] / 1e6 / dt) if dt > 0 else 0.0
    return st


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def compare(x, edge_index, train_mask, layouts, ssd_dir, opts):
    num_nodes, f = x.shape
    feature_bytes = f * 4
    nodes_per_page = max(1, PAGE_SIZE // feature_bytes)
    indptr, indices = build_csr(edge_index, num_nodes)
    train_idx = np.where(train_mask)[0]

    os.makedirs(ssd_dir, exist_ok=True)
    results = {}
    for method in layouts:
        print(f"\n=== layout: {method} ===")
        node_order, page_id = build_layout(
            method, edge_index, num_nodes, nodes_per_page, opts["seed"]
        )
        groups = page_node_groups(page_id)

        path = os.path.join(ssd_dir, f"{method}.bin")
        meta = write_feature_file(x, node_order, path, replicas=opts["replicas"])
        print(
            f"  store: {meta['file_bytes']/1e6:.1f} MB "
            f"({meta['replicas']}x{meta['copy_bytes']/1e6:.1f} MB), "
            f"{nodes_per_page} nodes/page"
        )

        if opts["drop_caches"]:
            print(f"  drop_caches: {'ok' if drop_caches() else 'unavailable'}")

        reader = SSDFeatureReader(
            path,
            node_order,
            meta,
            direct=opts["direct"],
            evict=opts["evict"],
        )

        nb = neighbor_batches(
            train_idx,
            indptr,
            indices,
            opts["batch_size"],
            opts["fanouts"],
            opts["seed"],
        )
        nb_stats = run_access(reader, nb, opts["time_limit"])
        print(
            f"  neighbor: RA_fetch={nb_stats['ra_physical']:.2f}x "
            f"RA_signal={nb_stats['ra_signal']:.2f}x "
            f"pages={nb_stats['pages_read']:,} ops={nb_stats['read_ops']:,} "
            f"direct={nb_stats['direct']}"
        )

        pb = page_batches(
            groups, page_id, train_mask, opts["pages_per_batch"], opts["seed"]
        )
        pb_stats = run_access(reader, pb, opts["time_limit"])
        print(
            f"  page:     RA_fetch={pb_stats['ra_physical']:.2f}x "
            f"RA_signal={pb_stats['ra_signal']:.2f}x "
            f"pages={pb_stats['pages_read']:,} ops={pb_stats['read_ops']:,}"
        )

        reader.close()
        results[method] = {"neighbor": nb_stats, "page": pb_stats}

    return results


def print_table(results):
    print("\n" + "=" * 72)
    print("Read amplification by layout (real device I/O)")
    print("=" * 72)
    print(
        f"{'layout':10s} {'nbr RA_fetch':>13s} {'nbr RA_signal':>14s} "
        f"{'page RA_fetch':>14s} {'page RA_signal':>15s}"
    )
    for method, r in results.items():
        print(
            f"{method:10s} {r['neighbor']['ra_physical']:>12.2f}x "
            f"{r['neighbor']['ra_signal']:>13.2f}x "
            f"{r['page']['ra_physical']:>13.2f}x "
            f"{r['page']['ra_signal']:>14.2f}x"
        )
    print(
        "\nThe neighbor RA_fetch column shows the value of locality "
        "(lower is better,\nnatural -> metis). The page RA_signal column "
        "shows the residual floor no\nlayout can close. Cross-check these "
        "against your own eBPF/iostat counts."
    )


# --------------------------------------------------------------------------
# Synthetic graph (smoke)
# --------------------------------------------------------------------------
def make_synthetic(num_nodes=6000, communities=100, deg=10, f=17, seed=0):
    """Community-structured graph where node id encodes community."""
    rng = np.random.default_rng(seed)
    per = num_nodes // communities
    comm = np.minimum(np.arange(num_nodes) // per, communities - 1)
    src, dst = [], []
    for u in range(num_nodes):
        c = comm[u]
        lo, hi = c * per, min((c + 1) * per, num_nodes)
        nbrs = rng.integers(lo, hi, size=deg)
        src.append(np.full(deg, u))
        dst.append(nbrs)
    edge_index = np.vstack([np.concatenate(src), np.concatenate(dst)])
    x = rng.standard_normal((num_nodes, f)).astype(np.float32)
    train_mask = rng.random(num_nodes) < 0.2
    return x, edge_index, train_mask


def default_opts(args):
    return {
        "seed": args.seed,
        "replicas": args.replicas,
        "direct": not args.no_direct,
        "evict": args.evict,
        "drop_caches": args.drop_caches,
        "batch_size": args.batch_size,
        "pages_per_batch": args.pages_per_batch,
        "fanouts": [int(v) for v in args.fanouts.split(",") if v],
        "time_limit": args.time,
    }


def main():
    ap = argparse.ArgumentParser(description="Force-SSD RA comparison")
    ap.add_argument("--dataset", default="dgraphfin")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--layouts", default="natural,bfs,metis")
    ap.add_argument("--ssd-dir", default="./gnn_ssd_store")
    ap.add_argument("--no-direct", action="store_true", help="disable O_DIRECT")
    ap.add_argument("--evict", action="store_true", help="fadvise DONTNEED")
    ap.add_argument("--drop-caches", action="store_true")
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--inflate-gb", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--pages-per-batch", type=int, default=32)
    ap.add_argument("--fanouts", default="10,5")
    ap.add_argument("--time", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        x, edge_index, train_mask = make_synthetic(
            num_nodes=30000, communities=500, deg=8
        )
        layouts = ["natural", "random"]
        # Small seed batches keep the neighbor working set local so the
        # layout contrast is unambiguous on the tiny synthetic graph.
        args.batch_size = min(args.batch_size, 128)
        args.pages_per_batch = min(args.pages_per_batch, 8)
    else:
        import torch  # noqa: F401
        from datasets import load_dataset

        data = load_dataset(args.dataset, args.data_dir)
        x = data.x.numpy()
        edge_index = data.edge_index.numpy()
        train_mask = data.train_mask.numpy()
        layouts = [m for m in args.layouts.split(",") if m]

    opts = default_opts(args)

    # Translate inflate-gb into a replica count.
    if args.inflate_gb > 0:
        copy_bytes = (
            (x.shape[0] * x.shape[1] * 4 + PAGE_SIZE - 1) // PAGE_SIZE
        ) * PAGE_SIZE
        need = int(args.inflate_gb * (1 << 30))
        opts["replicas"] = max(opts["replicas"], -(-need // max(1, copy_bytes)))
        print(f"inflate: {args.inflate_gb} GiB -> {opts['replicas']} replicas")

    results = compare(x, edge_index, train_mask, layouts, args.ssd_dir, opts)
    print_table(results)

    if args.out_json:
        meta = {
            "dataset": "synthetic-smoke" if args.smoke else args.dataset,
            "smoke": bool(args.smoke),
            "layouts": layouts,
            "direct_requested": not args.no_direct,
            "drop_caches": bool(args.drop_caches),
            "fanouts": opts["fanouts"],
            "batch_size": opts["batch_size"],
            "pages_per_batch": opts["pages_per_batch"],
            "replicas": opts["replicas"],
            "time_limit": opts["time_limit"],
        }
        payload = {"meta": meta, "results": results}
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.out_json}")

    if args.smoke:
        nat = results["natural"]["neighbor"]["ra_physical"]
        rnd = results["random"]["neighbor"]["ra_physical"]
        assert nat < rnd, f"expected natural<random neighbor RA, got {nat} vs {rnd}"
        print(f"\nsmoke OK: neighbor RA natural {nat:.2f}x < random {rnd:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
