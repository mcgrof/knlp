#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Config-driven GNN runner: the entry point ``make`` calls for MODEL_GNN.

Reads the generated config.py, prepares dependencies and the dataset,
builds the page layout if needed, then dispatches to the in-RAM
benchmark (benchmark.py) or, when GNN_FRAUD_FORCE_SSD is set, the
real-I/O read-amplification comparison (benchmark_ssd.py). This is what
makes ``make defconfig-gnn-dgraphfin && make`` a push-button run.

    python gnn/run_gnn.py            # honor .config
    python gnn/run_gnn.py --dry-run  # print the plan, run nothing heavy

The runner itself stays free of torch: the heavy work happens in
subprocesses, so --dry-run can be validated anywhere.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)  # generated config.py lives at the repo root
sys.path.insert(0, os.path.join(HERE, "scripts"))  # setup_gnn_deps


def load_cfg():
    try:
        from config import config as cfg

        return cfg
    except Exception as exc:
        print(f"ERROR: could not import generated config.py ({exc}).")
        print("Run: make defconfig-gnn-dgraphfin   (then make)")
        sys.exit(1)


def cfgget(cfg, key, default):
    val = getattr(cfg, key, default)
    return default if val is None else val


def abspath_in_repo(p):
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(REPO, p))


def runcmd(cmd, cwd, dry):
    printable = " ".join(str(c) for c in cmd)
    print(f"RUN ({cwd}): {printable}")
    if dry:
        return 0
    return subprocess.call([str(c) for c in cmd], cwd=cwd)


def ensure_dataset(dataset, data_dir, auto_fetch, dry):
    if dataset != "dgraphfin":
        return  # other datasets self-fetch in their loaders
    npz = os.path.join(data_dir, "dgraphfin.npz")
    if os.path.exists(npz):
        print(f"dataset present: {npz}")
        return
    if not auto_fetch:
        print(f"dataset missing and auto-fetch disabled: {npz}")
        return
    print(f"PLAN: fetch {npz} from HuggingFace")
    if dry:
        return
    sys.path.insert(0, HERE)
    from datasets import fetch_dgraphfin

    fetch_dgraphfin(data_dir)


def ensure_layout(dataset, data_dir, layout_path, dry):
    if os.path.exists(layout_path):
        print(f"layout present: {layout_path}")
        return
    cmd = [
        sys.executable,
        os.path.join("scripts", "build_graph_layout.py"),
        "--dataset",
        dataset,
        "--data-dir",
        data_dir,
        "--method",
        "metis",
        "--output",
        layout_path,
    ]
    runcmd(cmd, cwd=HERE, dry=dry)


def build_inram_cmd(cfg, dataset, data_dir, layout_path):
    method = cfgget(cfg, "GNN_FRAUD_METHOD", "both")
    cmd = [
        sys.executable,
        "benchmark.py",
        "--time",
        cfgget(cfg, "GNN_FRAUD_TIME_LIMIT", 3600),
        "--hidden-channels",
        cfgget(cfg, "GNN_FRAUD_HIDDEN_CHANNELS", 128),
        "--lr",
        cfgget(cfg, "GNN_FRAUD_LEARNING_RATE", "0.003"),
        "--weight-decay-baseline",
        cfgget(cfg, "GNN_FRAUD_WEIGHT_DECAY_BASELINE", "5e-7"),
        "--weight-decay-pageaware",
        cfgget(cfg, "GNN_FRAUD_WEIGHT_DECAY_PAGEAWARE", "5e-7"),
        "--batch-size",
        cfgget(cfg, "GNN_FRAUD_BATCH_SIZE", 1024),
        "--pages-per-batch",
        cfgget(cfg, "GNN_FRAUD_PAGES_PER_BATCH", 32),
        "--boundary-budget",
        cfgget(cfg, "GNN_FRAUD_BOUNDARY_BUDGET", "0.0"),
        "--layout",
        layout_path,
        "--data-dir",
        data_dir,
    ]
    nn = str(cfgget(cfg, "GNN_FRAUD_NUM_NEIGHBORS", "10,5")).replace(" ", "")
    cmd += ["--num-neighbors", *nn.split(",")]
    if not cfgget(cfg, "GNN_FRAUD_USE_WANDB", False):
        cmd.append("--no-wandb")
    if method == "neighborloader":
        cmd.append("--only-baseline")
    elif method == "pageaware":
        cmd.append("--only-pageaware")
    return cmd


def build_ssd_cmd(cfg, dataset, data_dir):
    cmd = [
        sys.executable,
        "benchmark_ssd.py",
        "--dataset",
        dataset,
        "--data-dir",
        data_dir,
        "--layouts",
        cfgget(cfg, "GNN_FRAUD_SSD_LAYOUTS", "natural,bfs,metis"),
        "--ssd-dir",
        cfgget(cfg, "GNN_FRAUD_SSD_DIR", "./gnn_ssd_store"),
        "--time",
        cfgget(cfg, "GNN_FRAUD_TIME_LIMIT", 3600),
        "--batch-size",
        cfgget(cfg, "GNN_FRAUD_BATCH_SIZE", 1024),
        "--pages-per-batch",
        cfgget(cfg, "GNN_FRAUD_PAGES_PER_BATCH", 32),
        "--inflate-gb",
        cfgget(cfg, "GNN_FRAUD_SSD_INFLATE_GB", 0),
        "--fanouts",
        str(cfgget(cfg, "GNN_FRAUD_NUM_NEIGHBORS", "10,5")).replace(" ", ""),
        "--out-json",
        abspath_in_repo("results/gnn/force_ssd_ra.json"),
    ]
    if not cfgget(cfg, "GNN_FRAUD_SSD_DIRECT", True):
        cmd.append("--no-direct")
    if cfgget(cfg, "GNN_FRAUD_SSD_DROP_CACHES", False):
        cmd.append("--drop-caches")
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Config-driven GNN runner")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and run nothing heavy",
    )
    args = ap.parse_args()

    cfg = load_cfg()
    if not cfgget(cfg, "MODEL_GNN", False):
        print("config.py does not select the GNN model (MODEL_GNN unset).")
        print("Run: make defconfig-gnn-dgraphfin")
        return 1

    dataset = cfgget(cfg, "GNN_FRAUD_DATASET", "dgraphfin")
    data_dir = abspath_in_repo(cfgget(cfg, "GNN_FRAUD_DATA_DIR", "."))
    layout_path = abspath_in_repo(
        cfgget(cfg, "GNN_FRAUD_LAYOUT_PATH", "layout_metis_bfs.npz")
    )
    auto_fetch = bool(cfgget(cfg, "GNN_FRAUD_AUTO_FETCH", True))
    auto_deps = bool(cfgget(cfg, "GNN_FRAUD_AUTO_DEPS", True))
    force_ssd = bool(cfgget(cfg, "GNN_FRAUD_FORCE_SSD", False))

    os.environ["GNN_AUTO_FETCH"] = "1" if auto_fetch else "0"

    print(
        f"GNN runner: dataset={dataset} data_dir={data_dir} "
        f"force_ssd={force_ssd} auto_deps={auto_deps} auto_fetch={auto_fetch}"
    )

    if auto_deps:
        import setup_gnn_deps

        ok = setup_gnn_deps.ensure_all(install=not args.dry_run)
        if not ok and not args.dry_run:
            print("Dependencies incomplete; aborting (see messages above).")
            return 1

    ensure_dataset(dataset, data_dir, auto_fetch, args.dry_run)

    if force_ssd:
        cmd = build_ssd_cmd(cfg, dataset, data_dir)
        return runcmd(cmd, cwd=HERE, dry=args.dry_run)

    ensure_layout(dataset, data_dir, layout_path, args.dry_run)
    cmd = build_inram_cmd(cfg, dataset, data_dir, layout_path)
    return runcmd(cmd, cwd=HERE, dry=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
