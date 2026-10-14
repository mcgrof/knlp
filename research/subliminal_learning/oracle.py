# SPDX-License-Identifier: MIT
"""Run the unmodified reference MNIST script and keep its per-model results.

upstream/run_mnist_experiment.py.orig is a byte-for-byte copy of
scripts/run_mnist_experiment.py from github.com/MinhxLe/subliminal-learning
at commit db04f4150edf940559b5f3147f65d808e9313efd (MIT license, see
upstream/LICENSE); the .orig suffix keeps code formatters from
rewriting it. This wrapper checks its git blob hash, executes it as
__main__ with a non-interactive matplotlib backend, and reads the
per-model accuracies and the cross-model permutation out of the finished
module namespace. It changes nothing the script computes; in particular
it does not reseed or touch any random number generator.

The permutation audit matters: the reference forms its different-
initialization controls with randperm(100), whose fixed points pair a
teacher with a student that shares its initialization after all.
"""

import argparse
import csv
import hashlib
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "upstream", "run_mnist_experiment.py.orig")
PINNED_BLOB = "3d250567780d42f05aa3de54f69bbdcc3b1e14d9"
PINNED_COMMIT = "db04f4150edf940559b5f3147f65d808e9313efd"

COLUMNS = {
    "ref_acc": "reference",
    "teach_acc": "teacher",
    "acc_sg": "student_aux",
    "acc_sa": "student_all",
    "acc_xg": "cross_model_aux",
    "acc_xa": "cross_model_all",
}


def git_blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for the oracle receipt")
    a = ap.parse_args()
    blob = git_blob_sha1(SCRIPT)
    if blob != PINNED_BLOB:
        raise SystemExit(f"vendored script blob {blob} != pinned {PINNED_BLOB}")
    os.makedirs(a.out, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    import torch

    t0 = time.time()
    g = runpy.run_path(SCRIPT, run_name="__main__")
    elapsed = time.time() - t0
    perm = g["perm"].tolist()
    fixed = [i for i, p in enumerate(perm) if i == p]
    rows = []
    for m in range(g["N_MODELS"]):
        rows.append({col: g[var][m] for var, col in COLUMNS.items()} | {"model": m})
    with open(os.path.join(a.out, "oracle_models.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", *COLUMNS.values()])
        w.writeheader()
        w.writerows(rows)
    means = {col: sum(r[col] for r in rows) / len(rows) for col in COLUMNS.values()}
    receipt = {
        "source": "github.com/MinhxLe/subliminal-learning",
        "commit": PINNED_COMMIT,
        "script_blob": blob,
        "n_models": g["N_MODELS"],
        "device": g["DEVICE"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "elapsed_s": elapsed,
        "means": means,
        "cross_model_perm": perm,
        "cross_model_fixed_points": fixed,
        "argv": sys.argv,
    }
    with open(os.path.join(a.out, "oracle_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    print(
        json.dumps(
            {k: receipt[k] for k in ("means", "cross_model_fixed_points", "elapsed_s")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
