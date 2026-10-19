# SPDX-License-Identifier: MIT
"""Build external routing-replay inputs from public data and owned checkpoints."""

import argparse
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from research.layerrope_beta.time_budget.protocol import freeze
from research.layerrope_beta.time_budget.run import job
from research.layerrope_beta.time_budget.state import json_hash
from research.layerrope_beta.time_budget.train import source_identity
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.data import DATA_REV, TOKEN_REV
from research.tinystories_depth.train import state_hash

TARGETS = (0.0, 200.0, 787.0)


def forward_source(source):
    """Keep the model/evaluation files and the complete vendored Python bundle."""
    required = {
        "research/layerrope_beta/model.py",
        "research/tinystories_depth/model.py",
        "research/tinystories_depth/float_quant.py",
        "research/tinystories_depth/kv_cache.py",
    }
    if not required <= source.keys():
        raise ValueError("Missing recorded forward source")
    return {
        k: v
        for k, v in source.items()
        if k in required or k.startswith("research/tinystories_depth/upstream/")
    }


def training_data(dataset, out):
    """Adapt public TinyStories arrays without reading its separate final array."""
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    metadata = read(dataset / "DATA.json")
    if (
        metadata["dataset_revision"] != DATA_REV
        or metadata["tokenizer_revision"] != TOKEN_REV
    ):
        raise ValueError("Pinned public data revisions differ")
    for name, count in (("train.npy", 40_000_000), ("curve.npy", 32769)):
        if sha(dataset / name) != metadata["files"][name]:
            raise ValueError("Public data digest differs")
        value = np.load(dataset / name, mmap_mode="r")
        if value.dtype != np.uint16 or value.shape != (count,):
            raise ValueError("Public data geometry differs")
    out.mkdir(parents=True, exist_ok=False)
    for src, dest in (("train.npy", "train.npy"), ("curve.npy", "validation.npy")):
        shutil.copyfile(dataset / src, out / dest)
    manifest = dict(
        schema="equal-time-training-capability.v1",
        files={n: sha(out / n) for n in ("train.npy", "validation.npy")},
        dataset_revision=DATA_REV,
        tokenizer_revision=TOKEN_REV,
        public_data_manifest_sha256=sha(dataset / "DATA.json"),
        scope="Regenerated diagnostic data; reuses public curve partition, not the separately excluded equal-time holdout",
    )
    write(out / "TRAINING.json", manifest)
    return manifest


def jobs(out, seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Use a nonnegative integer seed")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for row in ("M", "ML"):
        write(out / (row + ".json"), job(row, seed, 0.001, "qualification"))


def manifest(data, results, out, diagnostic_seed=77790):
    """Generate receipts without recomputing or replacing recorded validation CE."""
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if len(results) != 2:
        raise ValueError("Exactly two M/ML training results are required")
    training = read(data / "TRAINING.json")
    if set(training["files"]) != {"train.npy", "validation.npy"}:
        raise ValueError("Training input must contain train and validation only")
    for name, digest in training["files"].items():
        if sha(data / name) != digest:
            raise ValueError("Training input digest differs")
    validation = np.load(data / "validation.npy", mmap_mode="r")
    if validation.dtype != np.uint16 or validation.shape != (32769,):
        raise ValueError("Validation geometry differs")
    current = source_identity()
    required = forward_source(current)
    records, provenance, rows, seeds = [], [], set(), set()
    for path in results:
        path = Path(path).resolve()
        result = read(path)
        identity = result["identity"]
        recipe = identity["job"]
        row = recipe["arm"]
        if row not in ("M", "ML") or row in rows or result["status"] != "complete":
            raise ValueError("Need one complete result for each of M and ML")
        if recipe != job(row, recipe["seed"], 0.001, "qualification") or identity[
            "protocol_sha256"
        ] != json_hash(freeze()):
            raise ValueError("Qualification model, rate or protocol differs")
        if identity["data_sha256"] != sha(data / "TRAINING.json"):
            raise ValueError("Result belongs to different training inputs")
        if forward_source(identity["source"]) != required:
            raise ValueError("Recorded forward source differs from current files")
        rows.add(row)
        seeds.add(recipe["seed"])
        provenance.append(dict(path=str(path), sha256=sha(path), job=recipe))
        for target in TARGETS:
            landmarks = [
                v
                for v in result["history"]["landmarks"]
                if v["target_seconds"] == target
            ]
            if len(landmarks) != 1 or not landmarks[0]["timing_valid"]:
                raise ValueError("Missing, duplicate or invalid clock landmark")
            point = landmarks[0]
            receipt = point["checkpoint"]
            checkpoint = Path(receipt["path"])
            if not checkpoint.is_file():
                checkpoint = path.parent / checkpoint.name
            checkpoint = checkpoint.resolve()
            if sha(checkpoint) != receipt["sha256"]:
                raise ValueError("Checkpoint file digest differs")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (
                state["identity"] != identity
                or state_hash(state["model"]) != receipt["model_sha256"]
                or state["step"] != receipt["step"]
                or state["training_seconds"] != receipt["training_seconds"]
                or point["step"] != receipt["step"]
                or point["training_seconds"] != receipt["training_seconds"]
            ):
                raise ValueError("Checkpoint payload differs from its result")
            expected = point["validation"]
            if (
                len(expected["window_ce"]) != 128
                or not all(math.isfinite(v) for v in expected["window_ce"])
                or not math.isfinite(expected["ce"])
            ):
                raise ValueError("Recorded validation coverage or finiteness differs")
            records.append(
                dict(
                    path=str(checkpoint),
                    sha256=receipt["sha256"],
                    model_sha256=receipt["model_sha256"],
                    identity_sha256=json_hash(identity),
                    row=row,
                    seed=recipe["seed"],
                    step=receipt["step"],
                    training_seconds=receipt["training_seconds"],
                    target_seconds=target,
                    validation=expected,
                )
            )
            del state
    if len(seeds) != 1:
        raise ValueError("The two qualification rows must use the same seed")
    card = dict(
        schema="finite-routing-audit-inputs.v1",
        source=current,
        executed_forward_source=required,
        validation=dict(
            path=str(data / "validation.npy"), sha256=sha(data / "validation.npy")
        ),
        batch_size=32,
        window=256,
        diagnostic_seed=diagnostic_seed,
        checkpoints=sorted(records, key=lambda v: (v["row"], v["target_seconds"])),
        provenance=provenance,
        baseline_policy="Preserve original per-window losses; exact replay may reject a different runtime. Regenerated training is a new experiment, not reconstruction of original checkpoint bytes.",
    )
    write(out, card)
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    data = actions.add_parser("training-data")
    data.add_argument("--dataset", type=Path, required=True)
    data.add_argument("--out", type=Path, required=True)
    init = actions.add_parser("jobs")
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--out", type=Path, required=True)
    audit = actions.add_parser("manifest")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--result", type=Path, action="append", required=True)
    audit.add_argument("--diagnostic-seed", type=int, default=77790)
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "training-data":
        training_data(args.dataset, args.out)
    elif args.action == "jobs":
        jobs(args.out, args.seed)
    else:
        manifest(args.data, args.result, args.out, args.diagnostic_seed)


if __name__ == "__main__":
    main()
