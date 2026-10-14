#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the public Google FDE encoder with isolated benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LEGACY = ROOT / "benchmarks/muvera/scripts"
sys.path.insert(0, str(LEGACY))
import _muvera_config as cf


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(argv, **kwargs):
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, cwd=ROOT, **kwargs)


def build(build_dir):
    run_command(
        [sys.executable, HERE / "build_google_fde.py", "--build-dir", build_dir]
    )
    return build_dir / "google_fde"


def test(build_dir):
    binary = build(build_dir)
    env = os.environ.copy()
    env["GOOGLE_FDE_BINARY"] = str(binary)
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            ROOT / "benchmarks/google_muvera/tests",
            "-v",
        ],
        env=env,
    )
    return binary


def config_key(config_path, cfg):
    return hashlib.sha256(
        json.dumps([str(config_path.resolve()), cfg], sort_keys=True).encode()
    ).hexdigest()


def select_run(args, cfg, fresh=False):
    key = config_key(args.config, cfg)
    pointer = args.build_dir / "runs" / f"{key}.json"
    if args.run_dir:
        path = args.run_dir.expanduser().resolve()
        if fresh and path.exists() and any(path.iterdir()):
            raise ValueError(f"refusing to overwrite nonempty run directory: {path}")
    elif fresh:
        base = Path(cfg.get("CONFIG_MUVERA_OUTPUT_BASE", "/data/results-archive"))
        base = base.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        prefix = "google-muvera-small-" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ-"
        )
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    elif pointer.exists():
        path = Path(json.loads(pointer.read_text())["run_dir"])
        if not path.is_dir():
            raise ValueError(f"run directory no longer exists: {path}")
    else:
        raise ValueError(
            "run setup first, or set GOOGLE_MUVERA_RUN_DIR to an existing run"
        )
    path.mkdir(parents=True, exist_ok=True)
    if fresh:
        (path / "requested_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(json.dumps({"run_dir": str(path)}) + "\n")
    elif (path / "requested_config.json").exists():
        stored = json.loads((path / "requested_config.json").read_text())
        if stored != cfg:
            raise ValueError("configuration differs from this run; create a new run")
    else:
        raise ValueError(
            f"not a Google MUVERA run (missing requested_config.json): {path}"
        )
    return path


def effective_config(run_dir, cfg):
    effective = dict(cfg, CONFIG_MUVERA_RUN_DIR=str(run_dir))
    path = run_dir / "effective.config"
    if any("\n" in str(value) or "\r" in str(value) for value in effective.values()):
        raise ValueError("configuration values must fit on one line")
    # The shared parser strips outer quotes without decoding JSON escapes.
    path.write_text(
        "".join(f'{key}="{value}"\n' for key, value in sorted(effective.items()))
    )
    return path


def stamp(run_dir, cfg):
    cf.stamp_environment(cfg, run_dir)
    # The output captures the exact scripts even for a local dirty checkout.
    sources = list(HERE.glob("*.py")) + list(LEGACY.glob("*.py"))
    (run_dir / "harness_sources.json").write_text(
        json.dumps({str(p.relative_to(ROOT)): sha256(p) for p in sources}, indent=2)
        + "\n"
    )


def setup(run_dir, cfg):
    stamp(run_dir, cfg)
    cache = cfg.get("CONFIG_GOOGLE_MUVERA_EMBEDDING_CACHE", "")
    if cache:
        source = Path(cache).expanduser().resolve()
        names = (
            ["embeddings.npz"]
            if (source / "embeddings.npz").is_file()
            else [
                "colbertv2_embeddings.pt",
                "bge_embeddings.pt",
                "scifact_raw.json",
            ]
        )
        if not (source / names[0]).is_file():
            raise ValueError(f"missing embeddings in {source}")
        if (source / "synthetic_fixture.json").is_file():
            names.append("synthetic_fixture.json")
        records = {}
        for name in names:
            src = source / name
            if not src.is_file():
                continue
            dst = run_dir / name
            shutil.copy2(src, dst)
            records[name] = {"source": str(src), "sha256": sha256(dst)}
        (run_dir / "input_cache.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"Copied exact input cache to {run_dir}")
    else:
        config = effective_config(run_dir, cfg)
        for script in (
            "fetch_scifact.py",
            "build_colbertv2_embeddings.py",
            "build_bge_embeddings.py",
        ):
            run_command([sys.executable, LEGACY / script, "--config", config])
        records = {
            name: {"sha256": sha256(run_dir / name)}
            for name in (
                "colbertv2_embeddings.pt",
                "bge_embeddings.pt",
                "scifact_raw.json",
            )
            if (run_dir / name).is_file()
        }
        (run_dir / "input_cache.json").write_text(json.dumps(records, indent=2) + "\n")


def retrieval(run_dir, cfg, binary):
    from evaluate import evaluate

    if (run_dir / "metrics.json").exists():
        raise ValueError(
            "retrieval results already exist; create a new run to avoid overwriting"
        )
    stamp(run_dir, cfg)
    return evaluate(run_dir, run_dir, cfg, binary)


def systems(run_dir, cfg):
    if not cf.kbool(cfg, "CONFIG_MUVERA_SYSTEMS_BENCH", True):
        print("Memory benchmark disabled by configuration")
        return
    if (run_dir / "systems_byte_floor.csv").exists():
        raise ValueError("systems results already exist; create a new run")
    run_command(
        [
            sys.executable,
            LEGACY / "run_systems_byte_bench.py",
            "--config",
            effective_config(run_dir, cfg),
        ]
    )


def report(run_dir):
    source = run_dir / "metrics.csv"
    if not source.is_file():
        raise ValueError("run retrieval before generating a report")
    with source.open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    text = [
        "# Google MUVERA small benchmark",
        "",
        "This run uses Google's fixed dimensional encoding (FDE) implementation.",
        "The knlp harness performs exact FDE search and Chamfer reranking.",
        "These results do not reproduce the paper's DiskANN latency measurements.",
        "",
        "Read metrics.json and provenance.json for the input hashes, dimensions,",
        "seeds, denominators and byte accounting. The packed representation files",
        "measure payload storage. FP16 and INT8 scores use reconstructed FP32",
        "documents; their timings do not measure native compressed search kernels.",
        "Original token vectors are still needed for reranking.",
        "",
    ]
    if (run_dir / "synthetic_fixture.json").exists():
        text += [
            "**Synthetic smoke test only. These are not SciFact quality results.**",
            "",
        ]
    columns = [
        "method",
        "stage",
        "format",
        "dimension",
        "seed",
        "record_bytes",
        "candidate_budget",
        "recall_at_10",
        "recall_at_100",
        "exact_top10_candidate_recall",
    ]
    text += [
        "## Retrieval measurements",
        "",
        "The table summarizes metrics.csv. That file also records every construction",
        "parameter, hit rate, candidate recall, denominator and timing. Empty cells",
        "mean the measurement does not apply to that row.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    text += [
        "| "
        + " | ".join(str(row.get(c, "")).replace("|", "\\|") for c in columns)
        + " |"
        for row in rows
    ]
    text += [
        "",
        "## Memory and storage",
        "",
        (
            "systems_byte_floor.csv contains the optional memory/cache microbenchmark."
            if (run_dir / "systems_byte_floor.csv").exists()
            else "The memory/cache microbenchmark was not run."
        ),
        "No cold NVMe measurement is produced by this target. A modeled 4 KiB",
        "indirection-unit cost is distinct from host-transfer bytes and media reads.",
        "",
        "## Next tests",
        "",
        "Repeat the same fixed input cache with base seeds 1, 1001 and 2001.",
        "Each repetition uses base seed plus its index; space base seeds by at least",
        "the largest repetition count so trials do not reuse blocks. Inspect candidate",
        "fidelity before interpreting relevance scores. Use length-bucket_fidelity.csv",
        "to compare exact-neighbor recovery by the retained token length of the",
        "exact best document; all buckets search the same full corpus. A single",
        "corpus does not establish a general document-length dimension bound.",
        "",
        "Select dimensions on development data, then evaluate held-out queries.",
        "Measure native packed search separately before making a speed or resident",
        "memory claim. Measure cold storage with host and media bytes separately.",
        "See benchmarks/google_muvera/README.md for controls and decision rules.",
    ]
    (run_dir / "report.md").write_text("\n".join(text) + "\n")
    print(f"Report: {run_dir / 'report.md'}")


def synthetic_fixture(run_dir):
    import numpy as np

    rng = np.random.default_rng(1349)
    docs = []
    for length in (2, 4, 8, 16, 32, 64, 96, 128) * 4:
        points = rng.normal(size=(length, 16)).astype(np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        docs.append(points)
    queries = [d[:2].copy() for d in docs[:8]]
    doc_ids = np.array([str(i) for i in range(len(docs))])
    query_ids = doc_ids[: len(queries)].copy()
    np.savez(
        run_dir / "embeddings.npz",
        doc_values=np.concatenate(docs),
        doc_offsets=np.array([0] + list(np.cumsum([len(d) for d in docs]))),
        query_values=np.concatenate(queries),
        query_offsets=np.arange(0, 2 * (len(queries) + 1), 2),
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrel_query_ids=query_ids,
        qrel_doc_ids=query_ids,
        qrel_scores=np.ones(len(queries), dtype=np.int32),
        model=np.array("synthetic-unit-vectors"),
    )
    (run_dir / "synthetic_fixture.json").write_text(
        json.dumps(
            {"seed": 1349, "purpose": "pipeline validation, not research evidence"}
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "run",
            "build",
            "test",
            "setup",
            "retrieval",
            "systems",
            "report",
            "smoke",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".cache/google-muvera")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    args.build_dir = args.build_dir.expanduser().resolve()
    if args.stage == "build":
        build(args.build_dir)
        return
    if args.stage == "test":
        test(args.build_dir)
        return
    if not args.config.is_file():
        parser.error("missing configuration; run make defconfig-google-muvera-small")
    cfg = cf.parse_kconfig(args.config)
    if not cf.kbool(cfg, "CONFIG_GOOGLE_MUVERA_SMALL"):
        parser.error("configuration does not select Google MUVERA")
    if args.stage == "smoke":
        cfg = dict(
            cfg,
            CONFIG_GOOGLE_MUVERA_FDE_CONFIGS="2:2:8 3:3:8",
            CONFIG_GOOGLE_MUVERA_SEEDS="1 101",
            CONFIG_MUVERA_SYSTEMS_BENCH="n",
        )
    if args.stage in ("run", "smoke"):
        binary = test(args.build_dir)
    elif args.stage == "retrieval":
        binary = build(args.build_dir)
    run_dir = select_run(args, cfg, fresh=args.stage in ("run", "setup", "smoke"))
    print(f"Run directory: {run_dir}", flush=True)
    if args.stage == "smoke":
        stamp(run_dir, cfg)
        synthetic_fixture(run_dir)
        retrieval(run_dir, cfg, binary)
        report(run_dir)
    elif args.stage == "run":
        setup(run_dir, cfg)
        retrieval(run_dir, cfg, binary)
        systems(run_dir, cfg)
        report(run_dir)
    elif args.stage == "setup":
        setup(run_dir, cfg)
    elif args.stage == "retrieval":
        retrieval(run_dir, cfg, binary)
    elif args.stage == "systems":
        systems(run_dir, cfg)
    elif args.stage == "report":
        report(run_dir)


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        sys.exit(f"google-muvera-small: {error}")
