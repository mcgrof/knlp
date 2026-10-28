#!/usr/bin/env python3
"""Run the first bounded weight-loading I/O matrix under the NVMe tracer.

The matrix is generated as a list of named runs, shuffled with a fixed
seed so that arm order is not confounded with time, and executed one at
a time. A run whose summary already exists is skipped, so the matrix can
be resumed. Each run calls traced_replay.py and leaves a run directory
with the raw device trace, the replay windows and the attributed summary.

Groups:
  layers   first/middle/last decoder layer of each local checkpoint under
           pread, mmap and direct arms, cold and warm page cache
  rows     single embedding rows of the small control model
  chunks   one layer re-read as contiguous chunks of several sizes and
           queue depths under direct I/O, cold
  awq      selected-neuron slices of the AWQ MLP tensors, clustered and
           scattered, cold, pread and direct
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", required=True, help="experiment root with models/, stageA/, runs/"
    )
    ap.add_argument("--tracer", required=True)
    ap.add_argument("--disk", required=True)
    ap.add_argument("--part-start-sectors", type=int, required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--groups", nargs="*", default=["layers", "rows", "chunks", "awq"])
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--models-dir", default=None, help="root of <org>__<name>/full checkouts"
    )
    ap.add_argument(
        "--pins-dir", default=None, help="hub pins with each repository's config"
    )
    ap.add_argument(
        "--intent-tracer", default=None, help="syscall_monitor, passed to the wrapper"
    )
    ap.add_argument(
        "--fault-tracer", default=None, help="mmap_readamp, passed to the wrapper"
    )
    args = ap.parse_args()
    root = Path(args.root)
    sel = load_jsonl(root / "stageA/selection.jsonl")
    py = sys.executable
    models_dir = Path(args.models_dir) if args.models_dir else root / "models"
    pins_dir = Path(args.pins_dir) if args.pins_dir else root / "stageA/pins"

    # the repositories present as whole checkpoints are the ones replayed
    local = {}
    for repo in sorted({s["repo"] for s in sel}):
        d = models_dir / repo.replace("/", "__") / "full"
        if d.is_dir():
            local[repo] = d
    if not local:
        sys.exit(f"no <org>__<name>/full checkpoint under {models_dir}")

    def quant_method(repo):
        p = pins_dir / (repo.replace("/", "__") + ".pin.json")
        if not p.exists():
            return None
        cfg = json.loads(p.read_text()).get("config.json") or {}
        return (cfg.get("quantization_config") or {}).get("quant_method")

    def layer_bytes(repo):
        return max(
            (
                s["nbytes"]
                for s in sel
                if s["repo"] == repo and s["unit"].startswith("layer")
            ),
            default=0,
        )

    dense = [r for r in local if quant_method(r) is None]
    awq_repos = [r for r in local if quant_method(r) == "awq"]

    def unit_file(repo, unit):
        files = sorted(
            {s["file"] for s in sel if s["repo"] == repo and s["unit"] == unit}
        )
        if len(files) != 1:
            return None
        return local[repo] / files[0]

    runs = []
    if "layers" in args.groups:
        for repo in sorted(local):
            units = sorted(
                {
                    s["unit"]
                    for s in sel
                    if s["repo"] == repo and s["unit"].startswith("layer")
                },
                key=lambda u: int(u[5:]),
            )
            for unit in units:
                f = unit_file(repo, unit)
                if f is None:
                    print(f"skip {repo} {unit}: spans shards", file=sys.stderr)
                    continue
                for arm in ("pread", "mmap", "direct"):
                    for cache in ("cold", "warm"):
                        rid = f"layers/{repo.split('/')[1]}/{unit}/{arm}-{cache}"
                        runs.append(
                            (
                                rid,
                                [
                                    "--file",
                                    str(f),
                                    "--selection",
                                    str(root / "stageA/selection.jsonl"),
                                    "--repo",
                                    repo,
                                    "--unit",
                                    unit,
                                    "--arm",
                                    arm,
                                    "--cache",
                                    cache,
                                    "--trials",
                                    str(args.trials),
                                    "--label",
                                    rid,
                                ],
                            )
                        )
    if "rows" in args.groups:
        # the checkpoint with the smallest embedding row
        row_bytes = {
            r: min(
                (
                    s["nbytes"]
                    for s in sel
                    if s["repo"] == r and s["unit"].startswith("embedding_row")
                ),
                default=None,
            )
            for r in local
        }
        row_bytes = {r: b for r, b in row_bytes.items() if b}
        repo = min(row_bytes, key=row_bytes.get) if row_bytes else None
        for unit in sorted(
            {
                s["unit"]
                for s in sel
                if repo and s["repo"] == repo and s["unit"].startswith("embedding_row")
            }
        ):
            f = unit_file(repo, unit)
            for arm in ("pread", "direct", "mmap"):
                for cache in ("cold", "warm"):
                    rid = f"rows/{unit}/{arm}-{cache}"
                    runs.append(
                        (
                            rid,
                            [
                                "--file",
                                str(f),
                                "--selection",
                                str(root / "stageA/selection.jsonl"),
                                "--repo",
                                repo,
                                "--unit",
                                unit,
                                "--arm",
                                arm,
                                "--cache",
                                cache,
                                "--trials",
                                str(args.trials),
                                "--label",
                                rid,
                            ],
                        )
                    )
    if "chunks" in args.groups and dense:
        # the largest unquantized layer is the bandwidth target
        repo = max(dense, key=layer_bytes)
        units = sorted(
            {
                s["unit"]
                for s in sel
                if s["repo"] == repo and s["unit"].startswith("layer")
            },
            key=lambda u: int(u[5:]),
        )
        unit = units[len(units) // 2]
        f = unit_file(repo, unit)
        for chunk in (4096, 16384, 65536, 131072, 262144, 1 << 20):
            for qd in (1, 4, 16, 32):
                rid = f"chunks/{unit}/c{chunk}-qd{qd}"
                runs.append(
                    (
                        rid,
                        [
                            "--file",
                            str(f),
                            "--selection",
                            str(root / "stageA/selection.jsonl"),
                            "--repo",
                            repo,
                            "--unit",
                            unit,
                            "--arm",
                            "direct",
                            "--cache",
                            "cold",
                            "--chunk",
                            str(chunk),
                            "--qd",
                            str(qd),
                            "--trials",
                            str(args.trials),
                            "--label",
                            rid,
                        ],
                    )
                )
    if "awq" in args.groups and awq_repos:
        awq_repo = awq_repos[0]
        mdir = local[awq_repo]
        units = sorted(
            {
                s["unit"]
                for s in sel
                if s["repo"] == awq_repo and s["unit"].startswith("layer")
            },
            key=lambda u: int(u[5:]),
        )
        layer = int(units[len(units) // 2][5:])
        for proj in ("gate_proj", "down_proj"):
            for n in (1, 8, 32, 128):
                for pattern in ("clustered", "scattered"):
                    rdir = root / "runs/awq" / f"{proj}-n{n}-{pattern}"
                    rdir.mkdir(parents=True, exist_ok=True)
                    rj = rdir / "ranges.json"
                    if not rj.exists() and not args.dry_run:
                        subprocess.run(
                            [
                                py,
                                str(HERE / "awq_slices.py"),
                                "--model-dir",
                                str(mdir),
                                "--layer",
                                str(layer),
                                "--proj",
                                proj,
                                "--neurons",
                                str(n),
                                "--pattern",
                                pattern,
                                "--out",
                                str(rj),
                            ],
                            check=True,
                        )
                    for arm in ("pread", "direct", "direct-merged"):
                        rid = f"awq/{proj}-n{n}-{pattern}/{arm}-cold"
                        rargs = [
                            "--file",
                            "@" + str(rj),
                            "--ranges",
                            "@" + str(rj),
                            "--arm",
                            arm.split("-")[0],
                            "--cache",
                            "cold",
                            "--trials",
                            str(max(3, args.trials - 2)),
                            "--label",
                            rid,
                        ]
                        if arm == "direct-merged":
                            rargs.append("--merge")
                        runs.append((rid, rargs))

    rng = random.Random(args.seed)
    rng.shuffle(runs)
    order_path = root / "runs/stageB_order.json"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.write_text(json.dumps([r for r, _ in runs], indent=1))
    print(f"{len(runs)} runs")
    for rid, rargs in runs:
        rd = root / "runs" / rid
        if (rd / "summary.json").exists():
            continue
        # late-bind AWQ range files
        if rargs[1].startswith("@") and not args.dry_run:
            doc = json.loads(Path(rargs[1][1:]).read_text())
            rargs = [
                "--file",
                doc["file"],
                "--ranges",
                rargs[3],
            ] + rargs[4:]
        cmd = [
            py,
            str(HERE / "traced_replay.py"),
            "--tracer",
            args.tracer,
            "--disk",
            args.disk,
            "--run-dir",
            str(rd),
            *(["--intent-tracer", args.intent_tracer] if args.intent_tracer else []),
            *(["--fault-tracer", args.fault_tracer] if args.fault_tracer else []),
            "--",
            *rargs,
            "--part-start-sectors",
            str(args.part_start_sectors),
        ]
        print(f"== {rid}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd)
        if rc.returncode != 0:
            print(f"FAILED {rid} rc={rc.returncode}", flush=True)


if __name__ == "__main__":
    main()
