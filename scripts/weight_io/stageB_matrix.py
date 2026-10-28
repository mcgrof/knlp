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
    args = ap.parse_args()
    root = Path(args.root)
    sel = load_jsonl(root / "stageA/selection.jsonl")
    py = sys.executable

    local = {
        "Qwen/Qwen2.5-7B-Instruct": root / "models/Qwen__Qwen2.5-7B-Instruct/full",
        "Qwen/Qwen2.5-7B-Instruct-AWQ": root
        / "models/Qwen__Qwen2.5-7B-Instruct-AWQ/full",
        "Qwen/Qwen3-0.6B": root / "models/Qwen__Qwen3-0.6B/full",
    }

    def unit_file(repo, unit):
        files = sorted(
            {s["file"] for s in sel if s["repo"] == repo and s["unit"] == unit}
        )
        if len(files) != 1:
            return None
        return local[repo] / files[0]

    runs = []
    if "layers" in args.groups:
        for repo in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-AWQ"):
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
        repo = "Qwen/Qwen3-0.6B"
        for unit in sorted(
            {
                s["unit"]
                for s in sel
                if s["repo"] == repo and s["unit"].startswith("embedding_row")
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
    if "chunks" in args.groups:
        repo = "Qwen/Qwen2.5-7B-Instruct"
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
    if "awq" in args.groups:
        mdir = local["Qwen/Qwen2.5-7B-Instruct-AWQ"]
        units = sorted(
            {
                s["unit"]
                for s in sel
                if s["repo"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
                and s["unit"].startswith("layer")
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
                    for arm in ("pread", "direct"):
                        rid = f"awq/{proj}-n{n}-{pattern}/{arm}-cold"
                        runs.append(
                            (
                                rid,
                                [
                                    "--file",
                                    "@" + str(rj),
                                    "--ranges",
                                    "@" + str(rj),
                                    "--arm",
                                    arm,
                                    "--cache",
                                    "cold",
                                    "--trials",
                                    str(max(3, args.trials - 2)),
                                    "--label",
                                    rid,
                                ],
                            )
                        )

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
