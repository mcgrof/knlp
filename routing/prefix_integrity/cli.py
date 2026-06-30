# SPDX-License-Identifier: GPL-2.0
"""Command-line entry: `python -m routing.prefix_integrity.cli validate ...`.

Runs the selector-mode prefix-integrity suite against one cartridge and writes
result.json / result.md / block_survival.csv. Semantic drift (the GPU step) is
produced separately by semantic_drift.py and folded in via --semantic-json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from routing.prefix_integrity.harness import run_validate
from routing.prefix_integrity.report import render_markdown, write_reports


def _load_queries(path):
    if not path:
        return [{"id": f"q{i}", "query": f"synthetic query {i}"} for i in range(16)]
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"query": line}
            obj.setdefault("id", f"q{i}")
            obj.setdefault("query", obj.get("question", obj.get("text", obj["id"])))
            out.append(obj)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="prefix_integrity")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="grade an algorithm's prefix integrity")
    v.add_argument(
        "--cartridge",
        required=True,
        help="cartridge dir or .pt (id=path also accepted)",
    )
    v.add_argument(
        "--algorithm", required=True, help="builtin name or package.module:factory"
    )
    v.add_argument(
        "--algorithm-config",
        default=None,
        help="JSON file of kwargs for the adapter factory",
    )
    v.add_argument("--queries", default=None, help="jsonl of suffix/query requests")
    v.add_argument(
        "--budgets", default="16", help="comma list of K budgets, e.g. 8,16,32"
    )
    v.add_argument("--block-size", type=int, default=16)
    v.add_argument("--pins", default="A1R2", help="structural pins e.g. A1R2K13")
    v.add_argument(
        "--repeats", type=int, default=3, help="repeats per query for determinism check"
    )
    v.add_argument(
        "--semantic-json",
        default=None,
        help="JSON {kl, top1, repairable} from a GPU drift run",
    )
    v.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    if args.cmd != "validate":
        ap.error("unknown command")

    cart = args.cartridge.split("=", 1)[1] if "=" in args.cartridge else args.cartridge
    config = {}
    if args.algorithm_config:
        with open(args.algorithm_config) as f:
            config = json.load(f)
    queries = _load_queries(args.queries)
    semantic = None
    if args.semantic_json:
        with open(args.semantic_json) as f:
            semantic = json.load(f)

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    all_results = []
    for k in budgets:
        res = run_validate(
            cartridge=cart,
            adapter_spec=args.algorithm,
            queries=queries,
            budget_k=k,
            block_size=args.block_size,
            pins=args.pins,
            repeats=args.repeats,
            adapter_config=config,
            semantic=semantic,
        )
        out_dir = os.path.join(args.out, f"K{k}")
        paths = write_reports(res, out_dir)
        all_results.append({"budget_k": k, "result": res, "paths": paths})
        print(
            f"K={k:<4} status={res['status']:<5} "
            f"{res['classification']} danger={res['danger_score']}"
        )

    summary = os.path.join(args.out, "summary.json")
    os.makedirs(args.out, exist_ok=True)
    with open(summary, "w") as f:
        json.dump(
            [
                {
                    "budget_k": r["budget_k"],
                    "status": r["result"]["status"],
                    "classification": r["result"]["classification"],
                    "danger_score": r["result"]["danger_score"],
                }
                for r in all_results
            ],
            f,
            indent=2,
        )
    # Echo the first report so a human sees the verdict immediately.
    if all_results:
        print()
        print(render_markdown(all_results[0]["result"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
