#!/usr/bin/env python3
"""Score the full model and every expert pool on every domain (GPU).

A request sent to the wrong pool runs on experts calibrated for another
domain. With several domains, which wrong picks hurt depends on the pair,
so this script scores each pool on each domain's test requests, and the
full model on all of them, giving a pool-by-domain matrix of per-item
results. A domain may have no pool of its own (a domain that no pool was
built for); its row then shows what any pick costs.

Items are multiple-choice and scored as in score_screen.py: the summed
log probability of each choice given the context; the primary metric
picks the highest log probability per choice character, the secondary
the raw sum. Every model runs the fused MoE kernel with the tuned tile
table. The model is reloaded before each pool is applied, since applying
a pool frees the other experts.

The summary pairs each cell against the full model on the same items.

Usage:
    pool_matrix.py MODEL_DIR OUT_DIR --configs TUNED.json \\
        --domain arc TEST.json --domain health TEST.json ... \\
        --pool arc POOL.json --pool health POOL.json ...
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import apply_pool, load_model, runtime_identity, write_json  # noqa
from pool_quality import paired  # noqa
from score_screen import score_item  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--domain", nargs=2, action="append", required=True)
    ap.add_argument("--pool", nargs=2, action="append", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from fused_moe import ConfigTable, install_fused

    domains = {}
    for name, path in args.domain:
        with open(path) as f:
            domains[name] = json.load(f)["items"]
    pools = {}
    for name, path in args.pool:
        with open(path) as f:
            pools[name] = json.load(f)
    res = {"runtime": runtime_identity(), "args": vars(args), "models": {}}
    out_path = os.path.join(args.out_dir, "pool-matrix.json")

    for name in ["full"] + list(pools):
        model, tok = load_model(args.model_dir)
        if name != "full":
            apply_pool(model, pools[name])
        install_fused(model, ConfigTable(args.configs))
        gc.collect()
        torch.cuda.empty_cache()
        cells = {}
        for dom, items in domains.items():
            t0 = time.time()
            rows = [score_item(model, tok, it) for it in items]
            torch.cuda.synchronize()
            el = time.time() - t0
            acc = sum(r["correct_per_char"] for r in rows)
            print(
                f"{name} on {dom}: {acc}/{len(rows)} per-char, {el:.0f} s", flush=True
            )
            cells[dom] = {"rows": rows, "seconds": el}
        res["models"][name] = cells
        write_json(out_path, res)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    full = res["models"]["full"]
    summ = {"full": {}}
    for dom in domains:
        rows = full[dom]["rows"]
        summ["full"][dom] = {
            "per_char": sum(r["correct_per_char"] for r in rows),
            "raw": sum(r["correct_raw"] for r in rows),
            "n": len(rows),
        }
    for name in pools:
        summ[name] = {}
        for dom in domains:
            a, b = full[dom]["rows"], res["models"][name][dom]["rows"]
            summ[name][dom] = {
                "per_char": paired(a, b, "correct_per_char"),
                "raw": paired(a, b, "correct_raw"),
            }
    res["summary"] = summ
    write_json(out_path, res)
    for name in summ:
        line = []
        for dom in domains:
            c = summ[name][dom]
            if name == "full":
                line.append(f"{dom} {c['per_char']}")
            else:
                p = c["per_char"]
                line.append(f"{dom} {p['correct']} ({-p['net_losses']:+d})")
        print(f"{name:>12}: " + ", ".join(line))


if __name__ == "__main__":
    main()
