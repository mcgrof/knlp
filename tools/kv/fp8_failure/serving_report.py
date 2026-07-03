#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Aggregate Phase-10 serving-confirmation cells into the atlas table.

Reads the per-(model,cell) JSONs written by serving_confirm.py and, for each
model, computes vs the native cell:
  - ppl and delta-ppl
  - top1_agreement: fraction of holdout positions whose argmax token matches
    the native cell's argmax (1.0 = identical greedy next-token everywhere)
  - greedy_divergence_pos: mean index of the first token where the greedy
    continuation departs from native (higher = closer to native; capped at len)

The headline the atlas predicts: on fragile-key models (Qwen, Phi) the
symmetric k8v8 cell blows up ppl and drops top1-agreement, while the asym
k16v8 cell stays at/near native; a biasless control (Llama) tolerates both.
"""

import argparse
import glob
import json
import os


def top1_agreement(ref, cur):
    n = m = 0
    for rd, cd in zip(ref, cur):
        for a, b in zip(rd, cd):
            n += 1
            m += int(a == b)
    return (m / n) if n else float("nan")


def greedy_div(ref, cur):
    pos = []
    for rd, cd in zip(ref, cur):
        d = len(rd)
        for i, (a, b) in enumerate(zip(rd, cd)):
            if a != b:
                d = i
                break
        pos.append(d)
    return sum(pos) / len(pos) if pos else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = {}
    for fp in glob.glob(os.path.join(args.in_dir, "*.json")):
        d = json.load(open(fp))
        cells.setdefault(d["model"], {})[d["cell"]] = d

    report = {}
    for model, cs in sorted(cells.items()):
        nat = cs.get("native")
        row = {}
        for cell, d in cs.items():
            entry = {"ppl": d["ppl"], "n_tokens": d.get("n_tokens")}
            if nat is not None and cell != "native":
                entry["delta_ppl"] = d["ppl"] - nat["ppl"]
                entry["top1_agreement"] = top1_agreement(
                    nat["per_position_top1"], d["per_position_top1"]
                )
                entry["greedy_divergence_pos"] = greedy_div(
                    nat["greedy_tokens"], d["greedy_tokens"]
                )
            row[cell] = entry
        report[model] = row

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # pretty table
    print(f"\n{'model':<16}{'cell':<9}{'ppl':>10}{'dppl':>10}{'top1agr':>9}{'gdiv':>7}")
    print("-" * 61)
    for model, row in report.items():
        for cell in ("native", "k8v8", "k16v8"):
            if cell not in row:
                continue
            e = row[cell]
            dppl = f"{e.get('delta_ppl', 0):+.2f}" if "delta_ppl" in e else "  -"
            t1 = f"{e.get('top1_agreement'):.3f}" if "top1_agreement" in e else "  -"
            gd = f"{e.get('greedy_divergence_pos'):.0f}" if "greedy_divergence_pos" in e else " -"
            print(f"{model:<16}{cell:<9}{e['ppl']:>10.3f}{dppl:>10}{t1:>9}{gd:>7}")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
