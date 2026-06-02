"""Re-run only the B_min bisection + gamma fit on an existing
eval_rows.jsonl produced by eval_l2m_scaling.py.

Useful when a refactor of the analysis code lands and we don't want
to re-run the multi-hour sweep that produced the raw rows.

Usage:
  python -m src.analyze_l2m_gamma --input results/l2m_full/eval_rows.jsonl \
      --output_dir results/l2m_full

Writes:
  bmin_table.csv
  gamma_fit.csv
  summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_l2m_scaling import bisect_bmin, fit_gamma  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to eval_rows.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epsilon", default="0.01,0.03,0.1,0.3")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(l) for l in open(args.input)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    epsilons = [float(x) for x in args.epsilon.split(",")]
    bmin_rows = bisect_bmin(rows, epsilons)
    bmin_csv = out / "bmin_table.csv"
    with bmin_csv.open("w", newline="") as fh:
        if bmin_rows:
            w = csv.DictWriter(fh, fieldnames=list(bmin_rows[0].keys()))
            w.writeheader()
            for r in bmin_rows:
                w.writerow(r)

    gamma_rows = fit_gamma(bmin_rows)
    gamma_csv = out / "gamma_fit.csv"
    with gamma_csv.open("w", newline="") as fh:
        if gamma_rows:
            w = csv.DictWriter(fh, fieldnames=list(gamma_rows[0].keys()))
            w.writeheader()
            for r in gamma_rows:
                w.writerow(r)

    # Quick summary
    print(f"\n{len(rows)} eval rows  ->  {len(bmin_rows)} B_min rows  ->  {len(gamma_rows)} gamma fits\n")
    print("=== gamma fits (achieved B_min, fit across L) ===")
    for r in sorted(gamma_rows, key=lambda r: (r["model"], r["policy"], r["epsilon"])):
        g = r["gamma"]
        r2 = r["r2"]
        g_s = f"{g:+.3f}" if g is not None else "  n/a"
        r2_s = f"{r2:.3f}" if r2 is not None else "  n/a"
        print(
            f"  {r['model'][:38]:38s} policy={r['policy']:14s} "
            f"eps={r['epsilon']:.2f}  n={r['n_points']}  gamma={g_s}  r2={r2_s}  {r['note']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
