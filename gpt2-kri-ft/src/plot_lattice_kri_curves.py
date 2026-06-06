"""Plot Lattice-KRI screen results: KL vs *actual* retained KV.

Reads the JSONL written by eval_lattice_kri.py and, for each
(model, dataset, seq_len) group, produces:

  1. a PNG of KL-to-full (decode region) against the actual retained
     fraction — one line per router, so routers are compared at the KV
     they really keep, not the nominal budget;
  2. a "best router at matched retained KV" table (markdown + CSV): at
     each retained-fraction target, every router's KL is linearly
     interpolated onto that target and the lowest wins.

The matched-KV table is the gate artifact: it answers "does any Lattice
variant beat KRI-Q+N at the same actual retained KV?" directly. A router
that only looks good because it quietly kept more cache is exposed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Routers we always want to see called out in the matched-KV table.
KEY_ROUTERS = ("kri_q", "kri_q_novelty", "lattice_kri_rel_orth", "lattice_kri_mmr")
MATCH_TARGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
X_KEY = "retained_frac_decode"
Y_KEY = "kl_decode"


def _interp(points, x):
    """Linear interpolation of y at x over (x_i, y_i) points sorted by x.
    Returns None if x is outside the router's measured range (no
    extrapolation — a router that never reaches that KV simply doesn't
    compete there)."""
    pts = sorted(points)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return min(y0, y1)
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def load_rows(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_key(r):
    return (r["model"], r.get("dataset", "?"), r["seq_len"])


def plot_group(key, rows, out_dir):
    model, dataset, L = key
    by_router = defaultdict(list)
    for r in rows:
        by_router[r["router"]].append((r[X_KEY], r[Y_KEY]))
    fig, ax = plt.subplots(figsize=(7, 5))
    for router in sorted(by_router):
        pts = sorted(by_router[router])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = "-o" if router.startswith("lattice") else "--s"
        lw = 2.4 if router.startswith("lattice") else 1.3
        ax.plot(xs, ys, style, label=router, linewidth=lw, markersize=4)
    ax.set_xlabel("actual retained KV fraction (decode region)")
    ax.set_ylabel("KL to full cache (decode)")
    ax.set_title(f"{model}  {dataset}  L={L}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    safe = f"{model}_{dataset}_L{L}".replace("/", "-").replace(":", "-")
    png = out_dir / f"kl_vs_retained_{safe}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return png


def matched_table(key, rows):
    """For each retained-KV target, interpolate every router's KL and rank."""
    by_router = defaultdict(list)
    for r in rows:
        by_router[r["router"]].append((r[X_KEY], r[Y_KEY]))
    out = []
    for target in MATCH_TARGETS:
        scored = []
        for router, pts in by_router.items():
            kl = _interp(pts, target)
            if kl is not None:
                scored.append((kl, router))
        if not scored:
            continue
        scored.sort()
        best_kl, best_router = scored[0]
        kqn = dict((r, _interp(p, target)) for r, p in by_router.items())
        out.append(
            {
                "model": key[0],
                "dataset": key[1],
                "seq_len": key[2],
                "retained_target": target,
                "best_router": best_router,
                "best_kl": round(best_kl, 5),
                "kri_q_novelty_kl": (
                    round(kqn["kri_q_novelty"], 5)
                    if kqn.get("kri_q_novelty") is not None
                    else None
                ),
                "lattice_rel_orth_kl": (
                    round(kqn["lattice_kri_rel_orth"], 5)
                    if kqn.get("lattice_kri_rel_orth") is not None
                    else None
                ),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="eval_lattice_kri JSONL")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.results)
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    pngs, table = [], []
    for key, grp in sorted(groups.items()):
        pngs.append(plot_group(key, grp, out_dir))
        table.extend(matched_table(key, grp))

    # matched-KV table -> CSV + markdown
    if table:
        csv_path = out_dir / "matched_kv_best_router.csv"
        with csv_path.open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        md = out_dir / "matched_kv_best_router.md"
        with md.open("w") as mf:
            mf.write("# Best router at matched actual retained KV\n\n")
            mf.write(
                "| model | dataset | L | retained | best | best KL | "
                "KRI-Q+N KL | lattice_rel_orth KL |\n"
            )
            mf.write("|---|---|---|---|---|---|---|---|\n")
            for t in table:
                mf.write(
                    f"| {t['model']} | {t['dataset']} | {t['seq_len']} | "
                    f"{t['retained_target']:.2f} | {t['best_router']} | "
                    f"{t['best_kl']} | {t['kri_q_novelty_kl']} | "
                    f"{t['lattice_rel_orth_kl']} |\n"
                )
        print(f"wrote {len(pngs)} PNGs + {csv_path.name} + {md.name} -> {out_dir}")
    else:
        print("no rows to tabulate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
