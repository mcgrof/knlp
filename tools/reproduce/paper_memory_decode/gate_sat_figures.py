#!/usr/bin/env python3
"""Saturation figure generator.

Reads sweep + Hill fit JSON and generates two publication-quality figures:

  fig_sat_curves.pdf   — tok/s vs batch size per context length with
                         Hill model overlay (one curve per ctx length)
  fig_sat_surface.pdf  — 2D heatmap of measured tok/s over the
                         (batch, context) grid

Output directory: KNLP_FIG_DIR (default /tmp/sat_figures/).

Prints figure paths on stdout.  Exits 0 on success, 2 if matplotlib
is unavailable (treated as a warning — figures are optional).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SWEEP_PATH = os.environ.get("KNLP_SWEEP_PATH", "/tmp/sat_sweep_results.json")
FIT_PATH = os.environ.get("KNLP_FIT_PATH", "/tmp/sat_hillfit_results.json")
FIG_DIR = Path(os.environ.get("KNLP_FIG_DIR", "/tmp/sat_figures"))


def _hill(x: float, s_max: float, k_m: float) -> float:
    return s_max * x / (k_m + x)


def _plot_curves(sweep: dict, fit: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = sweep.get("cells", [])
    per_ctx = {e["ctx"]: e for e in fit.get("per_ctx", [])}
    ctx_lengths = sorted({c["ctx"] for c in cells})
    batch_sizes = sorted({c["batch"] for c in cells})

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ctx_lengths)))

    for i, ctx in enumerate(ctx_lengths):
        ctx_cells = sorted(
            [
                c
                for c in cells
                if c["ctx"] == ctx and not c.get("failed") and c.get("tps")
            ],
            key=lambda c: c["batch"],
        )
        if not ctx_cells:
            continue
        bs = [c["batch"] for c in ctx_cells]
        tps = [c["tps"] for c in ctx_cells]
        label = f"T={ctx//1024}K"
        ax.scatter(bs, tps, color=colors[i], s=25, zorder=3)
        # Hill overlay.
        if ctx in per_ctx:
            e = per_ctx[ctx]
            bs_fine = np.linspace(min(bs), max(bs) * 1.1, 200)
            tps_fit = [_hill(b, e["s_max"], e["k_m"]) for b in bs_fine]
            ax.plot(bs_fine, tps_fit, color=colors[i], lw=1.5, label=label)
        else:
            ax.plot(bs, tps, color=colors[i], lw=1.5, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Qwen2.5-7B decode throughput — Hill saturation fit")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def _plot_surface(sweep: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = sweep.get("cells", [])
    ctx_lengths = sorted({c["ctx"] for c in cells})
    batch_sizes = sorted({c["batch"] for c in cells})

    grid = np.full((len(ctx_lengths), len(batch_sizes)), float("nan"))
    for c in cells:
        if c.get("failed") or c.get("tps") is None:
            continue
        ri = ctx_lengths.index(c["ctx"])
        ci = batch_sizes.index(c["batch"])
        grid[ri, ci] = c["tps"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(ctx_lengths)))
    ax.set_yticklabels([f"{t//1024}K" for t in ctx_lengths])
    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel("Context length (T)")
    ax.set_title("Throughput (tok/s) heatmap — Qwen2.5-7B")
    plt.colorbar(im, ax=ax, label="tok/s")

    # Annotate cells.
    for ri in range(len(ctx_lengths)):
        for ci in range(len(batch_sizes)):
            val = grid[ri, ci]
            if not np.isnan(val):
                ax.text(
                    ci,
                    ri,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if val < grid.max() * 0.6 else "black",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def main() -> int:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("SKIP: matplotlib not available; saturation figures skipped", flush=True)
        return 2

    missing = [p for p in [SWEEP_PATH, FIT_PATH] if not os.path.exists(p)]
    if missing:
        print(f"SKIP: required files missing: {missing}", flush=True)
        return 2

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(SWEEP_PATH) as f:
        sweep = json.load(f)
    with open(FIT_PATH) as f:
        fit = json.load(f)

    _plot_curves(sweep, fit, FIG_DIR / "fig_sat_curves.pdf")
    _plot_surface(sweep, FIG_DIR / "fig_sat_surface.pdf")

    print("SAT FIGURES DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
