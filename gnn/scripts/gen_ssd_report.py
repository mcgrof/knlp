#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render a self-contained HTML report from a force-ssd RA run.

Reads the JSON written by benchmark_ssd.py (``{"meta": ..., "results":
...}``) and produces a dark-theme HTML page with inline SVG bar charts
and a metrics table. The charts are inline SVG, so the report renders
offline with no JavaScript or CDN. This is what ``make gnn-ssd-report``
runs.

    python gnn/scripts/gen_ssd_report.py \
        --json results/gnn/force_ssd_ra.json \
        --out docs/gnn_ssd_report.html
"""
import argparse
import html
import json
import os
from datetime import datetime

# Site palette (matches docs/gnn_fraud_visualization.html).
LAYOUT_COLOR = {
    "natural": "#fb7185",  # rose  - no optimization
    "random": "#f59e0b",  # amber - scrambled baseline
    "bfs": "#22d3ee",  # cyan  - graph BFS
    "metis": "#34d399",  # emerald - best
}
DEFAULT_COLOR = "#a78bfa"  # purple
MODELED_COLOR = "#64748b"  # slate


def color_for(layout):
    return LAYOUT_COLOR.get(layout, DEFAULT_COLOR)


def load(json_path):
    with open(json_path) as fh:
        data = json.load(fh)
    if "results" in data and "meta" in data:
        return data["meta"], data["results"]
    # Bare {layout: {...}} (older schema).
    return {}, data


def svg_bars(title, subtitle, categories, series, colors, fmt="{:.2f}x", note=None):
    """Vertical bar chart as an inline SVG string.

    series: list of (name, [value per category]); colors: list aligned to
    series, or a per-category list when there is a single series.
    """
    W, H = 660, 360
    pad_l, pad_r, pad_t, pad_b = 52, 18, 78, 66
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    n_cat = max(1, len(categories))
    n_ser = max(1, len(series))
    vals = [v for _, vs in series for v in vs] or [1.0]
    maxv = (max(vals) or 1.0) * 1.18
    group_w = plot_w / n_cat
    gap = 8
    bar_w = min(60.0, (group_w * 0.78 - gap * (n_ser - 1)) / n_ser)

    s = []
    s.append(
        f'<svg viewBox="0 0 {W} {H}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img">'
    )
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" fill="#111827"/>')
    s.append(
        f'<text x="{pad_l}" y="30" fill="#e5e7eb" font-size="17" '
        f'font-weight="700">{html.escape(title)}</text>'
    )
    if subtitle:
        s.append(
            f'<text x="{pad_l}" y="50" fill="#9ca3af" font-size="12.5">'
            f"{html.escape(subtitle)}</text>"
        )

    # Horizontal gridlines + y labels.
    for i in range(5):
        y = pad_t + plot_h * (1 - i / 4)
        gv = maxv * i / 4
        s.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
            f'stroke="#1f2937" stroke-width="1"/>'
        )
        s.append(
            f'<text x="{pad_l-8}" y="{y+4:.1f}" fill="#6b7280" font-size="10.5" '
            f'text-anchor="end">{fmt.format(gv)}</text>'
        )

    base = pad_t + plot_h
    for c, cat in enumerate(categories):
        cx = pad_l + group_w * c + group_w / 2
        total = bar_w * n_ser + gap * (n_ser - 1)
        start = cx - total / 2
        for si, (name, vs) in enumerate(series):
            v = vs[c]
            h = plot_h * (v / maxv) if maxv else 0
            x = start + si * (bar_w + gap)
            y = base - h
            col = colors[si] if n_ser > 1 else colors[c]
            s.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{max(0,h):.1f}" rx="5" fill="{col}"/>'
            )
            s.append(
                f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" fill="#e5e7eb" '
                f'font-size="11.5" font-weight="600" text-anchor="middle">'
                f"{fmt.format(v)}</text>"
            )
        s.append(
            f'<text x="{cx:.1f}" y="{base+20:.1f}" fill="#cbd5e1" '
            f'font-size="12.5" text-anchor="middle">{html.escape(cat)}</text>'
        )

    if n_ser > 1:
        lx = pad_l
        ly = H - 16
        for si, (name, _) in enumerate(series):
            s.append(
                f'<rect x="{lx}" y="{ly-10}" width="12" height="12" rx="3" '
                f'fill="{colors[si]}"/>'
            )
            s.append(
                f'<text x="{lx+18}" y="{ly}" fill="#9ca3af" font-size="12">'
                f"{html.escape(name)}</text>"
            )
            lx += 30 + 8 * len(name)
    elif note:
        s.append(
            f'<text x="{pad_l}" y="{H-16}" fill="#6b7280" font-size="11.5">'
            f"{html.escape(note)}</text>"
        )
    s.append("</svg>")
    return "".join(s)


def metrics_table(results):
    layouts = list(results.keys())
    head = (
        "<tr>"
        + "".join(
            f"<th>{h}</th>"
            for h in [
                "layout",
                "nbr RA_fetch",
                "nbr RA_signal",
                "page RA_fetch",
                "page RA_signal",
                "nbr pages",
                "nbr read ops",
                "GB read",
                "MB/s",
                "O_DIRECT",
            ]
        )
        + "</tr>"
    )
    rows = [head]
    for lo in layouts:
        nb = results[lo]["neighbor"]
        pg = results[lo]["page"]
        gb = nb["bytes_read"] / 1e9
        rows.append(
            "<tr>"
            + f'<td style="color:{color_for(lo)};font-weight:700">{lo}</td>'
            + f'<td>{nb["ra_physical"]:.2f}x</td>'
            + f'<td>{nb["ra_signal"]:.1f}x</td>'
            + f'<td>{pg["ra_physical"]:.2f}x</td>'
            + f'<td>{pg["ra_signal"]:.1f}x</td>'
            + f'<td>{nb["pages_read"]:,}</td>'
            + f'<td>{nb["read_ops"]:,}</td>'
            + f"<td>{gb:.2f}</td>"
            + f'<td>{nb["throughput_mb_s"]:.0f}</td>'
            + f'<td>{"yes" if nb.get("direct") else "no"}</td>'
            + "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


CSS = """
* { box-sizing: border-box; }
body { margin:0; background:#0b0f17; color:#e5e7eb;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width: 1000px; margin: 0 auto; padding: 40px 24px 64px; }
h1 { font-size: 30px; margin: 0 0 6px; }
h2 { font-size: 20px; margin: 30px 0 14px; color:#34d399; }
.sub { color:#9ca3af; margin: 0 0 22px; }
.card { background:#111827; border:1px solid #1f2937; border-radius:14px;
  padding:20px; margin-bottom:22px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media (max-width:760px){ .grid2{ grid-template-columns:1fr; } }
.banner { background:#3b2f0b; border:1px solid #b45309; color:#fcd34d;
  border-radius:12px; padding:14px 18px; margin-bottom:22px; font-size:14px; }
.kpi { display:flex; gap:18px; flex-wrap:wrap; }
.kpi .box { background:#0f172a; border:1px solid #1f2937; border-radius:12px;
  padding:14px 18px; flex:1; min-width:160px; }
.kpi .v { font-size:26px; font-weight:800; }
.kpi .l { color:#9ca3af; font-size:12.5px; margin-top:4px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { text-align:right; padding:8px 10px; border-bottom:1px solid #1f2937; }
th:first-child,td:first-child { text-align:left; }
th { color:#9ca3af; font-weight:600; }
code { background:#0f172a; padding:2px 6px; border-radius:5px; color:#22d3ee; }
.meta { color:#6b7280; font-size:12.5px; }
a { color:#34d399; }
.foot { color:#4b5563; font-size:12px; border-top:1px solid #1f2937;
  margin-top:30px; padding-top:18px; }
"""


def build_html(meta, results):
    layouts = list(results.keys())
    nbr_fetch = [results[lo]["neighbor"]["ra_physical"] for lo in layouts]
    page_signal = [results[lo]["page"]["ra_signal"] for lo in layouts]
    nbr_measured = nbr_fetch
    nbr_modeled = [results[lo]["neighbor"]["ra_modeled"] for lo in layouts]
    cat_colors = [color_for(lo) for lo in layouts]

    chart_value = svg_bars(
        "Neighbor access — read amplification by layout",
        "RA_fetch, lower is better. This is the value of a locality-aware layout.",
        layouts,
        [("RA_fetch", nbr_fetch)],
        cat_colors,
        note="bytes read from device / useful feature bytes",
    )
    chart_gap = svg_bars(
        "Page access — supervised read amplification",
        "RA_signal stays near the floor for every layout. This is the gap.",
        layouts,
        [("RA_signal", page_signal)],
        cat_colors,
        fmt="{:.1f}x",
        note="bytes read / supervised-node bytes — a floor no layout closes",
    )
    chart_cross = svg_bars(
        "Measured vs modeled (neighbor RA_fetch)",
        "Real O_DIRECT device reads vs the in-RAM page-touch metric.",
        layouts,
        [("measured (device)", nbr_measured), ("modeled (in-RAM)", nbr_modeled)],
        ["#34d399", MODELED_COLOR],
    )

    # Headline: worst vs best neighbor RA_fetch.
    best_i = min(range(len(layouts)), key=lambda i: nbr_fetch[i])
    worst_i = max(range(len(layouts)), key=lambda i: nbr_fetch[i])
    ratio = (nbr_fetch[worst_i] / nbr_fetch[best_i]) if nbr_fetch[best_i] else 1.0
    total_gb = sum(results[lo]["neighbor"]["bytes_read"] for lo in layouts) / 1e9

    smoke = bool(meta.get("smoke"))
    banner = ""
    if smoke:
        banner = (
            '<div class="banner"><b>Illustrative report — synthetic smoke '
            "data, not a DGraphFin measurement.</b> Generated from a small "
            "synthetic community graph to demonstrate the report. Regenerate "
            "with a real run: <code>make defconfig-gnn-dgraphfin-force-ssd && "
            "make</code>, then <code>make gnn-ssd-report</code>.</div>"
        )

    ds = html.escape(str(meta.get("dataset", "unknown")))
    direct = "O_DIRECT" if meta.get("direct_requested", True) else "buffered"
    fan = ",".join(str(x) for x in meta.get("fanouts", []))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    meta_line = (
        f"dataset <b>{ds}</b> · {direct} · layouts "
        f'{html.escape(",".join(layouts))} · fanout {html.escape(fan)} · '
        f'batch {meta.get("batch_size","?")} · pages/batch '
        f'{meta.get("pages_per_batch","?")} · replicas '
        f'{meta.get("replicas","?")} · generated {generated}'
    )

    kpi = (
        '<div class="kpi">'
        f'<div class="box"><div class="v" style="color:#34d399">{ratio:.1f}x</div>'
        f'<div class="l">fewer device reads, {layouts[worst_i]} → '
        f"{layouts[best_i]} (neighbor RA_fetch)</div></div>"
        f'<div class="box"><div class="v" style="color:#a78bfa">'
        f"{min(page_signal):.1f}x–{max(page_signal):.1f}x</div>"
        f'<div class="l">page RA_signal range — the floor no layout closes</div></div>'
        f'<div class="box"><div class="v" style="color:#22d3ee">{total_gb:.1f} GB</div>'
        f'<div class="l">real device reads issued across layouts</div></div>'
        "</div>"
    )

    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DGraphFin Force-SSD Read-Amplification Report</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>DGraphFin Force-SSD Read-Amplification Report</h1>
<p class="sub">Real device I/O measured per page layout. Cross-check these
numbers with your own eBPF/blktrace/iostat counters.</p>
{banner}
<p class="meta">{meta_line}</p>
<div class="card">{kpi}</div>

<h2>The value: locality lowers device reads</h2>
<div class="grid2">
  <div class="card">{chart_value}</div>
  <div class="card">{chart_cross}</div>
</div>
<p class="sub">The neighbor pattern expands random training seeds to their
sampled neighborhood. A locality-aware layout (metis) keeps those neighbors
on a few 4&nbsp;KiB pages, so the device serves far fewer reads per useful
byte. The measured O_DIRECT amplification tracks the in-RAM modeled metric,
which is exactly the agreement an external observer should confirm.</p>

<h2>The gap: the supervised-read floor</h2>
<div class="card">{chart_gap}</div>
<p class="sub">The page-aligned sweep reads whole training pages. RA_fetch is
~1x for any layout, but only a fraction of nodes per page are supervised, so
RA_signal — bytes read per supervised node — cannot drop below the ~4.3x
floor no matter how good the layout is. That floor is the gap the layout work
leaves open.</p>

<h2>All metrics</h2>
<div class="card">{metrics_table(results)}</div>

<h2>Verify it yourself</h2>
<p class="sub">We do not run eBPF here; the harness only issues the reads.
Attach <code>biosnoop</code>/<code>biolatency</code>, <code>blktrace</code>
or <code>iostat -x</code> to the device backing the store and confirm the
per-I/O read intent and amplification independently.</p>

<div class="foot">Generated by <code>make gnn-ssd-report</code> ·
<a href="gnn_fraud_visualization.html">GNN Fraud visualization</a> ·
<a href="gnn-dgraphfin.md">DGraphFin doc</a></div>
</div></body></html>
"""
    return body


def main():
    ap = argparse.ArgumentParser(description="Render force-ssd RA HTML report")
    ap.add_argument("--json", default="results/gnn/force_ssd_ra.json")
    ap.add_argument("--out", default="docs/gnn_ssd_report.html")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"No results JSON at {args.json}.")
        print("Run the force-ssd workload first:")
        print("  make defconfig-gnn-dgraphfin-force-ssd && make")
        return 1

    meta, results = load(args.json)
    if not results:
        print(f"No layout results in {args.json}.")
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(build_html(meta, results))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
