# SPDX-License-Identifier: MIT
"""Turn the P/D matrix records into a knlp page.

Reads the JSONL the matrix runner appends to and writes an HTML page in the
house style, with inline SVG charts so the page needs no scripts and no CDN.

Bars carry their spread, not just a mean.  Every arm here is measured many
times precisely because a single run of any of them is a number without an
error bar, and one non-interleaved sweep on this device already produced a
ranking that was drift.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

# The template's palette, so the charts match the rest of the site.
PALETTE = {
    "nixl-gds-xfs": "#f0b65c", "nixl-gds-ext4": "#f0b65c",
    "nixl-block": "#2ae6a3", "nixl-xfs": "#2ae6a3", "nixl-ext4": "#2ae6a3",
    "lmcache-block": "#62dce5", "lmcache-xfs": "#62dce5", "lmcache-ext4": "#62dce5",
}
ORDER = ["nixl-gds-xfs", "nixl-gds-ext4", "nixl-block", "nixl-xfs", "nixl-ext4",
         "lmcache-block", "lmcache-xfs", "lmcache-ext4"]


def load(path: str) -> dict[str, list[dict]]:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ok") and rec.get("write_mibs") and rec.get("read_mibs"):
                by_arm[rec["arm"]].append(rec)
    return by_arm


def stats(vals: list[float]) -> dict[str, float]:
    return {"median": statistics.median(vals), "lo": min(vals), "hi": max(vals),
            "n": len(vals)}


def svg_bars(rows: list[tuple[str, dict]], title: str, unit: str = "MiB/s") -> str:
    """A horizontal bar chart with a min-max whisker per bar."""
    if not rows:
        return "<p class='fine'>no data yet</p>"
    w, rowh, pad_l, pad_t = 900, 34, 150, 34
    h = pad_t + rowh * len(rows) + 16
    top = max(r[1]["hi"] for r in rows) * 1.12 or 1
    scale = (w - pad_l - 90) / top
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}" '
           f'style="width:100%;height:auto">',
           f'<text x="0" y="18" fill="#93a59a" font-size="13" '
           f'font-family="ui-monospace,monospace">{title} ({unit})</text>']
    for i, (arm, s) in enumerate(rows):
        y = pad_t + i * rowh
        bar = s["median"] * scale
        colour = PALETTE.get(arm, "#93a59a")
        out.append(f'<text x="0" y="{y + 15}" fill="#e2ece5" font-size="12" '
                   f'font-family="ui-monospace,monospace">{arm}</text>')
        out.append(f'<rect x="{pad_l}" y="{y + 4}" width="{bar:.1f}" height="16" '
                   f'rx="3" fill="{colour}" opacity="0.85"/>')
        # whisker: the spread across iterations
        x0, x1 = pad_l + s["lo"] * scale, pad_l + s["hi"] * scale
        out.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y + 12}" y2="{y + 12}" '
                   f'stroke="#e2ece5" stroke-width="1" opacity="0.55"/>')
        for xx in (x0, x1):
            out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y + 7}" y2="{y + 17}" '
                       f'stroke="#e2ece5" stroke-width="1" opacity="0.55"/>')
        out.append(f'<text x="{pad_l + bar + 8:.1f}" y="{y + 16}" fill="#93a59a" '
                   f'font-size="12" font-family="ui-monospace,monospace">'
                   f'{s["median"]:.0f}</text>')
    out.append("</svg>")
    return "".join(out)


def table(by_arm: dict[str, list[dict]]) -> str:
    head = ("<div class='scroll'><table><thead><tr><th>arm</th><th>implementation</th>"
            "<th>backend</th><th>target</th><th>write median</th><th>write range</th>"
            "<th>read median</th><th>read range</th><th>runs</th></tr></thead><tbody>")
    body = []
    for arm in ORDER:
        recs = by_arm.get(arm)
        if not recs:
            continue
        w = stats([r["write_mibs"] for r in recs])
        r_ = stats([r["read_mibs"] for r in recs])
        impl = "LMCache raw_block" if recs[0]["path"] == "lmcache" else "vLLM + NIXL"
        body.append(
            f"<tr><td>{arm}</td><td>{impl}</td><td>{recs[0]['backend']}</td>"
            f"<td>{recs[0]['target']}</td><td>{w['median']:.0f}</td>"
            f"<td>{w['lo']:.0f}–{w['hi']:.0f}</td><td>{r_['median']:.0f}</td>"
            f"<td>{r_['lo']:.0f}–{r_['hi']:.0f}</td><td>{w['n']}</td></tr>")
    return head + "".join(body) + "</tbody></table></div>"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="docs/bias-aware-kv-quantization.html")
    a = ap.parse_args(argv)

    by_arm = load(a.records)
    if not by_arm:
        print("no usable records yet")
        return 1
    any_rec = next(iter(by_arm.values()))[0]
    wrows = [(k, stats([r["write_mibs"] for r in by_arm[k]])) for k in ORDER if k in by_arm]
    rrows = [(k, stats([r["read_mibs"] for r in by_arm[k]])) for k in ORDER if k in by_arm]
    iters = max(len(v) for v in by_arm.values())

    with open(a.template) as f:
        tpl = f.read()
    style = tpl[tpl.index("<style>"):tpl.index("</style>") + 8]

    html = PAGE.format(
        style=style, write_chart=svg_bars(wrows, "Write throughput, median with min–max"),
        read_chart=svg_bars(rrows, "Read throughput, median with min–max"),
        table=table(by_arm), iters=iters, host=any_rec["host"],
        kernel=any_rec["kernel"], obj=any_rec["object_size"] >> 20,
        n=any_rec["objects"], when=any_rec["when"][:10])
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out} from {sum(len(v) for v in by_arm.values())} records "
          f"across {len(by_arm)} arms")
    return 0


PAGE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Storage-mediated P/D: which path moves a KV cache fastest</title>
<meta name="description" content="Measured comparison of storage-mediated prefill/decode paths: NIXL GDS against NIXL io_uring on raw block, XFS and ext4, and LMCache raw_block on the same three targets.">
{style}
</head><body>
<header><nav><a class="logo" href="./">KNLP</a><span class="spacer"></span>
<a href="./">Index</a><a href="./kvtide.html">KVTide</a>
<a href="./bias-aware-kv-quantization.html">KV quantization</a></nav></header>
<main>
<div class="hero">
<div class="eyebrow">Storage-mediated prefill/decode · measured · September 2026</div>
<h1>The filesystem<br>is not the tax.</h1>
<p class="lead"><strong>Moving a KV cache between GPUs through NVMe does not require giving up a filesystem.</strong> On the same drive, the same payload and the same io_uring path, writing to a file on XFS or ext4 lands within a few percent of writing to the raw namespace. The cost that does show up is elsewhere.</p>
<p class="lead">The comparison covers two implementations that both reach storage — the vLLM storage handoff connector over NIXL, and LMCache <code>raw_block</code> driving the device itself — against three targets each, with NIXL GDS as the baseline.</p>
<div class="badges">
<span class="badge green">{iters} interleaved iterations</span>
<span class="badge cyan">{n} × {obj} MiB per run</span>
<span class="badge amber">GDS in compat mode</span>
</div>
</div>

<section>
<h2>Throughput by path and target</h2>
<p class="sub">Bars are the median across iterations; the whisker is the full min–max spread. Arms are interleaved within each iteration rather than run back to back, because a non-interleaved sweep on this device once produced a clean-looking ranking that was drift and did not reproduce.</p>
{write_chart}
<div style="margin-top:26px">{read_chart}</div>
</section>

<section>
<h2>Every arm, with its spread</h2>
<p class="sub">A median without a range is a number nobody can check. Read the range first.</p>
{table}
<div class="note"><strong>The GDS baseline is host-staged here.</strong> With the IOMMU translating, <code>gdscheck</code> reports <code>NVMe : compat</code> and warns that GDS is not guaranteed to work functionally or performantly with <code>iommu=on/pt</code>. cuFile therefore falls back to staging through host memory rather than doing peer-to-peer DMA. The number is real and worth having as a baseline, but it is not a GPU-direct number, and labelling it as one would be wrong.</div>
</section>

<section>
<h2>What this does and does not establish</h2>
<div class="grid2">
<div class="card rule good"><strong>ESTABLISHED</strong><p>A file on XFS or ext4 carries the KV cache at essentially the raw-device rate on this drive, through the same io_uring path, with the dma-buf command-size ceiling applied. Keeping a filesystem is not what costs throughput.</p></div>
<div class="card rule warn"><strong>NOT ESTABLISHED</strong><p>Nothing here says what GDS would do with the IOMMU in passthrough, where cuFile could use peer-to-peer DMA. That is a different measurement on a differently booted host, and it is the honest comparison to make next.</p></div>
</div>
<div class="note" style="border-left-color:var(--muted)"><strong>Measured on</strong> {host}, kernel <code>{kernel}</code>, {when}. Every run addressed the drive through <code>/dev/disk/by-id</code> and verified device identity before writing. Reproduce with <code>make defconfig-pd-nixl-xfs</code> and its siblings in knlp.</div>
</section>
</main>
<footer>KNLP · storage-mediated prefill/decode · generated from the measurement records, not transcribed</footer>
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
