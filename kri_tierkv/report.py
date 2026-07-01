# SPDX-License-Identifier: GPL-2.0
"""HTML report for KRI-TierKV emulation runs.

Renders the per-policy metrics (attention-mass recall, high-mass false-negative
rate, blocks fetched, bytes moved) as a table plus simple inline SVG bars, so a
run answers its questions at a glance: does KRI-D-sum beat FIFO / recent-only on
mass recall, does K16/V8 stay cheap, and is K8/V4 flagged unsafe.
"""

from __future__ import annotations

import html


def _bar(value, vmax, color):
    w = 0 if vmax <= 0 else int(200 * max(0.0, min(1.0, value / vmax)))
    return (
        f'<svg width="210" height="14">'
        f'<rect width="200" height="12" fill="#1e293b"/>'
        f'<rect width="{w}" height="12" fill="{color}"/></svg>'
    )


def write_html(metrics: dict, path: str):
    rows = metrics.get("policies", [])
    recalls = [r["attention_mass_recall"] for r in rows] or [1.0]
    bmax = max((r["bytes_moved_per_token"] for r in rows), default=1.0) or 1.0

    body = []
    body.append(
        "<tr><th>Policy</th><th>Mass recall</th><th>High-mass FNR</th>"
        "<th>Slow blocks fetched</th><th>Bytes moved / token</th></tr>"
    )
    for r in rows:
        recall = r["attention_mass_recall"]
        rc = "#4ade80" if recall >= 0.9 else "#fbbf24" if recall >= 0.6 else "#fb7185"
        body.append(
            "<tr>"
            f"<td>{html.escape(r['policy'])}</td>"
            f"<td>{recall:.3f} {_bar(recall, 1.0, rc)}</td>"
            f"<td>{r['high_mass_false_negative_rate']:.3f}</td>"
            f"<td>{r['blocks_fetched_per_step']}</td>"
            f"<td>{r['bytes_moved_per_token']/1e6:.1f} MB "
            f"{_bar(r['bytes_moved_per_token'], bmax, '#60a5fa')}</td>"
            "</tr>"
        )

    quant_note = ""
    if metrics.get("k_bits", 16) < 16 and metrics.get("v_bits", 16) <= 4:
        quant_note = (
            "<p style='color:#fb7185'>K8/V4 is experimental/unsafe: "
            "sub-16-bit keys collapse on fragile-key models.</p>"
        )
    qerr = metrics.get("quant_rel_err")
    qerr_s = f"{qerr:.4f}" if qerr is not None else "n/a"

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KRI-TierKV A100 Results</title>
<style>
body{{background:#0f172a;color:#cbd5e1;font-family:system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:2rem}}
h1{{color:#f1f5f9}} code{{background:#1e293b;padding:2px 6px;border-radius:4px}}
table{{width:100%;border-collapse:collapse;margin:1rem 0}}
th{{text-align:left;color:#94a3b8;font-size:12px;text-transform:uppercase;padding:8px;border-bottom:1px solid #334155}}
td{{padding:8px;border-bottom:1px solid #243244}}
.meta{{color:#94a3b8;font-size:14px;line-height:1.8}}
</style></head><body>
<h1>KRI-TierKV A100 Results</h1>
<p class="meta">Inspired by TTKV (arXiv:2604.19769 &mdash; verify id before citing).
Not a TTKV clone: the tiering policy is KRI-D-sum. FP8/quant is fake/storage
simulation only; no native FP8 speedup is claimed on A100.</p>
<p class="meta">
Model: <code>{html.escape(str(metrics.get('model')))}</code><br>
Hardware: <code>{html.escape(str(metrics.get('device_name')))}</code><br>
Context: {metrics.get('context_len')} tokens, {metrics.get('num_blocks')} blocks of
{metrics.get('block_size')} &middot; fast window {metrics.get('fast_window_tokens')} tokens
&middot; slow top-K {metrics.get('slow_topk_blocks')}<br>
Quant: <code>{html.escape(str(metrics.get('quant')))}</code>
(K{metrics.get('k_bits')}/V{metrics.get('v_bits')}), mean V rel-err {qerr_s}
</p>
<table>{''.join(body)}</table>
<h2>Conclusion</h2>
<p class="meta">Does KRI-D-sum beat FIFO / recent-only on attention-mass recall?
Compare the <code>kri_d_sum_kri_topk</code> row against <code>recent_only</code>
and <code>fifo_kri_topk</code>. Does K16/V8 preserve quality? Check the V rel-err
and the quant run's recall. {quant_note}</p>
</body></html>"""
    with open(path, "w") as f:
        f.write(doc)
    return path
