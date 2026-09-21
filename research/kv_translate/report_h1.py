# SPDX-License-Identifier: GPL-2.0
"""Turn the H1 and task-gate outputs into the tables a reader needs.

Kept separate from the runs that produce the numbers so that re-reading a
result never risks re-running it, and so the arithmetic that turns raw scores
into gap-closure and recovery fractions lives in one place rather than being
repeated in prose.

Env: none; reads JSON only.
"""

from __future__ import annotations

import argparse
import json
import os


def pct(x, digits=1):
    return "n/a" if x is None or x != x else f"{100 * x:.{digits}f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1-json", required=True)
    ap.add_argument("--smoke-json", default="")
    args = ap.parse_args()

    h1 = json.load(open(args.h1_json))
    s = h1["summary"]
    rep = h1.get("reports", {})
    lat = h1.get("latency", {})
    byt = h1.get("arm_bytes", {})
    floor = s["no_cache"]["kl"]

    print("## Held-out divergence, best first")
    print()
    print(
        "| arm | KL | delta NLL | argmax flip | closes | apply p50 | resident MiB | folds |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|:--:|")
    for a in sorted(s, key=lambda a: s[a]["kl"]):
        if a == "target_native":
            continue
        b = byt.get(a, {})
        folded = "yes" if b.get("folded") else ("no" if a in rep else "-")
        total = b.get("total_bytes")
        print(
            f"| `{a}` | {s[a]['kl']:.4f} | {s[a]['delta_nll']:+.4f} | "
            f"{s[a]['argmax_flip']:.4f} | {pct(s[a]['gap_closed'])} | "
            f"{lat.get(a, {}).get('p50_ms', float('nan')):.1f} ms | "
            f"{(total / 2**20 if total else float('nan')):.0f} | {folded} |"
        )
    print()
    print(f"No-cache floor KL {floor:.4f}; target-native is 0 by construction.")
    print()

    if rep:
        print("## What each correction cost to train")
        print()
        print(
            "| arm | rate chosen | train loss | dev loss | params | seconds | peak GiB |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        for a, r in sorted(rep.items()):
            print(
                f"| `{a}` | {r['lr']:g} | {r['first_loss']:.4f} -> "
                f"{r['last_loss']:.4f} | {r['dev_loss']:.4f} | "
                f"{r['residual_params'] / 1e6:.2f}M | {r['train_seconds']:.0f} | "
                f"{r['peak_mem_bytes'] / 2**30:.1f} |"
            )
        print()
        merges = {
            a: r["merge_max_abs_gap"]
            for a, r in rep.items()
            if "merge_max_abs_gap" in r
        }
        if merges:
            worst = max(merges.values())
            print(
                f"Every linear correction was folded into the map it corrects and "
                f"scored in folded form; the largest disagreement between an arm "
                f"and its folded self was {worst:.1e}."
            )
            print()

    if not args.smoke_json or not os.path.exists(args.smoke_json):
        return 0

    sm = json.load(open(args.smoke_json))
    print("## Task gate on fresh documents")
    print()
    w = sm["latency_matched_window"]
    print(
        f"{sm['n_items']} items over {len(sm['documents'])} documents, none of "
        f"which appear in the corpus the maps were fitted on. The promoted arm "
        f"costs {w['budget_ms']:.1f} ms to apply, which buys a native prefill of "
        f"{w['tokens']} tokens ({w['ms']:.1f} ms) -- that is the "
        f"`recent_{w['tokens']}` row."
    )
    print()
    for kind in ("retrieval", "cloze", "free"):
        if kind not in sm["summary"]:
            continue
        print(f"### {kind}")
        print()
        print("| condition | score | 95% CI | recovers native gain | CI |")
        print("|---|---:|---|---:|---|")
        for cond in sorted(sm["summary"][kind]):
            row = sm["summary"][kind][cond]
            rc = sm["recovery"].get(cond, {}).get(kind)
            r1 = pct(rc["point"]) if rc else "-"
            r2 = f"[{pct(rc['lo'])}, {pct(rc['hi'])}]" if rc else "-"
            print(
                f"| `{cond}` | {row['point']:.3f} | "
                f"[{row['lo']:.3f}, {row['hi']:.3f}] | {r1} | {r2} |"
            )
        print()

    print("### Gate")
    print()
    print(
        "| arm | prompt-specific | recovers gain | within 5 pts | no pathology "
        "| beats recent window | keeps latency | verdict |"
    )
    print("|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for a, g in sorted(sm["gates"].items()):

        def y(v):
            return "-" if v is None else ("yes" if v else "no")

        print(
            f"| `{a}` | {y(g['prompt_specific'])} | "
            f"{y(g['recovers_incremental_gain'])} | {y(g['within_absolute_points'])} | "
            f"{y(g['no_pathology_regression'])} | "
            f"{y(g.get('beats_latency_matched_native'))} | "
            f"{y(g['retains_h0_latency_gates'])} | "
            f"**{'PASS' if g['passes'] else 'FAIL'}** |"
        )
    print()
    ft = sm.get("first_token", {})
    if ft:
        nat = next(iter(ft.values()))["native_p50_ms"]
        print(
            f"### First-token path with the correction in place "
            f"(native prefill {nat:.1f} ms)"
        )
        print()
        print(
            "| arm | translated p50 | ratio p50 | ratio p95 | next-token KL | argmax |"
        )
        print("|---|---:|---:|---:|---:|:--:|")
        for a, f in sorted(ft.items(), key=lambda kv: kv[1]["ratio_p50"]):
            print(
                f"| `{a}` | {f['translated_p50_ms']:.1f} ms | {f['ratio_p50']:.3f} | "
                f"{f['ratio_p95']:.3f} | {f['next_token_kl']:.4f} | "
                f"{'agrees' if f['argmax_agrees'] else 'differs'} |"
            )
        print()
    print()
    print("### Margins against the controls (paired over documents)")
    print()
    print("| arm | vs empty cache | vs its wrong-prompt | vs recent window |")
    print("|---|---|---|---|")

    def band(d):
        if not d:
            return "-"
        return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

    for a, g in sorted(sm["gates"].items()):
        print(
            f"| `{a}` | {band(g.get('vs_empty_cache'))} | "
            f"{band(g.get('vs_wrong_prompt'))} | {band(g.get('vs_recent_window'))} |"
        )
    print()
    print(f"Passing arms: {sm['passing_arms'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
