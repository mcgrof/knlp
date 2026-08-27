#!/usr/bin/env python3
"""Summarize a cartridge optimizer ablation into one comparison report.

Reads each arm's run.json (loss history plus the CUDA-synchronized
wall-clock split written by cart_opt_ablation.py) and, when available,
the strict re-evaluation JSON from opcart_reeval.py, and writes a
markdown report plus a machine-readable JSON next to it.

The headline numbers are the loss reached at matched steps, the
optimizer-step share of total wall-clock (the overhead ratio that
decides whether a second-order optimizer is affordable in this regime),
and strict letter accuracy per saved checkpoint with parser-invalid and
cap-hit rates alongside, since accuracy without those is not reportable.
"""

import argparse
import json
from pathlib import Path


def load_arms(ablation_dir):
    arms = {}
    for run_json in sorted(Path(ablation_dir).glob("*/run.json")):
        run = json.loads(run_json.read_text())
        arms[run["optimizer"]] = run
    return arms


def loss_at(run, step):
    for entry in run["history"]:
        if entry["step"] == step:
            return entry["loss"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", required=True)
    ap.add_argument("--reeval", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = load_arms(args.ablation_dir)
    if not arms:
        raise SystemExit(f"no */run.json under {args.ablation_dir}")
    names = sorted(arms)
    ref = arms[names[0]]
    marks = sorted(
        {e["step"] for run in arms.values() for e in run["history"]}
        & {
            e["step"]
            for e in ref["history"]
            if e["step"] in (1, 10, 25, 50, 100, 200, 300, 500, ref["steps"])
        }
    )
    if ref["steps"] not in marks:
        marks.append(ref["steps"])

    lines = [
        "# Cartridge optimizer ablation",
        "",
        f"Model {ref['model']}, patient {ref['patient']}, "
        f"{ref['steps']} steps, accum {ref['accum']}, lr {ref['lr']}, "
        f"seed {ref['seed']}, {ref['trainable_params']} trainable "
        f"cartridge params, shared init `{Path(ref['init_cart']).name}`.",
        "",
        "## Training loss at matched steps",
        "",
        "| step | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for s in marks:
        vals = [loss_at(arms[n], s) for n in names]
        row = [f"{v:.4f}" if v is not None else "-" for v in vals]
        lines.append(f"| {s} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Wall-clock split (CUDA-synchronized)",
        "",
        "| arm | total s | fwd+bwd s | optimizer s | optimizer share | peak GB |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        c = arms[n]["cost"]
        lines.append(
            f"| {n} | {c['total_wall_s']:.0f} | {c['fwdbwd_s']:.0f} | "
            f"{c['update_s']:.1f} | {c['update_frac']:.2%} | "
            f"{c['peak_mem_gb']:.2f} |"
        )

    summary = dict(arms={n: arms[n]["cost"] for n in names}, loss_marks={})
    for s in marks:
        summary["loss_marks"][str(s)] = {n: loss_at(arms[n], s) for n in names}

    if args.reeval and Path(args.reeval).is_file():
        report = json.loads(Path(args.reeval).read_text())
        lines += [
            "",
            "## Strict letter eval per checkpoint",
            "",
            "| condition | cap | strict acc | invalid | cap hit | mean len |",
            "|---|---|---|---|---|---|",
        ]
        for r in report["results"]:
            lines.append(
                f"| {r['condition']} | {r['cap']} | {r['strict_acc']:.3f} | "
                f"{r['parser_invalid']:.2f} | {r['cap_hit']:.2f} | "
                f"{r['mean_len']:.1f} |"
            )
        summary["reeval"] = report["results"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"CART_OPT_REPORT {out}")


if __name__ == "__main__":
    main()
