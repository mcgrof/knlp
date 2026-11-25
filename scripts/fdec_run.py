#!/usr/bin/env python3
"""Run the fdec composition matrix named by a Kconfig .config.

`run` executes scripts/fdec_compose.py once per configured model and skips a
model whose result file already exists. `tasks` executes
scripts/fdec_task_acc.py once per model: GSM8K and MMLU for the dense model,
each enabled brick alone and the enabled bricks together. `report` reads the
result files and writes one markdown table per model plus a verdict line.

The task check is optional and takes hours per model on one GPU. When its
result exists, `report` adds it and applies one rule: no cell may be
significantly below the dense model. The test is the exact McNemar test on
the items that changed, with a Bonferroni correction over the cells of one
model and task, at 0.05.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_config(path):
    cfg = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^CONFIG_(FDEC_\w+)=(.*)$", line.strip())
        if m:
            cfg[m.group(1)] = m.group(2).strip('"')
    assert cfg.get("FDEC_ENABLE") == "y", "CONFIG_FDEC_ENABLE is not set"
    return cfg


def bricks_of(cfg):
    pairs = (
        ("FDEC_INT8_WEIGHTS", "w8"),
        ("FDEC_ASYM_KV", "kv"),
        ("FDEC_LMHEAD_CERT", "cert"),
    )
    out = [b for k, b in pairs if cfg.get(k) == "y"]
    assert out, "no fdec brick is enabled"
    return out


def result_path(cfg, model):
    return Path(cfg["FDEC_ROOT"]) / (model.replace("/", "__") + ".json")


def kv_scale(cfg):
    scale = cfg.get("FDEC_KV_SCALE", "one")
    assert scale in ("one", "tensor"), f"CONFIG_FDEC_KV_SCALE={scale!r}"
    return scale


TASKS = ("gsm8k", "mmlu")
TASK_ALPHA = 0.05


def task_path(cfg, model):
    return Path(cfg["FDEC_ROOT"]) / (model.replace("/", "__") + ".tasks.json")


def task_cells(cfg):
    """The dense model, each enabled brick alone and all of them together.
    With all three bricks, also the two that change the hidden states."""
    on = bricks_of(cfg)
    cells = ["base"] + on
    if on == ["w8", "kv", "cert"]:
        cells.append("w8+kv")
    if len(on) > 1:
        cells.append("+".join(on))
    return cells


def run(cfg):
    for model in cfg["FDEC_MODELS"].split():
        out = result_path(cfg, model)
        if out.exists():
            # A result written before the option existed used the
            # per-tensor scale.
            have = json.loads(out.read_text()).get("kv_scale", "tensor")
            assert have == kv_scale(cfg) or "kv" not in bricks_of(cfg), (
                f"{out} was measured with the FP8 scale {have!r}, the "
                f"configuration asks for {kv_scale(cfg)!r}; remove the file "
                "or change CONFIG_FDEC_KV_SCALE"
            )
            print(f"[fdec] {model}: result exists, skipping ({out})")
            continue
        cmd = [
            sys.executable,
            str(HERE / "fdec_compose.py"),
            "--model",
            model,
            "--device",
            cfg.get("FDEC_DEVICE", "cuda:0"),
            "--bricks",
            ",".join(bricks_of(cfg)),
            "--eval-seqs",
            cfg.get("FDEC_EVAL_SEQS", "16"),
            "--calib-seqs",
            cfg.get("FDEC_CALIB_SEQS", "128"),
            "--kv-scale",
            kv_scale(cfg),
            "--out",
            str(out),
        ]
        if cfg.get("FDEC_DEVICE", "cuda:0") == "cpu":
            cmd += ["--dtype", "float32"]
        subprocess.run(cmd, check=True)


def tasks(cfg):
    cells = task_cells(cfg)
    for model in cfg["FDEC_MODELS"].split():
        out = task_path(cfg, model)
        if out.exists():
            raw = json.loads(out.read_text()).get("raw", {})
            if all(c in raw.get(t, {}) for t in TASKS for c in cells):
                print(f"[fdec] {model}: task result exists, skipping ({out})")
                continue
        cmd = [
            sys.executable,
            str(HERE / "fdec_task_acc.py"),
            "--model",
            model,
            "--device",
            cfg.get("FDEC_DEVICE", "cuda:0"),
            "--arms",
            ",".join(cells),
            "--gsm-limit",
            cfg.get("FDEC_TASK_GSM_LIMIT", "0"),
            "--mmlu-per-subject",
            cfg.get("FDEC_TASK_MMLU_PER_SUBJECT", "0"),
            "--gsm-batch",
            cfg.get("FDEC_TASK_GSM_BATCH", "16"),
            "--mmlu-batch",
            cfg.get("FDEC_TASK_MMLU_BATCH", "8"),
            "--kv-scale",
            kv_scale(cfg),
            "--out",
            str(out),
        ]
        if cfg.get("FDEC_DEVICE", "cuda:0") == "cpu":
            cmd += ["--dtype", "float32"]
        subprocess.run(cmd, check=True)


def fmt_p(p):
    """A p-value with two decimals; three when two would print 0.00."""
    if p < 0.0005:
        return "<0.001"
    return f"{p:.3f}" if p < 0.005 else f"{p:.2f}"


def task_report(path):
    """Markdown lines for one task result file, and whether every cell
    passes the task rule."""
    r = json.loads(path.read_text())
    ok = True
    lines = [
        "| task | cell | accuracy | difference | 95% interval | lost | gained |"
        " p | passes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for task, per in r["summary"].items():
        tests = max(1, len(per) - 1)
        for cell, c in per.items():
            if cell == "base":
                lines.append(
                    f"| {task} | base | {100 * c['accuracy']:.2f}% | reference |"
                    f" | | | | |"
                )
                continue
            good = not (c["diff_vs_base"] < 0 and c["mcnemar_p"] * tests < TASK_ALPHA)
            ok = ok and good
            lines.append(
                f"| {task} | {cell} | {100 * c['accuracy']:.2f}% | "
                f"{100 * c['diff_vs_base']:+.2f} | "
                f"{100 * c['ci95'][0]:+.2f} to {100 * c['ci95'][1]:+.2f} | "
                f"{c['lost']} | {c['gained']} | {fmt_p(c['mcnemar_p'])} | "
                f"{'yes' if good else 'NO'} |"
            )
        for cell, x in r.get("nonadditivity", {}).get(task, {}).items():
            notes.append(
                f"{task}, {cell}: accuracy change minus the sum of its bricks "
                f"alone {100 * x['value']:+.2f} points "
                f"({100 * x['ci95'][0]:+.2f} to {100 * x['ci95'][1]:+.2f})."
            )
    return lines + [""] + notes + ([""] if notes else []), ok


def report(cfg):
    lines = ["# fdec composition matrix", ""]
    ok = True
    for model in cfg["FDEC_MODELS"].split():
        p = result_path(cfg, model)
        if not p.exists():
            lines += [f"## {model}", "", "No result.", ""]
            ok = False
            continue
        r = json.loads(p.read_text())
        ok = ok and r["all_compose"]
        lines += [
            f"## {model}",
            "",
            f"Baseline perplexity {r['baseline_ppl']:.3f} on "
            f"{r['eval']['tokens']} held-out tokens; certificate calibrated on "
            f"{r['calibration']['tokens']} tokens, rank {r['cert']['rank']}.",
            "",
            "| cell | perplexity | top-1 agreement | KL | certificate exact |"
            " head bytes read | decode bytes saved | composes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, c in r["cells"].items():
            if not c["ran"]:
                lines.append(f"| {name} | failed: {c['error']} | | | | | | no |")
                continue
            cm = c.get("cert_matched")
            lines.append(
                f"| {name} | {c['ppl']:.3f} | {c['top1_agreement']:.3f} | "
                f"{c['mean_kl']:.4f} | "
                f"{'yes' if c['cert_exact'] else 'n/a' if cm is None else 'NO'} | "
                f"{'%.3f' % cm['total_byteratio'] if cm else 'n/a'} | "
                f"{c['bytes_saved_pct']:.1f}% | "
                f"{'yes' if c['composes'] else 'NO'} |"
            )
        lines.append("")
        # One interaction term per set of bricks that change the logits.
        seen = set()
        for name, c in r["cells"].items():
            ci = c.get("interaction_ci95")
            key = tuple(b for b in c.get("bricks", []) if b != "cert")
            if not ci or key in seen:
                continue
            seen.add(key)
            lines += [
                f"Interaction term of {'+'.join(key)}: "
                f"{c['interaction_dlogppl']:+.4f} in log-perplexity, 95% "
                f"interval {ci[0]:+.4f} to {ci[1]:+.4f}; the limit for the "
                f"upper end is {r['interaction_tol']}.",
                "",
            ]
        tp = task_path(cfg, model)
        if tp.exists():
            more, good = task_report(tp)
            lines += [f"### Task accuracy, {model}", ""] + more
            ok = ok and good
    lines.append(f"All cells compose: {'yes' if ok else 'no'}.")
    out = Path(cfg["FDEC_ROOT"]) / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".config")
    ap.add_argument("step", choices=["run", "tasks", "report"])
    args = ap.parse_args()
    cfg = read_config(args.config)
    if args.step == "run":
        run(cfg)
    elif args.step == "tasks":
        tasks(cfg)
    else:
        sys.exit(report(cfg))


if __name__ == "__main__":
    main()
