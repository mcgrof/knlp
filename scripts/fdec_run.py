#!/usr/bin/env python3
"""Run the fdec composition matrix named by a Kconfig .config.

`run` executes scripts/fdec_compose.py once per configured model and skips a
model whose result file already exists. `report` reads the result files and
writes one markdown table per model plus a verdict line.
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


def run(cfg):
    for model in cfg["FDEC_MODELS"].split():
        out = result_path(cfg, model)
        if out.exists():
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
            "--out",
            str(out),
        ]
        if cfg.get("FDEC_DEVICE", "cuda:0") == "cpu":
            cmd += ["--dtype", "float32"]
        subprocess.run(cmd, check=True)


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
    lines.append(f"All cells compose: {'yes' if ok else 'no'}.")
    out = Path(cfg["FDEC_ROOT"]) / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".config")
    ap.add_argument("step", choices=["run", "report"])
    args = ap.parse_args()
    cfg = read_config(args.config)
    if args.step == "run":
        run(cfg)
    else:
        sys.exit(report(cfg))


if __name__ == "__main__":
    main()
