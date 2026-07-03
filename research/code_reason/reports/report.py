#!/usr/bin/env python3
"""Render a scored code-reason run into a markdown report.

Reads a run directory's metrics_summary.json plus the per-cell metrics and
provenance records and writes a report that answers the question the harness
exists to ask: does semi-formal certificate reasoning change accuracy, and at
what evidence-quality and token cost, relative to standard reasoning. The
report leads with the standard-vs-semiformal comparison per model, then the
per-task-type numbers, certificate evidence quality, cost, any addendum
dispositions, and a provenance footer pinning the config and git commit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load(path, default=None):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return default


def _pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "-"


def _walk_cells(run_dir):
    """Yield (task_id, model, mode, cell_dir) for every result cell."""
    tasks_root = os.path.join(run_dir, "tasks")
    if not os.path.isdir(tasks_root):
        return
    for task_id in sorted(os.listdir(tasks_root)):
        tdir = os.path.join(tasks_root, task_id)
        for model in sorted(os.listdir(tdir)):
            mdir = os.path.join(tdir, model)
            for mode in sorted(os.listdir(mdir)):
                yield task_id, model, mode, os.path.join(mdir, mode)


def _accuracy_of(block):
    """A single headline accuracy for a metrics block, if present."""
    if "patch_equiv" in block:
        return block["patch_equiv"].get("accuracy")
    if "fault_localization" in block:
        return block["fault_localization"].get("top_1_any")
    if "code_qa" in block:
        return block["code_qa"].get("rubric_score")
    return None


def _cert_quality(run_dir):
    """Aggregate certificate evidence quality across semiformal cells."""
    vals, unsupported, missing, cells = [], 0, 0, 0
    for _, _, mode, cell in _walk_cells(run_dir):
        m = _load(os.path.join(cell, "metrics.json"))
        cq = (m or {}).get("certificate_quality")
        if cq:
            cells += 1
            vals.append(cq.get("cited_evidence_validity", 0.0))
            unsupported += cq.get("unsupported_claim_count", 0)
            missing += cq.get("missing_trace_count", 0)
    if not cells:
        return None
    return {
        "cells": cells,
        "mean_evidence_validity": sum(vals) / len(vals),
        "unsupported_claims": unsupported,
        "missing_traces": missing,
    }


def _addendum_dispositions(run_dir):
    counts = {}
    for _, _, _, cell in _walk_cells(run_dir):
        adir = os.path.join(cell, "addendums")
        if not os.path.isdir(adir):
            continue
        for fn in sorted(os.listdir(adir)):
            rec = _load(os.path.join(adir, fn), {})
            name = rec.get("name", fn)
            d = counts.setdefault(name, {"enabled": 0, "disabled": 0})
            d["enabled" if rec.get("enabled") else "disabled"] += 1
    return counts


def build_report(run_dir):
    summary = _load(os.path.join(run_dir, "metrics_summary.json"), {})
    cfg = _load(os.path.join(run_dir, "config.json"), {})
    git = _load(os.path.join(run_dir, "git-state.json"), {})
    env = _load(os.path.join(run_dir, "environment.json"), {})

    lines = ["# Code Reasoning run report", ""]
    lines.append(
        "This run measures whether semi-formal certificate reasoning changes "
        "accuracy against standard reasoning, and at what evidence quality and "
        "token cost."
    )
    lines.append("")

    # standard vs semi-formal, per model
    models = sorted({k.split("/")[0] for k in summary})
    lines.append("## Standard vs semi-formal accuracy")
    lines.append("")
    lines.append("| Model | Standard | Semi-formal | Delta |")
    lines.append("|---|---|---|---|")
    for model in models:
        std = _accuracy_of(summary.get(f"{model}/standard", {}))
        semi = _accuracy_of(summary.get(f"{model}/semiformal", {}))
        delta = (
            f"{100 * (semi - std):+.1f} pp"
            if isinstance(std, (int, float)) and isinstance(semi, (int, float))
            else "-"
        )
        lines.append(f"| {model} | {_pct(std)} | {_pct(semi)} | {delta} |")
    lines.append("")

    # per (model, mode) task-type detail
    lines.append("## Per task type")
    lines.append("")
    for key in sorted(summary):
        block = summary[key]
        lines.append(f"### {key}")
        if "patch_equiv" in block:
            pe = block["patch_equiv"]
            lines.append(
                f"- patch equivalence: accuracy {_pct(pe.get('accuracy'))}, "
                f"false-equivalent {_pct(pe.get('false_equivalent_rate'))}, "
                f"false-non-equivalent {_pct(pe.get('false_non_equivalent_rate'))}"
            )
        if "fault_localization" in block:
            fl = block["fault_localization"]
            lines.append(
                f"- fault localization: top-1 {_pct(fl.get('top_1_any'))}, "
                f"top-3 {_pct(fl.get('top_3_any'))}, "
                f"top-5 {_pct(fl.get('top_5_any'))}"
            )
        if "code_qa" in block:
            lines.append(
                f"- code QA: rubric {_pct(block['code_qa'].get('rubric_score'))}"
            )
        cost = block.get("cost", {})
        lines.append(
            f"- cost: {cost.get('input_tokens', 0)} in / "
            f"{cost.get('output_tokens', 0)} out tokens"
        )
        lines.append("")

    # certificate quality
    cq = _cert_quality(run_dir)
    if cq:
        lines.append("## Certificate evidence quality")
        lines.append("")
        lines.append(
            f"Across {cq['cells']} semi-formal cells, mean cited-evidence "
            f"validity is {_pct(cq['mean_evidence_validity'])} "
            f"({cq['unsupported_claims']} unsupported claims, "
            f"{cq['missing_traces']} missing traces). Evidence validity is "
            "checked by re-reading every cited file and line against the repo."
        )
        lines.append("")

    # addendum dispositions
    disp = _addendum_dispositions(run_dir)
    if disp:
        lines.append("## Addendum dispositions")
        lines.append("")
        lines.append("| Addendum | Enabled | Disabled |")
        lines.append("|---|---|---|")
        for name in sorted(disp):
            d = disp[name]
            lines.append(f"| {name} | {d['enabled']} | {d['disabled']} |")
        lines.append("")

    # provenance
    lines.append("## Provenance")
    lines.append("")
    src = cfg.get("source", "-")
    aug = cfg.get("flags", {}).get("CONFIG_CODE_REASON_AUGMENTED", "-")
    lines.append(f"- config source: `{src}` (augmented={aug})")
    lines.append(
        f"- git: {git.get('commit', '-')[:12]} on {git.get('branch', '-')}"
        f"{' (dirty)' if git.get('dirty') else ''}"
    )
    lines.append(f"- python {env.get('python', '-')} on {env.get('platform', '-')}")
    lines.append("")
    return "\n".join(lines)


def write_report(run_dir, out=None):
    md = build_report(run_dir)
    out = out or os.path.join(run_dir, "report.md")
    with open(out, "w") as fh:
        fh.write(md if md.endswith("\n") else md + "\n")
    return out


def _self_test():
    import tempfile

    sys.path.insert(0, os.path.join(_ROOT, "datasets"))
    sys.path.insert(0, os.path.join(_ROOT, "runners"))
    from manifest_builder import load_dataset, build_manifest
    from runner import run_manifest
    from metrics import score_run

    ds_dir = os.path.join(_ROOT, "datasets", "fixtures", "smoke")
    ds = load_dataset(ds_dir)
    flags = {
        "CONFIG_CODE_REASON_TASK_PATCH_EQUIV": True,
        "CONFIG_CODE_REASON_TASK_FAULT_LOCALIZATION": True,
        "CONFIG_CODE_REASON_TASK_CODE_QA": True,
        "CONFIG_CODE_REASON_MODEL_OPUS_4_5": True,
        "CONFIG_CODE_REASON_MODE_STANDARD": True,
        "CONFIG_CODE_REASON_MODE_SEMIFORMAL": True,
        "CONFIG_CODE_REASON_AUGMENTED": True,
        "CONFIG_CODE_REASON_ADDENDUMS": True,
        "CONFIG_CODE_REASON_ADDENDUM_AST_RUNTIME": True,
        "CONFIG_CODE_REASON_ADDENDUM_SEMANTIC_REWRITE": True,
        "CONFIG_CODE_REASON_ADDENDUM_COCCINELLE": True,
        "CONFIG_CODE_REASON_ADDENDUM_COCCINELLE_AUTO_DISABLE_IF_UNSUPPORTED": True,
        "CONFIG_CODE_REASON_RECORD_ADDENDUMS": True,
    }
    rows = build_manifest(ds, flags)
    rd = tempfile.mkdtemp()
    run_manifest(rows, flags, rd, ds_dir, repo_root=_ROOT)
    score_run(rd, ds_dir, rows)
    out = write_report(rd)
    md = open(out).read()
    for section in (
        "# Code Reasoning run report",
        "## Standard vs semi-formal accuracy",
        "## Per task type",
        "## Certificate evidence quality",
        "## Addendum dispositions",
        "## Provenance",
        "claude-opus-4-5",
    ):
        assert section in md, f"missing section: {section}"
    # addendum artifacts were actually written and summarized
    assert "ast_runtime" in md, "addendum dispositions not summarized"
    print(f"[report] self-test PASS: full report at {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run-dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.run_dir:
        ap.error("--run-dir required (or --self-test)")
    out = write_report(args.run_dir, args.out)
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
