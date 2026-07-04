#!/usr/bin/env python3
"""Gate harness for the synthetic C first-fault benchmark.

Before spending on the full four-condition certificate comparison, this runs
the design-review GATE: only `surface_only` and `standard` prompts, to prove
the task is (a) not guessable from the function surface and (b) hard enough for
GPT-5.5 xhigh to leave headroom. It builds fixed, condition-neutral prompt
files (the grader later reads ONLY the final verdict, so no condition can win
on verbosity) and scores model responses.

Two modes:
  build  -- write per-item prompt files for each condition plus a private
            manifest (gold labels, never shown to the model).
  grade  -- read the model's verdict JSON (a list of {task_id, bug_present,
            category, first_fault_line, final_value}) and report bug_present
            MCC/accuracy per condition, category/line/value secondaries, and
            the gate decision.

The model calls themselves run over the Codex MCP channel (GPT-5.5, xhigh
reasoning, read-only sandbox so nothing executes) and are driven by the
conductor, not this script -- MCP tools cannot be invoked by a subprocess.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

CATEGORIES = [
    "signed-overflow",
    "shift-UB",
    "OOB-read",
    "use-after-free",
    "double-free",
    "null-deref",
    "none",
]

_SCHEMA = (
    "Output ONE JSON object and nothing else (no prose, no code fences):\n"
    '{"bug_present": true|false, "category": "<one of '
    + ", ".join(CATEGORIES)
    + '>", "first_fault_line": [start, end] or null, '
    '"final_value": <the integer the program prints if it is clean, '
    "else null>}"
)

_STD_HEAD = (
    "You are analyzing a self-contained C program by STATIC reasoning only. "
    "You CANNOT compile or run it; reason from the source. The program's "
    "main() calls the function under test with ONE fixed, hard-coded input. "
    "For THAT input, decide whether execution reaches undefined behavior "
    "before the program returns.\n"
    "- If it does: bug_present=true, report the FIRST UB category and the "
    "line range where it happens, final_value=null.\n"
    "- If it does not: bug_present=false, category=none, first_fault_line="
    "null, and final_value = the integer the program prints on stdout.\n"
    "Every source line is prefixed with its number as `<n>: ` -- cite those "
    "numbers.\n\n"
)

_SURF_HEAD = (
    "You are shown ONLY the include lines, function signatures, and brace "
    "structure of a C program -- the function bodies have been removed. From "
    "this surface alone, predict whether the program reaches undefined "
    "behavior for its (unseen) input. You cannot compute the exact value; set "
    "final_value=null. Answer the schema as best you can from the surface.\n\n"
)


def _numbered(src):
    return "\n".join(f"{i}: {ln}" for i, ln in enumerate(src.splitlines(), 1))


def build(items_dir, out_dir, shuffle_seed=1):
    """Write prompt files under OPAQUE, shuffled ids.

    The item directory names encode the label (`...-ub`, `...-clean`) and family,
    and the a/b pairing encodes the buggy/safe pattern -- so the model must never
    see them. Each confirmed item is assigned an opaque `qNNN` handle in a
    deterministically shuffled order; the model sees only `qNNN.txt`, and the
    private manifest maps qid -> task_id + gold.
    """
    idx = json.load(open(os.path.join(items_dir, "index.json")))
    for cond in ("standard", "surface_only"):
        os.makedirs(os.path.join(out_dir, cond), exist_ok=True)
    confirmed = []
    for tid in idx["items"]:
        meta = json.load(open(os.path.join(items_dir, tid, "meta.json")))
        if meta.get("confirmed"):
            confirmed.append((tid, meta))
    order = list(range(len(confirmed)))
    random.Random(shuffle_seed).shuffle(order)
    manifest = []
    for qnum, orig_i in enumerate(order):
        tid, meta = confirmed[orig_i]
        qid = f"q{qnum:03d}"
        d = os.path.join(items_dir, tid)
        src = open(os.path.join(d, "prog.c")).read()
        surf = open(os.path.join(d, "surface.txt")).read()
        std = _STD_HEAD + _numbered(src) + "\n\n" + _SCHEMA + "\n"
        sur = _SURF_HEAD + surf + "\n" + _SCHEMA + "\n"
        open(os.path.join(out_dir, "standard", qid + ".txt"), "w").write(std)
        open(os.path.join(out_dir, "surface_only", qid + ".txt"), "w").write(sur)
        manifest.append(
            {
                "qid": qid,
                "task_id": tid,
                "family": meta["family"],
                "pair_id": meta.get("pair_id"),
                "expected": meta["expected"],
                "bug_present": meta["expected"] == "ub",
                "category": meta["category"],
                "first_fault_line": meta["first_fault_line"],
                "final_value": meta["final_value"],
            }
        )
    json.dump(
        {"items": manifest}, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2
    )
    json.dump(
        [r["qid"] for r in manifest],
        open(os.path.join(out_dir, "qids.json"), "w"),
    )
    return manifest


def _mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den else 0.0


def _score(preds, gold):
    """preds: qid -> verdict dict. gold: qid -> manifest row."""
    tp = tn = fp = fn = 0
    n = 0
    cat_hits = cat_tot = 0
    val_hits = val_tot = 0
    for tid, g in gold.items():
        if tid not in preds:
            continue
        n += 1
        p = preds[tid]
        pb = bool(p.get("bug_present"))
        gb = bool(g["bug_present"])
        if gb and pb:
            tp += 1
        elif gb and not pb:
            fn += 1
        elif not gb and pb:
            fp += 1
        else:
            tn += 1
        if gb:
            cat_tot += 1
            if str(p.get("category")) == g["category"]:
                cat_hits += 1
        else:
            val_tot += 1
            if p.get("final_value") is not None and str(p.get("final_value")) == str(
                g["final_value"]
            ):
                val_hits += 1
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "mcc": round(_mcc(tp, tn, fp, fn), 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "category_acc_on_ub": round(cat_hits / cat_tot, 4) if cat_tot else None,
        "final_value_acc_on_clean": round(val_hits / val_tot, 4) if val_tot else None,
    }


def grade(out_dir, responses):
    """responses: {condition: [verdict, ...]} where each verdict has task_id."""
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))["items"]
    gold = {r["qid"]: r for r in manifest}
    report = {"n_items": len(gold), "conditions": {}}
    for cond, verdicts in responses.items():
        preds = {v["qid"]: v for v in verdicts if "qid" in v}
        report["conditions"][cond] = _score(preds, gold)
    su = report["conditions"].get("surface_only", {})
    st = report["conditions"].get("standard", {})
    surf_mcc = abs(su.get("mcc", 1.0))
    std_acc = st.get("accuracy", 0.0)
    report["gate"] = {
        "surface_only_abs_mcc": round(surf_mcc, 4),
        "standard_accuracy": std_acc,
        "surface_leak_ok": surf_mcc < 0.10,
        "headroom_ok": 0.40 <= std_acc <= 0.70,
        "too_easy": std_acc > 0.80,
        "too_hard": std_acc < 0.25,
        "pass": (surf_mcc < 0.10) and (0.40 <= std_acc <= 0.70),
    }
    json.dump(report, open(os.path.join(out_dir, "gate_report.json"), "w"), indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--items", required=True)
    b.add_argument("--out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--out", required=True)
    g.add_argument(
        "--responses",
        required=True,
        help="json {condition: [verdicts]} of model answers",
    )
    args = ap.parse_args()
    if args.cmd == "build":
        m = build(args.items, args.out)
        print(f"[cgate] built prompts for {len(m)} confirmed items -> {args.out}")
    else:
        resp = json.load(open(args.responses))
        rep = grade(args.out, resp)
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
