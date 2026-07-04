#!/usr/bin/env python3
"""Automated single-shot fault-localization run over the locked Defects4J set.

Runs five conditions per bug through one model backend, holding model settings
(reasoning effort, token budget) constant so only the PROMPT varies:

  standard         - reason, then answer
  standard_matched - reason extensively (length control for semiformal)
  semiformal       - premises + trace + formal conclusion (the paper's method)
  evidence_only    - cite file/line evidence per claim, no formal proof
  test_only        - failing-test name + candidate files, NO source (a leakage
                     / memorization baseline: high score here means the model
                     localizes without reading the code)

Every condition ends with the IDENTICAL ranked_locations schema, and the
grader (faultloc_grade) reads only that block, so semi-formal cannot win by
citing more lines. Reports Top-N All/Any per condition and the token cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from openai_client import OpenAIClient, call_cost  # noqa: E402
from faultloc_grade import parse_predictions, grade_bug, aggregate  # noqa: E402

_SCHEMA = (
    "Output ONE JSON object and nothing else (no markdown fences):\n"
    '{"ranked_locations": [{"file": "<path>", "line_start": N, '
    '"line_end": M}]}\n'
    "At most 5 regions, most likely first, using the exact file paths and "
    "line numbers shown."
)
_SEMI_SCHEMA = (
    "Output ONE JSON object, no fences:\n"
    '{"certificate": {"premises": [{"claim": "...", "file": "<path>", '
    '"line_start": N, "line_end": M}], "trace": "...", '
    '"formal_conclusion": "..."}, "ranked_locations": '
    '[{"file": "<path>", "line_start": N, "line_end": M}]}\n'
    "ranked_locations: at most 5, most likely first, exact paths/lines."
)

_HEAD = (
    "You are localizing a fault in Java code by STATIC reasoning only. You "
    "cannot run the code or its tests. Reason only from the source shown.\n\n"
    "Failing test(s): {tests}\n\n"
)
_SRC = (
    "The buggy source files are below; every line is prefixed with its line "
    "number as `<n>: `. Those numbers are authoritative -- cite them "
    "exactly.\n\n{source}\n\n"
)

CONDITIONS = [
    "standard",
    "standard_matched",
    "semiformal",
    "evidence_only",
    "test_only",
]

_TASK = {
    "standard": "TASK: Find where the fault is. " + _SCHEMA,
    "standard_matched": (
        "TASK: Find where the fault is. Reason step by step as thoroughly as "
        "you can: consider several candidate locations, trace how each relates "
        "to the failing test, and rule out the ones that do not fit, before "
        "answering. " + _SCHEMA
    ),
    "semiformal": (
        "TASK: Localize the fault with a semi-formal certificate. Build "
        "premises (each citing a file and line range you read), a trace from "
        "the failing test to the fault, and a formal conclusion. Then answer. "
        + _SEMI_SCHEMA
    ),
    "evidence_only": (
        "TASK: Find where the fault is. For each claim in your reasoning, cite "
        "the specific file and line range you read that supports it. Then "
        "answer. " + _SCHEMA
    ),
    "test_only": (
        "TASK: Based only on the failing test name(s) and the candidate file "
        "list, predict where the fault is. You are NOT shown the source. " + _SCHEMA
    ),
}


def _numbered(repo_dir, scope_files):
    blocks = []
    for rel in scope_files:
        with open(os.path.join(repo_dir, rel), errors="replace") as fh:
            lines = fh.read().splitlines()
        body = "\n".join(f"{i}: {ln}" for i, ln in enumerate(lines, 1))
        blocks.append(f"=== FILE: {rel} ===\n{body}")
    return "\n\n".join(blocks)


def build_prompt(bug, condition, repo_dir):
    tests = ", ".join(bug["payload"]["failing_tests"]) or "(unnamed)"
    head = _HEAD.format(tests=tests)
    if condition == "test_only":
        cand = "\n".join(f"- {f}" for f in bug["payload"]["scope_files"])
        return head + f"Candidate files:\n{cand}\n\n" + _TASK[condition]
    src = _SRC.format(source=_numbered(repo_dir, bug["payload"]["scope_files"]))
    return head + src + _TASK[condition]


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    # outermost object
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _one(bug, cond, repos_dir, client, raw_dir):
    tid = bug["task_id"]
    repo_dir = os.path.join(repos_dir, bug["repo"])
    gold = bug["gold"]["hunks"]
    prompt = build_prompt(bug, cond, repo_dir)
    res = client.complete(prompt)
    answer = _extract_json(res["text"])
    preds = parse_predictions(answer)
    flags = grade_bug(preds, gold)
    flags["task_id"] = tid
    with open(os.path.join(raw_dir, f"{tid}__{cond}.json"), "w") as fh:
        json.dump(
            {
                "task_id": tid,
                "condition": cond,
                "predictions": preds,
                "gold": gold,
                "flags": flags,
                "finish_reason": res["finish_reason"],
                "usage": res["usage"],
                "raw_text": res["text"],
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    return cond, flags, call_cost(res["usage"]), res["finish_reason"]


def run(manifest_rows, repos_dir, out_dir, client, conditions=CONDITIONS, workers=8):
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(out_dir, exist_ok=True)
    per_cond = {c: [] for c in conditions}
    total_cost, total_calls, truncated = 0.0, 0, 0
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    jobs = [(bug, cond) for bug in manifest_rows for cond in conditions]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_one, bug, cond, repos_dir, client, raw_dir) for bug, cond in jobs
        ]
        for f in futs:
            cond, flags, cost, finish = f.result()
            per_cond[cond].append(flags)
            total_cost += cost
            total_calls += 1
            if finish == "length":
                truncated += 1

    summary = {c: aggregate(per_cond[c]) for c in conditions}
    result = {
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "n_bugs": len(manifest_rows),
        "conditions": summary,
        "total_calls": total_calls,
        "truncated_calls": truncated,
        "est_cost_usd": round(total_cost, 2),
    }
    json.dump(
        result,
        open(os.path.join(out_dir, "results.json"), "w"),
        indent=2,
        sort_keys=True,
    )
    _write_report(result, out_dir)
    return result


def _write_report(result, out_dir):
    lines = ["# Fault localization: prompt-condition comparison", ""]
    lines.append(
        f"Model {result['model']} (reasoning={result['reasoning_effort']}), "
        f"{result['n_bugs']} locked Defects4J bugs, single-shot, "
        f"execution-free. Est. cost ${result['est_cost_usd']}, "
        f"{result['truncated_calls']}/{result['total_calls']} truncated."
    )
    lines.append("")
    lines.append(
        "| Condition | Top-1 Any | Top-3 Any | Top-5 Any | " "Top-1 All | Top-5 All |"
    )
    lines.append("|---|---|---|---|---|---|")
    for c in CONDITIONS:
        s = result["conditions"].get(c, {})

        def p(k):
            return f"{100 * s.get(k, 0):.1f}%"

        lines.append(
            f"| {c} | {p('top_1_any')} | {p('top_3_any')} | {p('top_5_any')} "
            f"| {p('top_1_all')} | {p('top_5_all')} |"
        )
    lines.append("")
    std = result["conditions"].get("standard", {})
    semi = result["conditions"].get("semiformal", {})
    for k in ("top_5_all", "top_5_any", "top_1_any"):
        if k in std and k in semi:
            d = 100 * (semi[k] - std[k])
            lines.append(f"- semiformal vs standard {k}: {d:+.1f} pp")
    to = result["conditions"].get("test_only", {})
    if to:
        lines.append(
            f"- test-only (leakage probe) Top-5 Any: "
            f"{100 * to.get('top_5_any', 0):.1f}% "
            "(high here means localization without reading code)"
        )
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.2-2025-12-11")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    client = OpenAIClient(model=args.model, reasoning_effort=args.reasoning)
    t0 = time.time()
    res = run(rows, args.repos, args.out, client)
    print(json.dumps(res["conditions"], indent=2, sort_keys=True))
    print(
        f"[faultloc_run] {res['total_calls']} calls, "
        f"${res['est_cost_usd']}, {time.time() - t0:.0f}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
