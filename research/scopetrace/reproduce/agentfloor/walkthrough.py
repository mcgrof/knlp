#!/usr/bin/env python3
"""Read one agent run out loud, or count how a model gets tool calling wrong.

The point of this is to see what actually happens when an open-weight model
uses a tool, rather than reading a pass rate and inferring it. A run record
holds the whole exchange: what the model emitted, whether the arguments
validated, whether the tool it named even exists, what the environment sent
back, and what the model did about it.

Two modes.

`walk` narrates a single run turn by turn. It is the one to read first, and
reading three of them teaches more about tool calling than any table.

`taxonomy` counts failure kinds across many runs and breaks them down by model.
That is where the shape of the problem shows up: small models mostly fail before
the call is ever dispatched, on schema and on inventing tools that do not exist,
while larger ones get the call out and then fail on what they did with the
answer.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Iterator

RUN_SUFFIX = ".json"
SCORE_SUFFIX = ".score.json"


def iter_runs(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield run records, skipping the sidecar score files."""
    for p in sorted(root.rglob("*" + RUN_SUFFIX)):
        if not p.name.endswith(SCORE_SUFFIX):
            yield p


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def score_for(path: pathlib.Path) -> dict[str, Any] | None:
    """The sidecar verdict for a run, when it was scored."""
    sc = path.with_suffix("").with_suffix(".score.json")
    if sc.exists():
        return json.loads(sc.read_text())
    return None


def brief(value: Any, width: int = 88) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def call_verdict(call: dict[str, Any]) -> str:
    """Say in one phrase what went wrong with a call, before its result."""
    if call.get("is_hallucinated"):
        return "INVENTED A TOOL THAT DOES NOT EXIST"
    if call.get("is_malformed"):
        return "MALFORMED ARGUMENTS"
    if call.get("is_dispatch_error"):
        return "DISPATCH ERROR"
    if not call.get("valid_schema", True):
        return "ARGUMENTS FAILED THE SCHEMA"
    return "well formed"


def walk(path: pathlib.Path) -> int:
    """Narrate one run.

    The model's own call ids and the harness's do not match, so calls are
    paired by order rather than by id. What the model emitted comes from the
    assistant message, including the raw argument string it produced and
    whether that string parsed; what happened to the call comes from the log.
    """
    run = load(path)
    log = run.get("call_log", [])
    variant = path.stem.split("__")[1] if "__" in path.stem else "?"

    print("=" * 78)
    print(f"{run['model']}   task {run['task_id']}   tier {run['level']}   "
          f"variant {variant}")
    print("=" * 78)

    turn = 0
    emitted = 0   # index into call_log, in emission order
    consumed = 0  # index into call_log, in result order

    for msg in run.get("messages", []):
        role = msg.get("role")

        if role == "user":
            print("\nTHE TASK")
            for line in msg.get("text", "").strip().splitlines():
                print(f"    {line}")

        elif role == "assistant":
            tcs = msg.get("tool_calls") or []
            if not tcs:
                if msg.get("text"):
                    print("\nMODEL ANSWERED IN PROSE, no tool call")
                    print(f"    {brief(msg['text'])}")
                continue
            turn += 1
            print(f"\nTURN {turn} — the model asks for {len(tcs)} tool call(s)")
            for tc in tcs:
                rec = log[emitted] if emitted < len(log) else {}
                emitted += 1
                name = tc.get("name") or rec.get("tool_name", "?")
                print(f"    it emitted: {name}")
                raw = tc.get("raw_arguments")
                if raw is not None:
                    print(f"      as text: {brief(raw, 64)}")
                if tc.get("parse_error"):
                    print(f"      THAT TEXT DID NOT PARSE: {brief(tc['parse_error'], 52)}")
                else:
                    print(f"      parsed to: {brief(tc.get('arguments', {}), 62)}")
                print(f"      verdict: {call_verdict(rec)}")

        elif role == "tool":
            rec = log[consumed] if consumed < len(log) else {}
            consumed += 1
            res = rec.get("response") or msg.get("tool_result") or {}
            print(f"    the environment replies: {res.get('status', '?')}")
            err = res.get("error") or {}
            if err:
                flag = "recoverable" if err.get("recoverable") else "fatal"
                print(f"      {err.get('type')}: {brief(err.get('message'), 62)}"
                      f"  ({flag})")
            elif res.get("result") is not None:
                print(f"      {brief(res['result'], 66)}")

    print(f"\nHOW IT ENDED: {run.get('termination')}")
    if run.get("termination_detail"):
        print(f"    {run['termination_detail']}")
    if run.get("final_text"):
        print(f"    final answer: {brief(run['final_text'], 66)}")
    print(f"    {run.get('total_turns')} turns, {len(log)} calls, "
          f"{run.get('total_latency_ms')} ms")

    sc = score_for(path)
    if sc is not None:
        print(f"    scored: {'PASS' if sc.get('tcr_pass') else 'FAIL'}"
              f" — {brief(sc.get('summary', ''), 58)}")
    return 0


def taxonomy(root: pathlib.Path) -> int:
    """Count how calls go wrong, per model."""
    per: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    runs_per: collections.Counter = collections.Counter()

    for path in iter_runs(root):
        run = load(path)
        model = run.get("model", "?")
        runs_per[model] += 1
        log = run.get("call_log", [])
        per[model]["calls"] += len(log)
        if not log:
            per[model]["runs_with_no_call_at_all"] += 1
        for c in log:
            if c.get("is_hallucinated"):
                per[model]["invented_tool"] += 1
            elif c.get("is_malformed") or not c.get("valid_schema", True):
                per[model]["malformed_args"] += 1
            elif c.get("is_dispatch_error"):
                per[model]["dispatch_error"] += 1
            if (c.get("response") or {}).get("status") == "error":
                per[model]["env_returned_error"] += 1

    if not per:
        print(f"no run records under {root}", file=sys.stderr)
        return 2

    cols = ("calls", "invented_tool", "malformed_args", "env_returned_error",
            "runs_with_no_call_at_all")
    print(f"{'model':22s} {'runs':>5s} " + " ".join(f"{c:>22s}" for c in cols))
    print("-" * (28 + 23 * len(cols)))
    for model in sorted(per):
        row = per[model]
        cells = []
        for c in cols:
            n = row[c]
            if c == "calls":
                cells.append(f"{n:>22d}")
            elif c == "runs_with_no_call_at_all":
                pct = 100 * n / runs_per[model]
                cells.append(f"{n:>15d} ({pct:3.0f}%)")
            else:
                pct = 100 * n / row["calls"] if row["calls"] else 0.0
                cells.append(f"{n:>15d} ({pct:3.0f}%)")
        print(f"{model:22s} {runs_per[model]:5d} " + " ".join(cells))

    print()
    print("invented_tool and malformed_args are counted against calls emitted, and")
    print("both are decided before the call is dispatched, so they are failures of")
    print("protocol rather than of the task. runs_with_no_call_at_all is counted")
    print("against runs: a model that answers in prose when a tool was required")
    print("never entered the protocol in the first place.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("walk", help="narrate one run record")
    w.add_argument("run", type=pathlib.Path)

    t = sub.add_parser("taxonomy", help="count call failures per model")
    t.add_argument("results_dir", type=pathlib.Path)

    args = ap.parse_args()
    if args.cmd == "walk":
        return walk(args.run)
    return taxonomy(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
