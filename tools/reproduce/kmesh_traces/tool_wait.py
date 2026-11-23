# SPDX-License-Identifier: MIT
"""Measure how long agents wait on tool calls in the FreeInference trace.

Reads each week's released tool_calls table. A call's latency_ms is the
release's gap from the end of the model response that made the call to
the start of the session's next LLM request, so it holds the tool's run
time plus the harness time around it. Calls in one response share that
gap. Calls without a next request have no gap and are counted apart.
Reports, per week, the share of tool-result tokens whose call waited at
least each threshold, and the token share and median wait per tool
category. Result tokens are the release's o200k_base counts, not the
tokens of any served model. No model, GPU or storage work.
"""

import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq

THRESHOLDS_MS = (250, 1000, 5000)
TOP_CATEGORIES = 8


def week_report(path):
    t = pq.read_table(
        path, columns=["category", "result_tokens", "latency_ms"]
    ).to_pandas()
    no_gap = t[t.latency_ms.isna() | t.result_tokens.isna()]
    t = t.dropna(subset=["latency_ms", "result_tokens"])
    total = float(t.result_tokens.sum())
    report = {
        "calls": int(len(t)),
        "calls_without_gap": int(len(no_gap)),
        "result_tokens": int(total),
        "wait_ms": {
            f"p{p}": float(np.percentile(t.latency_ms, p)) for p in (10, 50, 90)
        },
        "result_token_share_waiting_at_least_ms": {
            str(th): float(t[t.latency_ms >= th].result_tokens.sum() / total)
            for th in THRESHOLDS_MS
        },
    }
    g = t.groupby("category").agg(
        calls=("result_tokens", "size"),
        tokens=("result_tokens", "sum"),
        median_result_tokens=("result_tokens", "median"),
        median_wait_ms=("latency_ms", "median"),
    )
    g = g.sort_values("tokens", ascending=False).head(TOP_CATEGORIES)
    report["categories"] = {
        name: {
            "calls": int(row.calls),
            "token_share": float(row.tokens / total),
            "median_result_tokens": float(row.median_result_tokens),
            "median_wait_ms": float(row.median_wait_ms),
        }
        for name, row in g.iterrows()
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True, help="layout of week directories")
    ap.add_argument("--weeks", nargs="+", required=True, help="e.g. 2026-07-12")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report = {}
    for week in args.weeks:
        matches = [d for d in os.listdir(args.release) if d.startswith(week)]
        if len(matches) != 1:
            raise SystemExit(f"{week}: expected one week directory, found {matches}")
        path = os.path.join(args.release, matches[0], "tool_calls.parquet")
        report[matches[0]] = week_report(path)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    for week, r in report.items():
        shares = r["result_token_share_waiting_at_least_ms"]
        print(
            week,
            f"calls={r['calls']}",
            f"median_wait_ms={r['wait_ms']['p50']:.0f}",
            " ".join(f">={k}ms:{v:.3f}" for k, v in shares.items()),
        )


if __name__ == "__main__":
    main()
