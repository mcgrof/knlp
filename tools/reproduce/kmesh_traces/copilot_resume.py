# SPDX-License-Identifier: MIT
"""Measure user-turn resumes in the public GitHub Copilot coding agent traces.

Streams the released gzip JSONL shards for the given days. An LLM call's
timestamp is read as the END of the call, so it runs from timestamp minus
duration_ms to timestamp. Consecutive calls fit that reading in 99.94% of
81,519 checked pairs and the start reading in 75%; the release's own
trace_metrics.py uses the start reading. Tool-batch timestamps do not agree
with the call clock (a batch's timestamp is usually earlier than the end of
the call it names), so turn times use LLM calls only. For each session it
orders turns by their first call's start. At every turn boundary it records the idle gap since the previous turn's
last end, the first LLM call's prompt and cached tokens, whether its model
differs from the previous call's, and whether the prompt shrank by more than
30% after a prompt above 10,000 tokens (the release's compaction heuristic).
It also records long gaps inside turns and the peak number of overlapping
LLM calls per turn. Cached tokens are client labels from the deployment,
not a local cache hit rate. LLM call timestamps are whole seconds in this
release, so a call counts as overlapping another only beyond that
1 s resolution. No model, GPU or storage work.
"""

import argparse
import glob
import gzip
import json
import os
from datetime import datetime

import numpy as np

LONG_PROMPT = 32768
LONG_IDLE_S = 60.0


def ts(s):
    if not s:
        return None
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = head + "." + frac[:6]
    return datetime.fromisoformat(s).timestamp()


def pct(values, ps=(10, 50, 90, 99)):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    out = {"n": int(a.size), "mean": float(a.mean())}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def peak_overlap(intervals):
    ev = []
    for a, b in intervals:
        ev.append((a, 1))
        ev.append((b, -1))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data", required=True, help="directory holding date=YYYY-MM-DD folders"
    )
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sessions = turns_total = calls_total = 0
    prompt_all = uncached_all = 0
    b_idle, b_prompt, b_cached_frac = [], [], []
    b_switch = b_compact = 0
    past_cliff = {"n": 0, "prompt": 0, "uncached": 0}
    target = {
        "n": 0,
        "prompt": 0,
        "uncached": 0,
        "idle": [],
        "prompt_list": [],
        "cached_frac": [],
    }
    in_turn_long = {"n": 0, "prompt": 0, "uncached": 0}
    buckets = {}
    edges = [0, 10, 60, 300, 1800, 3600, 86400, float("inf")]
    labels = ["<10s", "10-60s", "1-5min", "5-30min", "30-60min", "1-24h", ">24h"]
    peaks = []
    for day in args.days:
        for path in sorted(
            glob.glob(os.path.join(args.data, f"date={day}", "shard-*.jsonl.gz"))
        ):
            with gzip.open(path, "rt") as f:
                for line in f:
                    d = json.loads(line)
                    sessions += 1
                    trs = []
                    for t in d["turns"]:
                        calls = []
                        for c in t["llm_calls"]:
                            s0 = ts(c.get("timestamp"))
                            if s0 is None:
                                continue
                            tok = c.get("tokens") or {}
                            calls.append(
                                (
                                    s0 - float(c.get("duration_ms") or 0.0) / 1000.0,
                                    s0,
                                    int(tok.get("prompt") or 0),
                                    int(tok.get("cached") or 0),
                                    c.get("model"),
                                )
                            )
                        ends = [e for _, e, _, _, _ in calls]
                        starts = [s for s, _, _, _, _ in calls]
                        if not calls:
                            continue
                        calls.sort()
                        trs.append((min(starts), max(ends), calls))
                        peaks.append(
                            peak_overlap(
                                # calls overlap only beyond the 1 s timestamp resolution
                                [(a, max(a, b - 1.0)) for a, b, _, _, _ in calls]
                            )
                        )
                    trs.sort(key=lambda x: x[0])
                    prev_call = None
                    prev_end = None
                    for i, (start, end, calls) in enumerate(trs):
                        turns_total += 1
                        for j, (s0, e0, p, c, m) in enumerate(calls):
                            calls_total += 1
                            prompt_all += p
                            uncached_all += max(0, p - c)
                            if j > 0:
                                gap = s0 - calls[j - 1][1]
                                if gap >= LONG_IDLE_S and p >= LONG_PROMPT:
                                    in_turn_long["n"] += 1
                                    in_turn_long["prompt"] += p
                                    in_turn_long["uncached"] += max(0, p - c)
                        first = calls[0]
                        if i > 0 and prev_call is not None:
                            idle = max(0.0, start - prev_end)
                            p, c, m = first[2], first[3], first[4]
                            switched = m != prev_call[4]
                            compacted = prev_call[2] > 10000 and p < 0.7 * prev_call[2]
                            b_idle.append(idle)
                            b_prompt.append(p)
                            b_switch += switched
                            b_compact += compacted
                            frac = c / p if p else 0.0
                            b_cached_frac.append(frac)
                            lab = next(
                                lb
                                for lb, lo, hi in zip(labels, edges, edges[1:])
                                if lo <= idle < hi
                            )
                            bk = buckets.setdefault(
                                lab,
                                {
                                    "n": 0,
                                    "same_model": [],
                                    "switched": [],
                                    "prompt": [],
                                },
                            )
                            bk["n"] += 1
                            bk["prompt"].append(p)
                            (bk["switched"] if switched else bk["same_model"]).append(
                                frac
                            )
                            if (
                                idle >= LONG_IDLE_S
                                and p >= LONG_PROMPT
                                and not switched
                                and not compacted
                            ):
                                target["n"] += 1
                                target["prompt"] += p
                                target["uncached"] += max(0, p - c)
                                target["idle"].append(idle)
                                target["prompt_list"].append(p)
                                target["cached_frac"].append(frac)
                                if idle >= 300.0:
                                    past_cliff["n"] += 1
                                    past_cliff["prompt"] += p
                                    past_cliff["uncached"] += max(0, p - c)
                        prev_call = calls[-1]
                        prev_end = end
    report = {
        "days": args.days,
        "sessions": sessions,
        "turns": turns_total,
        "llm_calls": calls_total,
        "prompt_tokens": prompt_all,
        "uncached_prompt_tokens": uncached_all,
        "turn_boundaries": {
            "n": len(b_idle),
            "idle_s": pct(b_idle),
            "prompt_tokens": pct(b_prompt),
            "cached_fraction": pct(b_cached_frac),
            "model_switched": b_switch,
            "compacted": b_compact,
            "by_idle_bucket": {
                k: {
                    "n": v["n"],
                    "prompt_tokens": pct(v["prompt"]),
                    "cached_fraction_same_model": pct(v["same_model"]),
                    "cached_fraction_switched": pct(v["switched"]),
                }
                for k, v in buckets.items()
            },
        },
        "store_target": {
            "definition": f"turn boundary, same model, not compacted, idle >= {LONG_IDLE_S:.0f} s, prompt >= {LONG_PROMPT}",
            "n": target["n"],
            "share_of_boundaries": target["n"] / len(b_idle) if b_idle else None,
            "share_of_calls": target["n"] / calls_total if calls_total else None,
            "share_of_prompt_tokens": (
                target["prompt"] / prompt_all if prompt_all else None
            ),
            "share_of_uncached_prompt_tokens": (
                target["uncached"] / uncached_all if uncached_all else None
            ),
            "idle_s": pct(target["idle"]),
            "prompt_tokens": pct(target["prompt_list"]),
            "cached_fraction": pct(target["cached_frac"]),
        },
        "store_target_idle_ge_300s": {
            "n": past_cliff["n"],
            "share_of_prompt_tokens": (
                past_cliff["prompt"] / prompt_all if prompt_all else None
            ),
            "share_of_uncached_prompt_tokens": (
                past_cliff["uncached"] / uncached_all if uncached_all else None
            ),
            "uncached_fraction_of_its_prompt": (
                past_cliff["uncached"] / past_cliff["prompt"]
                if past_cliff["prompt"]
                else None
            ),
        },
        "in_turn_long_gap": {
            "definition": f"call after a gap >= {LONG_IDLE_S:.0f} s inside a turn, prompt >= {LONG_PROMPT}",
            "n": in_turn_long["n"],
            "share_of_prompt_tokens": (
                in_turn_long["prompt"] / prompt_all if prompt_all else None
            ),
            "share_of_uncached_prompt_tokens": (
                in_turn_long["uncached"] / uncached_all if uncached_all else None
            ),
        },
        "peak_overlapping_llm_calls_per_turn": {
            "turns": len(peaks),
            "ge2": int(sum(1 for p in peaks if p >= 2)),
            "ge4": int(sum(1 for p in peaks if p >= 4)),
            "max": int(max(peaks) if peaks else 0),
        },
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("sessions", "turns", "llm_calls")}))
    print(json.dumps(report["store_target"], default=float)[:1500])
    print(json.dumps(report["peak_overlapping_llm_calls_per_turn"]))


if __name__ == "__main__":
    main()
