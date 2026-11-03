# SPDX-License-Identifier: MIT
"""Measure long-idle resumes of the main agent in the Weka Claude Code corpus.

For each main-agent request after the first, the gap is its start minus the
previous main request's end (start + api_time). A resume is a request after
a gap of at least 60 s with at least 32K input tokens. A resume is a
subagent wait when a subagent was spawned inside the gap, otherwise a user
or tool wait. Reports counts and the share of all main-agent input tokens,
for comparison with the subagent launch spans. Weka records no cache labels.
No model, GPU or storage work.
"""

import argparse
import json

LONG_PROMPT = 32768
LONG_IDLE_S = 60.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    total_in = 0
    child_first_in = 0
    out = {k: {"n": 0, "tokens": 0} for k in ("subagent_wait", "user_or_tool_wait")}
    with open(args.traces) as f:
        for line in f:
            d = json.loads(line)
            reqs = d["requests"]
            mains = sorted(
                (r for r in reqs if r.get("type") != "subagent"), key=lambda r: r["t"]
            )
            spawns = sorted(r["t"] for r in reqs if r.get("type") == "subagent")
            for r in reqs:
                if r.get("type") == "subagent" and r.get("requests"):
                    child_first_in += int(r["requests"][0].get("in") or 0)
            for i, r in enumerate(mains):
                total_in += int(r.get("in") or 0)
                if i == 0:
                    continue
                p = mains[i - 1]
                end = p["t"] + float(p.get("api_time") or 0.0)
                gap = r["t"] - end
                if gap < LONG_IDLE_S or int(r.get("in") or 0) < LONG_PROMPT:
                    continue
                kind = (
                    "subagent_wait"
                    if any(p["t"] <= s <= r["t"] for s in spawns)
                    else "user_or_tool_wait"
                )
                out[kind]["n"] += 1
                out[kind]["tokens"] += int(r.get("in") or 0)
    report = {
        "definition": f"main request after a gap >= {LONG_IDLE_S:.0f} s with input >= {LONG_PROMPT}",
        "main_input_tokens": total_in,
        "child_first_request_input_tokens": child_first_in,
        **{
            k: {**v, "share_of_main_input_tokens": v["tokens"] / total_in}
            for k, v in out.items()
        },
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
