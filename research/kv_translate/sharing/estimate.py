#!/usr/bin/env python3
"""Quality and first-token time, estimated the way the workload was frozen.

Quality. A context's score is the mean correctness of its fixed questions.
The deficit of a route is the receiver's own score minus that route's score,
paired by context, averaged within each stratum of domain and length, and
combined with the frozen weights. It is signed: a route that does better
than the receiver has a negative deficit. Uncertainty comes from resampling
source groups within strata, keeping everything a group produced together.

Health. A context is unhealthy under a route when any of its answers is
empty, malformed, repeating or never ends. The excess over the receiver is
bounded by an exact binomial upper limit for the route less an exact lower
limit for the receiver. That bound stays honest when no failure is seen,
where a resampled interval would collapse to zero width.

Time. Percentiles are taken over individual requests, each weighted so that
the strata carry their frozen weights. The saving of one route over another
is resampled by source group, with every repeat of a group kept together.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys

LENGTH_WEIGHTS = {4096: 0.25, 16384: 0.5, 32768: 0.25}
DOMAIN_WEIGHTS = {"document": 0.5, "code": 0.5}
RESAMPLES = 10000
SEED = 20260928
BASELINE = "restore"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def arm(row):
    return row["route"] if not row.get("mapper") else f"{row['route']}:{row['mapper']}"


def stratum_weight(domain, length):
    return DOMAIN_WEIGHTS[domain] * LENGTH_WEIGHTS[int(length)]


def context_scores(scores):
    """{arm: {context: {score, unhealthy, group, stratum, queries}}}"""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in scores:
        if s["correct"] is None:
            raise ValueError(
                f"{s['query_id']} has no score; refusing to count it as wrong"
            )
        by[arm(s)][s["context_id"]].append(s)
    out = {}
    for a, ctxs in by.items():
        out[a] = {}
        for cid, rows in ctxs.items():
            out[a][cid] = {
                "score": sum(bool(r["correct"]) for r in rows) / len(rows),
                "unhealthy": any(r["health_events"] for r in rows),
                "group": rows[0]["source_group_id"],
                "stratum": (rows[0]["domain"], int(rows[0]["length_bucket"])),
                "queries": len(rows),
            }
    return out


def weighted(values_by_stratum):
    """Frozen-weight mean of stratum means, over the strata that are present."""
    num = den = 0.0
    for (domain, length), vals in values_by_stratum.items():
        if vals:
            w = stratum_weight(domain, length)
            num += w * sum(vals) / len(vals)
            den += w
    return num / den if den else float("nan")


def paired_deficit(base, other, *, resamples=RESAMPLES, seed=SEED):
    import numpy as np

    shared = sorted(set(base) & set(other))
    if set(base) != set(other):
        missing = sorted(set(base) ^ set(other))
        raise ValueError(
            f"the two routes do not cover the same contexts: {missing[:4]}"
        )
    strata = collections.defaultdict(list)
    for cid in shared:
        strata[base[cid]["stratum"]].append(cid)
    diff = {cid: base[cid]["score"] - other[cid]["score"] for cid in shared}
    point = weighted({s: [diff[c] for c in cids] for s, cids in strata.items()})
    rng = np.random.default_rng(seed)
    groups = {s: sorted({base[c]["group"] for c in cids}) for s, cids in strata.items()}
    members = collections.defaultdict(list)
    for cid in shared:
        members[(base[cid]["stratum"], base[cid]["group"])].append(cid)
    draws = np.empty(resamples)
    for i in range(resamples):
        vals = {}
        for s, gs in groups.items():
            pick = rng.integers(0, len(gs), size=len(gs))
            vals[s] = [diff[c] for j in pick for c in members[(s, gs[j])]]
        draws[i] = weighted(vals)
    per = {
        f"{d}/{n}": {
            "contexts": len(cids),
            "groups": len(groups[(d, n)]),
            "baseline": sum(base[c]["score"] for c in cids) / len(cids),
            "route": sum(other[c]["score"] for c in cids) / len(cids),
            "deficit": sum(diff[c] for c in cids) / len(cids),
        }
        for (d, n), cids in sorted(strata.items())
    }
    return {
        "contexts": len(shared),
        "groups": len({base[c]["group"] for c in shared}),
        "baseline_score": weighted(
            {s: [base[c]["score"] for c in cids] for s, cids in strata.items()}
        ),
        "route_score": weighted(
            {s: [other[c]["score"] for c in cids] for s, cids in strata.items()}
        ),
        "deficit": point,
        "deficit_upper_95_one_sided": float(np.quantile(draws, 0.95)),
        "deficit_interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "largest_stratum_deficit": max(v["deficit"] for v in per.values()),
        "strata": per,
        "contexts_better": sum(1 for c in shared if diff[c] < 0),
        "contexts_worse": sum(1 for c in shared if diff[c] > 0),
        "contexts_equal": sum(1 for c in shared if diff[c] == 0),
    }


def binomial_upper(k, n, alpha):
    """Exact upper limit for a proportion, by bisection on the binomial tail."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        tail = sum(
            math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(k + 1)
        )
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return hi


def binomial_lower(k, n, alpha):
    return 1.0 - binomial_upper(n - k, n, alpha)


def health_excess(base, other):
    shared = sorted(set(base) & set(other))
    n = len(shared)
    kb = sum(base[c]["unhealthy"] for c in shared)
    ko = sum(other[c]["unhealthy"] for c in shared)
    return {
        "contexts": n,
        "baseline_unhealthy": kb,
        "route_unhealthy": ko,
        "excess": (ko - kb) / n,
        # each side at 2.5 percent, so the difference holds at 95 together
        "excess_upper_95_conservative": binomial_upper(ko, n, 0.025)
        - binomial_lower(kb, n, 0.025),
        "method": "exact binomial upper limit for the route less exact lower limit for the receiver",
    }


def agreement(scores, generations, a, b):
    """How often two routes wrote the same thing, and scored the same."""
    text = {(g["query_id"], arm(g)): g["tokens"] for g in generations}
    right = {(s["query_id"], arm(s)): s["correct"] for s in scores}
    qs = sorted({q for q, x in text if x == a} & {q for q, x in text if x == b})
    same = sum(text[(q, a)] == text[(q, b)] for q in qs)
    first = sum(text[(q, a)][:1] == text[(q, b)][:1] for q in qs)
    verdict = sum(right.get((q, a)) == right.get((q, b)) for q in qs)
    return {
        "queries": len(qs),
        "identical_output": same,
        "identical_first_token": first,
        "same_verdict": verdict,
    }


def weighted_quantile(pairs, q):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * total - 1e-12:
            return v
    return pairs[-1][0]


def request_weights(rows):
    per = collections.Counter((r["domain"], int(r["length_bucket"])) for r in rows)
    return [
        stratum_weight(r["domain"], r["length_bucket"])
        / per[(r["domain"], int(r["length_bucket"]))]
        for r in rows
    ]


def latency(timings, state, *, lengths=None, resamples=2000, seed=SEED):
    """Percentiles per route and the saving of each stored route over recomputing."""
    import numpy as np

    want = {"direct": ("direct", "none"), "page_cache": ("page_cache", "none")}[state]
    rows = [
        t
        for t in timings
        if t["read_state"] in want
        and (lengths is None or int(t["length_bucket"]) in lengths)
    ]
    by = collections.defaultdict(list)
    for t in rows:
        by[t["route"]].append(t)
    out = {
        "read_state": state,
        "lengths": sorted(lengths) if lengths else "all",
        "routes": {},
    }

    def pct(rs, q):
        return weighted_quantile(
            list(zip([r["seconds_total"] for r in rs], request_weights(rs))), q
        )

    for route, rs in sorted(by.items()):
        stages = sorted(
            k for k in rs[0] if k.startswith("seconds_") and k != "seconds_total"
        )
        out["routes"][route] = {
            "requests": len(rs),
            "groups": len({r["source_group_id"] for r in rs}),
            "p50": pct(rs, 0.5),
            "p95": pct(rs, 0.95),
            "mean_stage_seconds": {
                s: sum(r.get(s, 0.0) for r in rs) / len(rs) for s in stages
            },
        }
    if "recompute" not in by:
        return out
    rng = np.random.default_rng(seed)
    groups = sorted({r["source_group_id"] for r in rows})
    members = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        members[r["source_group_id"]][r["route"]].append(r)
    out["against_recompute"] = {}
    for route in sorted(by):
        if route == "recompute":
            continue
        s50, s95 = [], []
        for _ in range(resamples):
            pick = rng.integers(0, len(groups), size=len(groups))
            a = [r for j in pick for r in members[groups[j]]["recompute"]]
            b = [r for j in pick for r in members[groups[j]][route]]
            if not a or not b:
                continue
            s50.append(1.0 - pct(b, 0.5) / pct(a, 0.5))
            s95.append(pct(b, 0.95) - pct(a, 0.95))
        base, this = out["routes"]["recompute"], out["routes"][route]
        out["against_recompute"][route] = {
            "p50_saving": 1.0 - this["p50"] / base["p50"],
            "p50_saving_interval_95": [
                float(np.quantile(s50, 0.025)),
                float(np.quantile(s50, 0.975)),
            ],
            "p95_seconds_more": this["p95"] - base["p95"],
            "p95_seconds_more_interval_95": [
                float(np.quantile(s95, 0.025)),
                float(np.quantile(s95, 0.975)),
            ],
        }
    return out


def report(scores, generations, timings, lead):
    ctx = context_scores(scores)
    if BASELINE not in ctx:
        raise ValueError("the receiver's own from-cache route was not scored")
    out = {
        "baseline": BASELINE,
        "lead": lead,
        "quality": {},
        "health": {},
        "agreement": {},
    }
    for a in sorted(ctx):
        if a == BASELINE:
            continue
        base = {c: v for c, v in ctx[BASELINE].items() if c in ctx[a]}
        out["quality"][a] = paired_deficit(base, ctx[a])
        out["health"][a] = health_excess(base, ctx[a])
        out["agreement"][a] = agreement(scores, generations, BASELINE, a)
    out["contexts"] = len(ctx[BASELINE])
    out["queries"] = sum(v["queries"] for v in ctx[BASELINE].values())
    out["groups"] = len({v["group"] for v in ctx[BASELINE].values()})
    if timings:
        out["latency"] = {}
        for state in ("direct", "page_cache"):
            if any(t["read_state"] == state for t in timings):
                out["latency"][state] = {
                    "all_lengths": latency(timings, state),
                    **{
                        str(n): latency(timings, state, lengths={n})
                        for n in sorted({int(t["length_bucket"]) for t in timings})
                    },
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--timings", default="")
    ap.add_argument("--lead", default="translate:cache_bridge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = report(
        load_jsonl(args.scores),
        load_jsonl(args.generations),
        load_jsonl(args.timings) if args.timings else [],
        args.lead,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("contexts", "queries", "groups")}))
    for a, q in out["quality"].items():
        print(
            a,
            "deficit %.4f upper %.4f largest stratum %.4f" % (
                q["deficit"], q["deficit_upper_95_one_sided"], q["largest_stratum_deficit"]
            ),
            "health", out["health"][a]["route_unhealthy"], "/", out["health"][a]["contexts"],
        )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
