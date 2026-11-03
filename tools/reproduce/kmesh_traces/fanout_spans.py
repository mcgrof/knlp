# SPDX-License-Identifier: MIT
"""Count wide agent launches and the spans their children share.

For every launch (one parent step that starts children) in the Weka
Claude Code corpus or one FreeInference week, this computes from each
child's first request: the span common to all children (group span L),
the span the first child shares with a child of another launch (preamble
span X), the span it shares with any earlier request (prior span E), and
from those the saved family ancestor S = max(0, min(L, E) - X) and the
new family span N = L - max(E, X). Spans are equal leading block ids,
converted to tokens with the corpus block size. The definitions are in
README.md. One JSON report with one row per launch
plus a summary. No model, GPU or storage work.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

WIDE = 4
ANCESTOR_TOKENS = 16384


def lcp(a, b, cap=None):
    n = min(len(a), len(b))
    if cap is not None:
        n = min(n, cap)
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def group_lcp(seqs):
    if not seqs:
        return 0
    n = len(seqs[0])
    for s in seqs[1:]:
        n = lcp(seqs[0], s, n)
    return n


def spans(firsts, others, prior, foreign=None):
    """firsts: block lists of this launch's children (first requests).
    others: first-request block lists of children of other launches.
    prior: (end time, block list) of requests that started before the launch.
    foreign: first-request block lists of children launched by other users.

    Returns the spans in blocks and the latest end time of a prior request
    that holds the whole saved shared span S2 = min(L, E)."""
    L = group_lcp(firsts)
    c0 = firsts[0]
    X = max((lcp(c0, o, L) for o in others), default=None)
    E = 0
    for _, p in prior:
        if p and p[0] == c0[0]:
            E = max(E, lcp(c0, p, L))
            if E == L:
                break
    S2 = min(L, E)
    last_end = None
    if S2:
        for t_end, p in prior:
            if p and p[0] == c0[0] and lcp(c0, p, S2) == S2:
                last_end = t_end if last_end is None else max(last_end, t_end)
    x = X or 0
    out = {
        "L": L,
        "X": X,
        "E": E,
        "F": max(0, L - x),
        "S": max(0, S2 - x),
        "N": max(0, L - max(E, x)),
        "S2": S2,
    }
    if foreign is not None:
        out["Xf"] = max((lcp(c0, o, L) for o in foreign), default=None)
    return out, last_end


def pct(values, ps=(50, 90, 99)):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    out = {"n": int(a.size), "mean": float(a.mean()), "max": float(a.max())}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def weka_launches(path, window_s):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            bt = d.get("block_size", 64)
            reqs = d["requests"]
            mains = [r for r in reqs if r.get("type") != "subagent"]
            subs = sorted(
                (r for r in reqs if r.get("type") == "subagent" and r.get("requests")),
                key=lambda r: r["t"],
            )
            groups, cur = [], []
            for r in subs:
                if cur and r["t"] - cur[-1]["t"] > window_s:
                    groups.append(cur)
                    cur = []
                cur.append(r)
            if cur:
                groups.append(cur)
            firsts_of = [
                [s["requests"][0].get("hash_ids") or [] for s in g] for g in groups
            ]
            # every request with its start time, for the prior span
            timed = [
                (
                    r["t"],
                    r["t"] + float(r.get("api_time") or 0.0),
                    r.get("hash_ids") or [],
                )
                for r in mains
            ]
            for s in subs:
                for q in s["requests"]:
                    timed.append(
                        (
                            q["t"],
                            q["t"] + float(q.get("api_time") or 0.0),
                            q.get("hash_ids") or [],
                        )
                    )
            timed.sort(key=lambda x: x[0])
            for gi, g in enumerate(groups):
                firsts = [x for x in firsts_of[gi] if x]
                if not firsts:
                    continue
                t0 = g[0]["t"]
                others = [
                    x for gj, fl in enumerate(firsts_of) if gj != gi for x in fl if x
                ]
                prior = [(e, h) for t, e, h in timed if t < t0]
                sp, last_end = spans(firsts, others, prior)
                prev = [r for r in mains if r["t"] <= t0]
                idle = None
                if prev:
                    p = prev[-1]
                    idle = t0 - (p["t"] + float(p.get("api_time") or 0.0))
                parent_tokens = (
                    len(prev[-1].get("hash_ids") or []) * bt if prev else None
                )
                pshare = (
                    min(lcp(prev[-1].get("hash_ids") or [], x) for x in firsts) * bt
                    if prev
                    else None
                )
                rows.append(
                    {
                        "corpus": "weka",
                        "trace": d["id"],
                        "harness": "claude-code",
                        "width": len(firsts),
                        "child_first_tokens_p50": float(
                            np.median([len(x) for x in firsts]) * bt
                        ),
                        "parent_tokens": parent_tokens,
                        "parent_share_tokens": pshare,
                        "parent_idle_s": idle,
                        "spawn_spread_s": g[-1]["t"] - t0,
                        "saved_idle_s": None if last_end is None else t0 - last_end,
                        **{
                            k + "_tokens": (None if v is None else v * bt)
                            for k, v in sp.items()
                        },
                    }
                )
    return rows


def freeinference_launches(release, week):
    from release import reader

    week_dir = reader.week_directories(release, [week])[0]
    rel = reader.load(release, [week])
    S, R = rel.sessions, rel.requests
    sub = S[S.parent_session_id.notna()]
    harness_of = S.set_index("session_id").harness.to_dict()
    user_of = S.set_index("session_id").user_id.to_dict()
    parent_of = sub.set_index("session_id").parent_session_id.to_dict()

    def root(sid):
        seen = 0
        while sid in parent_of and seen < 64:
            sid = parent_of[sid]
            seen += 1
        return sid

    family_sessions = set(sub.session_id) | set(sub.parent_session_id)
    R = R[R.session_id.isin(family_sessions)]
    want = set(zip(R.session_id, R.step.astype(int)))
    blocks = {}
    pf = reader.open_blocks(week_dir)
    for batch in pf.iter_batches(columns=["session_id", "step", "block_ids"]):
        sid = batch.column("session_id").to_pylist()
        st = batch.column("step").to_pylist()
        bl = batch.column("block_ids")
        for i, key in enumerate(zip(sid, st)):
            if key in want:
                blocks[key] = bl[i].as_py()
    start = {(s, int(t)): int(ms) for s, t, ms in zip(R.session_id, R.step, R.start_ms)}
    end = {(s, int(t)): int(ms) for s, t, ms in zip(R.session_id, R.step, R.end_ms)}
    by_root = defaultdict(list)
    for key, ms in start.items():
        by_root[root(key[0])].append((ms, key))
    for v in by_root.values():
        v.sort()
    fam = sub.groupby(["parent_session_id", "spawned_by_step"]).session_id.apply(list)
    launches = []
    for (pid, pstep), kids in fam.items():
        firsts = [(k, blocks.get((k, 0))) for k in kids]
        firsts = [(k, b) for k, b in firsts if b]
        if firsts:
            launches.append(((pid, int(pstep)), firsts))
    rows = []
    bt = 16
    for li, ((pid, pstep), firsts) in enumerate(launches):
        h = harness_of.get(firsts[0][0])
        others = [
            b
            for lj, (_, fl) in enumerate(launches)
            if lj != li
            for k, b in fl
            if harness_of.get(k) == h
        ]
        foreign = [
            b
            for lj, ((pj, _), fl) in enumerate(launches)
            if lj != li and user_of.get(pj) != user_of.get(pid)
            for k, b in fl
            if harness_of.get(k) == h
        ]
        t0 = min(start.get((k, 0), 1 << 62) for k, _ in firsts)
        prior = [
            (end[key], blocks[key])
            for ms, key in by_root[root(pid)]
            if ms < t0 and key in blocks
        ]
        sp, last_end = spans([b for _, b in firsts], others, prior, foreign)
        pb = blocks.get((pid, pstep))
        rows.append(
            {
                "corpus": "freeinference-" + week,
                "trace": pid,
                "harness": h,
                "width": len(firsts),
                "child_first_tokens_p50": float(
                    np.median([len(b) for _, b in firsts]) * bt
                ),
                "parent_tokens": len(pb) * bt if pb else None,
                "parent_share_tokens": (
                    min(lcp(pb, b) for _, b in firsts) * bt if pb else None
                ),
                "parent_idle_s": (
                    (t0 - end[(pid, pstep)]) / 1000.0 if (pid, pstep) in end else None
                ),
                "saved_idle_s": None if last_end is None else (t0 - last_end) / 1000.0,
                **{
                    k + "_tokens": (None if v is None else v * bt)
                    for k, v in sp.items()
                },
            }
        )
    return rows


def summarize(rows):
    n = len(rows)
    kids = sum(r["width"] for r in rows)
    out = {"launches": n, "children": kids}
    buckets = {
        "1": lambda w: w == 1,
        "2-3": lambda w: 2 <= w <= 3,
        "4+": lambda w: w >= WIDE,
    }
    out["by_width"] = {}
    for name, sel in buckets.items():
        rs = [r for r in rows if sel(r["width"])]
        out["by_width"][name] = {
            "launches": len(rs),
            "children": sum(r["width"] for r in rs),
            **{
                k: pct([r.get(k) for r in rs])
                for k in (
                    "L_tokens",
                    "X_tokens",
                    "E_tokens",
                    "F_tokens",
                    "S_tokens",
                    "N_tokens",
                    "parent_tokens",
                    "parent_share_tokens",
                    "parent_idle_s",
                    "child_first_tokens_p50",
                    "S2_tokens",
                    "Xf_tokens",
                    "saved_idle_s",
                    "spawn_spread_s",
                )
            },
            "preamble_unknown": sum(1 for r in rs if r["X_tokens"] is None),
        }
    hit = [r for r in rows if r["width"] >= WIDE and r["S_tokens"] >= ANCESTOR_TOKENS]
    hit_known = [r for r in hit if r["X_tokens"] is not None]
    out["rule"] = {
        "wide_with_saved_ancestor_launches": len(hit),
        "of_which_preamble_known": len(hit_known),
        "launch_share": len(hit) / n if n else None,
        "child_share": sum(r["width"] for r in hit) / kids if kids else None,
        "launch_share_preamble_known": len(hit_known) / n if n else None,
        "child_share_preamble_known": (
            sum(r["width"] for r in hit_known) / kids if kids else None
        ),
        "pass": bool(
            n and (len(hit) / n >= 0.05 or sum(r["width"] for r in hit) / kids >= 0.10)
        ),
    }
    # sensitivity check written after the rule failed: count the whole saved
    # shared span (preamble included) when it sat idle long enough to be cold
    out["cold_shared_span"] = {}
    for idle_min in (60, 300):
        cold = [
            r
            for r in rows
            if r["width"] >= WIDE
            and r["S2_tokens"] >= ANCESTOR_TOKENS
            and r["saved_idle_s"] is not None
            and r["saved_idle_s"] >= idle_min
        ]
        out["cold_shared_span"][f"idle_ge_{idle_min}s"] = {
            "launches": len(cold),
            "children": sum(r["width"] for r in cold),
            "launch_share": len(cold) / n if n else None,
            "child_share": sum(r["width"] for r in cold) / kids if kids else None,
            "redundant_tokens": float(
                sum((r["width"] - 1) * r["S2_tokens"] for r in cold)
            ),
        }
    # children-weighted redundant reads storage cascade would remove
    out["redundant_saved_ancestor_tokens"] = {
        "all_launches": float(sum((r["width"] - 1) * r["S_tokens"] for r in rows)),
        "wide_launches": float(
            sum((r["width"] - 1) * r["S_tokens"] for r in rows if r["width"] >= WIDE)
        ),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weka")
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--release")
    ap.add_argument("--week")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.weka:
        rows = weka_launches(args.weka, args.window_s)
    else:
        rows = freeinference_launches(args.release, args.week)
    report = {"summary": summarize(rows), "rows": rows}
    by_h = defaultdict(list)
    for r in rows:
        by_h[r["harness"]].append(r)
    report["by_harness"] = {
        h: summarize(v) for h, v in sorted(by_h.items(), key=lambda x: str(x[0]))
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    s = report["summary"]
    print(
        json.dumps(
            {
                "launches": s["launches"],
                "children": s["children"],
                "rule": s["rule"],
                "wide": {
                    k: s["by_width"]["4+"][k]
                    for k in ("launches", "children", "preamble_unknown")
                },
            }
        )
    )


if __name__ == "__main__":
    main()
