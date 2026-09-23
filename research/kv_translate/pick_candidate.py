# SPDX-License-Identifier: GPL-2.0
"""Choose what the next recovery stage runs, from the screen's own numbers.

The screen produces one verdict per repair condition and the next stage runs
exactly one of them. Deciding which by reading a table on a rented card is how
a preference becomes a result: the arm that looked best gets carried, and the
rule that picked it is reconstructed afterwards. So the rule is written down
here, before the screen runs, and the choice is computed.

The order of preference is the parent plan's, not a ranking by score. Even
position coverage is preferred whenever it matches the best ranked selector,
because it needs no pass over the question and a selector that cannot beat
free coverage has not earned its cost. Among ranked selectors only the one
reading the translated cache is deployable; the one reading the receiver's own
cache is an oracle and can be carried no further than a diagnostic. Between two
budgets that both pass, the smaller is carried.

Emits the condition files for the stages that follow, so no one hand-edits a
matrix while a card is billing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from research.kv_translate.recovery_gate import (  # noqa: E402
    SCREEN_MAX_CLOZE_DROP,
    SCREEN_MIN_NET_RETRIEVAL,
    Invalid,
    cell,
    health,
    load_rows,
)

# In preference order. A tuple that appears earlier is carried even when a
# later one scores higher, unless the later one is strictly better than every
# earlier one that passed -- which for the first entry means "matches or wins".
FAMILY_ORDER = ("uniform", "mapped_attn", "native_attn")


def screen_one(rows, docs, cand, unrepaired):
    """The screen's three checks for one condition, as the gate computes them."""
    cand_r = cell(rows, "retrieval", cand, docs)
    unrep_r = cell(rows, "retrieval", unrepaired, docs)
    net = float(cand_r.sum() - unrep_r.sum())
    drop = float(
        cell(rows, "cloze", unrepaired, docs).mean()
        - cell(rows, "cloze", cand, docs).mean()
    )
    ch, uh = health(rows, cand, docs), health(rows, unrepaired, docs)
    return {
        "condition": cand,
        "net_retrieval_over_unrepaired": net,
        "cloze_drop": drop,
        "unhealthy": ch["unhealthy"],
        "gross_repetition_failures": ch["gross_repetition_failures"],
        "meets_net_retrieval": bool(net >= SCREEN_MIN_NET_RETRIEVAL),
        "meets_cloze": bool(drop <= SCREEN_MAX_CLOZE_DROP),
        "unhealthy_not_worse": bool(ch["unhealthy"] <= uh["unhealthy"] + 1e-12),
        "gross_repetition_not_worse": bool(
            ch["gross_repetition_failures"] <= uh["gross_repetition_failures"] + 1e-12
        ),
    }


def passes(s):
    return (
        s["meets_net_retrieval"]
        and s["meets_cloze"]
        and s["unhealthy_not_worse"]
        and s["gross_repetition_not_worse"]
    )


def family(defs, name):
    return defs[name].get("selector")


def choose(screens, defs):
    """Apply the preference order to the conditions that passed."""
    ok = [s for s in screens if passes(s)]
    if not ok:
        return (
            None,
            "NO_LOCAL_REPAIR_SIGNAL",
            (
                "no repair condition cleared the screen, so no mask is worth "
                "recomputing; the plan's remaining branch is a single "
                "last-quarter-layer donor diagnostic and then terminal"
            ),
        )
    by_fam = {}
    for s in ok:
        by_fam.setdefault(family(defs, s["condition"]), []).append(s)

    def best(group):
        # Most net retrieval; the smaller budget breaks a tie, since it is
        # cheaper to recompute and cheaper to move.
        return sorted(
            group,
            key=lambda s: (
                -s["net_retrieval_over_unrepaired"],
                defs[s["condition"]]["blocks"],
            ),
        )[0]

    ranked = [best(by_fam[f]) for f in FAMILY_ORDER if f in by_fam]
    top = max(s["net_retrieval_over_unrepaired"] for s in ranked)
    if (
        "uniform" in by_fam
        and best(by_fam["uniform"])["net_retrieval_over_unrepaired"] >= top
    ):
        return (
            best(by_fam["uniform"]),
            "SCREEN_SIGNAL_UNIFORM",
            "even coverage matched or beat every ranked selector, so it is "
            "carried and the pass over the question is not paid for",
        )
    if "mapped_attn" in by_fam:
        return (
            best(by_fam["mapped_attn"]),
            "SCREEN_SIGNAL_MAPPED_RANKING",
            "ranking on the translated cache cleared the screen and is "
            "available at serving time",
        )
    return (
        best(by_fam["native_attn"]),
        "ORACLE_RANKING_ONLY",
        "only ranking on the receiver's own cache cleared the screen. That is "
        "not available at serving time, so it cannot be carried into a cost "
        "claim; the plan allows at most one shallow receiver selector here "
        "and then terminal",
    )


def r2_conditions(winner, defs, block_total):
    """The same mask, recomputed for real, beside the arms it is read against."""
    spec = dict(defs[winner["condition"]])
    k, sel = spec["blocks"], spec["selector"]
    d = {
        "target_native": {"repair": "native"},
        "empty_cache": {"repair": "empty"},
        "mapped_unrepaired": {"repair": "none"},
        "anchor77": {"repair": "none", "arm": "anchor77"},
        "candidate": {"repair": "replay", "selector": sel, "blocks": k},
        "donor_bound": {"repair": "oracle", "selector": sel, "blocks": k},
        "span_native": {"repair": "span", "selector": "lexical", "blocks": k},
    }
    m = [
        "target_native",
        "empty_cache",
        "mapped_unrepaired",
        "candidate",
        "donor_bound",
        "span_native",
    ]
    return {
        "stage": "R2",
        "purpose": (
            "Recompute the winning mask for real and price the whole path. The "
            "donor row is the bound the same mask reached under substitution; "
            "the raw-text row is the baseline that buys context with the same "
            "budget and no map at all."
        ),
        "block_tokens": 16,
        "total_blocks": block_total,
        "carried_from_R1": winner["condition"],
        "definitions": d,
        "matrix": {
            "retrieval": m,
            "cloze": m,
            "free": [
                "target_native",
                "empty_cache",
                "mapped_unrepaired",
                "anchor77",
                "candidate",
            ],
        },
    }


def r3_conditions(winner, defs, block_total):
    spec = defs[winner["condition"]]
    k, sel = spec["blocks"], spec["selector"]
    d = {
        "target_native": {"repair": "native"},
        "empty_cache": {"repair": "empty"},
        "anchor77": {"repair": "none", "arm": "anchor77"},
        "candidate": {"repair": "replay", "selector": sel, "blocks": k},
        "candidate|wrong": {
            "repair": "replay",
            "selector": sel,
            "blocks": k,
            "source": "wrong",
        },
        "span_native": {"repair": "span", "selector": "lexical", "blocks": k},
    }
    m = [
        "target_native",
        "empty_cache",
        "candidate",
        "candidate|wrong",
        "anchor77",
        "span_native",
    ]
    return {
        "stage": "R3",
        "purpose": "The frozen development gate over all sixty-four documents.",
        "block_tokens": 16,
        "total_blocks": block_total,
        "carried_from_R1": winner["condition"],
        "definitions": d,
        "matrix": {
            "retrieval": m,
            "cloze": m,
            "free": ["target_native", "empty_cache", "candidate", "anchor77"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="the screen's scored rows")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--conditions", required=True, help="the screen's own file")
    ap.add_argument("--unrepaired", default="mapped_unrepaired")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.conditions))
    defs, matrix = cfg["definitions"], cfg["matrix"]
    manifest = json.load(open(args.manifest))
    try:
        rows, docs = load_rows(args.rows, manifest, matrix, "R1")
    except Invalid as e:
        json.dump(
            {
                "valid_evidence": False,
                "reason": str(e),
                "verdict": "INVALID_EVIDENCE",
                "advances": False,
            },
            open(os.path.join(args.out_dir, "CANDIDATE.json"), "w"),
            indent=2,
        )
        print(f"INVALID_EVIDENCE: {e}", file=sys.stderr)
        return 2

    repairs = sorted(
        {
            c
            for cs in matrix.values()
            for c in cs
            if defs[c].get("selector")
            and defs[c].get("blocks", 0) < cfg["total_blocks"]
        }
    )
    screens = [screen_one(rows, docs, c, args.unrepaired) for c in repairs]
    winner, verdict, why = choose(screens, defs)
    out = {
        "contract": "recovery_candidate_v1",
        "documents": len(docs),
        "screened": screens,
        "rule": (
            "even coverage is carried whenever it matches the best ranked "
            "selector; otherwise ranking on the translated cache; ranking on "
            "the receiver's own cache is an oracle and terminal. Smaller "
            "budget breaks a tie."
        ),
        "preference_order": list(FAMILY_ORDER),
        "verdict": verdict,
        "reason": why,
        "candidate": winner["condition"] if winner else None,
        "advances": winner is not None and verdict != "ORACLE_RANKING_ONLY",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(args.out_dir, "CANDIDATE.json"), "w"), indent=2)
    if winner and out["advances"]:
        for stage, fn in (("r2", r2_conditions), ("r3", r3_conditions)):
            json.dump(
                fn(winner, defs, cfg["total_blocks"]),
                open(os.path.join(args.out_dir, f"conditions_{stage}.json"), "w"),
                indent=2,
            )
    print(f"{verdict}: {out['candidate']}")
    for s in screens:
        mark = "pass" if passes(s) else "    "
        print(
            f"  {mark} {s['condition']:18s} net {s['net_retrieval_over_unrepaired']:+5.1f} "
            f"cloze {s['cloze_drop']:+.3f} unhealthy {s['unhealthy']:.3f}"
        )
    return 0 if out["advances"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
