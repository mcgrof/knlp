# SPDX-License-Identifier: GPL-2.0
"""Decide a recovery stage, and refuse to decide on evidence that is not there.

The gate this replaces accepted scored evidence it should have rejected. It
overwrote duplicate cells, intersected mismatched document sets and reported a
paired comparison over the single document both arms happened to share, filled
missing health fields with zero, and trusted a `healthy` flag that contradicted
the row it sat in. Every one of those returned an advancing verdict on a
fixture built to be invalid.

So completeness is checked before anything is computed. The manifest and the
condition matrix are inputs, not conventions: the gate is told which documents
and which cells must exist, and an evidence set that does not match is an
explicit invalid result rather than a smaller valid one. A missing row is never
resolved by dropping that document from the other arms.

The recovery stages have their own schema. The earlier gate is scoped to its
own experiment and its advancement label does not carry over to this one.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys

import numpy as np

TASKS = ("retrieval", "cloze")
N_BOOT = 2000
SEED = 0

# R1: a resource-release screen, not a deployment decision.
SCREEN_MIN_NET_RETRIEVAL = 2
SCREEN_MAX_CLOZE_DROP = 0.03

# R3: the inherited development gate.
ABSOLUTE_POINTS = 0.05
RECOVERY_POINT = 0.75
RECOVERY_FLOOR = 0.50
UNHEALTHY_EXCESS = 0.02
NATIVE_RETRIEVAL_MIN = 0.90
CLOZE_HEADROOM_MIN = 0.10

HEALTH_FIELDS = ("healthy", "empty", "too_short", "degenerate", "stopped", "cap_hit")


class Invalid(Exception):
    """Evidence that cannot support any verdict, valid or otherwise."""


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_rows(path, manifest, matrix, stage):
    """Ingest scored rows, or refuse.

    Every check here corresponds to a fixture that previously advanced. The
    duplicate check is on the raw stream rather than on a dict built from it,
    because building the dict is what hid the duplicate in the first place.
    """
    docs = [d["doc"] for d in manifest["documents"]]
    if len(set(docs)) != len(docs):
        raise Invalid("the manifest itself names a document twice")
    want_docs = set(docs)

    seen = collections.Counter()
    rows = {}
    versions = set()
    for ln, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise Invalid(f"row {ln} is not JSON: {e}")
        for f in ("kind", "condition", "doc"):
            if f not in r:
                raise Invalid(f"row {ln} has no {f!r}")
        versions.add(r.get("manifest_version"))
        key = (r["kind"], r["condition"], r["doc"])
        seen[key] += 1
        rows[key] = r

    dupes = sorted(k for k, n in seen.items() if n > 1)
    if dupes:
        raise Invalid(
            f"{len(dupes)} cells appear more than once, first {dupes[:3]}. A "
            "later row silently replacing an earlier one is how a rerun of one "
            "condition gets mistaken for a complete set."
        )
    if len(versions) > 1:
        raise Invalid(f"rows carry mixed manifest versions: {sorted(versions)}")

    missing, extra = [], []
    for kind, conditions in matrix.items():
        for cond in conditions:
            for doc in docs:
                if (kind, cond, doc) not in rows:
                    missing.append((kind, cond, doc))
    allowed = {(k, c) for k, cs in matrix.items() for c in cs}
    for kind, cond, doc in rows:
        if (kind, cond) not in allowed:
            extra.append((kind, cond, doc))
        elif doc not in want_docs:
            extra.append((kind, cond, doc))
    if missing:
        raise Invalid(
            f"{len(missing)} required cells are absent, first {missing[:3]}. "
            "An incomplete experiment stays incomplete; the remedy is not to "
            "drop those documents from the arms that do have them."
        )
    if extra:
        raise Invalid(
            f"{len(extra)} rows are outside the declared matrix, first {extra[:3]}"
        )

    for key, r in rows.items():
        kind = key[0]
        if kind == "free":
            for f in HEALTH_FIELDS:
                if f not in r:
                    raise Invalid(
                        f"{key}: health field {f!r} is absent. An absent "
                        "gross-repetition field is not zero failures."
                    )
                if not _finite(r[f]):
                    raise Invalid(f"{key}: health field {f!r} is not finite")
            # A row cannot be healthy and also empty, or healthy and too short.
            if float(r["healthy"]) > 0 and (
                float(r["empty"]) > 0 or float(r["too_short"]) > 0
            ):
                raise Invalid(
                    f"{key}: marked healthy while also empty or too short. The "
                    "derived label contradicts the raw fields it summarises."
                )
        else:
            if "primary" not in r or not _finite(r["primary"]):
                raise Invalid(f"{key}: primary score absent or not finite")
    return rows, docs


def cell(rows, kind, cond, docs, field="primary"):
    return np.array([float(rows[(kind, cond, d)][field]) for d in docs])


def recovery_interval(rows, kind, cand, docs):
    """Joint resample of candidate, native and empty, with a strict rule.

    A draw whose native-minus-empty gain is not positive has no incremental
    gain to recover, so its ratio is not defined. Discarding those draws and
    reporting percentiles of the rest describes uncertainty conditional on the
    benchmark having worked, which is not the uncertainty being claimed: a
    constructed case with genuine headroom dropped eighteen percent of its
    draws and reported a perfect interval.

    So a single invalid draw makes the interval undefined. That is stricter
    than the alternative and deliberately so; it removes the ambiguity without
    moving any performance threshold.
    """
    c = cell(rows, kind, cand, docs)
    n = cell(rows, kind, "target_native", docs)
    e = cell(rows, kind, "empty_cache", docs)
    denom = float(n.mean() - e.mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(docs), size=(N_BOOT, len(docs)))
    dn = n[idx].mean(axis=1) - e[idx].mean(axis=1)
    invalid = int((dn <= 0).sum())
    out = {
        "n_docs": len(docs),
        "observed_denominator": denom,
        "draws": N_BOOT,
        "draws_with_nonpositive_headroom": invalid,
        "seed": SEED,
        "rule": (
            "any draw with nonpositive native-minus-empty headroom makes the "
            "interval undefined; a positive-draw-only interval is never used "
            "to advance"
        ),
    }
    if denom <= 0 or invalid > 0:
        out["undefined"] = True
        out["point"] = None
        out["label"] = "BENCHMARK_UNINFORMATIVE_FOR_RECOVERY_GATE"
        return out
    ratios = (c[idx].mean(axis=1) - e[idx].mean(axis=1)) / dn
    out["undefined"] = False
    out["point"] = float((c.mean() - e.mean()) / denom)
    out["lo"] = float(np.percentile(ratios, 2.5))
    out["hi"] = float(np.percentile(ratios, 97.5))
    return out


def paired(rows, kind, a, b, docs):
    d = cell(rows, kind, a, docs) - cell(rows, kind, b, docs)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    bs = d[idx].mean(axis=1)
    return {
        "n": int(d.size),
        "point": float(d.mean()),
        "lo": float(np.percentile(bs, 2.5)),
        "hi": float(np.percentile(bs, 97.5)),
    }


def health(rows, cond, docs):
    out = {"n": len(docs)}
    for f in HEALTH_FIELDS:
        out[f] = float(np.mean([float(rows[("free", cond, d)][f]) for d in docs]))
    out["unhealthy"] = 1.0 - out["healthy"]
    out["gross_repetition_failures"] = out["degenerate"]
    out["note"] = (
        "stopping by end-of-sequence alone is reported, never counted unhealthy"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--manifest", required=True, help="the frozen document manifest")
    ap.add_argument("--matrix", required=True, help="the required condition matrix")
    ap.add_argument("--stage", required=True, choices=("R1", "R3"))
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--unrepaired", default="mapped_unrepaired")
    ap.add_argument("--anchor", default="anchor77")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    matrix = json.load(open(args.matrix))
    report = {
        "contract": "recovery_gate_v1",
        "stage": args.stage,
        "candidate": args.candidate,
        "manifest": {
            "path": args.manifest,
            "version": manifest.get("version", manifest.get("contract")),
            "joint_sha256": manifest.get("joint_sha256"),
            "documents": len(manifest["documents"]),
        },
        "scope": (
            "recovery stages have their own schema and thresholds; the earlier "
            "experiment's advancement label does not carry over"
        ),
    }
    try:
        rows, docs = load_rows(args.rows, manifest, matrix, args.stage)
    except Invalid as e:
        report.update(
            {
                "valid_evidence": False,
                "reason": str(e),
                "verdict": "INVALID_EVIDENCE",
                "advances": False,
            }
        )
        json.dump(report, open(args.out, "w"), indent=2, sort_keys=True, default=str)
        print(f"INVALID_EVIDENCE: {e}", file=sys.stderr)
        return 2
    report["valid_evidence"] = True

    nat = {t: cell(rows, t, "target_native", docs).mean() for t in TASKS}
    emp = {t: cell(rows, t, "empty_cache", docs).mean() for t in TASKS}
    informative = {
        "native_retrieval": float(nat["retrieval"]),
        "native_retrieval_min": NATIVE_RETRIEVAL_MIN,
        "cloze_headroom": float(nat["cloze"] - emp["cloze"]),
        "cloze_headroom_min": CLOZE_HEADROOM_MIN,
    }
    informative["passed"] = bool(
        informative["native_retrieval"] >= NATIVE_RETRIEVAL_MIN
        and informative["cloze_headroom"] >= CLOZE_HEADROOM_MIN
    )
    report["benchmark_informative"] = informative

    cand_h = health(rows, args.candidate, docs)
    report["health"] = {
        "candidate": cand_h,
        "native": health(rows, "target_native", docs),
    }
    scores = {t: float(cell(rows, t, args.candidate, docs).mean()) for t in TASKS}
    report["scores"] = scores

    if args.stage == "R1":
        unrep_h = health(rows, args.unrepaired, docs)
        report["health"]["unrepaired"] = unrep_h
        cand_r = cell(rows, "retrieval", args.candidate, docs)
        unrep_r = cell(rows, "retrieval", args.unrepaired, docs)
        net = float(cand_r.sum() - unrep_r.sum())
        cloze_drop = float(
            cell(rows, "cloze", args.unrepaired, docs).mean() - scores["cloze"]
        )
        checks = {
            "net_retrieval_over_unrepaired": net,
            "net_retrieval_required": SCREEN_MIN_NET_RETRIEVAL,
            "meets_net_retrieval": bool(net >= SCREEN_MIN_NET_RETRIEVAL),
            "cloze_drop": cloze_drop,
            "cloze_drop_allowed": SCREEN_MAX_CLOZE_DROP,
            "meets_cloze": bool(cloze_drop <= SCREEN_MAX_CLOZE_DROP),
            "unhealthy_not_worse": bool(
                cand_h["unhealthy"] <= unrep_h["unhealthy"] + 1e-12
            ),
            "gross_repetition_not_worse": bool(
                cand_h["gross_repetition_failures"]
                <= unrep_h["gross_repetition_failures"] + 1e-12
            ),
        }
        report["screen"] = checks
        passed = informative["passed"] and all(
            v
            for k, v in checks.items()
            if k.startswith("meets_") or k.endswith("_not_worse")
        )
        report["verdict"] = (
            "SCREEN_SIGNAL"
            if passed
            else (
                "BENCHMARK_UNINFORMATIVE"
                if not informative["passed"]
                else "NO_SCREEN_SIGNAL"
            )
        )
        report["advances"] = bool(passed)
        report["note"] = (
            "a resource-release heuristic, not significance, deployment or a "
            "measured speedup; native-donor rows remain oracle rows"
        )
    else:
        anchor_h = health(rows, args.anchor, docs)
        report["health"]["anchor"] = anchor_h
        tasks = {}
        for t in TASKS:
            ri = recovery_interval(rows, t, args.candidate, docs)
            vw = paired(rows, t, args.candidate, f"{args.candidate}|wrong", docs)
            tasks[t] = {
                "score": scores[t],
                "native": float(nat[t]),
                "points_behind_native": float((nat[t] - scores[t]) * 100.0),
                "within_five_points": bool(scores[t] >= nat[t] - ABSOLUTE_POINTS),
                "recovery_interval": ri,
                "recovers_enough": bool(
                    not ri.get("undefined")
                    and ri.get("point") is not None
                    and ri["point"] >= RECOVERY_POINT
                    and ri["lo"] > RECOVERY_FLOOR
                ),
                "vs_own_wrong_source": vw,
                "beats_own_wrong_source": bool(vw["lo"] > 0.0),
            }
        excess = cand_h["unhealthy"] - report["health"]["native"]["unhealthy"]
        h = {
            "excess_unhealthy": excess,
            "excess_limit": UNHEALTHY_EXCESS,
            "within_limit": bool(excess <= UNHEALTHY_EXCESS),
            "gross_repetition_vs_anchor": cand_h["gross_repetition_failures"]
            - anchor_h["gross_repetition_failures"],
            "no_gross_repetition_increase": bool(
                cand_h["gross_repetition_failures"]
                <= anchor_h["gross_repetition_failures"] + 1e-12
            ),
        }
        h["passed"] = bool(h["within_limit"] and h["no_gross_repetition_increase"])
        report["tasks"] = tasks
        report["health_gate"] = h
        reqs = {
            f"{t}_{k}": tasks[t][k]
            for t in TASKS
            for k in ("within_five_points", "recovers_enough", "beats_own_wrong_source")
        }
        reqs["health"] = h["passed"]
        report["requirements"] = reqs
        report["failed_requirements"] = sorted(k for k, v in reqs.items() if not v)
        undefined = [t for t in TASKS if tasks[t]["recovery_interval"].get("undefined")]
        if undefined:
            report["verdict"] = "BENCHMARK_UNINFORMATIVE_FOR_RECOVERY_GATE"
            report["undefined_recovery_tasks"] = undefined
            report["advances"] = False
        else:
            ok = informative["passed"] and not report["failed_requirements"]
            report["verdict"] = "RECOVERY_DEVELOPMENT_SIGNAL" if ok else "NO_SIGNAL"
            report["advances"] = bool(ok)
        report["note"] = (
            "a development signal is not deployment; byte compliance, fresh "
            "confirmation and integration are not released by it"
        )

    report["rows_sha256"] = hashlib.sha256(open(args.rows, "rb").read()).hexdigest()
    json.dump(report, open(args.out, "w"), indent=2, sort_keys=True, default=str)
    print(
        f"stage {args.stage} verdict {report['verdict']} advances={report['advances']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
