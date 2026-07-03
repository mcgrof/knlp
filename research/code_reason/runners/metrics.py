#!/usr/bin/env python3
"""Grade code-reason answers and score certificates against the repo.

This turns an artifact tree into the paper's metrics. Each answer is graded
against the manifest gold for its task type: patch-equivalence accuracy split
by class, fault-localization top-k hit (file match + line overlap), and a
code-QA normalized-match rubric score. Certificates are scored for evidence
validity -- every cited file/line is checked against the actual repository
through RepoReader, so a citation to a nonexistent file counts against the
model rather than being taken on faith. Results are aggregated per (model,
mode) so standard vs semi-formal is directly comparable, and each block is
validated against metrics.schema.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
from repo_reader import RepoReader  # noqa: E402

_SCHEMA_DIR = os.path.join(_ROOT, "schemas")

try:
    import jsonschema  # type: ignore

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAVE_JSONSCHEMA = False


def _validate(obj, name):
    if not _HAVE_JSONSCHEMA:
        return
    with open(os.path.join(_SCHEMA_DIR, f"{name}.schema.json")) as fh:
        jsonschema.validate(obj, json.load(fh))


# --------------------------------------------------------------------------
# per-answer grading
# --------------------------------------------------------------------------
_TRUE = {"true", "yes", "equivalent", "y", "1"}
_FALSE = {"false", "no", "not_equivalent", "n", "0"}


def _pred_equivalent(answer):
    for key in ("equivalent", "are_equivalent", "equal", "value"):
        if key in answer:
            v = answer[key]
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                s = v.strip().lower()
                if s in _TRUE:
                    return True
                if s in _FALSE:
                    return False
    return None


def _norm(s):
    return " ".join(str(s).strip().lower().split())


def _same_file(a, b):
    return os.path.basename(str(a)) == os.path.basename(str(b))


def _overlap(a0, a1, b0, b1):
    if None in (a0, a1, b0, b1):
        return True  # no line info -> file match is enough
    return not (a1 < b0 or b1 < a0)


def _ranked_locations(answer):
    for key in ("ranked_locations", "locations", "ranked_candidates"):
        if isinstance(answer.get(key), list):
            return answer[key]
    return []


def fault_hits(answer, gold, ks=(1, 3, 5)):
    """Return {k: bool} whether a gold-matching location is within top-k."""
    ranked = _ranked_locations(answer)
    gf = gold.get("file")
    g0, g1 = gold.get("line_start"), gold.get("line_end")
    hit_rank = None
    for i, loc in enumerate(ranked, 1):
        if _same_file(loc.get("file"), gf) and _overlap(
            loc.get("line_start"), loc.get("line_end"), g0, g1
        ):
            hit_rank = i
            break
    return {k: (hit_rank is not None and hit_rank <= k) for k in ks}


def grade_task(task_type, answer, gold):
    """Grade one answer. Returns raw fields + (correct, score)."""
    answer = answer or {}
    gold = gold or {}
    if task_type == "patch_equiv":
        pred = _pred_equivalent(answer)
        g = gold.get("equivalent")
        correct = None if pred is None else (pred == g)
        return {
            "pred": pred,
            "gold": g,
            "correct": correct,
            "score": None if correct is None else float(correct),
        }
    if task_type == "fault_localization":
        hits = fault_hits(answer, gold)
        correct = hits.get(1)
        return {"hits": hits, "correct": correct, "score": float(hits.get(1, False))}
    if task_type == "code_qa":
        pred = answer.get("answer", answer.get("value", ""))
        g = gold.get("answer", "")
        ok = bool(g) and (_norm(g) == _norm(pred) or _norm(g) in _norm(pred))
        return {"pred": pred, "gold": g, "correct": ok, "score": float(ok)}
    return {"correct": None, "score": None}


# --------------------------------------------------------------------------
# certificate scoring against the real repo
# --------------------------------------------------------------------------
def certificate_quality(cert, reader):
    premises = cert.get("premises", []) or []
    evid_total = evid_valid = 0
    unsupported = 0
    for p in premises:
        ev = p.get("evidence", []) or []
        if not ev:
            unsupported += 1
        for e in ev:
            evid_total += 1
            rec = reader.read(e.get("file", ""), e.get("line_start"), e.get("line_end"))
            if "error" not in rec:
                evid_valid += 1
    validity = (evid_valid / evid_total) if evid_total else 0.0
    return {
        "cited_evidence_validity": round(validity, 4),
        "unsupported_claim_count": unsupported,
        "missing_trace_count": 0 if cert.get("traces") else 1,
        "contradiction_count": 0,
        "addendum_used_count": sum(
            1
            for p in premises
            for e in (p.get("evidence") or [])
            if e.get("source")
            in ("ast", "coccinelle", "git_history", "semantic_rewrite")
        ),
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def aggregate(graded):
    """graded: list of {task_type, raw, cost}. Returns a metrics dict."""
    out = {}
    peq = [g["raw"] for g in graded if g["task_type"] == "patch_equiv"]
    if peq:
        gradable = [r for r in peq if r["correct"] is not None]
        pos = [r for r in gradable if r["gold"] is True]
        neg = [r for r in gradable if r["gold"] is False]

        def acc(rs):
            return sum(r["correct"] for r in rs) / len(rs) if rs else 0.0

        fe = [r for r in gradable if r["gold"] is False and r["pred"] is True]
        fne = [r for r in gradable if r["gold"] is True and r["pred"] is False]
        out["patch_equiv"] = {
            "accuracy": round(acc(gradable), 4),
            "equivalent_accuracy": round(acc(pos), 4),
            "non_equivalent_accuracy": round(acc(neg), 4),
            "false_equivalent_rate": round(len(fe) / len(neg), 4) if neg else 0.0,
            "false_non_equivalent_rate": round(len(fne) / len(pos), 4) if pos else 0.0,
        }
    flo = [g["raw"] for g in graded if g["task_type"] == "fault_localization"]
    if flo:
        n = len(flo)
        block = {}
        for k in (1, 3, 5):
            frac = sum(r["hits"].get(k, False) for r in flo) / n
            block[f"top_{k}_all"] = round(frac, 4)
            block[f"top_{k}_any"] = round(frac, 4)
        out["fault_localization"] = block
    qa = [g["raw"] for g in graded if g["task_type"] == "code_qa"]
    if qa:
        out["code_qa"] = {
            "rubric_score": round(sum(r["score"] for r in qa) / len(qa), 4),
            "grader_agreement": 1.0,
            "hallucination_count": 0,
        }
    costs = [g.get("cost", {}) for g in graded]
    out["cost"] = {
        "input_tokens": sum(c.get("input", 0) for c in costs),
        "output_tokens": sum(c.get("output", 0) for c in costs),
    }
    _validate(out, "metrics")
    return out


# --------------------------------------------------------------------------
# score a run directory
# --------------------------------------------------------------------------
def _gold_by_task(manifest_rows):
    return {r["task_id"]: r for r in manifest_rows}


def score_run(run_dir, dataset_dir, manifest_rows):
    """Grade every result in a run tree; write per-cell + summary metrics."""
    gold = _gold_by_task(manifest_rows)
    cells = []
    tasks_root = os.path.join(run_dir, "tasks")
    for task_id in sorted(os.listdir(tasks_root)):
        row = gold.get(task_id, {})
        tt = row.get("task_type")
        repo = os.path.join(dataset_dir, row.get("repo", "repo"))
        reader = RepoReader(repo)
        tdir = os.path.join(tasks_root, task_id)
        for model in sorted(os.listdir(tdir)):
            for mode in sorted(os.listdir(os.path.join(tdir, model))):
                cell = os.path.join(tdir, model, mode)
                answer = json.load(open(os.path.join(cell, "answer.json"))).get(
                    "answer", {}
                )
                raw = grade_task(tt, answer, row.get("gold", {}))
                cost = {}
                cpath = os.path.join(cell, "costs.json")
                if os.path.exists(cpath):
                    cost = json.load(open(cpath))
                m = {"task_type": tt, "raw": raw, "cost": cost}
                cert_path = os.path.join(cell, "certificate.json")
                if os.path.exists(cert_path):
                    cq = certificate_quality(json.load(open(cert_path)), reader)
                    per = aggregate([m])
                    per["certificate_quality"] = cq
                else:
                    per = aggregate([m])
                with open(os.path.join(cell, "metrics.json"), "w") as fh:
                    json.dump(per, fh, indent=2, sort_keys=True)
                m.update(model=model, mode=mode, correct=raw.get("correct"))
                cells.append(m)
    # per (model, mode) summary
    summary = {}
    keys = sorted({(c["model"], c["mode"]) for c in cells})
    for model, mode in keys:
        subset = [c for c in cells if c["model"] == model and c["mode"] == mode]
        summary[f"{model}/{mode}"] = aggregate(subset)
    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def _self_test():
    import tempfile

    # patch_equiv: both correct
    a = grade_task("patch_equiv", {"equivalent": True}, {"equivalent": True})
    b = grade_task("patch_equiv", {"value": "no"}, {"equivalent": False})
    assert a["correct"] and b["correct"], (a, b)
    agg = aggregate(
        [{"task_type": "patch_equiv", "raw": a}, {"task_type": "patch_equiv", "raw": b}]
    )
    assert agg["patch_equiv"]["accuracy"] == 1.0
    # a false-equivalent (predict equiv, gold not)
    c = grade_task("patch_equiv", {"equivalent": True}, {"equivalent": False})
    agg2 = aggregate([{"task_type": "patch_equiv", "raw": c}])
    assert agg2["patch_equiv"]["false_equivalent_rate"] == 1.0
    # fault localization: gold at rank 1
    fa = grade_task(
        "fault_localization",
        {"ranked_locations": [{"file": "calc.py", "line_start": 13, "line_end": 18}]},
        {"file": "calc.py", "line_start": 13, "line_end": 18},
    )
    assert fa["hits"] == {1: True, 3: True, 5: True}
    # miss
    fm = grade_task(
        "fault_localization",
        {"ranked_locations": [{"file": "strutil.py", "line_start": 1, "line_end": 2}]},
        {"file": "calc.py", "line_start": 13, "line_end": 18},
    )
    assert fm["hits"][1] is False
    # code_qa exact + substring
    assert grade_task("code_qa", {"answer": "hi"}, {"answer": "hi"})["correct"]
    assert grade_task("code_qa", {"value": "returns hi"}, {"answer": "hi"})["correct"]
    # certificate validity against the smoke repo
    repo = os.path.join(_ROOT, "datasets", "fixtures", "smoke", "repo")
    reader = RepoReader(repo)
    good = certificate_quality(
        {
            "premises": [
                {
                    "id": "P1",
                    "claim": "x",
                    "evidence": [
                        {
                            "file": "calc.py",
                            "line_start": 1,
                            "line_end": 3,
                            "source": "repo_read",
                        }
                    ],
                }
            ],
            "traces": ["t"],
        },
        reader,
    )
    assert good["cited_evidence_validity"] == 1.0
    bad = certificate_quality(
        {
            "premises": [
                {
                    "id": "P1",
                    "claim": "x",
                    "evidence": [{"file": "nope.py", "source": "ast"}],
                }
            ]
        },
        reader,
    )
    assert bad["cited_evidence_validity"] == 0.0 and bad["missing_trace_count"] == 1

    # end-to-end: build manifest, run (mock), score the tree
    sys.path.insert(0, os.path.join(_ROOT, "datasets"))
    sys.path.insert(0, _HERE)
    from manifest_builder import load_dataset, build_manifest
    from runner import run_manifest

    ds_dir = os.path.join(_ROOT, "datasets", "fixtures", "smoke")
    ds = load_dataset(ds_dir)
    flags = {
        "CONFIG_CODE_REASON_TASK_PATCH_EQUIV": True,
        "CONFIG_CODE_REASON_TASK_FAULT_LOCALIZATION": True,
        "CONFIG_CODE_REASON_TASK_CODE_QA": True,
        "CONFIG_CODE_REASON_MODEL_OPUS_4_5": True,
        "CONFIG_CODE_REASON_MODE_STANDARD": True,
        "CONFIG_CODE_REASON_MODE_SEMIFORMAL": True,
    }
    rows = build_manifest(ds, flags)
    rd = tempfile.mkdtemp()
    run_manifest(rows, flags, rd, ds_dir, repo_root=_ROOT)
    summary = score_run(rd, ds_dir, rows)
    assert "claude-opus-4-5/standard" in summary
    assert os.path.exists(os.path.join(rd, "metrics_summary.json"))
    # every cell has a schema-valid metrics.json
    for root, _, fs in os.walk(rd):
        if "metrics.json" in fs:
            _validate(json.load(open(os.path.join(root, "metrics.json"))), "metrics")
    js = "with-schema" if _HAVE_JSONSCHEMA else "no-jsonschema (soft)"
    print(f"[metrics] self-test PASS ({js}): grading + cert validity + score_run")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run-dir")
    ap.add_argument("--dataset-dir")
    ap.add_argument("--manifest")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not (args.run_dir and args.dataset_dir and args.manifest):
        ap.error("--run-dir, --dataset-dir, --manifest required (or --self-test)")
    rows = [json.loads(l) for l in open(args.manifest)]
    summary = score_run(args.run_dir, args.dataset_dir, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
