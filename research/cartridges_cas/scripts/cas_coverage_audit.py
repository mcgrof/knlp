#!/usr/bin/env python3
"""Coverage audit — is the CAS cartridge quality gap a QUESTION-COVERAGE problem?

Every synthesis lever we tried (thinking-on, more steps, more data) plateaued at
~0.55 while the full-document cart reaches 0.86. That pattern says the induced
self-study data may simply never ASK about the facts the eval tests. This audit
checks it directly, per the reviewers' cheap-first step: for each eval question,
measure how much of its CORRECT answer's salient content (numbers, dates, proper
nouns, medical terms) appears anywhere in that patient's self-study corpus
(questions + teacher answers), then split by whether the trained cart got the
question right or wrong.

  * wrong questions systematically LOWER coverage than right ones -> the bottleneck
    is coverage; the fix is coverage-directed question generation.
  * wrong questions have coverage COMPARABLE to right ones -> the facts were studied
    but not written into the cart; the problem is training/writing, not coverage.

Env: PATIENTS, SELF_STUDY_DIR (self-study parquets), EVAL_JSON (a cas_eval_matched
records file giving per-(patient,qid) correctness), CARTRIDGES_DIR.
"""

import os
import re
import sys
import glob
import json
import statistics

os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import pandas as pd
from cartridges.data.longhealth.utils import load_longhealth_dataset

PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
SELF_STUDY_DIR = os.environ.get(
    "SELF_STUDY_DIR", os.path.expanduser("~/cas_out/synth_faithful")
)
EVAL_JSON = os.environ.get(
    "EVAL_JSON", os.path.expanduser("~/cas_out/eval_matched_faithful.json")
)

STOP = set(
    "the a an and or of to in on for with was were is are be been has had have "
    "at by from as that this his her their they she he it patient mg ml".split()
)


def salient_tokens(text):
    """Facts worth checking: numbers/dates, and content words >3 chars."""
    toks = set()
    for m in re.findall(r"\d[\d.,/:-]*\d|\d", text):  # numbers, dates, doses
        toks.add(m.lower())
    for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", text):
        wl = w.lower()
        if wl not in STOP:
            toks.add(wl)
    return toks


def corpus_for(patient_id):
    """All self-study text (questions + teacher answers) for a patient, lowercased."""
    pstr = patient_id.replace("patient_", "p")
    files = glob.glob(
        f"{SELF_STUDY_DIR}/*/synth_decoupled_14bq_8ba_{pstr}_n*/artifact/dataset.parquet"
    )
    assert files, f"no self-study parquet for {patient_id} under {SELF_STUDY_DIR}"
    df = pd.read_parquet(sorted(files)[-1])
    chunks = []
    for _, row in df.iterrows():
        msgs = row.get("messages")
        if msgs is not None:
            for m in msgs:
                c = m.get("content") if isinstance(m, dict) else None
                if isinstance(c, str):
                    chunks.append(c)
        sp = row.get("system_prompt")
        if isinstance(sp, str):
            chunks.append(sp)
    return " ".join(chunks).lower()


def main():
    eval_recs = json.load(open(EVAL_JSON))["records"]
    correct_by = {
        (r["patient"], r["qid"]): r.get("tail", r.get("correct")) for r in eval_recs
    }

    patients = load_longhealth_dataset(PATIENTS)
    rows = []
    for p in patients:
        corpus = corpus_for(p.patient_id)
        for q in p.questions[:20]:
            key = (p.patient_id, q.question_id)
            if key not in correct_by:
                continue
            toks = salient_tokens(q.correct)
            if not toks:
                continue
            cov = sum(1 for t in toks if t in corpus) / len(toks)
            rows.append(
                (p.patient_id, q.question_id, int(correct_by[key]), cov, len(toks))
            )

    right = [r[3] for r in rows if r[2] == 1]
    wrong = [r[3] for r in rows if r[2] == 0]
    print(f"self-study: {SELF_STUDY_DIR}")
    print(f"eval: {EVAL_JSON}   questions audited: {len(rows)}")
    print(
        f"  RIGHT questions: n={len(right)}  mean answer-fact coverage={statistics.mean(right):.3f}"
        if right
        else "  no right"
    )
    print(
        f"  WRONG questions: n={len(wrong)}  mean answer-fact coverage={statistics.mean(wrong):.3f}"
        if wrong
        else "  no wrong"
    )
    if right and wrong:
        print(
            f"  delta (right - wrong) = {statistics.mean(right) - statistics.mean(wrong):+.3f}"
        )
    lowcov_wrong = sum(1 for r in rows if r[2] == 0 and r[3] < 0.5)
    print(
        f"  wrong questions with <50% answer-fact coverage: {lowcov_wrong}/{len(wrong)}"
    )
    # per-patient
    print("  per-patient (right_cov / wrong_cov):")
    for p in PATIENTS:
        pr = [r[3] for r in rows if r[0] == p and r[2] == 1]
        pw = [r[3] for r in rows if r[0] == p and r[2] == 0]
        rc = f"{statistics.mean(pr):.2f}" if pr else "-"
        wc = f"{statistics.mean(pw):.2f}" if pw else "-"
        print(f"    {p}: {rc} / {wc}  (n_right={len(pr)} n_wrong={len(pw)})")


if __name__ == "__main__":
    main()
