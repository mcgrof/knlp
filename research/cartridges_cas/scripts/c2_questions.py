#!/usr/bin/env python3
"""Freeze the question set the receiver-initializer pilot will be scored on.

The pilot compares three ways of initializing a trainable prefix for a frozen
receiver, and a comparison is only as trustworthy as the thing it is measured
against. So the questions are fixed once, here, before any arm exists: their
identities, their option texts and which option is the recorded answer, all
hashed into one number that every later artifact binds to.

Three problems this closes, each of which has already cost this program a
result.

The benchmark is read from a file and hashed, never from a moving reference.
An upstream edit would renumber questions under a recorded id list with no
error anywhere.

The five option texts are written out verbatim. The previous screen recorded
only the generated text and whether it was right, so its rows cannot be
re-scored offline -- a later reviewer who wants to check the scorer against the
raw generations has nothing to check it with. Options in the frozen set make
every row re-scorable forever.

And the answer is resolved to an option *index* here rather than at scoring
time, with a hard failure unless exactly one option matches. The recorded
answer in this benchmark is option *text*, not a letter, so a question whose
answer matches two options or none cannot be scored at all, and finding that
out during a paid run is finding it out too late.

The selection itself reuses ``select_two_per_patient`` from the source-utility
screen rather than reimplementing it, so the two stages cannot silently drift
apart. For a patient with exactly as many scorable questions as requested that
function is an identity: it takes all of them and returns them in question-number
order. That is the case here, and it is asserted rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from research.cartridges_cas.scripts.c1_source_utility import (  # noqa: E402
    select_two_per_patient,
)
from research.cartridges_cas.scripts.score_longhealth import norm  # noqa: E402

TAG = "c2-receiver-initializer-v1"
OPTION_SLOTS = "abcde"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha256(obj):
    """Hash a structure the way two different machines will agree on."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_patients_plain(path, patient_ids, want_sha256=""):
    """The pinned benchmark as plain objects, without the library.

    ``load_patients_pinned`` in the source-utility screen returns the library's
    ``LongHealthPatient``, which means importing the library, which is not
    installed everywhere this needs to run -- and this stage does no modelling,
    so it should not need a modelling stack. The attribute surface
    ``select_two_per_patient`` actually touches is small and is reproduced
    exactly: ``patient_id``, ``questions``, and ``question_id`` plus
    ``answer_a`` through ``answer_e`` and ``correct`` on each question.
    """
    got = sha256_file(path)
    if want_sha256 and got != want_sha256:
        raise SystemExit(f"INVALID: benchmark hashes to {got}, expected {want_sha256}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for pid, row in sorted(data.items()):
        if patient_ids is not None and pid not in patient_ids:
            continue
        questions = [
            SimpleNamespace(question_id=f"{pid}_{q['No']}", **q)
            for q in row["questions"]
        ]
        out.append(SimpleNamespace(patient_id=pid, questions=questions, raw=row))
    if patient_ids is not None:
        missing = sorted(set(patient_ids) - {p.patient_id for p in out})
        if missing:
            raise SystemExit(f"INVALID: benchmark has no {', '.join(missing)}")
    return got, out


def resolve_gold(q):
    """Which option slot the recorded answer is, or a hard failure.

    Scoring compares generated text against option text, so an answer that
    matches two options is unscorable and an answer that matches none means the
    question and its answer key have drifted apart. Neither is something to
    discover halfway through a run.
    """
    options = [getattr(q, f"answer_{c}") for c in OPTION_SLOTS]
    normalised = [norm(o) for o in options]
    target = norm(q.correct)
    hits = [i for i, o in enumerate(normalised) if o == target]
    if len(hits) != 1:
        raise SystemExit(
            f"INVALID: {q.question_id} has {len(hits)} options matching its "
            f"recorded answer, so it cannot be scored"
        )
    return options, normalised, hits[0]


def freeze(benchmark, patient, want_sha256="", tag=TAG, per_patient=20):
    got, patients = load_patients_plain(benchmark, [patient], want_sha256=want_sha256)
    picked = select_two_per_patient(patients, tag=tag, per_patient=per_patient)

    available = len(patients[0].questions)
    if len(picked) != per_patient:
        raise SystemExit(
            f"INVALID: asked for {per_patient} questions and the selector "
            f"returned {len(picked)}"
        )

    questions = []
    for q in picked:
        options, normalised, gold = resolve_gold(q)
        questions.append(
            {
                "question_id": q.question_id,
                "number": int(q.No),
                "question": q.question,
                "options": options,
                "options_normalised": normalised,
                "gold_index": gold,
                "gold_text": q.correct,
            }
        )

    return {
        "contract": "c2_question_set_v1",
        "tag": tag,
        "patient": patient,
        "benchmark_path": os.path.abspath(benchmark),
        "benchmark_sha256": got,
        "questions_available": available,
        "questions_selected": len(questions),
        "selection_is_exhaustive": len(questions) == available,
        "selection_rule": (
            "select_two_per_patient at per_patient equal to the number of "
            "scorable questions, which returns all of them in question-number "
            "order; the hash ranking is inert at that size and is recorded for "
            "provenance rather than because it chose anything"
        ),
        "questions": questions,
        "question_set_sha256": canonical_sha256(questions),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, help="the pinned benchmark json")
    ap.add_argument(
        "--dataset-sha256",
        default="",
        help="refuse to proceed unless the benchmark hashes to this",
    )
    ap.add_argument("--patient", default="patient_02")
    ap.add_argument("--per-patient", type=int, default=20)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--out", required=True, help="where to write QUESTIONS.json")
    args = ap.parse_args()

    frozen = freeze(
        args.benchmark,
        args.patient,
        want_sha256=args.dataset_sha256,
        tag=args.tag,
        per_patient=args.per_patient,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print(f"{frozen['patient']}: {frozen['questions_selected']} questions frozen")
    print(f"  benchmark      {frozen['benchmark_sha256']}")
    print(f"  question set   {frozen['question_set_sha256']}")
    print(
        f"  ids            {frozen['questions'][0]['question_id']} .. "
        f"{frozen['questions'][-1]['question_id']}"
    )
    if not frozen["selection_is_exhaustive"]:
        print(
            f"  NOTE: {frozen['questions_available']} scorable questions exist; "
            f"{frozen['questions_selected']} were taken, so the hash ranking chose"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
