#!/usr/bin/env python3
"""CAS self-study phase 1: question generation with a decoupled M_Q.

Faithful to Cartridges-at-Scale (arXiv:2606.04557) Appendix I: a stronger,
cross-family question model (here Llama-3.1-70B, standing in for GPT-OSS-120B)
receives one clinical NOTE in its system prompt and emits a batch of diverse,
coverage-directed prompts as a JSON array. Four seed-prompt rounds feed training
(question / structuring / summarization / use_case; creative is excluded per the
paper's footnote 12). Documents are sampled proportional to length; for multi-
note records one note is drawn per prompt for fine-grained coverage.

Output: a JSON list of records {patient_id, note_id, round, note_context, prompt}
consumed by phase 2 (cas_synth_answer.py), which answers each with the Qwen3-8B
teacher and writes the distillation parquet.

Env: VLLM_URL (default http://localhost:8009/v1), MODEL (default llama70b-qgen),
PATIENTS, TARGET_PER_ROUND (prompts per round across the corpus, default 500 for
a pilot; paper is 10000), N_PER_CALL (default 20), CONCURRENCY, OUT_JSON, SEED.
"""

import os
import json
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/home/mcgrof/cas_out")

from openai import OpenAI
from cartridges.data.longhealth.utils import load_longhealth_dataset

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8009/v1")
MODEL = os.environ.get("MODEL", "llama70b-qgen")
PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
TARGET_PER_ROUND = int(os.environ.get("TARGET_PER_ROUND", "500"))
N_PER_CALL = int(os.environ.get("N_PER_CALL", "20"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
OUT_JSON = os.environ.get("OUT_JSON", "/home/mcgrof/cas_out/qgen/qgen_pilot.json")
SEED = int(os.environ.get("SEED", "0"))

# One clinical note in the question model's system prompt (Appendix H/I: the
# question model receives the document context in its system prompt).
NOTE_SYSTEM = (
    "You are studying a section of a patient's medical record.\n\n"
    "Patient: {name} (ID: {pid}), born {birthday}, diagnosis: {diagnosis}.\n"
    "Note {note_id}:\n<note>\n{note}\n</note>"
)

# Round instructions. The "question" round is verbatim from Appendix I; the other
# three follow the paper's one-line descriptions of each seed type.
ROUND_INSTR = {
    "question": (
        "Generate {n} diverse questions that test knowledge of the information "
        "in the note above. Each question should cover a different fact, detail, "
        "or aspect of the note. Vary the style: mix factual recall, comparison, "
        "reasoning, and detail-oriented questions. Include specific details "
        "(ids, names, titles, dates, numerical values, etc.) in each question so "
        "it is clear what you are asking about."
    ),
    "structuring": (
        "Generate {n} diverse requests to organize information from the note "
        "above into a structured format (e.g. JSON, YAML, a table, or a "
        "timeline). Each request should target a different set of facts, "
        "details, or aspects, and name the specific fields, dates, values, or "
        "entities to structure."
    ),
    "summarization": (
        "Generate {n} diverse requests to summarize specific sections or aspects "
        "of the note above. Each request should target a different section, "
        "topic, or detail (e.g. a specific diagnosis, treatment, lab result, or "
        "time period) and make clear what to summarize."
    ),
    "use_case": (
        "Generate {n} diverse practical, downstream-task requests that a "
        "clinician or system might make using the information in the note above "
        "(e.g. drafting a referral, checking a medication, planning follow-up). "
        "Each should reference specific facts, names, dates, or values from the "
        "note."
    ),
}
JSON_TAIL = (
    "\n\nOutput ONLY a JSON array of strings, e.g. "
    '["item 1", "item 2", ...]. No other text, no markdown fences, '
    "no explanation."
)

ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_array(text):
    """Extract a JSON array of strings; tolerate a stray fence or prose."""
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
    except Exception:  # noqa: BLE001
        pass
    m = ARRAY_RE.search(text)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
        except Exception:  # noqa: BLE001
            return []
    return []


def main():
    rng = random.Random(SEED)
    client = OpenAI(base_url=VLLM_URL, api_key="EMPTY", timeout=600)
    patients = load_longhealth_dataset(PATIENTS)

    # note pool with proportional-to-length sampling weights
    notes = []
    for p in patients:
        for note_id, text in p.texts.items():
            notes.append(
                {
                    "patient_id": p.patient_id,
                    "note_id": note_id,
                    "text": text,
                    "sys": NOTE_SYSTEM.format(
                        name=p.name,
                        pid=p.patient_id,
                        birthday=p.birthday,
                        diagnosis=p.diagnosis,
                        note_id=note_id,
                        note=text,
                    ),
                    "len": len(text),
                }
            )
    min_len = min(n["len"] for n in notes)
    weights = [n["len"] / min_len for n in notes]
    print(
        f"[qgen] {len(patients)} patients, {len(notes)} notes; "
        f"target {TARGET_PER_ROUND}/round x {len(ROUND_INSTR)} rounds",
        flush=True,
    )

    def one_call(job):
        rnd, note = job
        instr = ROUND_INSTR[rnd].format(n=N_PER_CALL) + JSON_TAIL
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": note["sys"]},
                    {"role": "user", "content": instr},
                ],
                temperature=0.6,
                top_p=0.95,
                max_completion_tokens=4096,
                extra_body={"top_k": 20},
            )
            items = parse_array(resp.choices[0].message.content or "")
            return rnd, note, items, None
        except Exception as e:  # noqa: BLE001
            return rnd, note, [], str(e)[:200]

    records = []
    t0 = time.time()
    for rnd in ROUND_INSTR:
        got = 0
        parse_fail = err = 0
        # enough calls to reach the target, each ~N_PER_CALL after parse loss
        n_calls = int(TARGET_PER_ROUND / (N_PER_CALL * 0.9)) + 4
        jobs = [
            (rnd, rng.choices(notes, weights=weights, k=1)[0]) for _ in range(n_calls)
        ]
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            for rnd_, note, items, e in ex.map(one_call, jobs):
                if e:
                    err += 1
                    continue
                if not items:
                    parse_fail += 1
                    continue
                for prompt in items:
                    records.append(
                        {
                            "patient_id": note["patient_id"],
                            "note_id": note["note_id"],
                            "round": rnd_,
                            "note_context": note["sys"],
                            "prompt": prompt,
                        }
                    )
                    got += 1
                if got >= TARGET_PER_ROUND:
                    break
        print(
            f"  [{rnd}] got {got} prompts (parse_fail={parse_fail} err={err}) "
            f"t={time.time() - t0:.0f}s",
            flush=True,
        )

    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    json.dump(records, open(OUT_JSON, "w"))
    by_patient = {}
    for r in records:
        by_patient[r["patient_id"]] = by_patient.get(r["patient_id"], 0) + 1
    print(f"CAS_QGEN_DONE {len(records)} prompts -> {OUT_JSON}")
    print("per-patient:", by_patient)


if __name__ == "__main__":
    main()
