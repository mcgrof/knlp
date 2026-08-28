#!/usr/bin/env python3
"""Floor and ceiling for the document-advantage metric.

A cartridge's document advantage -- its forced-choice margin on its own
patient's questions minus its margin on other patients' -- is only
interpretable against two references measured the same way.

The floor is the base model with no context at all.  Its advantage
should be near zero: with no document, the target patient's questions
are just questions.  Anything much above zero means the metric is
picking up something other than document knowledge, and every cartridge
number has to be discounted by it.

The ceiling is the base model with the whole record in the prompt.
That is the information actually being compressed, so it bounds what
any cartridge for this document could achieve, and the ratio between a
cartridge and this number says how much of the document a cartridge of
that size captures.

Uses the stock model rather than the cache-surgery path: no cartridge
is involved in either reference, and an empty trainable cache is not a
valid object to hand the flex model.

Env: MODEL, PATIENT, LONGHEALTH_JSON, RECORD, CONTROL_PATIENTS, OUT_JSON.
"""

import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PATIENT = os.environ.get("PATIENT", "patient_02")
LONGHEALTH_JSON = os.environ["LONGHEALTH_JSON"]
RECORD = os.environ["RECORD"]
CONTROL_PATIENTS = int(os.environ.get("CONTROL_PATIENTS", "6"))
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/choice_reference.json")
DEVICE = "cuda"
LETTERS = "ABCDE"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .to(DEVICE)
    .eval()
)
model.requires_grad_(False)

LETTER_IDS = {}
for L in LETTERS:
    enc = tok(L, add_special_tokens=False).input_ids
    assert len(enc) == 1
    LETTER_IDS[L] = enc[0]

data = json.loads(Path(LONGHEALTH_JSON).read_text())
record = Path(RECORD).read_text()


def questions_for(pid):
    out = []
    for i, q in enumerate(data[pid]["questions"]):
        amap = {L: q["answer_" + L.lower()] for L in LETTERS}
        correct = next((L for L, v in amap.items() if v == q["correct"]), None)
        if correct is None:
            continue
        out.append(
            dict(
                qid=f"{pid}_{i}",
                patient=pid,
                text=q["question"],
                amap=amap,
                correct=correct,
            )
        )
    return out


target_qs = questions_for(PATIENT)
others = [p for p in sorted(data) if p != PATIENT][:CONTROL_PATIENTS]
control_qs = [q for p in others for q in questions_for(p)]


def prompt_for(q, with_record):
    body = (
        f"Question: {q['text']}\n"
        + "\n".join(f"{L}) {q['amap'][L]}" for L in LETTERS)
        + "\n\nAnswer with ONLY the letter (A, B, C, D, or E). Do not explain."
    )
    return (record + "\n\n" + body) if with_record else body


@torch.no_grad()
def score(qs, with_record):
    rows = []
    for q in qs:
        ids = (
            tok.apply_chat_template(
                [{"role": "user", "content": prompt_for(q, with_record)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            .to(DEVICE)
            .flatten()
            .unsqueeze(0)
        )
        out = model(input_ids=ids)
        lp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
        lps = {L: float(lp[LETTER_IDS[L]]) for L in LETTERS}
        c = q["correct"]
        best_wrong = max(v for L, v in lps.items() if L != c)
        rows.append(
            dict(
                qid=q["qid"],
                correct=c,
                pred=max(lps, key=lps.get),
                hit=max(lps, key=lps.get) == c,
                margin=lps[c] - best_wrong,
            )
        )
    return rows


def summarize(rows):
    return dict(
        n=len(rows),
        acc=sum(r["hit"] for r in rows) / len(rows),
        margin_mean=statistics.fmean(r["margin"] for r in rows),
    )


def main():
    report = dict(model=MODEL, patient=PATIENT, control_patients=others, conditions={})
    for name, with_record in (
        ("no_context_floor", False),
        ("full_context_ceiling", True),
    ):
        t_rows = score(target_qs, with_record)
        c_rows = score(control_qs, with_record)
        t, c = summarize(t_rows), summarize(c_rows)
        report["conditions"][name] = dict(
            target=t,
            control=c,
            advantage=t["margin_mean"] - c["margin_mean"],
            target_rows=t_rows,
            control_rows=c_rows,
        )
        print(
            f"[ref] {name}: target acc={t['acc']:.3f} margin={t['margin_mean']:+.3f} "
            f"| control margin={c['margin_mean']:+.3f} "
            f"| advantage={report['conditions'][name]['advantage']:+.3f}",
            flush=True,
        )
        Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(report, f, indent=1)
    print(f"REFERENCE_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()
