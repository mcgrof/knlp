#!/usr/bin/env python3
"""Document-utility audit (ChatGPT-Pro's #1 synthetic-data strategy, 2026-07-29).

Coverage told us the needed facts APPEAR in the corpus. It did not tell us whether
each training example actually REQUIRES the document to answer -- many self-study
prompts (esp. summarization / use_case) produce polished, generic clinical prose
the model could write with no patient record at all, so the cartridge spends
capacity imitating style instead of document-specific bindings.

Measure per prompt the document utility over the answer-bearing tokens:
  U_doc = mean_t [ log p(a_t | doc, q, a_<t) - log p(a_t | q, a_<t) ]
via the STANDARD Qwen3-8B (doc in the prompt vs not) -- no cartridge path needed.
High U_doc = the document genuinely determines the answer; U_doc ~ 0 = generic.
Report the distribution and the fraction of low-utility (filler) examples.

Env: PATIENTS, SYNTH_DIR, N (prompts sampled per patient, default 80), DEVICE,
MODEL, SEED.
"""

import os
import sys
import glob
import random
import statistics

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/home/mcgrof/cas_out")
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset

PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
SYNTH_DIR = os.environ.get("SYNTH_DIR", "/home/mcgrof/cas_out/synth_diverse")
N = int(os.environ.get("N", "80"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
SEED = int(os.environ.get("SEED", "0"))
MAX_A = int(os.environ.get("MAX_A", "700"))  # cap answer tokens scored

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(
    DEVICE
)
model.eval()


@torch.no_grad()
def answer_logprob(context_user, answer):
    """mean log p(answer | context_user) over answer tokens, standard model."""
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": context_user}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).flatten()
    ans = tok(answer, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].flatten()[:MAX_A]
    full = torch.cat([prompt, ans]).unsqueeze(0).to(DEVICE)
    logits = model(full).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    s = 0.0
    for j in range(prompt.shape[0], full.shape[1]):
        s += float(lp[j - 1, int(full[0, j])])
    return s / max(ans.shape[0], 1)


def main():
    rng = random.Random(SEED)
    patients = {p.patient_id: p for p in load_longhealth_dataset(PATIENTS)}
    allU = []
    for pid in PATIENTS:
        pstr = pid.replace("patient_", "p")
        g = glob.glob(
            f"{SYNTH_DIR}/*/synth_*_14bq_8ba_{pstr}_n*/artifact/dataset.parquet"
        )
        if not g or pid not in patients:
            print(f"  {pid}: skip")
            continue
        df = pd.read_parquet(sorted(g)[-1])
        doc = "\n\n".join(patients[pid].texts.values())
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        idxs = idxs[:N]
        us = []
        for i in idxs:
            msgs = df.iloc[i]["messages"]
            q = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            a = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
            if not q or not a:
                continue
            # strip the teacher's <think> so we score the ACTUAL answer content
            a_ans = a.split("</think>")[-1].strip() if "</think>" in a else a
            if not a_ans:
                continue
            pd_ = answer_logprob(f"{doc}\n\n{q}", a_ans)
            p0_ = answer_logprob(q, a_ans)
            us.append(pd_ - p0_)
        allU += us
        m = statistics.mean(us)
        low = sum(1 for u in us if u < 0.05) / max(len(us), 1)
        print(
            f"  {pid}: n={len(us)} U_doc mean={m:+.3f} median={statistics.median(us):+.3f} "
            f"frac_low(<0.05)={low:.1%}",
            flush=True,
        )
    print(
        f"OVERALL U_doc mean={statistics.mean(allU):+.3f}  "
        f"frac_low(<0.05)={sum(1 for u in allU if u<0.05)/max(len(allU),1):.1%}  n={len(allU)}"
    )
    print(
        "(U_doc ~ 0 => the document doesn't change the answer = filler; high => doc-determined)"
    )


if __name__ == "__main__":
    main()
