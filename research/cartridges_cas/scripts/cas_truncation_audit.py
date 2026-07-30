#!/usr/bin/env python3
"""Training-truncation audit (ChatGPT-Pro's #1 suspect, 2026-07-29).

cas_train_isolated packs training sequences to packed_seq_length=2048 with
packing_mode="truncate", but the answer teacher generates up to 2048 completion
tokens. So a long thinking answer + the question + chat framing cannot fit in a
2048-token training sequence, and the packer drops the overflow -- potentially
the end of the reasoning, </think>, the FINAL ANSWER, and EOS. Training a cart to
imitate reasoning that never reaches an answer would exactly produce the observed
serving symptom (carts that begin reasoning and wander past the eval budget
without answering).

This measures, per self-study example (no GPU, tokenization only): the full
serialized [user, assistant] length, whether </think> and the final <answer> tag
survive before the 2048 boundary, and whether the teacher itself hit its 2048
completion cap (an unfinished thought). Reports the fraction of examples and of
teacher answer tokens that a 2048 pack would drop.

Env: PATIENTS, SYNTH_DIRS (comma-sep self-study roots to audit), BOUND (2048).
"""

import os
import sys
import glob

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
SYNTH_DIRS = os.environ.get(
    "SYNTH_DIRS",
    "/home/mcgrof/cas_out/synth_diverse,/home/mcgrof/cas_out/synth_faithful",
).split(",")
BOUND = int(os.environ.get("BOUND", "2048"))
MODEL = "Qwen/Qwen3-8B"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def audit_dir(root):
    files = []
    for p in PATIENTS:
        pstr = p.replace("patient_", "p")
        g = glob.glob(f"{root}/*/synth_*_14bq_8ba_{pstr}_n*/artifact/dataset.parquet")
        if g:
            files.append(sorted(g)[-1])
    if not files:
        print(f"  {root}: no parquets")
        return
    tot = 0
    over_bound = 0
    think_dropped = 0
    answer_dropped = 0
    teacher_capped = 0
    lens = []
    ans_tok_total = 0
    ans_tok_dropped = 0
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            msgs = row["messages"]
            uq = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            aa = next(
                (m["content"] for m in msgs if m.get("role") == "assistant"), None
            )
            if not uq or not aa:
                continue
            tot += 1
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": uq}],
                tokenize=True,
                add_generation_prompt=True,
            )
            full = tok.apply_chat_template(
                [{"role": "user", "content": uq}, {"role": "assistant", "content": aa}],
                tokenize=True,
                add_generation_prompt=False,
            )
            L = len(full)
            lens.append(L)
            ans_len = L - len(prompt)
            ans_tok_total += ans_len
            if L > BOUND:
                over_bound += 1
                ans_tok_dropped += L - BOUND

            # find </think> and final <answer> char positions -> token position
            # via re-tokenizing the assistant text prefix up to each marker.
            def tok_pos_of(marker):
                idx = aa.rfind(marker)
                if idx < 0:
                    return None
                pre = aa[: idx + len(marker)]
                return len(prompt) + len(
                    tok(pre, add_special_tokens=False)["input_ids"]
                )

            tp = tok_pos_of("</think>")
            ap = tok_pos_of("</answer>")
            if tp is None or tp > BOUND:
                think_dropped += 1
            if ap is None or ap > BOUND:
                answer_dropped += 1
            # teacher-cap heuristic: no </think> or no </answer> and answer near 2048
            if ans_len >= 2000 and (tp is None or ap is None):
                teacher_capped += 1
    lens = np.array(lens)
    print(f"  {root}")
    print(
        f"    examples={tot}  seq-len mean={lens.mean():.0f} p50={np.percentile(lens,50):.0f} "
        f"p90={np.percentile(lens,90):.0f} p99={np.percentile(lens,99):.0f} max={lens.max()}"
    )
    print(
        f"    seq > {BOUND} (answer tail truncated):      {over_bound}/{tot} = {over_bound/tot:.1%}"
    )
    print(
        f"    </think> dropped (missing or > {BOUND}):     {think_dropped}/{tot} = {think_dropped/tot:.1%}"
    )
    print(
        f"    </answer> dropped (missing or > {BOUND}):    {answer_dropped}/{tot} = {answer_dropped/tot:.1%}"
    )
    print(
        f"    teacher likely hit 2048 cap (unfinished):   {teacher_capped}/{tot} = {teacher_capped/tot:.1%}"
    )
    print(
        f"    teacher answer TOKENS dropped by pack:      {ans_tok_dropped}/{ans_tok_total} = {ans_tok_dropped/max(ans_tok_total,1):.1%}"
    )


def main():
    for d in SYNTH_DIRS:
        audit_dir(d.strip())


if __name__ == "__main__":
    main()
