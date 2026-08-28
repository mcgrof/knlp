#!/usr/bin/env python3
"""Per-question forced-choice scoring for cartridge checkpoints.

The screen compared arms on accuracy over twenty questions, which is a
count with 0.05 granularity and binomial noise wide enough to swallow
every difference it was asked about.  Every arm read between 0.450 and
0.550 and the lane concluded knowledge was flat.  That conclusion is
not safe: the same twenty questions also yield a continuous margin per
question -- the correct letter's log probability minus the best wrong
one's -- and a paired comparison of those margins is far sharper than
comparing two counts.  The earlier evaluator computed the margin and
then threw the per-question values away, keeping only the mean.

This scorer keeps them, so arms can be compared pairwise on the same
questions.  It also adds the control the screen lacked: questions
belonging to OTHER patients, which the target patient's record cannot
answer.  A cartridge that encodes document knowledge should move its
own patient's margins and leave the others alone; one that merely
learned to take multiple-choice tests should move both.  Without that
contrast an improvement on the target patient is uninterpretable.

Scoring is one forward pass per question with no generation, so the
whole sweep is minutes even on one consumer GPU, and it is unaffected
by the generation-length and parsing pathologies that dominate the
generation evaluation.

Env: MODEL, CARTS (comma list name=path), PATIENT, LONGHEALTH_JSON,
CONTROL_PATIENTS (count of other patients to sample, default 6),
OUT_JSON.
"""

import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/data/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/ca_choice")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CARTS = os.environ["CARTS"]
PATIENT = os.environ.get("PATIENT", "patient_02")
LONGHEALTH_JSON = os.environ["LONGHEALTH_JSON"]
CONTROL_PATIENTS = int(os.environ.get("CONTROL_PATIENTS", "6"))
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/ca_choice/choice.json")
DEVICE = "cuda"
LETTERS = "ABCDE"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)

LETTER_IDS = {}
for L in LETTERS:
    enc = tok(L, add_special_tokens=False).input_ids
    assert len(enc) == 1, f"letter {L} is not one token: {enc}"
    LETTER_IDS[L] = enc[0]


def load_cart(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    tk = [t(p) for p in ck["trainable_keys"]]
    tv = [t(p) for p in ck["trainable_values"]]
    fk = ck.get("frozen_keys") or []
    fv = ck.get("frozen_values") or []
    if fk:
        ik = [torch.cat([t(fk[i]), tk[i]], dim=2) for i in range(len(tk))]
        iv = [torch.cat([t(fv[i]), tv[i]], dim=2) for i in range(len(tv))]
        nfrozen = t(fk[0]).shape[2]
    else:
        ik, iv, nfrozen = tk, tv, 0
    return TrainableCache(
        config=ac, init_keys=ik, init_values=iv, num_frozen_tokens=nfrozen
    ).to(DEVICE)


data = json.loads(Path(LONGHEALTH_JSON).read_text())


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
print(
    f"[choice] target {PATIENT}: {len(target_qs)} questions; "
    f"control {others}: {len(control_qs)} questions",
    flush=True,
)


def prompt_for(q):
    return (
        f"Question: {q['text']}\n"
        + "\n".join(f"{L}) {q['amap'][L]}" for L in LETTERS)
        + "\n\nAnswer with ONLY the letter (A, B, C, D, or E). Do not explain."
    )


@torch.no_grad()
def score(cache, qs):
    rows = []
    for q in qs:
        ids = (
            tok.apply_chat_template(
                [{"role": "user", "content": prompt_for(q)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            .to(DEVICE)
            .flatten()
        )
        cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=ids,
                seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
                position_ids=torch.arange(ids.shape[0], device=DEVICE),
                use_cache=True,
                past_key_values=cache,
            )
        lp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
        lps = {L: float(lp[LETTER_IDS[L]]) for L in LETTERS}
        c = q["correct"]
        best_wrong = max(v for L, v in lps.items() if L != c)
        pred = max(lps, key=lps.get)
        rows.append(
            dict(
                qid=q["qid"],
                patient=q["patient"],
                correct=c,
                pred=pred,
                hit=pred == c,
                margin=lps[c] - best_wrong,
                lp_correct=lps[c],
                lps=lps,
            )
        )
    return rows


def summarize(rows):
    if not rows:
        return None
    m = [r["margin"] for r in rows]
    return dict(
        n=len(rows),
        acc=sum(r["hit"] for r in rows) / len(rows),
        margin_mean=statistics.fmean(m),
        margin_median=statistics.median(m),
    )


def main():
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    report = dict(model=MODEL, patient=PATIENT, control_patients=others, conditions={})
    for spec in CARTS.split(","):
        name, path = spec.split("=", 1)
        cache = load_cart(path)
        t_rows = score(cache, target_qs)
        c_rows = score(cache, control_qs)
        report["conditions"][name] = dict(
            target=summarize(t_rows),
            control=summarize(c_rows),
            target_rows=t_rows,
            control_rows=c_rows,
        )
        ts, cs = (
            report["conditions"][name]["target"],
            report["conditions"][name]["control"],
        )
        print(
            f"[choice] {name}: target acc={ts['acc']:.3f} margin={ts['margin_mean']:+.3f} "
            f"| control acc={cs['acc']:.3f} margin={cs['margin_mean']:+.3f}",
            flush=True,
        )
        with open(OUT_JSON, "w") as f:
            json.dump(report, f, indent=1)
        del cache
        torch.cuda.empty_cache()
    print(f"CHOICE_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()
