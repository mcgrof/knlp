#!/usr/bin/env python3
"""Option-likelihood diagnostic — does the cartridge CONTAIN the answer, separate
from whether it can finish a long reasoning trace?

The generative eval couples two things: (1) whether the cartridge encodes the
discriminating fact, and (2) whether the model can complete up to ~2048 tokens of
thinking and emit a clean <answer> tag (30% cap-hit says it often can't). This
diagnostic isolates (1): with NO thinking, it scores each of the five option
texts by the cartridge-conditioned mean log-likelihood of a
"<answer> {option} </answer>" continuation and picks the argmax. No generation,
no reasoning trace.

  * high option-logit accuracy (>> strict-generative ~0.48) -> the cartridge HAS
    the information; the generative gap is a reasoning/decoding-length problem.
  * option-logit accuracy also ~0.48 -> the cartridge genuinely lacks the
    discriminating facts; it is an induction/write problem.

Not the paper metric -- a diagnostic. Env: CART_DIR, PATIENTS, MAX_Q, DEVICE,
SINK_MAX, OUT_JSON.
"""

import os
import sys
import json
import statistics

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/home/mcgrof/cas_out")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import torch
from transformers import AutoTokenizer

import cas_eval_table15 as E
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset

CART_DIR = os.environ["CART_DIR"]
PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
MAX_Q = int(os.environ.get("MAX_Q", "20"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))
MODEL = E.MODEL
OUT_JSON = os.environ.get("OUT_JSON", "/home/mcgrof/cas_out/eval_optlogit.json")


def load_cart_kv(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    fk = ck.get("frozen_keys") or []
    nfrozen = t(fk[0]).shape[2] if fk else 0
    use_frozen = 0 < nfrozen <= SINK_MAX

    def cat(fro, tra):
        tt = [t(p) for p in tra]
        if fro and use_frozen:
            ff = [t(p) for p in fro]
            return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
        return tt

    return (
        cat(ck.get("frozen_keys"), ck["trainable_keys"]),
        cat(ck.get("frozen_values"), ck["trainable_values"]),
    )


def main():
    di = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
    torch.cuda.set_device(di)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
    model.eval()
    ac = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )

    def make_cache(path):
        k, v = load_cart_kv(path)
        return TrainableCache(config=ac, init_keys=k, init_values=v).to(DEVICE)

    @torch.no_grad()
    def option_logprob(cache, prefix_ids, opt_ids):
        """Mean log-prob of opt_ids as a continuation of prefix_ids, conditioned
        on the cartridge cache."""
        full = torch.cat([prefix_ids, opt_ids]).to(DEVICE)
        cache.clear()
        pos = torch.arange(full.shape[0], device=DEVICE)
        sids = torch.zeros(full.shape[0], dtype=torch.long, device=DEVICE)
        o = model(
            input_ids=full,
            seq_ids=sids,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            mode="generate",
        )
        cache.clear()
        logits = o.logits[0].float()  # [seq, vocab]
        lp = torch.log_softmax(logits, dim=-1)
        n_pre = prefix_ids.shape[0]
        total = 0.0
        for j in range(opt_ids.shape[0]):
            # token at absolute position n_pre+j is predicted by logits[n_pre+j-1]
            tid = int(opt_ids[j])
            total += float(lp[n_pre + j - 1, tid])
        return total / max(opt_ids.shape[0], 1)

    patients = load_longhealth_dataset(PATIENTS)
    present = [
        p
        for p in patients
        if os.path.exists(os.path.join(CART_DIR, f"{p.patient_id}.pt"))
    ]
    records = []
    for patient in present:
        cache = make_cache(os.path.join(CART_DIR, f"{patient.patient_id}.pt"))
        for q in patient.questions[:MAX_Q]:
            # prompt WITHOUT thinking; assistant begins the answer tag
            prefix = tok.apply_chat_template(
                [
                    {"role": "system", "content": E.SYSTEM_PROMPT},
                    {"role": "user", "content": E.user_prompt(q)},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            ).flatten()
            tag_open = tok("<answer> ", add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].flatten()
            prefix = torch.cat([prefix, tag_open])
            options = [q.answer_a, q.answer_b, q.answer_c, q.answer_d, q.answer_e]
            scores = []
            for opt in options:
                opt_ids = tok(
                    opt + " </answer>", add_special_tokens=False, return_tensors="pt"
                )["input_ids"].flatten()
                scores.append(option_logprob(cache, prefix, opt_ids))
            pred = int(max(range(5), key=lambda i: scores[i]))
            ci = E.correct_index(q)
            records.append(
                {
                    "patient": patient.patient_id,
                    "qid": q.question_id,
                    "pred": pred,
                    "correct_idx": ci,
                    "hit": int(pred == ci),
                }
            )
        del cache
        torch.cuda.empty_cache()
        print(f"  {patient.patient_id} done", flush=True)

    acc = sum(r["hit"] for r in records) / max(len(records), 1)
    per_patient = {}
    for p in PATIENTS:
        pr = [r["hit"] for r in records if r["patient"] == p]
        if pr:
            per_patient[p] = round(statistics.mean(pr), 3)
    out = {
        "cart_dir": CART_DIR,
        "n": len(records),
        "option_logit_acc": round(acc, 4),
        "per_patient": per_patient,
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"OPTLOGIT_ACC={acc:.4f}  n={len(records)}  per_patient={per_patient}")
    print(f"CAS_OPTLOGIT_DONE -> {OUT_JSON}")


if __name__ == "__main__":
    main()
